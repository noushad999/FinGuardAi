"""Graph analytics over the transaction ecosystem (NetworkX).

Graph model
-----------
Nodes: customers (``C:``), devices (``D:``), merchants/agents (``M:``).
Edges:
* ``sends_money_to``  customer -> customer (P2P, weighted by amount)
* ``uses_device``     customer -- device
* ``pays_merchant``   customer -> merchant (merchant payments / cash-out at agents)

What we compute
---------------
* degree / weighted degree / PageRank / (sampled) betweenness on the P2P money-flow graph,
* Louvain communities on the P2P + shared-device graph (the relations that tie fraud rings together),
* device-sharing components (accounts linked through common handsets),
* an unsupervised, explainable **mule score** per customer,
* a **suspicious cluster** ranking of communities.

Mule/cluster quality is measured against the simulator's hidden ground truth
(``customer_ground_truth.is_mule``), which the algorithms never see.

Why no GNN? With a few thousand labelled mules at best, hand-crafted graph
features + gradient boosting are competitive, explainable and cheap. A GNN
(e.g. GraphSAGE) becomes worthwhile when relational patterns are too complex
for summary statistics *and* there are enough labels - listed as future work.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from common.config import Settings, get_settings
from common.logging import get_logger
from ml.common.bundle import sanitize, write_json
from ml.common.metrics import classification_report, precision_recall_at_k

log = get_logger(__name__)

BUSINESS_HUB_PREFIXES = ("BIL_", "TEL_", "BNK_")  # national billers/operators/banks: not informative


def build_graph(tx: pd.DataFrame) -> tuple[nx.DiGraph, nx.Graph, nx.Graph]:
    """Return (money-flow digraph, P2P+device community graph, customer-device bipartite graph)."""
    ok = tx[tx["status"] == "SUCCESS"]
    p2p = ok[ok["transaction_type"] == "P2P"]
    flow = p2p.groupby(["sender_id", "receiver_id"])["amount"].agg(["sum", "count"]).reset_index()
    money = nx.DiGraph()
    money.add_nodes_from(tx["customer_id"].unique())
    for s, r, amt, cnt in flow.itertuples(index=False):
        money.add_edge(s, r, weight=float(amt), count=int(cnt), kind="sends_money_to")

    dev = tx[tx["device_id"].notna()].groupby(["customer_id", "device_id"]).size().reset_index(name="n")
    bip = nx.Graph()
    for c, d, n in dev.itertuples(index=False):
        bip.add_node(c, kind="customer")
        bip.add_node("D:" + d, kind="device")
        bip.add_edge(c, "D:" + d, weight=int(n), kind="uses_device")

    comm = nx.Graph()
    comm.add_nodes_from(money.nodes)
    for s, r, data in money.edges(data=True):
        w = comm[s][r]["weight"] + data["count"] if comm.has_edge(s, r) else data["count"]
        comm.add_edge(s, r, weight=w)
    for d in [n for n, a in bip.nodes(data=True) if a["kind"] == "device"]:
        accounts = list(bip.neighbors(d))
        if len(accounts) > 1:  # only shared devices create customer links
            comm.add_node(d)
            for a in accounts:
                comm.add_edge(a, d, weight=3.0)
    return money, comm, bip


def _pass_through_share(tx: pd.DataFrame, window_s: int = 3600) -> pd.Series:
    """Share of a customer's inflows followed by an outflow within ``window_s``."""
    ok = tx[tx["status"] == "SUCCESS"]
    inflow = ok[ok["transaction_type"] == "P2P"][["receiver_id", "event_time"]].rename(
        columns={"receiver_id": "customer_id"}
    )
    outflow = ok[ok["transaction_type"].isin(["P2P", "CASH_OUT"])][["customer_id", "event_time"]]
    inflow = inflow[inflow["customer_id"].str.startswith("CUS_")].sort_values("event_time")
    outflow = outflow.sort_values("event_time").rename(columns={"event_time": "out_time"})
    m = pd.merge_asof(
        inflow,
        outflow,
        left_on="event_time",
        right_on="out_time",
        by="customer_id",
        direction="forward",
        tolerance=pd.Timedelta(seconds=window_s),
    )
    return m.assign(fast=m["out_time"].notna()).groupby("customer_id")["fast"].mean()


def customer_graph_metrics(tx: pd.DataFrame, customers: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, nx.Graph]:
    money, comm, bip = build_graph(tx)
    ids = customers["customer_id"].to_numpy()
    df = pd.DataFrame({"customer_id": ids}).set_index("customer_id")
    df["in_degree"] = pd.Series(dict(money.in_degree()))
    df["out_degree"] = pd.Series(dict(money.out_degree()))
    df["weighted_in"] = pd.Series(dict(money.in_degree(weight="weight")))
    df["weighted_out"] = pd.Series(dict(money.out_degree(weight="weight")))
    df["pagerank"] = pd.Series(nx.pagerank(money, weight="weight", alpha=0.85))
    k = min(300, money.number_of_nodes())
    df["betweenness"] = pd.Series(nx.betweenness_centrality(money, k=k, seed=seed, normalized=True))

    communities = nx.community.louvain_communities(comm, weight="weight", seed=seed, resolution=3.0)
    comm_id: dict[str, int] = {}
    for i, members in enumerate(sorted(communities, key=len, reverse=True)):
        for n in members:
            comm_id[n] = i
    df["community_id"] = pd.Series({n: c for n, c in comm_id.items() if not n.startswith("D:")})
    sizes = pd.Series(comm_id).loc[lambda s: ~s.index.str.startswith("D:")].value_counts()
    df["community_size"] = df["community_id"].map(sizes)

    # device sharing
    cust_proj = nx.Graph()
    max_acc: dict[str, int] = {}
    for d in [n for n, a in bip.nodes(data=True) if a["kind"] == "device"]:
        accounts = list(bip.neighbors(d))
        for a in accounts:
            max_acc[a] = max(max_acc.get(a, 1), len(accounts))
        for i in range(len(accounts)):
            for j in range(i + 1, len(accounts)):
                cust_proj.add_edge(accounts[i], accounts[j])
    comp_size = {n: len(c) for c in nx.connected_components(cust_proj) for n in c}
    df["device_share_max_accounts"] = pd.Series(max_acc)
    df["device_component_size"] = pd.Series(comp_size)

    # behavioural flow statistics used by the mule score
    ok = tx[tx["status"] == "SUCCESS"]
    cash_out = ok[ok["transaction_type"] == "CASH_OUT"].groupby("customer_id")["amount"].sum()
    spend = ok[ok["transaction_type"].isin(["MERCHANT_PAYMENT", "BILL_PAYMENT", "MOBILE_RECHARGE"])]
    df["cash_out_amount"] = cash_out
    df["everyday_spend_count"] = spend.groupby("customer_id").size()
    df["pass_through_share"] = _pass_through_share(tx)
    reg = pd.to_datetime(customers.set_index("customer_id")["registration_date"], utc=True)
    first_in = ok[ok["transaction_type"] == "P2P"].groupby("receiver_id")["event_time"].min()
    first_in = pd.to_datetime(first_in, utc=True).reindex(df.index)
    df["account_age_at_first_inflow_days"] = (first_in - reg.reindex(df.index)).dt.days
    df = df.fillna(
        {
            "in_degree": 0,
            "out_degree": 0,
            "weighted_in": 0,
            "weighted_out": 0,
            "pagerank": 0,
            "betweenness": 0,
            "device_share_max_accounts": 1,
            "device_component_size": 1,
            "cash_out_amount": 0,
            "everyday_spend_count": 0,
            "pass_through_share": 0,
            "community_id": -1,
            "community_size": 1,
        }
    )
    df["account_age_at_first_inflow_days"] = df["account_age_at_first_inflow_days"].fillna(9999)
    return df.reset_index(), comm


def mule_score(m: pd.DataFrame) -> pd.Series:
    """Unsupervised, explainable mule score in [0, 1] (mean of percentile ranks).

    Money mules typically: receive from many unrelated senders, move money out
    quickly (pass-through), cash out a large share of inflows, share devices
    with other accounts, are young accounts and have little everyday spending.
    """
    inflow = m["weighted_in"].clip(lower=1)
    components = pd.DataFrame(
        {
            "many_senders": m["in_degree"],
            "pass_through": m["pass_through_share"],
            "cash_out_ratio": (m["cash_out_amount"] / inflow).clip(upper=2),
            "shared_device": m["device_share_max_accounts"],
            "young_account": -m["account_age_at_first_inflow_days"].clip(upper=3650),
            "low_everyday_spend": -(m["everyday_spend_count"] / (1 + m["in_degree"])),
        }
    )
    ranks = components.rank(pct=True)
    # Accounts that never received money can't be mules; damp their score.
    receives = (m["in_degree"] > 0).astype(float)
    return (ranks.mean(axis=1) * (0.4 + 0.6 * receives)).rename("mule_score")


def suspicious_communities(
    metrics: pd.DataFrame, flagged_by_customer: pd.Series, min_size: int = 3, top_quantile: float = 0.9
) -> pd.DataFrame:
    g = metrics.groupby("community_id")
    comm = pd.DataFrame(
        {
            "size": g.size(),
            "mule_score_mean": g["mule_score"].mean(),
            "mule_score_max": g["mule_score"].max(),
            "max_device_sharing": g["device_share_max_accounts"].max(),
            "total_inflow": g["weighted_in"].sum(),
        }
    )
    flagged = (
        metrics.assign(flagged=metrics["customer_id"].map(flagged_by_customer).fillna(0))
        .groupby("community_id")["flagged"]
        .sum()
    )
    comm["flagged_txn_count"] = flagged
    comm["flagged_per_member"] = comm["flagged_txn_count"] / comm["size"]
    eligible = comm["size"] >= min_size
    comm["risk_score"] = 0.0
    comm.loc[eligible, "risk_score"] = (
        comm.loc[eligible, "mule_score_mean"].rank(pct=True) * 0.4
        + comm.loc[eligible, "flagged_per_member"].rank(pct=True) * 0.4
        + comm.loc[eligible, "max_device_sharing"].rank(pct=True) * 0.2
    )
    cutoff = comm.loc[eligible, "risk_score"].quantile(top_quantile) if eligible.any() else 1.0
    comm["is_suspicious"] = eligible & (comm["risk_score"] >= cutoff)
    return comm.reset_index().sort_values("risk_score", ascending=False)


def run(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    synth, processed, reports = (
        settings.path("synthetic_dir"),
        settings.path("processed_dir"),
        settings.path("reports_dir"),
    )
    tx = pd.read_parquet(synth / "transactions.parquet")
    customers = pd.read_parquet(synth / "customers.parquet")
    truth = pd.read_parquet(synth / "customer_ground_truth.parquet")

    metrics, comm_graph = customer_graph_metrics(tx, customers, seed=int(settings["training"]["random_state"]))
    metrics["mule_score"] = mule_score(metrics).to_numpy()
    metrics["mule_score_percentile"] = metrics["mule_score"].rank(pct=True) * 100

    # transactions flagged by the fraud model (if trained) per customer
    flagged = pd.Series(dtype=float)
    scores_path = processed / "fraud_scores.parquet"
    if scores_path.exists():
        from common.config import load_policy

        s = pd.read_parquet(scores_path, columns=["transaction_id", "fraud_probability"])
        thr = load_policy()["thresholds"]["review"] / 100.0
        hits = tx.merge(s, on="transaction_id")
        hits = hits[hits["fraud_probability"] >= thr]
        involved = pd.concat(
            [hits["customer_id"], hits["receiver_id"].where(hits["receiver_id"].str.startswith("CUS_"))]
        )
        flagged = involved.dropna().value_counts()
    comm = suspicious_communities(metrics, flagged)
    metrics = metrics.merge(
        comm[["community_id", "is_suspicious", "risk_score"]].rename(
            columns={"is_suspicious": "in_suspicious_cluster", "risk_score": "cluster_risk_score"}
        ),
        on="community_id",
        how="left",
    )
    hub_cut = metrics["pagerank"].quantile(0.99)
    metrics["is_suspicious_hub"] = (metrics["pagerank"] >= hub_cut) & (metrics["mule_score_percentile"] >= 90)

    # ---- evaluation against hidden ground truth ----
    ev = metrics.merge(truth[["customer_id", "is_mule", "mule_ring_id"]], on="customer_id")
    y = ev["is_mule"].astype(int).to_numpy()
    k = int(y.sum())
    rep = classification_report(
        y, ev["mule_score"].to_numpy(), threshold=float(np.sort(ev["mule_score"])[-k]), ks=(k, 2 * k)
    )
    p_in, r_in = precision_recall_at_k(y, ev["in_degree"].to_numpy(), k)
    susp = ev["in_suspicious_cluster"].fillna(False).to_numpy(dtype=bool)
    evaluation = {
        "n_customers": int(len(ev)),
        "true_mules": k,
        "mule_score": {
            "pr_auc": rep["pr_auc"],
            "roc_auc": rep["roc_auc"],
            f"precision_at_{k}": rep[f"precision_at_{k}"],
            f"recall_at_{k}": rep[f"recall_at_{k}"],
            f"recall_at_{2 * k}": rep[f"recall_at_{2 * k}"],
        },
        "baseline_in_degree_only": {f"precision_at_{k}": p_in, f"recall_at_{k}": r_in},
        "suspicious_clusters": {
            "n_clusters": int(comm["is_suspicious"].sum()),
            "members": int(susp.sum()),
            "mule_share_in_suspicious_clusters": float(y[susp].mean()) if susp.any() else 0.0,
            "mule_recall_of_suspicious_clusters": float(susp[y == 1].mean()) if k else 0.0,
            "base_mule_rate": float(y.mean()),
        },
        "graph": {
            "nodes": comm_graph.number_of_nodes(),
            "edges": comm_graph.number_of_edges(),
            "communities": int(metrics["community_id"].nunique()),
        },
    }

    # ---- outputs ----
    processed.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(processed / "graph_customer_metrics.parquet", index=False)
    comm.to_parquet(processed / "graph_communities.parquet", index=False)
    top = set(comm[comm["is_suspicious"]].head(15)["community_id"])
    nodes_top = {n for n, c in _community_members(metrics, comm_graph, top)}
    edges = [(u, v, float(d.get("weight", 1.0))) for u, v, d in comm_graph.subgraph(nodes_top).edges(data=True)]
    pd.DataFrame(edges, columns=["source", "target", "weight"]).assign(
        edge_type=lambda e: np.where(
            e["source"].str.startswith("D:") | e["target"].str.startswith("D:"), "uses_device", "sends_money_to"
        )
    ).to_parquet(processed / "graph_suspicious_edges.parquet", index=False)
    write_json(reports / "graph_analysis.json", sanitize(evaluation))
    log.info(
        "graph_analysis_done",
        extra={
            "mule_pr_auc": evaluation["mule_score"]["pr_auc"],
            "suspicious_clusters": evaluation["suspicious_clusters"]["n_clusters"],
        },
    )
    return evaluation


def _community_members(metrics: pd.DataFrame, g: nx.Graph, communities: set[int]):
    members = set(metrics.loc[metrics["community_id"].isin(communities), "customer_id"])
    for n in list(members):
        yield n, None
        for nb in g.neighbors(n):
            if nb.startswith("D:"):
                yield nb, None


if __name__ == "__main__":  # pragma: no cover
    run()
