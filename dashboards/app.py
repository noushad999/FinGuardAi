"""FinGuard analytics & operations dashboard (Streamlit).

    streamlit run dashboards/app.py

Pages are built for decisions, not decoration: each page opens with the
questions it answers and highlights what changed or needs action.
Data: dbt marts in PostgreSQL (``analytics.*``), ML outputs (``ml.*``),
operational tables (``ops.*``), the FinGuard API and Prometheus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D  # noqa: E402

# ---------------------------------------------------------------- visual system
# Reference data-viz palette (validated categorical order) + fixed status colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {
    "APPROVE": "#0ca30c",
    "REVIEW": "#fab219",
    "BLOCK": "#d03b3b",
    "Low": "#0ca30c",
    "Medium": "#fab219",
    "High": "#d03b3b",
}
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
px.defaults.color_discrete_sequence = SERIES  # fixed categorical order for every Plotly Express chart

st.set_page_config(page_title="FinGuard", page_icon="🛡️", layout="wide")


def style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=78 if legend else 44, b=8),
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=12),
        title=dict(y=0.98, yanchor="top", x=0.01, xanchor="left", font=dict(color=INK, size=14)),
        hoverlabel=dict(bgcolor="white", font_color=INK),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
        bargap=0.25,
        colorway=SERIES,
    )
    fig.update_xaxes(showgrid=False, linecolor=BASE, tickfont_color=MUTED, title_font_color=INK2)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=BASE, tickfont_color=MUTED, title_font_color=INK2)
    # raw column names are not axis titles
    for axis in (fig.layout.xaxis, fig.layout.yaxis):
        t = axis.title.text
        if t in ("date", "time", "window_start", "value", "n", "series"):
            axis.title.text = None
        elif t:
            axis.title.text = t.replace("_", " ")
    fig.update_traces(selector=dict(type="scatter"), line_width=2)
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    return fig


def full_weeks(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Weekly resample without the partial first/last week (they would look like dips)."""
    w = df.set_index(date_col).resample("W").sum(numeric_only=True).reset_index()
    return w.iloc[1:-1] if len(w) > 2 else w


@st.cache_data(ttl=30, show_spinner=False)
def q(sql: str) -> pd.DataFrame:
    return D.query(sql)


def fmt_bdt(v: float) -> str:
    v = float(v or 0)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= div:
            return f"৳{v / div:,.1f}{unit}"
    return f"৳{v:,.0f}"


def delta_pct(cur: float, prev: float) -> str | None:
    if not prev:
        return None
    return f"{(cur - prev) / prev:+.1%} vs prior 7 days"


def guard() -> bool:
    try:
        if not D.table_exists("analytics.mart_daily_kpis"):
            st.warning("The warehouse marts are not built yet. Run `make pipeline` (or the Airflow DAGs) first.")
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cannot reach the warehouse ({D.DATABASE_URL.split('@')[-1]}): {exc}")
        return False


# ---------------------------------------------------------------- pages
def executive() -> None:
    st.header("Executive overview")
    st.caption("Is the platform growing, is fraud under control, and is service quality holding?")
    k = q("SELECT * FROM analytics.mart_daily_kpis ORDER BY date")
    k["date"] = pd.to_datetime(k["date"])
    last = k["date"].max()
    cur = k[k["date"] > last - pd.Timedelta(days=7)]
    prev = k[(k["date"] <= last - pd.Timedelta(days=7)) & (k["date"] > last - pd.Timedelta(days=14))]
    active = q(
        "SELECT count(DISTINCT customer_id) AS n FROM analytics.fact_transactions "
        "WHERE event_time > (SELECT max(event_time) FROM analytics.fact_transactions) - interval '30 days'"
    )
    c = st.columns(6)
    c[0].metric(
        "Transactions (total)",
        f"{int(k['transactions'].sum()):,}",
        delta_pct(cur["transactions"].sum(), prev["transactions"].sum()),
    )
    c[1].metric(
        "Transaction value",
        fmt_bdt(k["transaction_value"].sum()),
        delta_pct(cur["transaction_value"].sum(), prev["transaction_value"].sum()),
    )
    c[2].metric("Active users (30d)", f"{int(active['n'].iloc[0]):,}")
    fr_cur = cur["reported_fraud_transactions"].sum() / max(cur["transactions"].sum(), 1)
    fr_all = k["reported_fraud_transactions"].sum() / max(k["transactions"].sum(), 1)
    c[3].metric(
        "Reported fraud rate", f"{fr_all:.2%}", f"{fr_cur - fr_all:+.2%} last 7d vs overall", delta_color="inverse"
    )
    c[4].metric("Reported fraud value", fmt_bdt(k["reported_fraud_value"].sum()))
    fail_all = (k["failed_rate"] * k["transactions"]).sum() / max(k["transactions"].sum(), 1)
    c[5].metric("Failed transaction rate", f"{fail_all:.2%}")

    col1, col2 = st.columns(2)
    weekly = full_weeks(k)
    fig = px.bar(weekly, x="date", y="transaction_value", title="Weekly transaction value (BDT)")
    fig.update_traces(marker_color=SERIES[0], hovertemplate="Week of %{x|%d %b}<br>৳%{y:,.0f}<extra></extra>")
    col1.plotly_chart(style(fig, legend=False), use_container_width=True)
    dec = weekly.melt("date", ["review_decisions", "block_decisions"], var_name="decision", value_name="n")
    dec["decision"] = dec["decision"].map({"review_decisions": "REVIEW", "block_decisions": "BLOCK"})
    fig = px.bar(
        dec,
        x="date",
        y="n",
        color="decision",
        color_discrete_map=STATUS,
        title="Weekly risk decisions requiring action",
    )
    col2.plotly_chart(style(fig), use_container_width=True)
    st.info(
        "Reported fraud reflects labels received from case management; the most recent weeks are "
        "under-reported because fraud reports arrive with a delay (median ≈ 6 days)."
    )


def fraud() -> None:
    st.header("Fraud intelligence")
    st.caption("Where is fraud concentrated, how well do the models separate it, and who should analysts look at?")
    k = q("SELECT date, reported_fraud_transactions, transactions FROM analytics.mart_daily_kpis ORDER BY date")
    k["date"] = pd.to_datetime(k["date"])
    w = full_weeks(k)
    w["fraud_per_10k"] = 1e4 * w["reported_fraud_transactions"] / w["transactions"]
    fig = px.line(
        w,
        x="date",
        y="fraud_per_10k",
        markers=True,
        title="Reported fraud per 10,000 transactions (weekly; latest weeks under-reported until labels mature)",
    )
    fig.update_traces(line_color=SERIES[7], marker_size=8)
    st.plotly_chart(style(fig, 280, legend=False), use_container_width=True)

    b = q("SELECT * FROM analytics.mart_fraud_breakdown")
    cols = st.columns(3)
    for col, (dim, title) in zip(
        cols,
        (
            ("transaction_type", "Fraud rate by transaction type"),
            ("device_type", "Fraud rate by device type"),
            ("channel", "Fraud rate by channel"),
        ),
    ):
        d = b[b["dimension"] == dim].sort_values("fraud_rate")
        fig = px.bar(
            d,
            x="fraud_rate",
            y="value",
            orientation="h",
            title=title,
            hover_data={"transactions": ":,", "fraud_transactions": ":,"},
        )
        fig.update_traces(marker_color=SERIES[0])
        fig.update_xaxes(tickformat=".1%", title=None)
        fig.update_yaxes(title=None)
        col.plotly_chart(style(fig, 300, legend=False), use_container_width=True)

    col1, col2 = st.columns(2)
    d = b[b["dimension"] == "district"].sort_values("fraud_value", ascending=False).head(12).sort_values("fraud_value")
    fig = px.bar(
        d,
        x="fraud_value",
        y="value",
        orientation="h",
        title="Reported fraud value by district (top 12)",
        hover_data={"fraud_rate": ":.2%", "transactions": ":,"},
    )
    fig.update_traces(marker_color=SERIES[1])
    fig.update_yaxes(title=None)
    fig.update_xaxes(title="BDT")
    col1.plotly_chart(style(fig, 380, legend=False), use_container_width=True)
    rs = q("SELECT * FROM analytics.mart_risk_score_distribution")
    pol = D.api_get("/model/info") or {}
    thr = (pol.get("policy") or {}).get("thresholds", {"review": 30, "block": 90})
    fig = go.Figure()
    fig.add_bar(x=rs["risk_score_bucket"], y=rs["transactions"], name="All transactions", marker_color=SEQ_BLUE[2])
    fig.add_bar(x=rs["risk_score_bucket"], y=rs["reported_fraud"], name="Reported fraud", marker_color=SERIES[7])
    fig.update_yaxes(type="log", title="transactions (log scale)")
    fig.update_xaxes(title="risk score bucket")
    for name, v, colr in (("review", thr["review"], STATUS["REVIEW"]), ("block", thr["block"], STATUS["BLOCK"])):
        fig.add_vline(
            x=v, line_dash="dash", line_color=colr, annotation_text=f"{name} ≥ {v}", annotation_font_color=INK2
        )
    fig.update_layout(title="Risk score distribution vs. policy thresholds", barmode="overlay")
    col2.plotly_chart(style(fig, 380), use_container_width=True)

    st.subheader("Analyst work queues")
    col1, col2 = st.columns(2)
    cust = q(
        "SELECT customer_id, risk_level, risk_score, flagged_txn_30d, round(mule_score_percentile) AS mule_pct, "
        "in_suspicious_cluster, segment_name, district, risk_reasons "
        "FROM analytics.mart_top_suspicious_customers LIMIT 50"
    )
    col1.markdown("**Top suspicious customers**")
    col1.dataframe(cust, use_container_width=True, hide_index=True, height=360)
    mer = q(
        "SELECT merchant_id, merchant_category, risk_category, registration_date::date AS registered, "
        "transactions, distinct_customers, round(avg_fraud_probability::numeric, 3) AS avg_fraud_p, "
        "reported_fraud_transactions FROM analytics.mart_top_suspicious_merchants LIMIT 30"
    )
    col2.markdown("**Top suspicious merchants** (by average model fraud probability)")
    col2.dataframe(mer, use_container_width=True, hide_index=True, height=360)
    if D.table_exists("ops.fraud_alerts"):
        alerts = q(
            "SELECT created_at, transaction_id, customer_id, risk_score, decision, reasons, status "
            "FROM ops.fraud_alerts ORDER BY created_at DESC LIMIT 100"
        )
        st.markdown(f"**Live alert queue** (real-time risk engine) - {len(alerts)} most recent")
        st.dataframe(alerts, use_container_width=True, hide_index=True, height=300)


def customers() -> None:
    st.header("Customer intelligence")
    st.caption("Who are our customers, which segments are at risk of churning, and what does one customer look like?")
    seg = q("SELECT * FROM analytics.mart_segment_summary")
    col1, col2 = st.columns(2)
    s = seg[seg["segment_name"] != "Unsegmented (low activity)"].sort_values("customers")
    fig = px.bar(s, x="customers", y="segment_name", orientation="h", title="Customers per behavioural segment")
    fig.update_traces(marker_color=SERIES[0])
    fig.update_yaxes(title=None)
    col1.plotly_chart(style(fig, 340, legend=False), use_container_width=True)
    s2 = s.sort_values("avg_churn_probability")
    fig = px.bar(
        s2,
        x="avg_churn_probability",
        y="segment_name",
        orientation="h",
        title="Average churn probability (next 30 days)",
        hover_data={"customers_high_churn_risk": True},
    )
    fig.update_traces(marker_color=SERIES[1])
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(title=None)
    col2.plotly_chart(style(fig, 340, legend=False), use_container_width=True)
    st.dataframe(seg, use_container_width=True, hide_index=True)

    act = q(
        "SELECT d.date, count(DISTINCT f.customer_id) AS active_customers FROM analytics.fact_customer_activity f "
        "JOIN analytics.dim_date d USING (date_key) GROUP BY 1 ORDER BY 1"
    )
    act["date"] = pd.to_datetime(act["date"])
    wk = act.set_index("date").resample("W").mean().reset_index()
    fig = px.line(wk, x="date", y="active_customers", markers=True, title="Average daily active customers (weekly)")
    fig.update_traces(line_color=SERIES[2], marker_size=8)
    st.plotly_chart(style(fig, 260, legend=False), use_container_width=True)

    st.subheader("Customer lookup (live API)")
    cid = st.text_input("Customer id", value="CUS_0000042")
    if cid:
        prof = D.api_get(f"/customer/{cid}")
        if prof is None:
            st.warning("API not reachable.")
        elif prof.get("_status"):
            st.warning(prof.get("detail", "not found"))
        else:
            c = st.columns(5)
            c[0].metric("Risk", f"{prof.get('risk_level')} ({prof.get('risk_score')})")
            c[1].metric("Fraud probability (30d max)", f"{(prof.get('fraud_probability') or 0):.3f}")
            c[2].metric("ATO probability (30d max)", f"{(prof.get('account_takeover_probability') or 0):.3f}")
            ch = prof.get("churn_probability")
            c[3].metric("Churn probability", "n/a" if ch is None else f"{ch:.2f}")
            c[4].metric("Segment", prof.get("customer_segment") or "-")
            st.write(
                {
                    k: prof.get(k)
                    for k in (
                        "behavior",
                        "recent_activity",
                        "mule_score_percentile",
                        "in_suspicious_cluster",
                        "risk_reasons",
                        "live_state",
                    )
                }
            )


def network() -> None:
    st.header("Network intelligence")
    st.caption("Which clusters of accounts and devices behave like coordinated fraud or money-mule rings?")
    comm = q(
        "SELECT community_id, size, round(mule_score_mean::numeric, 3) AS mule_score_mean, max_device_sharing, "
        "flagged_txn_count, round(risk_score::numeric, 3) AS risk_score, is_suspicious "
        "FROM ml.graph_communities WHERE size >= 3 ORDER BY risk_score DESC LIMIT 25"
    )
    col1, col2 = st.columns([2, 3])
    col1.markdown("**Suspicious transaction clusters** (Louvain communities on P2P + shared-device graph)")
    col1.dataframe(comm, use_container_width=True, hide_index=True, height=420)
    edges = q("SELECT source, target, weight, edge_type FROM ml.graph_suspicious_edges")
    if not edges.empty:
        top = set(comm[comm["is_suspicious"]].head(5)["community_id"])
        members = q("SELECT customer_id, community_id, mule_score_percentile FROM ml.graph_customer_metrics")
        keep = set(members.loc[members["community_id"].isin(top), "customer_id"])
        e = edges[edges["source"].isin(keep) | edges["target"].isin(keep)]
        g = nx.from_pandas_edgelist(e, "source", "target", ["edge_type"])
        pos = nx.spring_layout(g, seed=7, k=0.35)
        fig = go.Figure()
        for etype, colr in (("sends_money_to", BASE), ("uses_device", SERIES[3])):
            xs, ys = [], []
            for u, v, d in g.edges(data=True):
                if d["edge_type"] == etype:
                    xs += [pos[u][0], pos[v][0], None]
                    ys += [pos[u][1], pos[v][1], None]
            fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=colr, width=1.2), name=etype, hoverinfo="skip")
        mp = members.set_index("customer_id")["mule_score_percentile"]
        cust_nodes = [n for n in g.nodes if not n.startswith("D:")]
        dev_nodes = [n for n in g.nodes if n.startswith("D:")]
        fig.add_scatter(
            x=[pos[n][0] for n in cust_nodes],
            y=[pos[n][1] for n in cust_nodes],
            mode="markers",
            name="customer (colour = mule score pct)",
            text=cust_nodes,
            marker=dict(
                size=10,
                color=[mp.get(n, 0) for n in cust_nodes],
                colorscale=SEQ_BLUE,
                cmin=0,
                cmax=100,
                line=dict(color="white", width=2),
                colorbar=dict(title="mule pct", thickness=10),
            ),
            hovertemplate="%{text}<br>mule score pct %{marker.color:.0f}<extra></extra>",
        )
        fig.add_scatter(
            x=[pos[n][0] for n in dev_nodes],
            y=[pos[n][1] for n in dev_nodes],
            mode="markers",
            name="shared device",
            text=dev_nodes,
            marker=dict(size=9, symbol="diamond", color=SERIES[3], line=dict(color="white", width=2)),
            hovertemplate="%{text}<extra></extra>",
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title="Top-5 suspicious clusters: money flows and shared devices")
        style(fig, 470)
        fig.update_layout(margin=dict(t=44, b=60), legend=dict(yanchor="top", y=-0.02))
        col2.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    dev = q(
        "SELECT device_id, device_type, os, accounts_seen, first_seen::date AS first_seen "
        "FROM analytics.dim_device WHERE accounts_seen >= 3 ORDER BY accounts_seen DESC LIMIT 30"
    )
    col1.markdown(f"**Device-account networks** - {len(dev)} handsets used by 3+ accounts")
    col1.dataframe(dev, use_container_width=True, hide_index=True, height=320)
    hubs = q(
        "SELECT customer_id, in_degree, round(weighted_in::numeric) AS inflow_bdt, "
        "round(pagerank::numeric, 5) AS pagerank, round(mule_score_percentile::numeric, 1) AS mule_pct, "
        "in_suspicious_cluster FROM ml.graph_customer_metrics ORDER BY mule_score DESC LIMIT 30"
    )
    col2.markdown("**Suspicious hubs** (highest mule scores)")
    col2.dataframe(hubs, use_container_width=True, hide_index=True, height=320)


def operations() -> None:
    st.header("Operations & model health")
    st.caption("Is the real-time pipeline keeping up, are services healthy, and are the models drifting?")
    health = D.api_get("/health")
    c = st.columns(5)
    c[0].metric("API", "healthy" if health and health.get("status") == "ok" else ("degraded" if health else "down"))
    lag = D.prom("finguard_kafka_consumer_lag_total")
    c[1].metric("Kafka consumer lag", "n/a" if lag is None else f"{lag:,.0f} msgs")
    eps = D.prom('sum(rate(finguard_stream_events_total{outcome="valid"}[1m]))')
    c[2].metric("Risk-engine throughput", "n/a" if eps is None else f"{eps:,.1f} ev/s")
    p95 = D.prom("histogram_quantile(0.95, sum(rate(finguard_scoring_seconds_bucket[5m])) by (le))")
    c[3].metric("Scoring latency p95", "n/a" if p95 is None else f"{p95 * 1000:,.1f} ms")
    err = D.prom(
        'sum(rate(finguard_api_requests_total{status=~"5.."}[5m])) / '
        "clamp_min(sum(rate(finguard_api_requests_total[5m])), 1e-9)"
    )
    c[4].metric("API 5xx rate", "n/a" if err is None else f"{err:.2%}")

    col1, col2 = st.columns(2)
    thr = D.prom_range("sum(rate(finguard_stream_events_total[1m])) by (outcome)")
    if not thr.empty:
        fig = px.line(thr, x="time", y="value", color="series", title="Kafka events consumed per second")
        col1.plotly_chart(style(fig, 280), use_container_width=True)
    lagdf = D.prom_range("finguard_kafka_consumer_lag_total")
    if not lagdf.empty:
        fig = px.line(lagdf, x="time", y="value", title="Consumer lag (messages)")
        fig.update_traces(line_color=SERIES[1])
        col2.plotly_chart(style(fig, 280, legend=False), use_container_width=True)
    if D.table_exists("ops.stream_metrics_minute"):
        sm = q("SELECT window_start, txn_count, failed_count FROM ops.stream_metrics_minute ORDER BY window_start")
        if not sm.empty:
            sm["window_start"] = pd.to_datetime(sm["window_start"])
            hourly = sm.set_index("window_start").resample("h").sum().reset_index()
            fig = px.bar(
                hourly,
                x="window_start",
                y="txn_count",
                title="Spark Structured Streaming: transactions per event-time hour",
            )
            fig.update_traces(marker_color=SERIES[0])
            st.plotly_chart(style(fig, 260, legend=False), use_container_width=True)
    dec = (
        q(
            "SELECT decision, count(*) AS n, round(avg(latency_ms)::numeric, 2) AS avg_latency_ms "
            "FROM ops.risk_decisions GROUP BY 1"
        )
        if D.table_exists("ops.risk_decisions")
        else pd.DataFrame()
    )
    col1, col2 = st.columns(2)
    if not dec.empty:
        fig = px.bar(
            dec,
            x="decision",
            y="n",
            color="decision",
            color_discrete_map=STATUS,
            title="Real-time decisions (all time)",
        )
        col1.plotly_chart(style(fig, 280), use_container_width=True)
    runs = q(
        "SELECT pipeline, started_at, finished_at, status, rows_written FROM ops.pipeline_runs "
        "ORDER BY started_at DESC LIMIT 10"
    )
    col2.markdown("**Pipeline runs**")
    col2.dataframe(runs, use_container_width=True, hide_index=True)

    st.subheader("Data quality & model monitoring")
    col1, col2 = st.columns(2)
    dq = q(
        "SELECT dataset, check_name, severity, passed, failing_rows FROM ops.data_quality_results "
        "WHERE run_at = (SELECT max(run_at) FROM ops.data_quality_results)"
    )
    col1.markdown("**Latest data-quality run**")
    col1.dataframe(dq, use_container_width=True, hide_index=True, height=320)
    mm = q(
        "SELECT name AS feature, round(value::numeric, 3) AS psi, status FROM ops.model_monitoring "
        "WHERE metric_type = 'feature_psi' AND run_at = (SELECT max(run_at) FROM ops.model_monitoring) "
        "ORDER BY value DESC NULLS LAST LIMIT 15"
    )
    col2.markdown("**Feature drift (PSI vs. training)** - >0.25 significant, 0.1-0.25 moderate")
    col2.dataframe(mm, use_container_width=True, hide_index=True, height=320)
    info = D.api_get("/model/info")
    if info:
        st.markdown("**Deployed models**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "model": k,
                        **{x: v.get(x) for x in ("version", "created_at", "n_features")},
                        **(v.get("test_metrics") or {}),
                    }
                    for k, v in info["models"].items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )


PAGES = {
    "Executive overview": executive,
    "Fraud intelligence": fraud,
    "Customer intelligence": customers,
    "Network intelligence": network,
    "Operations & model health": operations,
}

st.sidebar.title("🛡️ FinGuard")
st.sidebar.caption("Synthetic MFS risk & analytics platform. All data is simulated.")
page = st.sidebar.radio("Page", list(PAGES))
if guard():
    PAGES[page]()
