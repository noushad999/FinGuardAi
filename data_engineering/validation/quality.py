"""Data-quality checks for transaction data and its reference entities.

Each check returns a :class:`CheckResult` (name, severity, pass/fail, number of
failing rows, sample offending ids). ``error`` checks fail the pipeline (Airflow
task fails -> downstream loads do not run); ``warning`` checks are reported only.

Covered: schema drift, nulls, duplicates, invalid amounts, invalid timestamps,
impossible balances (sign + ledger arithmetic), invalid customer ids,
referential integrity (customers, devices, merchants, labels), enumerations and
geographic bounds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simulator.constants import CHANNELS, FAILURE_REASONS, TRANSACTION_COLUMNS, TRANSACTION_TYPES

EXPECTED_TYPES: dict[str, str] = {
    "transaction_id": "string",
    "event_time": "datetime",
    "sender_id": "string",
    "receiver_id": "string",
    "customer_id": "string",
    "merchant_id": "string",
    "amount": "float",
    "fee": "float",
    "currency": "string",
    "transaction_type": "string",
    "channel": "string",
    "device_id": "string",
    "district": "string",
    "latitude": "float",
    "longitude": "float",
    "status": "string",
    "failure_reason": "string",
    "balance_before": "float",
    "balance_after": "float",
    "metadata": "string",
}
assert set(EXPECTED_TYPES) == set(TRANSACTION_COLUMNS)


@dataclass
class CheckResult:
    name: str
    severity: str  # error | warning
    passed: bool
    failing_rows: int
    description: str
    sample: list[str] = field(default_factory=list)


def _kind(dtype: Any) -> str:
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_float_dtype(dtype) or pd.api.types.is_integer_dtype(dtype):
        return "float"
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    return "string"


def _result(name: str, severity: str, mask: pd.Series | np.ndarray, ids: pd.Series, description: str) -> CheckResult:
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    return CheckResult(name, severity, n == 0, n, description, [str(x) for x in ids[mask].head(5)])


def check_transactions(
    tx: pd.DataFrame,
    customers: pd.DataFrame | None = None,
    devices: pd.DataFrame | None = None,
    merchants: pd.DataFrame | None = None,
    labels: pd.DataFrame | None = None,
    now: pd.Timestamp | None = None,
) -> list[CheckResult]:
    now = now or pd.Timestamp.now(tz="UTC")
    out: list[CheckResult] = []

    # ---- schema drift ----
    missing = [c for c in EXPECTED_TYPES if c not in tx.columns]
    extra = [c for c in tx.columns if c not in EXPECTED_TYPES and c not in ("event_date",)]
    type_changes = [
        f"{c}:{_kind(tx[c].dtype)}!={t}"
        for c, t in EXPECTED_TYPES.items()
        if c in tx.columns and _kind(tx[c].dtype) != t
    ]
    drift = missing + [f"+{c}" for c in extra] + type_changes
    out.append(
        CheckResult(
            "schema_drift",
            "error",
            not drift,
            len(drift),
            "Columns and types match the transaction contract",
            drift[:10],
        )
    )
    if missing:
        return out  # remaining checks need the full schema
    ids = tx["transaction_id"].astype(str)

    # ---- completeness ----
    required = ["transaction_id", "event_time", "customer_id", "amount", "transaction_type", "status"]
    out.append(
        _result(
            "required_not_null", "error", tx[required].isna().any(axis=1), ids, f"No nulls in {', '.join(required)}"
        )
    )
    out.append(
        _result(
            "merchant_id_present_for_merchant_payments",
            "error",
            (tx["transaction_type"] == "MERCHANT_PAYMENT") & tx["merchant_id"].isna(),
            ids,
            "Merchant payments reference a merchant",
        )
    )
    out.append(
        _result(
            "failure_reason_iff_failed",
            "error",
            (tx["status"] == "FAILED") != tx["failure_reason"].notna(),
            ids,
            "failure_reason is set exactly for FAILED transactions",
        )
    )

    # ---- uniqueness ----
    out.append(
        _result(
            "duplicate_transaction_id",
            "error",
            tx["transaction_id"].duplicated(keep="first"),
            ids,
            "transaction_id is unique",
        )
    )
    content_key = ["customer_id", "event_time", "amount", "receiver_id", "transaction_type"]
    out.append(
        _result(
            "suspected_duplicate_submission",
            "warning",
            # exact re-deliveries (same id) are already covered above
            tx.duplicated(subset=content_key, keep="first") & ~tx["transaction_id"].duplicated(keep=False),
            ids,
            "Same customer/time/amount/receiver/type under different transaction ids",
        )
    )

    # ---- validity ----
    amt = pd.to_numeric(tx["amount"], errors="coerce")
    out.append(
        _result(
            "invalid_amount", "error", ~(amt > 0) | amt.isna() | (amt > 10_000_000), ids, "0 < amount <= 10,000,000 BDT"
        )
    )
    et = pd.to_datetime(tx["event_time"], utc=True, errors="coerce")
    out.append(
        _result(
            "invalid_timestamp",
            "error",
            et.isna() | (et > now + pd.Timedelta(minutes=5)) | (et < pd.Timestamp("2010-01-01", tz="UTC")),
            ids,
            "event_time parseable, not in the future, not before platform launch",
        )
    )
    out.append(
        _result(
            "invalid_enumeration",
            "error",
            ~tx["transaction_type"].isin(TRANSACTION_TYPES)
            | ~tx["channel"].isin(CHANNELS)
            | ~tx["status"].isin(["SUCCESS", "FAILED"])
            | (tx["failure_reason"].notna() & ~tx["failure_reason"].isin(FAILURE_REASONS)),
            ids,
            "transaction_type / channel / status / failure_reason in allowed sets",
        )
    )
    out.append(
        _result(
            "invalid_customer_id_format",
            "error",
            ~tx["customer_id"].astype(str).str.fullmatch(r"CUS_\d{7}"),
            ids,
            "customer_id matches CUS_#######",
        )
    )
    geo_bad = tx["latitude"].notna() & ~(tx["latitude"].between(20.5, 26.8) & tx["longitude"].between(88.0, 92.8))
    out.append(_result("geo_out_of_bounds", "warning", geo_bad, ids, "Coordinates inside Bangladesh bounding box"))

    # ---- balances ----
    bb, ba = tx["balance_before"], tx["balance_after"]
    fee = tx["fee"].fillna(0)
    ok = tx["status"] == "SUCCESS"
    debit = tx["transaction_type"] != "CASH_IN"
    expected = np.where(~ok, bb, np.where(debit, bb - amt - fee, bb + amt))
    out.append(_result("negative_balance", "error", (bb < 0) | (ba < 0), ids, "Balances are never negative"))
    out.append(
        _result(
            "ledger_arithmetic",
            "error",
            np.abs(ba - expected) > 0.05,
            ids,
            "balance_after = balance_before -/+ amount (- fee); unchanged when failed",
        )
    )

    # ---- referential integrity ----
    if customers is not None:
        known = set(customers["customer_id"])
        out.append(
            _result("unknown_customer", "error", ~tx["customer_id"].isin(known), ids, "customer_id exists in customers")
        )
        p2p = tx["transaction_type"] == "P2P"
        out.append(
            _result(
                "unknown_p2p_receiver",
                "error",
                p2p & ~tx["receiver_id"].isin(known),
                ids,
                "P2P receivers exist in customers",
            )
        )
        reg = tx["customer_id"].map(customers.set_index("customer_id")["registration_date"])
        out.append(
            _result(
                "transaction_before_registration",
                "error",
                et < pd.to_datetime(reg, utc=True),
                ids,
                "No activity before customer registration",
            )
        )
    if devices is not None:
        out.append(
            _result(
                "unknown_device",
                "error",
                tx["device_id"].notna() & ~tx["device_id"].isin(set(devices["device_id"])),
                ids,
                "device_id exists in devices",
            )
        )
    if merchants is not None:
        out.append(
            _result(
                "unknown_merchant",
                "error",
                tx["merchant_id"].notna() & ~tx["merchant_id"].isin(set(merchants["merchant_id"])),
                ids,
                "merchant_id exists in merchants",
            )
        )
    if labels is not None:
        lab_ids = labels["transaction_id"].astype(str)
        orphan = ~labels["transaction_id"].isin(set(tx["transaction_id"]))
        out.append(_result("orphan_labels", "error", orphan, lab_ids, "Every label refers to a known transaction"))
    return out


@dataclass
class QualityReport:
    dataset: str
    run_at: str
    rows: int
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "run_at": self.run_at,
            "rows": self.rows,
            "passed": self.passed,
            "errors_failed": sum(1 for c in self.checks if c.severity == "error" and not c.passed),
            "warnings_failed": sum(1 for c in self.checks if c.severity == "warning" and not c.passed),
            "checks": [asdict(c) for c in self.checks],
        }

    def to_markdown(self) -> str:
        d = self.to_dict()
        lines = [
            f"# Data-quality report: {self.dataset}",
            "",
            f"Run at {self.run_at} - {self.rows:,} rows - overall: **{'PASS' if self.passed else 'FAIL'}** "
            f"({d['errors_failed']} error checks failed, {d['warnings_failed']} warnings)",
            "",
            "| Check | Severity | Result | Failing rows | Description | Sample |",
            "|---|---|---|---|---|---|",
        ]
        for c in self.checks:
            lines.append(
                f"| {c.name} | {c.severity} | {'PASS' if c.passed else 'FAIL'} | {c.failing_rows:,} | "
                f"{c.description} | {', '.join(c.sample[:3])} |"
            )
        return "\n".join(lines) + "\n"


def build_report(dataset: str, tx: pd.DataFrame, **refs: Any) -> QualityReport:
    return QualityReport(dataset, datetime.now(UTC).isoformat(), len(tx), check_transactions(tx, **refs))


def write_report(report: QualityReport, reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"data_quality_{report.dataset.replace('/', '_')}"
    j, md = reports_dir / f"{stem}.json", reports_dir / f"{stem}.md"
    j.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    md.write_text(report.to_markdown(), encoding="utf-8")
    return j, md
