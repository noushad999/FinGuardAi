"""FinGuard FastAPI service.

Run locally:  ``uvicorn api.main:app --reload``  (interactive docs at /docs)

Endpoints
---------
GET  /health                     liveness + dependency status
POST /risk/score                 real-time transaction risk scoring (ML + policy)
GET  /customer/{customer_id}     unified customer intelligence profile
GET  /transaction/{transaction_id}
GET  /fraud/alerts               analyst alert queue
GET  /model/info                 deployed model versions, metrics, lineage, policy
GET  /metrics                    Prometheus metrics
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.repository import FileRepository, Repository, SqlRepository
from api.schemas import (
    AlertsResponse,
    CustomerProfileResponse,
    HealthResponse,
    LiveState,
    ModelInfoResponse,
    RiskScoreResponse,
    TransactionResponse,
    TransactionScoreRequest,
)
from common.config import get_settings
from common.logging import get_logger
from features.definitions import FEATURE_NAMES
from features.store import InMemoryStateStore, RedisStateStore
from monitoring import metrics as m
from services.risk_engine.scorer import RiskScorer

log = get_logger("api")


def _default_repository() -> Repository:
    settings = get_settings()
    try:
        from common.db import get_engine

        repo = SqlRepository(get_engine())
        if repo.ok():
            return repo
        log.warning("database_unreachable_falling_back_to_files")
    except Exception as exc:  # noqa: BLE001
        log.warning("database_init_failed", extra={"error": str(exc)[:200]})
    return FileRepository(settings.path("synthetic_dir"), settings.path("processed_dir"))


def create_app(scorer: RiskScorer | None = None, repository: Repository | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.scorer = scorer or RiskScorer.from_settings()
        app.state.repo = repository or _default_repository()
        log.info("api_started", extra={"repository": app.state.repo.name, "models": app.state.scorer.models.versions()})
        yield

    app = FastAPI(
        title="FinGuard Risk API",
        version="1.0.0",
        description="Real-time transaction risk scoring, customer intelligence and fraud alerts for a simulated "
        "Mobile Financial Services platform. All data is synthetic.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = time.perf_counter()
        route = request.url.path
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            r = request.scope.get("route")
            route = getattr(r, "path", route)
            m.API_REQUESTS.labels(request.method, route, str(status)).inc()
            m.API_LATENCY.labels(request.method, route).observe(time.perf_counter() - start)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        log.info("request_validation_failed", extra={"path": request.url.path, "errors": len(exc.errors())})
        return JSONResponse(
            status_code=422, content={"error": "validation_error", "detail": jsonable_errors(exc.errors())}
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled_error", extra={"path": request.url.path})
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "unexpected error"})

    # ------------------------------------------------------------------ routes
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        sc: RiskScorer = request.app.state.scorer
        repo: Repository = request.app.state.repo
        store = sc.engine.store
        store_ok = True
        if isinstance(store, RedisStateStore):
            try:
                store_ok = store.ping()
            except Exception:  # noqa: BLE001
                store_ok = False
        repo_ok = repo.ok()
        return HealthResponse(
            status="ok" if store_ok and repo_ok else "degraded",
            models_loaded=True,
            model_versions=sc.models.versions(),
            state_store=type(store).__name__,
            state_store_ok=store_ok,
            repository=repo.name,
            repository_ok=repo_ok,
        )

    @app.post("/risk/score", response_model=RiskScoreResponse, tags=["risk"])
    def score(
        request: Request,
        txn: TransactionScoreRequest,
        commit: bool | None = Query(
            None, description="Fold this transaction into online state (use when the API is the system of record)"
        ),
    ):
        sc: RiskScorer = request.app.state.scorer
        commit = get_settings()["api"]["commit_state_on_score"] if commit is None else commit
        payload = txn.to_engine_dict()
        if not payload["transaction_id"]:
            payload["transaction_id"] = f"API_{uuid.uuid4().hex[:16]}"
        t0 = time.perf_counter()
        result = sc.score(payload, commit=bool(commit), include_features=True)
        m.SCORING_LATENCY.labels("api").observe(time.perf_counter() - t0)
        m.DECISIONS.labels("api", result["decision"]).inc()
        m.RISK_SCORE.labels("api").observe(result["risk_score"])
        m.FRAUD_PROBABILITY.labels("api").observe(result["fraud_probability"])
        result["state_committed"] = bool(commit)
        record = dict(
            result,
            source="api",
            event_time=txn.event_time.isoformat(),
            transaction_type=txn.transaction_type,
            amount=txn.amount,
        )
        try:
            request.app.state.repo.save_decision(record)
        except Exception as exc:  # noqa: BLE001 - scoring must not fail because audit storage is down
            log.warning("decision_persist_failed", extra={"error": str(exc)[:200]})
        log.info(
            "transaction_scored",
            extra={
                "transaction_id": result["transaction_id"],
                "decision": result["decision"],
                "risk_score": result["risk_score"],
                "latency_ms": result["latency_ms"],
            },
        )
        result.pop("features", None)
        return result

    @app.get("/customer/{customer_id}", response_model=CustomerProfileResponse, tags=["customers"])
    def customer(request: Request, customer_id: str) -> CustomerProfileResponse:
        repo: Repository = request.app.state.repo
        sc: RiskScorer = request.app.state.scorer
        prof = repo.customer_profile(customer_id)
        state = sc.engine.customer_state(customer_id)
        if prof is None and state is None and not repo.customer_exists(customer_id):
            raise HTTPException(status_code=404, detail=f"customer {customer_id} not found")
        prof = prof or {}
        live = None
        if state:
            now = state.get("last_ts") or 0
            live = LiveState(
                prior_transactions=state["n"],
                last_transaction_at=datetime.fromtimestamp(state["last_ts"], tz=UTC) if state.get("last_ts") else None,
                known_devices=len(state["devices"]),
                known_counterparties=len(state["recipients"]),
                known_districts=len(state["districts"]),
                transactions_last_24h=sum(1 for r in state["recent"] if now - r[0] <= 86400),
            )
        reasons = prof.get("risk_reasons") or []
        if isinstance(reasons, str):
            import json

            reasons = json.loads(reasons)
        return CustomerProfileResponse(
            customer_id=customer_id,
            risk_level=prof.get("risk_level"),
            risk_score=prof.get("risk_score"),
            fraud_probability=prof.get("fraud_probability_30d_max"),
            account_takeover_probability=prof.get("ato_probability_30d_max"),
            churn_probability=prof.get("churn_probability"),
            customer_segment=prof.get("segment_name"),
            behavior=prof.get("behavior"),
            recent_activity=prof.get("recent_activity"),
            mule_score_percentile=prof.get("mule_score_percentile"),
            in_suspicious_cluster=prof.get("in_suspicious_cluster"),
            flagged_transactions_30d=prof.get("flagged_txn_30d"),
            risk_reasons=list(reasons),
            district=prof.get("district"),
            product_tier=prof.get("product_tier"),
            profile_as_of=prof.get("as_of"),
            live_state=live,
            recent_decisions=repo.recent_decisions(customer_id),
        )

    @app.get("/transaction/{transaction_id}", response_model=TransactionResponse, tags=["transactions"])
    def transaction(request: Request, transaction_id: str) -> TransactionResponse:
        repo: Repository = request.app.state.repo
        txn = repo.transaction(transaction_id)
        preds = repo.predictions(transaction_id)
        if txn is None and not preds:
            raise HTTPException(status_code=404, detail=f"transaction {transaction_id} not found")
        return TransactionResponse(transaction=txn or {"transaction_id": transaction_id}, predictions=preds)

    @app.get("/fraud/alerts", response_model=AlertsResponse, tags=["risk"])
    def alerts(
        request: Request,
        limit: int = Query(50, ge=1, le=500),
        decision: str | None = Query(None, pattern="^(REVIEW|BLOCK)$"),
        customer_id: str | None = None,
    ) -> AlertsResponse:
        rows = request.app.state.repo.alerts(limit, decision, customer_id)
        return AlertsResponse(count=len(rows), alerts=rows)

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
    def model_info(request: Request) -> ModelInfoResponse:
        sc: RiskScorer = request.app.state.scorer
        info: dict[str, dict[str, Any]] = {}
        for key, model in (
            ("fraud", sc.models.fraud),
            ("account_takeover", sc.models.ato),
            ("isolation_forest", sc.models.iforest),
        ):
            if model is None:
                continue
            meta = model.meta
            test = meta.metrics.get("test", {})
            summary = test.get("at_policy_review", test)
            info[key] = {
                "version": meta.version,
                "model_type": meta.model_type,
                "created_at": meta.created_at,
                "n_features": len(meta.feature_names),
                "dataset": meta.data,
                "test_metrics": {
                    k: summary.get(k) for k in ("pr_auc", "roc_auc", "precision", "recall", "f1") if k in summary
                },
                "thresholds": meta.thresholds,
                "mlflow_run_id": meta.extra.get("mlflow_run_id"),
            }
        if sc.models.behavioral is not None:
            info["behavioral_anomaly"] = {
                "version": sc.models.behavioral.version,
                "model_type": "customer_baseline_fisher",
            }
        return ModelInfoResponse(models=info, policy=sc.policy.config, feature_count=len(FEATURE_NAMES))

    @app.get("/metrics", tags=["ops"])
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def jsonable_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"loc": list(e.get("loc", [])), "msg": str(e.get("msg")), "type": e.get("type")} for e in errors]


app = create_app()
__all__ = ["app", "create_app", "InMemoryStateStore"]
