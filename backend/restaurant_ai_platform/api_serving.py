from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, List
import time
import uuid


DEFAULT_RESTAURANT_ID = "restaurant_001"
DEFAULT_LOCATION_ID = "location_001"


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _log(message: str) -> None:
    print(f"[{_utc_ts()}] {message}")


def _response_ok(data: Dict[str, Any], status_code: int = 200) -> Any:
    from flask import jsonify  # type: ignore

    payload = {
        "ok": True,
        "timestamp": _utc_ts(),
        **data,
    }
    return jsonify(payload), status_code


def _response_error(
    message: str,
    status_code: int = 500,
    *,
    details: Optional[Dict[str, Any]] = None,
) -> Any:
    from flask import jsonify  # type: ignore

    payload: Dict[str, Any] = {
        "ok": False,
        "error": {"message": message},
        "timestamp": _utc_ts(),
    }
    if details:
        payload["error"]["details"] = details
    return jsonify(payload), status_code


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    parts = [p.strip() for p in v.split(",")]
    parts = [p for p in parts if p]
    return parts or None


def _collect_tenant_from_query(args: Any) -> Dict[str, str]:
    restaurant_id = (args.get("restaurant_id") or "").strip() or DEFAULT_RESTAURANT_ID
    location_id = (args.get("location_id") or "").strip() or DEFAULT_LOCATION_ID
    return {
        "restaurant_id": restaurant_id,
        "location_id": location_id,
    }


def _collect_tenant_from_json(payload: Dict[str, Any]) -> Dict[str, str]:
    restaurant_id = payload.get("restaurant_id")
    location_id = payload.get("location_id")

    restaurant_id_str = str(restaurant_id).strip() if restaurant_id is not None else ""
    location_id_str = str(location_id).strip() if location_id is not None else ""

    return {
        "restaurant_id": restaurant_id_str or DEFAULT_RESTAURANT_ID,
        "location_id": location_id_str or DEFAULT_LOCATION_ID,
    }


def _collect_options_from_query(args: Any) -> Dict[str, Any]:
    options: Dict[str, Any] = {}

    dry_run = _parse_bool(args.get("dry_run"))
    if dry_run is not None:
        options["dry_run"] = dry_run

    strict = _parse_bool(args.get("strict"))
    if strict is not None:
        options["strict"] = strict

    steps = _parse_csv(args.get("steps"))
    if steps is not None:
        options["steps"] = steps

    exclude = _parse_csv(args.get("exclude"))
    if exclude is not None:
        options["exclude"] = exclude

    start_at = (args.get("start_at") or "").strip()
    if start_at:
        options["start_at"] = start_at

    stop_after = (args.get("stop_after") or "").strip()
    if stop_after:
        options["stop_after"] = stop_after

    stop_on_error = _parse_bool(args.get("stop_on_error"))
    if stop_on_error is not None:
        options["stop_on_error"] = stop_on_error

    options.update(_collect_tenant_from_query(args))
    return options


def _collect_options_from_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    options: Dict[str, Any] = {}

    if isinstance(payload.get("dry_run"), bool):
        options["dry_run"] = payload["dry_run"]

    if isinstance(payload.get("strict"), bool):
        options["strict"] = payload["strict"]

    steps = payload.get("steps")
    if isinstance(steps, list) and all(isinstance(x, str) for x in steps):
        options["steps"] = steps
    elif isinstance(steps, str):
        parsed = _parse_csv(steps)
        if parsed is not None:
            options["steps"] = parsed

    exclude = payload.get("exclude")
    if isinstance(exclude, list) and all(isinstance(x, str) for x in exclude):
        options["exclude"] = exclude
    elif isinstance(exclude, str):
        parsed = _parse_csv(exclude)
        if parsed is not None:
            options["exclude"] = parsed

    start_at = payload.get("start_at")
    if isinstance(start_at, str) and start_at.strip():
        options["start_at"] = start_at.strip()

    stop_after = payload.get("stop_after")
    if isinstance(stop_after, str) and stop_after.strip():
        options["stop_after"] = stop_after.strip()

    if isinstance(payload.get("stop_on_error"), bool):
        options["stop_on_error"] = payload["stop_on_error"]

    options.update(_collect_tenant_from_json(payload))
    return options


def _run_pipeline(orchestrator: Any, options: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return orchestrator.run_pipeline(options)  # type: ignore[misc]
    except TypeError:
        try:
            return orchestrator.run_pipeline(**options)  # type: ignore[misc]
        except TypeError:
            return orchestrator.run_pipeline()  # type: ignore[misc]


_PERSIST_AVAILABLE = False
try:
    from .core import persistence  # type: ignore

    _PERSIST_AVAILABLE = True
except Exception as e:
    _log(f"persistence_not_available: {type(e).__name__}: {e}")
    _PERSIST_AVAILABLE = False


def _persist_save(payload: Dict[str, Any]) -> None:
    if not _PERSIST_AVAILABLE:
        return
    try:
        persistence.save_run(payload)  # type: ignore[attr-defined]
    except Exception as e:
        _log(f"persistence_save_failed: {type(e).__name__}: {e}")


def _persist_get_last(
    restaurant_id: Optional[str] = None,
    location_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not _PERSIST_AVAILABLE:
        return None
    try:
        return persistence.get_last_run(restaurant_id=restaurant_id, location_id=location_id)  # type: ignore[attr-defined]
    except TypeError:
        try:
            return persistence.get_last_run()  # type: ignore[attr-defined]
        except Exception as e:
            _log(f"persistence_read_failed: {type(e).__name__}: {e}")
            return None
    except Exception as e:
        _log(f"persistence_read_failed: {type(e).__name__}: {e}")
        return None


bp: Optional[Any] = None

try:
    from flask import Blueprint, request  # type: ignore

    bp = Blueprint("api_serving", __name__)

    @bp.get("/health")
    @bp.get("/api/v1/health")
    def health() -> Any:
        return _response_ok(
            {
                "service": "restaurant-ai-platform",
                "status": "ok",
            }
        )

    @bp.get("/pipeline/status")
    @bp.get("/api/v1/pipeline/status")
    def pipeline_status() -> Any:
        return _response_ok(
            {
                "service": "restaurant-ai-platform",
                "pipeline": {
                    "status": "ready",
                    "persistence": {
                        "enabled": bool(_PERSIST_AVAILABLE),
                        "module": "core.persistence",
                    },
                    "supported_options_now": {
                        "dry_run": True,
                        "steps": True,
                        "exclude": True,
                        "start_at": True,
                        "stop_after": True,
                        "stop_on_error": True,
                        "strict": True,
                        "restaurant_id": True,
                        "location_id": True,
                    },
                    "endpoints": {
                        "health": ["/health", "/api/v1/health"],
                        "pipeline_status": ["/pipeline/status", "/api/v1/pipeline/status"],
                        "pipeline_last_run": [
                            "/pipeline/last-run",
                            "/api/v1/pipeline/last-run",
                            "/pipeline/last_run",
                            "/api/v1/pipeline/last_run",
                            "/pipeline/lastrun",
                            "/api/v1/pipeline/lastrun",
                        ],
                        "pipeline_run_post": [
                            "/pipeline/run (POST)",
                            "/api/v1/pipeline/run (POST)",
                        ],
                        "pipeline_run_browser": [
                            "/pipeline/run?execute=1&confirm=yes (GET)",
                            "/api/v1/pipeline/run?execute=1&confirm=yes (GET)",
                        ],
                    },
                    "examples": {
                        "run_all": "/api/v1/pipeline/run?execute=1&confirm=yes",
                        "run_for_tenant": "/api/v1/pipeline/run?execute=1&confirm=yes&restaurant_id=restaurant_001&location_id=location_001",
                        "dry_run": "/api/v1/pipeline/run?execute=1&confirm=yes&dry_run=1",
                        "last_run": "/api/v1/pipeline/last-run",
                        "last_run_for_tenant": "/api/v1/pipeline/last-run?restaurant_id=restaurant_001&location_id=location_001",
                        "subset": "/api/v1/pipeline/run?execute=1&confirm=yes&steps=1_data_ingestion,2_data_warehouse",
                        "exclude": "/api/v1/pipeline/run?execute=1&confirm=yes&exclude=7_api_serving",
                        "range": "/api/v1/pipeline/run?execute=1&confirm=yes&start_at=3_feature_engineering&stop_after=5_ml_prediction",
                        "stop_on_error": "/api/v1/pipeline/run?execute=1&confirm=yes&stop_on_error=1",
                        "non_strict": "/api/v1/pipeline/run?execute=1&confirm=yes&strict=0",
                    },
                },
            }
        )

    @bp.get("/pipeline/last-run")
    @bp.get("/api/v1/pipeline/last-run")
    @bp.get("/pipeline/last_run")
    @bp.get("/api/v1/pipeline/last_run")
    @bp.get("/pipeline/lastrun")
    @bp.get("/api/v1/pipeline/lastrun")
    def pipeline_last_run() -> Any:
        tenant = _collect_tenant_from_query(request.args)
        last_run = _persist_get_last(
            restaurant_id=tenant["restaurant_id"],
            location_id=tenant["location_id"],
        )

        if last_run is None:
            return _response_ok(
                {
                    "service": "restaurant-ai-platform",
                    "restaurant_id": tenant["restaurant_id"],
                    "location_id": tenant["location_id"],
                    "has_last_run": False,
                    "last_run": None,
                    "message": "No pipeline run has been stored yet for this tenant. Run the pipeline first.",
                }
            )

        return _response_ok(
            {
                "service": "restaurant-ai-platform",
                "restaurant_id": tenant["restaurant_id"],
                "location_id": tenant["location_id"],
                "has_last_run": True,
                "last_run": last_run,
            }
        )

    @bp.get("/pipeline/run")
    @bp.get("/api/v1/pipeline/run")
    def pipeline_run_browser() -> Any:
        execute = (request.args.get("execute") or "").strip().lower()
        confirm = (request.args.get("confirm") or "").strip().lower()

        if execute in {"1", "true", "yes"} and confirm == "yes":
            from . import orchestrator

            options = _collect_options_from_query(request.args)

            started = time.time()
            request_id = uuid.uuid4().hex
            try:
                result = _run_pipeline(orchestrator, options)
                duration_ms = int((time.time() - started) * 1000)
                run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request_id[:8]}"

                stored_status = "ok"
                if isinstance(result, dict) and isinstance(result.get("status"), str):
                    stored_status = str(result["status"])

                response_payload = {
                    "request_id": request_id,
                    "run_id": run_id,
                    "status": stored_status,
                    "duration_ms": duration_ms,
                    "restaurant_id": options.get("restaurant_id", DEFAULT_RESTAURANT_ID),
                    "location_id": options.get("location_id", DEFAULT_LOCATION_ID),
                    "requested_payload": {"_via": "browser_get", **options},
                    "orchestrator_options_used": options,
                    "result": result,
                }

                _persist_save(response_payload)
                return _response_ok(response_payload)
            except Exception as e:
                _log(f"pipeline_run_browser failed: {type(e).__name__}: {e}")
                err_payload = {
                    "request_id": request_id,
                    "status": "error",
                    "restaurant_id": options.get("restaurant_id", DEFAULT_RESTAURANT_ID),
                    "location_id": options.get("location_id", DEFAULT_LOCATION_ID),
                    "requested_payload": {"_via": "browser_get", **options},
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
                _persist_save(err_payload)
                return _response_error("Pipeline execution failed", 500, details=err_payload)

        return _response_ok(
            {
                "service": "restaurant-ai-platform",
                "message": (
                    "Use POST to /api/v1/pipeline/run for normal clients. "
                    "For Safari browser testing, call GET with ?execute=1&confirm=yes."
                ),
                "how_to_test_in_browser": {
                    "safe_check": "/api/v1/pipeline/status",
                    "execute_pipeline": "/api/v1/pipeline/run?execute=1&confirm=yes",
                    "execute_pipeline_for_tenant": "/api/v1/pipeline/run?execute=1&confirm=yes&restaurant_id=restaurant_001&location_id=location_001",
                    "dry_run_example": "/api/v1/pipeline/run?execute=1&confirm=yes&dry_run=1",
                    "last_run": "/api/v1/pipeline/last-run",
                    "last_run_for_tenant": "/api/v1/pipeline/last-run?restaurant_id=restaurant_001&location_id=location_001",
                    "steps_example": "/api/v1/pipeline/run?execute=1&confirm=yes&steps=1_data_ingestion,2_data_warehouse",
                    "range_example": "/api/v1/pipeline/run?execute=1&confirm=yes&start_at=3_feature_engineering&stop_after=5_ml_prediction",
                },
            }
        )

    @bp.post("/pipeline/run")
    @bp.post("/api/v1/pipeline/run")
    def pipeline_run_post() -> Any:
        from . import orchestrator

        started = time.time()
        request_id = uuid.uuid4().hex

        try:
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, dict):
                payload = {"_raw": payload}

            options = _collect_options_from_json(payload)
            result = _run_pipeline(orchestrator, options)

            duration_ms = int((time.time() - started) * 1000)
            run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request_id[:8]}"

            stored_status = "ok"
            if isinstance(result, dict) and isinstance(result.get("status"), str):
                stored_status = str(result["status"])

            response_payload = {
                "request_id": request_id,
                "run_id": run_id,
                "status": stored_status,
                "duration_ms": duration_ms,
                "restaurant_id": options.get("restaurant_id", DEFAULT_RESTAURANT_ID),
                "location_id": options.get("location_id", DEFAULT_LOCATION_ID),
                "requested_payload": payload,
                "orchestrator_options_used": options,
                "result": result,
            }

            _persist_save(response_payload)
            return _response_ok(response_payload)

        except Exception as e:
            _log(f"pipeline_run_post failed: {type(e).__name__}: {e}")
            err_payload = {
                "request_id": request_id,
                "status": "error",
                "error_type": type(e).__name__,
                "error": str(e),
            }
            _persist_save(err_payload)
            return _response_error("Pipeline execution failed", 500, details=err_payload)

except Exception as e:
    _log(f"Flask blueprint not available: {type(e).__name__}: {e}")


def register(app: Any) -> bool:
    if bp is None:
        return False

    try:
        app.register_blueprint(bp)
        return True
    except Exception as e:
        _log(f"Blueprint registration failed: {type(e).__name__}: {e}")
        return False


def run() -> Dict[str, Any]:
    _log("api_serving.run() started")

    if _PERSIST_AVAILABLE:
        try:
            persistence.init_db()  # type: ignore[attr-defined]
        except Exception as e:
            _log(f"persistence_init_failed: {type(e).__name__}: {e}")

    result: Dict[str, Any] = {
        "api_serving_status": "ok",
        "blueprint_available": bp is not None,
        "persistence": {
            "enabled": bool(_PERSIST_AVAILABLE),
            "module": "core.persistence",
        },
        "expected_endpoints": [
            "/health",
            "/pipeline/status",
            "/pipeline/last-run",
            "/pipeline/last_run",
            "/pipeline/lastrun",
            "/pipeline/run (POST)",
            "/pipeline/run?execute=1&confirm=yes (GET)",
            "/api/v1/health",
            "/api/v1/pipeline/status",
            "/api/v1/pipeline/last-run",
            "/api/v1/pipeline/last_run",
            "/api/v1/pipeline/lastrun",
            "/api/v1/pipeline/run (POST)",
            "/api/v1/pipeline/run?execute=1&confirm=yes (GET)",
        ],
        "timestamp": _utc_ts(),
    }

    _log("api_serving.run() completed")
    return result