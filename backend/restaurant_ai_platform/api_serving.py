from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
import time
import uuid


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


def _collect_options_from_query(args: Any) -> Dict[str, Any]:
    """
    Query params supported (passed to orchestrator):
      - dry_run=1
      - strict=0
      - steps=1_data_ingestion,2_data_warehouse
      - exclude=7_api_serving
      - start_at=3_feature_engineering
      - stop_after=5_ml_prediction
      - stop_on_error=1
    """
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

    return options


def _collect_options_from_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    JSON body keys supported (passed to orchestrator):
      - dry_run: bool
      - strict: bool
      - steps: list[str] or comma string
      - exclude: list[str] or comma string
      - start_at: str
      - stop_after: str
      - stop_on_error: bool
    """
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

    return options


def _run_pipeline(orchestrator: Any, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preferred: pass dict options (new orchestrator supports this).
    Fallbacks included for older signatures.
    """
    try:
        # New orchestrator: run_pipeline(options_dict)
        return orchestrator.run_pipeline(options)  # type: ignore[misc]
    except TypeError:
        # Older orchestrator might expect kwargs
        try:
            return orchestrator.run_pipeline(**options)  # type: ignore[misc]
        except TypeError:
            # Oldest: no-arg
            return orchestrator.run_pipeline()  # type: ignore[misc]


bp: Optional[Any] = None

try:
    from flask import Blueprint, request  # type: ignore

    bp = Blueprint("api_serving", __name__)

    # ----------------------------
    # Health
    # ----------------------------
    @bp.get("/health")
    @bp.get("/api/v1/health")
    def health() -> Any:
        return _response_ok(
            {
                "service": "restaurant-ai-platform",
                "status": "ok",
            }
        )

    # ----------------------------
    # Pipeline status (docs-ish)
    # ----------------------------
    @bp.get("/pipeline/status")
    @bp.get("/api/v1/pipeline/status")
    def pipeline_status() -> Any:
        return _response_ok(
            {
                "service": "restaurant-ai-platform",
                "pipeline": {
                    "status": "ready",
                    "supported_options_now": {
                        "dry_run": True,
                        "steps": True,
                        "exclude": True,
                        "start_at": True,
                        "stop_after": True,
                        "stop_on_error": True,
                        "strict": True,
                    },
                    "endpoints": {
                        "health": ["/health", "/api/v1/health"],
                        "pipeline_status": ["/pipeline/status", "/api/v1/pipeline/status"],
                        "pipeline_run_post": ["/pipeline/run (POST)", "/api/v1/pipeline/run (POST)"],
                        "pipeline_run_browser": [
                            "/pipeline/run?execute=1&confirm=yes (GET)",
                            "/api/v1/pipeline/run?execute=1&confirm=yes (GET)",
                        ],
                    },
                    "examples": {
                        "run_all": "/api/v1/pipeline/run?execute=1&confirm=yes",
                        "dry_run": "/api/v1/pipeline/run?execute=1&confirm=yes&dry_run=1",
                        "subset": "/api/v1/pipeline/run?execute=1&confirm=yes&steps=1_data_ingestion,2_data_warehouse",
                        "exclude": "/api/v1/pipeline/run?execute=1&confirm=yes&exclude=7_api_serving",
                        "range": "/api/v1/pipeline/run?execute=1&confirm=yes&start_at=3_feature_engineering&stop_after=5_ml_prediction",
                        "stop_on_error": "/api/v1/pipeline/run?execute=1&confirm=yes&stop_on_error=1",
                        "non_strict": "/api/v1/pipeline/run?execute=1&confirm=yes&strict=0",
                    },
                },
            }
        )

    # ----------------------------
    # Pipeline run (GET helper for browsers)
    # ----------------------------
    @bp.get("/pipeline/run")
    @bp.get("/api/v1/pipeline/run")
    def pipeline_run_browser() -> Any:
        """
        Safari opens URLs with GET.
        Default shows instructions.
        To execute via browser:
            ?execute=1&confirm=yes

        Supported options:
            &dry_run=1
            &strict=0
            &steps=...
            &exclude=...
            &start_at=...
            &stop_after=...
            &stop_on_error=1
        """
        execute = (request.args.get("execute") or "").strip().lower()
        confirm = (request.args.get("confirm") or "").strip().lower()

        if execute in {"1", "true", "yes"} and confirm == "yes":
            from . import orchestrator  # local import to avoid circular dependency

            options = _collect_options_from_query(request.args)

            started = time.time()
            request_id = uuid.uuid4().hex
            try:
                result = _run_pipeline(orchestrator, options)
                duration_ms = int((time.time() - started) * 1000)
                run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request_id[:8]}"

                return _response_ok(
                    {
                        "request_id": request_id,
                        "run_id": run_id,
                        "duration_ms": duration_ms,
                        "requested_payload": {"_via": "browser_get", **options},
                        "orchestrator_options_used": options,
                        "result": result,
                    }
                )
            except Exception as e:
                _log(f"pipeline_run_browser failed: {type(e).__name__}: {e}")
                return _response_error(
                    "Pipeline execution failed",
                    500,
                    details={"request_id": request_id, "error_type": type(e).__name__, "error": str(e)},
                )

        return _response_ok(
            {
                "service": "restaurant-ai-platform",
                "message": "Use POST to /api/v1/pipeline/run for normal clients. "
                "For Safari browser testing, call GET with ?execute=1&confirm=yes.",
                "how_to_test_in_browser": {
                    "safe_check": "/api/v1/pipeline/status",
                    "execute_pipeline": "/api/v1/pipeline/run?execute=1&confirm=yes",
                    "dry_run_example": "/api/v1/pipeline/run?execute=1&confirm=yes&dry_run=1",
                    "steps_example": "/api/v1/pipeline/run?execute=1&confirm=yes&steps=1_data_ingestion,2_data_warehouse",
                    "range_example": "/api/v1/pipeline/run?execute=1&confirm=yes&start_at=3_feature_engineering&stop_after=5_ml_prediction",
                },
            }
        )

    # ----------------------------
    # Pipeline run (POST)
    # ----------------------------
    @bp.post("/pipeline/run")
    @bp.post("/api/v1/pipeline/run")
    def pipeline_run_post() -> Any:
        from . import orchestrator  # local import to avoid circular dependency

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

            return _response_ok(
                {
                    "request_id": request_id,
                    "run_id": run_id,
                    "duration_ms": duration_ms,
                    "requested_payload": payload,
                    "orchestrator_options_used": options,
                    "result": result,
                }
            )

        except Exception as e:
            _log(f"pipeline_run_post failed: {type(e).__name__}: {e}")
            return _response_error(
                "Pipeline execution failed",
                500,
                details={"request_id": request_id, "error_type": type(e).__name__, "error": str(e)},
            )

except Exception as e:
    _log(f"Flask blueprint not available: {type(e).__name__}: {e}")


def register(app: Any) -> bool:
    """
    Call this from app_factory.py:

        from . import api_serving
        api_serving.register(app)

    Returns True if blueprint was registered.
    """
    if bp is None:
        return False

    try:
        app.register_blueprint(bp)
        return True
    except Exception as e:
        _log(f"Blueprint registration failed: {type(e).__name__}: {e}")
        return False


def run() -> Dict[str, Any]:
    """
    Pipeline step execution.
    Does NOT start the server.
    """
    _log("api_serving.run() started")

    result: Dict[str, Any] = {
        "api_serving_status": "ok",
        "blueprint_available": bp is not None,
        "expected_endpoints": [
            "/health",
            "/pipeline/status",
            "/pipeline/run (POST)",
            "/pipeline/run?execute=1&confirm=yes (GET)",
            "/api/v1/health",
            "/api/v1/pipeline/status",
            "/api/v1/pipeline/run (POST)",
            "/api/v1/pipeline/run?execute=1&confirm=yes (GET)",
        ],
        "timestamp": _utc_ts(),
    }

    _log("api_serving.run() completed")
    return result