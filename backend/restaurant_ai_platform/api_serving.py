from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
import time
import uuid


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _log(message: str) -> None:
    print(f"[{_utc_ts()}] {message}")


def _response_ok(data: Dict[str, Any], status_code: int = 200) -> Any:
    # Imported lazily so this module can still be imported without Flask
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
                    "endpoints": {
                        "health": ["/health", "/api/v1/health"],
                        "pipeline_status": ["/pipeline/status", "/api/v1/pipeline/status"],
                        "pipeline_run_post": ["/pipeline/run (POST)", "/api/v1/pipeline/run (POST)"],
                        "pipeline_run_browser": [
                            "/pipeline/run?execute=1&confirm=yes (GET)",
                            "/api/v1/pipeline/run?execute=1&confirm=yes (GET)",
                        ],
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
        Browsers (Safari) always send GET when opening a URL.
        This endpoint provides instructions by default.
        To actually execute via browser, call with:
            ?execute=1&confirm=yes
        """
        execute = (request.args.get("execute") or "").strip().lower()
        confirm = (request.args.get("confirm") or "").strip().lower()

        if execute in {"1", "true", "yes"} and confirm == "yes":
            # Run the pipeline (same logic as POST)
            from . import orchestrator  # local import to avoid circular dependency

            started = time.time()
            request_id = uuid.uuid4().hex
            try:
                result = orchestrator.run_pipeline()
                duration_ms = int((time.time() - started) * 1000)
                run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request_id[:8]}"

                return _response_ok(
                    {
                        "request_id": request_id,
                        "run_id": run_id,
                        "duration_ms": duration_ms,
                        "requested_payload": {"_via": "browser_get"},
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

        # Default: show instructions
        return _response_ok(
            {
                "service": "restaurant-ai-platform",
                "message": "This endpoint is POST-only for normal clients. Use POST to /api/v1/pipeline/run. "
                "For Safari browser testing, call with ?execute=1&confirm=yes.",
                "how_to_test_in_browser": {
                    "safe_check": "/api/v1/pipeline/status",
                    "execute_pipeline": "/api/v1/pipeline/run?execute=1&confirm=yes",
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

            result = orchestrator.run_pipeline()

            duration_ms = int((time.time() - started) * 1000)
            run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{request_id[:8]}"

            return _response_ok(
                {
                    "request_id": request_id,
                    "run_id": run_id,
                    "duration_ms": duration_ms,
                    "requested_payload": payload,
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