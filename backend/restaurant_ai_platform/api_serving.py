from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _log(message: str) -> None:
    print(f"[{_utc_ts()}] {message}")


# Optional Flask Blueprint support
bp = None  # type: ignore[assignment]

try:
    from flask import Blueprint, jsonify  # type: ignore

    bp = Blueprint("api_serving", __name__)

    @bp.get("/health")
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "service": "restaurant-ai-platform",
                "timestamp": _utc_ts(),
            }
        )

    @bp.get("/pipeline/run")
    def pipeline_run() -> Any:
        from . import orchestrator  # local import to avoid circular dependency

        result = orchestrator.run_pipeline()
        return jsonify(result)

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
        "expected_endpoints": ["/", "/health", "/pipeline/run"],
        "timestamp": _utc_ts(),
    }

    _log("api_serving.run() completed")
    return result