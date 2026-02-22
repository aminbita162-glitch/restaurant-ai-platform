from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List


PIPELINE_ORDER: List[str] = [
    "1_data_ingestion",
    "2_data_warehouse",
    "3_feature_engineering",
    "4_feature_store_sync",
    "5_ml_prediction",
    "6_optimization",
    "7_api_serving",
    "8_dashboard_update",
]


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _log(message: str) -> None:
    print(f"[{_utc_ts()}] {message}")


def run_step(step_name: str) -> Dict[str, Any]:
    """
    Executes one pipeline step and returns a structured payload.
    - Missing/unimplemented steps are skipped safely.
    - Exceptions are captured so the pipeline can continue.
    """
    _log(f"START {step_name}")

    registry: Dict[str, Callable[[], Dict[str, Any]]] = {
        "1_data_ingestion": _run_data_ingestion,
        "2_data_warehouse": _run_data_warehouse,
        "3_feature_engineering": _run_feature_engineering,
        "4_feature_store_sync": _run_feature_store_sync,
        "5_ml_prediction": _run_ml_prediction,
        "6_optimization": _run_optimization,
        "7_api_serving": _run_api_serving,
        "8_dashboard_update": _run_dashboard_update,
    }

    fn = registry.get(step_name)
    if fn is None:
        _log(f"SKIP  {step_name} (no handler registered)")
        return {
            "step": step_name,
            "status": "skipped",
            "reason": "no_handler",
            "timestamp": _utc_ts(),
        }

    try:
        result = fn()
        _log(f"DONE  {step_name}")
        return {
            "step": step_name,
            "status": "ok",
            "result": result,
            "timestamp": _utc_ts(),
        }
    except ImportError as e:
        # Covers ModuleNotFoundError and other import-related errors
        _log(f"SKIP  {step_name} (import error: {e})")
        return {
            "step": step_name,
            "status": "skipped",
            "reason": "import_error",
            "error": str(e),
            "timestamp": _utc_ts(),
        }
    except Exception as e:
        _log(f"ERROR {step_name} ({type(e).__name__}: {e})")
        return {
            "step": step_name,
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
            "timestamp": _utc_ts(),
        }


def run_pipeline() -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for step in PIPELINE_ORDER:
        results.append(run_step(step))
    return {"status": "ok", "results": results, "timestamp": _utc_ts()}


def _run_data_ingestion() -> Dict[str, Any]:
    from . import data_ingestion

    return data_ingestion.run()


def _run_data_warehouse() -> Dict[str, Any]:
    from . import data_warehouse

    return data_warehouse.run()


def _run_feature_engineering() -> Dict[str, Any]:
    from . import feature_engineering

    return feature_engineering.run()


def _run_feature_store_sync() -> Dict[str, Any]:
    from . import feature_store_sync

    return feature_store_sync.run()


def _run_ml_prediction() -> Dict[str, Any]:
    from . import ml_prediction

    return ml_prediction.run()


def _run_optimization() -> Dict[str, Any]:
    from . import optimization

    return optimization.run()


def _run_api_serving() -> Dict[str, Any]:
    from . import api_serving

    return api_serving.run()


def _run_dashboard_update() -> Dict[str, Any]:
    from . import dashboard_update

    return dashboard_update.run()


if __name__ == "__main__":
    output = run_pipeline()
    print(output)