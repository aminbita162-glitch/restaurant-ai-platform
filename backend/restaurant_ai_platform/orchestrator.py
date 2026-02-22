from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict


PIPELINE_ORDER = [
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
    Executes a pipeline step and returns a dict payload.
    Steps not implemented yet will be skipped safely.
    """
    _log(f"START {step_name}")

    registry: Dict[str, Callable[[], Dict[str, Any]]] = {
        "1_data_ingestion": _run_data_ingestion,
        "2_data_warehouse": _run_data_warehouse,
    }

    fn = registry.get(step_name)
    if fn is None:
        _log(f"SKIP  {step_name} (not implemented)")
        return {"step": step_name, "status": "skipped", "timestamp": _utc_ts()}

    result = fn()
    _log(f"DONE  {step_name}")
    return {"step": step_name, "status": "ok", "result": result, "timestamp": _utc_ts()}


def run_pipeline() -> Dict[str, Any]:
    results = []
    for step in PIPELINE_ORDER:
        results.append(run_step(step))
    return {"status": "ok", "results": results, "timestamp": _utc_ts()}


def _run_data_ingestion() -> Dict[str, Any]:
    # Local import to avoid hard failures if package layout changes later
    from . import data_ingestion

    return data_ingestion.run()


def _run_data_warehouse() -> Dict[str, Any]:
    # Local import to avoid hard failures if package layout changes later
    from . import data_warehouse

    return data_warehouse.run()


if __name__ == "__main__":
    run_pipeline()