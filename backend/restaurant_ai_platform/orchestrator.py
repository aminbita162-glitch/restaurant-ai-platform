from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, Any


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


def _log(message: str) -> None:
    print(f"[{datetime.utcnow().isoformat()}] {message}")


def run_step(step_name: str) -> Dict[str, Any]:
    """
    Executes a pipeline step and returns a dict payload.
    Steps not implemented yet will be skipped safely.
    """
    _log(f"START {step_name}")

    registry: Dict[str, Callable[[], Dict[str, Any]]] = {
        "1_data_ingestion": _run_data_ingestion,
    }

    fn = registry.get(step_name)
    if fn is None:
        _log(f"SKIP  {step_name} (not implemented)")
        return {"step": step_name, "status": "skipped", "timestamp": datetime.utcnow().isoformat()}

    result = fn()
    _log(f"DONE  {step_name}")
    return {"step": step_name, "status": "ok", "result": result, "timestamp": datetime.utcnow().isoformat()}


def run_pipeline() -> Dict[str, Any]:
    results = []
    for step in PIPELINE_ORDER:
        results.append(run_step(step))
    return {"status": "ok", "results": results, "timestamp": datetime.utcnow().isoformat()}


def _run_data_ingestion() -> Dict[str, Any]:
    # Local import to avoid hard failures if package layout changes later
    from . import data_ingestion

    return data_ingestion.run()


if __name__ == "__main__":
    run_pipeline()