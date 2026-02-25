import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, List


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _load_persistence_module():
    """
    Supports both layouts:
      - backend/core/persistence.py  -> from .core import persistence
      - backend/persistence.py       -> from . import persistence
    """
    try:
        from .core import persistence  # type: ignore
        return persistence
    except Exception:
        from . import persistence  # type: ignore
        return persistence


def _get_latest_pipeline_result() -> Optional[Dict[str, Any]]:
    """
    Persistence may store either:
      - the orchestrator result dict directly, OR
      - an API wrapper that contains {"result": <orchestrator_result>, ...}
    """
    persistence = _load_persistence_module()

    last_run = None
    if hasattr(persistence, "get_last_run"):
        last_run = persistence.get_last_run()  # type: ignore[attr-defined]
    if not isinstance(last_run, dict):
        return None

    if isinstance(last_run.get("result"), dict):
        return last_run["result"]  # type: ignore[return-value]

    return last_run


def _extract_sales_from_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = result.get("results")
    if not isinstance(results, list):
        return []

    for step in results:
        if not isinstance(step, dict):
            continue
        if step.get("step") != "1_data_ingestion":
            continue
        data = step.get("data")
        if not isinstance(data, dict):
            continue
        sales = data.get("sales")
        if isinstance(sales, list) and all(isinstance(x, dict) for x in sales):
            return sales  # type: ignore[return-value]

    return []


def train_and_save_model(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple demo model training.
    Calculates average daily sales and stores it.
    Expected input:
      features["sales"] -> list of {"daily_sales_total": number, "date": "..."}
    """
    sales_data = features.get("sales", [])
    if not isinstance(sales_data, list) or not sales_data:
        raise ValueError("No sales data provided for training")

    total = 0.0
    count = 0
    for item in sales_data:
        if not isinstance(item, dict):
            continue
        v = item.get("daily_sales_total")
        if isinstance(v, (int, float)):
            total += float(v)
            count += 1

    if count == 0:
        raise ValueError("Sales data did not contain any numeric daily_sales_total values")

    avg_daily_sales = total / count

    model = {
        "model_name": "average_daily_sales_model",
        "version": "1.0",
        "trained_at": _utc_ts(),
        "avg_daily_sales": avg_daily_sales,
        "training_rows": count,
    }

    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model, f)

    return model


def load_model() -> Dict[str, Any]:
    """
    Loads the saved model from disk.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found")

    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("Model file is invalid")

    return obj  # type: ignore[return-value]


def predict() -> Dict[str, Any]:
    """
    Uses stored average to simulate next day prediction.
    """
    model = load_model()

    avg = model.get("avg_daily_sales")
    if not isinstance(avg, (int, float)):
        raise ValueError("Model is missing avg_daily_sales")

    prediction = {
        "predicted_next_day_sales": round(float(avg) * 1.02, 2),
        "model_version": str(model.get("version", "unknown")),
        "prediction_timestamp": _utc_ts(),
    }
    return prediction


def run() -> Dict[str, Any]:
    """
    Pipeline step entrypoint.
    Trains (or refreshes) the demo model by reading sales from the latest stored pipeline run.
    """
    result = _get_latest_pipeline_result()
    if result is None:
        raise ValueError("No stored pipeline run found. Run 1_data_ingestion first.")

    sales = _extract_sales_from_result(result)
    if not sales:
        raise ValueError("Model registry requires non-empty sales data from 1_data_ingestion")

    model = train_and_save_model({"sales": sales})

    return {
        "model_registry_status": "ok",
        "model_saved": True,
        "model_path": os.path.basename(MODEL_PATH),
        "model": model,
        "timestamp": _utc_ts(),
    }