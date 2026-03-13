import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

DEFAULT_RESTAURANT_ID = "restaurant_001"
DEFAULT_LOCATION_ID = "location_001"


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _ensure_model_dir() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)


def _safe_id(value: str) -> str:
    return str(value).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def _build_model_path(restaurant_id: str, location_id: str) -> str:
    restaurant_part = _safe_id(restaurant_id or DEFAULT_RESTAURANT_ID)
    location_part = _safe_id(location_id or DEFAULT_LOCATION_ID)
    filename = f"model__{restaurant_part}__{location_part}.json"
    return os.path.join(MODEL_DIR, filename)


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


def _extract_restaurant_context_from_result(result: Dict[str, Any]) -> Tuple[str, str]:
    results = result.get("results")
    if not isinstance(results, list):
        return DEFAULT_RESTAURANT_ID, DEFAULT_LOCATION_ID

    for step in results:
        if not isinstance(step, dict):
            continue
        if step.get("step") != "1_data_ingestion":
            continue
        data = step.get("data")
        if not isinstance(data, dict):
            continue

        restaurant_id = str(data.get("restaurant_id") or DEFAULT_RESTAURANT_ID)
        location_id = str(data.get("location_id") or DEFAULT_LOCATION_ID)

        return restaurant_id, location_id

    return DEFAULT_RESTAURANT_ID, DEFAULT_LOCATION_ID


def train_and_save_model(
    features: Dict[str, Any],
    restaurant_id: Optional[str] = None,
    location_id: Optional[str] = None,
) -> Dict[str, Any]:
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

    restaurant_id = str(restaurant_id or DEFAULT_RESTAURANT_ID)
    location_id = str(location_id or DEFAULT_LOCATION_ID)

    avg_daily_sales = total / count

    model = {
        "model_name": "average_daily_sales_model",
        "version": "1.0",
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        "trained_at": _utc_ts(),
        "avg_daily_sales": avg_daily_sales,
        "training_rows": count,
    }

    _ensure_model_dir()
    model_path = _build_model_path(restaurant_id, location_id)

    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f)

    return model


def load_model(
    restaurant_id: Optional[str] = None,
    location_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Loads the saved model from disk.
    """
    restaurant_id = str(restaurant_id or DEFAULT_RESTAURANT_ID)
    location_id = str(location_id or DEFAULT_LOCATION_ID)

    model_path = _build_model_path(restaurant_id, location_id)

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found")

    with open(model_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if not isinstance(obj, dict):
        raise ValueError("Model file is invalid")

    return obj  # type: ignore[return-value]


def predict(
    restaurant_id: Optional[str] = None,
    location_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Uses stored average to simulate next day prediction.
    """
    model = load_model(restaurant_id=restaurant_id, location_id=location_id)

    avg = model.get("avg_daily_sales")
    if not isinstance(avg, (int, float)):
        raise ValueError("Model is missing avg_daily_sales")

    prediction = {
        "restaurant_id": model.get("restaurant_id", DEFAULT_RESTAURANT_ID),
        "location_id": model.get("location_id", DEFAULT_LOCATION_ID),
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

    restaurant_id, location_id = _extract_restaurant_context_from_result(result)

    model = train_and_save_model(
        {"sales": sales},
        restaurant_id=restaurant_id,
        location_id=location_id,
    )

    return {
        "model_registry_status": "ok",
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        "model_saved": True,
        "model_path": os.path.basename(_build_model_path(restaurant_id, location_id)),
        "model": model,
        "timestamp": _utc_ts(),
    }