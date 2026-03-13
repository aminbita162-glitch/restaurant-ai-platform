from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional


DEFAULT_RESTAURANT_ID = "restaurant_001"
DEFAULT_LOCATION_ID = "location_001"


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _extract_restaurant_context() -> tuple[str, str]:
    try:
        from . import model_registry

        result = model_registry._get_latest_pipeline_result()
        if not isinstance(result, dict):
            return DEFAULT_RESTAURANT_ID, DEFAULT_LOCATION_ID

        return model_registry._extract_restaurant_context_from_result(result)
    except Exception:
        return DEFAULT_RESTAURANT_ID, DEFAULT_LOCATION_ID


def run() -> Dict[str, Any]:
    """
    Uses the registered demo model from model_registry.py
    and produces a simple 7-day forecast based on avg_daily_sales.
    """
    horizon = 7

    try:
        from . import model_registry
    except Exception as e:
        return {
            "data": {
                "ml_prediction_status": "error",
                "reason": f"import_model_registry_failed:{type(e).__name__}:{e}",
                "timestamp": _utc_ts(),
            },
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

    restaurant_id, location_id = _extract_restaurant_context()

    try:
        model = model_registry.load_model(
            restaurant_id=restaurant_id,
            location_id=location_id,
        )
    except FileNotFoundError:
        return {
            "data": {
                "ml_prediction_status": "error",
                "restaurant_id": restaurant_id,
                "location_id": location_id,
                "reason": "model_not_found",
                "timestamp": _utc_ts(),
            },
            "errors": [],
            "warnings": [],
            "metrics": {},
        }
    except Exception as e:
        return {
            "data": {
                "ml_prediction_status": "error",
                "restaurant_id": restaurant_id,
                "location_id": location_id,
                "reason": f"model_load_failed:{type(e).__name__}:{e}",
                "timestamp": _utc_ts(),
            },
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

    try:
        base = float(model["avg_daily_sales"])
        model_name = str(model.get("model_name", "unknown_model"))
        model_version = str(model.get("version", "unknown_version"))
        trained_at = str(model.get("trained_at", "unknown_time"))
    except Exception as e:
        return {
            "data": {
                "ml_prediction_status": "error",
                "restaurant_id": restaurant_id,
                "location_id": location_id,
                "reason": f"invalid_model_format:{type(e).__name__}:{e}",
                "timestamp": _utc_ts(),
            },
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

    growth = 0.02
    forecast: List[Dict[str, float]] = []
    current = base

    for _ in range(horizon):
        current = round(current * (1.0 + growth), 2)
        forecast.append({"predicted_sales": current})

    return {
        "data": {
            "ml_prediction_status": "ok",
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "model_used": {
                "model_name": model_name,
                "model_version": model_version,
                "trained_at": trained_at,
            },
            "horizon": horizon,
            "forecast": forecast,
            "timestamp": _utc_ts(),
        },
        "errors": [],
        "warnings": [],
        "metrics": {},
    }