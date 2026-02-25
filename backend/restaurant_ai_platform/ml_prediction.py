from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def run() -> Dict[str, Any]:
    """
    Uses the registered demo model from model_registry.py (model.json)
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

    try:
        model = model_registry.load_model()
    except FileNotFoundError:
        return {
            "data": {
                "ml_prediction_status": "error",
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
                "reason": f"invalid_model_format:{type(e).__name__}:{e}",
                "timestamp": _utc_ts(),
            },
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

    # Simple forecast: assume 2% daily growth (demo)
    growth = 0.02
    forecast: List[Dict[str, float]] = []
    current = base
    for _ in range(horizon):
        current = round(current * (1.0 + growth), 2)
        forecast.append({"predicted_sales": current})

    return {
        "data": {
            "ml_prediction_status": "ok",
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