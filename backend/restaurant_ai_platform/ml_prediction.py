from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
import json
import os


MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/restaurant_ai_models")
MODEL_NAME = os.getenv("MODEL_NAME", "sales_forecast_baseline")
LATEST_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}__LATEST.json")


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _log(msg: str) -> None:
    print(f"[{_utc_ts()}] {msg}")


def _load_latest_model() -> Dict[str, Any]:
    if not os.path.exists(LATEST_PATH):
        return {"error": "model_not_found"}

    try:
        with open(LATEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"model_load_failed:{type(e).__name__}:{e}"}


def _fallback_prediction(horizon: int) -> List[float]:
    return [1000.0 for _ in range(horizon)]


def run() -> Dict[str, Any]:
    _log("ml_prediction.run() started")

    horizon = 7  # default forecast for next 7 days
    model = _load_latest_model()

    if "error" in model:
        preds = _fallback_prediction(horizon)
        result = {
            "ml_prediction_status": "fallback",
            "reason": model["error"],
            "horizon": horizon,
            "predictions": preds,
            "timestamp": _utc_ts(),
        }
        _log("ml_prediction.run() fallback completed")
        return {"data": result, "errors": [], "warnings": [{"message": model["error"]}], "metrics": {}}

    median_value = float(model.get("median_value", 0.0))
    preds = [median_value for _ in range(horizon)]

    result = {
        "ml_prediction_status": "ok",
        "model_used": {
            "model_name": model.get("model_name"),
            "model_version": model.get("model_version"),
            "trained_at": model.get("trained_at"),
        },
        "horizon": horizon,
        "predictions": preds,
        "timestamp": _utc_ts(),
    }

    _log("ml_prediction.run() completed")
    return {"data": result, "errors": [], "warnings": [], "metrics": {}}