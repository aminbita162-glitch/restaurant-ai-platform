from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
import json
import os


MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/restaurant_ai_models")
MODEL_NAME = os.getenv("MODEL_NAME", "sales_forecast_pro")
LATEST_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}__LATEST.json")


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _load_model() -> Dict[str, Any]:
    if not os.path.exists(LATEST_PATH):
        return {"error": "model_not_found"}
    try:
        with open(LATEST_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"model_load_failed:{type(e).__name__}:{e}"}


def run() -> Dict[str, Any]:
    model = _load_model()
    horizon = 7

    if "error" in model:
        return {
            "data": {
                "ml_prediction_status": "error",
                "reason": model["error"],
                "timestamp": _utc_ts(),
            },
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

    intercept = float(model["intercept"])
    slope = float(model["slope"])
    std_error = float(model["std_error"])
    start_index = int(model["rows"])

    preds: List[float] = []
    intervals: List[Dict[str, float]] = []

    for i in range(horizon):
        t = start_index + i
        prediction = intercept + slope * t

        lower = max(0.0, prediction - 1.96 * std_error)
        upper = prediction + 1.96 * std_error

        preds.append(prediction)
        intervals.append({
            "prediction": prediction,
            "lower_95": lower,
            "upper_95": upper,
        })

    return {
        "data": {
            "ml_prediction_status": "ok",
            "model_used": {
                "model_name": model["model_name"],
                "model_version": model["model_version"],
                "trained_at": model["trained_at"],
            },
            "confidence_level": "95%",
            "horizon": horizon,
            "forecast": intervals,
            "timestamp": _utc_ts(),
        },
        "errors": [],
        "warnings": [],
        "metrics": {},
    }