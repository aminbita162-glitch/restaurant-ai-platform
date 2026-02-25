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
    weekday_factors = model["weekday_factors"]
    start_index = int(model["rows"])

    preds: List[float] = []

    for i in range(horizon):
        t = start_index + i
        base = intercept + slope * t
        season = weekday_factors.get(str(t % 7), 0.0)
        preds.append(max(0.0, base + season))

    return {
        "data": {
            "ml_prediction_status": "ok",
            "model_used": {
                "model_name": model["model_name"],
                "model_version": model["model_version"],
                "trained_at": model["trained_at"],
            },
            "horizon": horizon,
            "predictions": preds,
            "timestamp": _utc_ts(),
        },
        "errors": [],
        "warnings": [],
        "metrics": {},
    }