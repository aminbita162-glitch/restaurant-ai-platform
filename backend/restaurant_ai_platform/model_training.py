from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple
import json
import math
import os
import sqlite3
import uuid


DEFAULT_DB_PATH = os.getenv("PIPELINE_DB_PATH", "/tmp/restaurant_ai_pipeline.db")
MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/restaurant_ai_models")
MODEL_NAME = os.getenv("MODEL_NAME", "sales_forecast_pro")


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _synthetic_series(n: int = 365) -> List[float]:
    ys = []
    base = 1200.0
    for i in range(n):
        trend = base + i * 0.9
        weekly = 1.0 + 0.2 * math.sin(2 * math.pi * (i % 7) / 7)
        noise = math.sin(i * 0.13) * 30
        ys.append(max(0.0, trend * weekly + noise))
    return ys


def _linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    slope = num / den if den != 0 else 0.0
    intercept = mean_y - slope * mean_x
    return intercept, slope


def _residual_std(xs: List[float], ys: List[float], intercept: float, slope: float) -> float:
    residuals = []
    for i in range(len(xs)):
        pred = intercept + slope * xs[i]
        residuals.append(ys[i] - pred)
    mean_res = sum(residuals) / len(residuals)
    var = sum((r - mean_res) ** 2 for r in residuals) / len(residuals)
    return math.sqrt(var)


def run() -> Dict[str, Any]:
    series = _synthetic_series()
    xs = list(range(len(series)))

    intercept, slope = _linear_regression(xs, series)
    std_error = _residual_std(xs, series, intercept, slope)

    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    artifact = {
        "model_name": MODEL_NAME,
        "model_version": version,
        "intercept": intercept,
        "slope": slope,
        "std_error": std_error,
        "rows": len(series),
        "trained_at": _utc_ts(),
    }

    _ensure_dir(MODEL_DIR)

    path = os.path.join(MODEL_DIR, f"{MODEL_NAME}__{version}.json")
    latest = os.path.join(MODEL_DIR, f"{MODEL_NAME}__LATEST.json")

    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)

    with open(latest, "w") as f:
        json.dump(artifact, f, indent=2)

    return {
        "data": {
            "model_saved": artifact,
            "training_status": "ok",
        },
        "errors": [],
        "warnings": [],
        "metrics": {},
    }