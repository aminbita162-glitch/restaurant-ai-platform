from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
import json
import math
import os
import uuid


MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/restaurant_ai_models")
MODEL_NAME = os.getenv("MODEL_NAME", "sales_forecast_pro")

DEFAULT_RESTAURANT_ID = "restaurant_001"
DEFAULT_LOCATION_ID = "location_001"


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_id(value: str) -> str:
    return str(value).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def _build_model_paths(restaurant_id: str, location_id: str, version: str) -> Tuple[str, str]:
    restaurant_part = _safe_id(restaurant_id or DEFAULT_RESTAURANT_ID)
    location_part = _safe_id(location_id or DEFAULT_LOCATION_ID)

    versioned_path = os.path.join(
        MODEL_DIR,
        f"{MODEL_NAME}__{restaurant_part}__{location_part}__{version}.json",
    )
    latest_path = os.path.join(
        MODEL_DIR,
        f"{MODEL_NAME}__{restaurant_part}__{location_part}__LATEST.json",
    )

    return versioned_path, latest_path


def _load_persistence_module():
    try:
        from .core import persistence  # type: ignore
        return persistence
    except Exception:
        from . import persistence  # type: ignore
        return persistence


def _get_latest_pipeline_result() -> Optional[Dict[str, Any]]:
    persistence = _load_persistence_module()

    last_run = None
    if hasattr(persistence, "get_last_run"):
        last_run = persistence.get_last_run()  # type: ignore[attr-defined]

    if not isinstance(last_run, dict):
        return None

    if isinstance(last_run.get("result"), dict):
        return last_run["result"]  # type: ignore[return-value]

    return last_run


def _extract_ingestion_payload(result: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    results = result.get("results")
    if not isinstance(results, list):
        return DEFAULT_RESTAURANT_ID, DEFAULT_LOCATION_ID, []

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
        sales = data.get("sales")

        if isinstance(sales, list):
            clean_sales = [x for x in sales if isinstance(x, dict)]
            return restaurant_id, location_id, clean_sales

        return restaurant_id, location_id, []

    return DEFAULT_RESTAURANT_ID, DEFAULT_LOCATION_ID, []


def _synthetic_series(n: int = 365) -> List[float]:
    ys = []
    base = 1200.0
    for i in range(n):
        trend = base + i * 0.9
        weekly = 1.0 + 0.2 * math.sin(2 * math.pi * (i % 7) / 7)
        noise = math.sin(i * 0.13) * 30
        ys.append(max(0.0, trend * weekly + noise))
    return ys


def _extract_real_series(rows: List[Dict[str, Any]]) -> List[float]:
    series: List[float] = []

    for row in rows:
        value = row.get("daily_sales_total")
        if isinstance(value, (int, float)):
            series.append(float(value))

    return series


def _linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return ys[0], 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))

    slope = num / den if den != 0 else 0.0
    intercept = mean_y - slope * mean_x

    return intercept, slope


def _residual_std(xs: List[float], ys: List[float], intercept: float, slope: float) -> float:
    if not xs or not ys:
        return 0.0

    residuals = []
    for i in range(len(xs)):
        pred = intercept + slope * xs[i]
        residuals.append(ys[i] - pred)

    mean_res = sum(residuals) / len(residuals)
    var = sum((r - mean_res) ** 2 for r in residuals) / len(residuals)

    return math.sqrt(var)


def run() -> Dict[str, Any]:
    result = _get_latest_pipeline_result()

    restaurant_id = DEFAULT_RESTAURANT_ID
    location_id = DEFAULT_LOCATION_ID
    training_source = "synthetic_fallback"
    warnings: List[Dict[str, Any]] = []

    if isinstance(result, dict):
        restaurant_id, location_id, sales_rows = _extract_ingestion_payload(result)
        series = _extract_real_series(sales_rows)

        if len(series) >= 2:
            training_source = "real_pipeline_sales"
        else:
            series = _synthetic_series()
            warnings.append(
                {
                    "type": "FallbackUsed",
                    "message": "Real sales data unavailable or insufficient. Synthetic fallback used.",
                }
            )
    else:
        series = _synthetic_series()
        warnings.append(
            {
                "type": "FallbackUsed",
                "message": "No persisted pipeline result found. Synthetic fallback used.",
            }
        )

    xs = list(range(len(series)))

    intercept, slope = _linear_regression(xs, series)
    std_error = _residual_std(xs, series, intercept, slope)

    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    artifact = {
        "model_name": MODEL_NAME,
        "model_version": version,
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        "training_source": training_source,
        "intercept": intercept,
        "slope": slope,
        "std_error": std_error,
        "rows": len(series),
        "trained_at": _utc_ts(),
    }

    _ensure_dir(MODEL_DIR)

    versioned_path, latest_path = _build_model_paths(
        restaurant_id=restaurant_id,
        location_id=location_id,
        version=version,
    )

    with open(versioned_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    return {
        "data": {
            "training_status": "ok",
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "training_source": training_source,
            "model_saved": artifact,
            "versioned_model_path": versioned_path,
            "latest_model_path": latest_path,
        },
        "errors": [],
        "warnings": warnings,
        "metrics": {},
    }