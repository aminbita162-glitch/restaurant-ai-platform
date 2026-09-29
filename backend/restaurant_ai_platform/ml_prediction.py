from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

# Stable public error codes for forecast failures.
FORECAST_ERROR_MISSING_SALES = "FORECAST_MISSING_SALES"
FORECAST_ERROR_INSUFFICIENT_DATA = "FORECAST_INSUFFICIENT_DATA"
FORECAST_ERROR_INVALID_SALES = "FORECAST_INVALID_SALES"

# Minimum number of valid daily totals required to produce a forecast.
MIN_SALES_ROWS = 7


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _extract_daily_totals(sales: Any) -> Optional[List[float]]:
    """Return a list of float daily_sales_total values, or None if data is unusable."""
    if not isinstance(sales, list) or not sales:
        return None

    totals: List[float] = []
    for row in sales:
        if not isinstance(row, dict):
            return None
        v = row.get("daily_sales_total")
        if not isinstance(v, (int, float)):
            return None
        totals.append(float(v))

    return totals if totals else None


def run(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Produce a 7-day heuristic sales forecast from actual sales rows in context.

    Method: simple average of all supplied daily totals, compounded at a fixed
    2 % daily growth factor. This is a heuristic — it is NOT a trained ML model.
    The output field `method` is always set to `heuristic` to reflect this.
    """
    ctx = context or {}
    restaurant_id = ctx.get("restaurant_id")
    location_id = ctx.get("location_id")
    sales = ctx.get("sales")
    horizon = 7

    # Guard: sales must be present.
    if sales is None or (isinstance(sales, list) and len(sales) == 0):
        code = FORECAST_ERROR_MISSING_SALES
        print(f"[{_utc_ts()}] ml_prediction status=error code={code}")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    # Guard: sales must be valid numeric rows.
    totals = _extract_daily_totals(sales)
    if totals is None:
        code = FORECAST_ERROR_INVALID_SALES
        print(f"[{_utc_ts()}] ml_prediction status=error code={code}")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    # Guard: require at least MIN_SALES_ROWS rows to avoid thin-data forecasts.
    if len(totals) < MIN_SALES_ROWS:
        code = FORECAST_ERROR_INSUFFICIENT_DATA
        print(
            f"[{_utc_ts()}] ml_prediction status=error code={code}"
            f" rows={len(totals)} required={MIN_SALES_ROWS}"
        )
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "sales_rows_provided": len(totals),
            "sales_rows_required": MIN_SALES_ROWS,
            "timestamp": _utc_ts(),
        }

    # Heuristic: average of all supplied daily totals.
    avg = sum(totals) / len(totals)
    growth = 0.02

    forecast: List[Dict[str, float]] = []
    current = avg
    for _ in range(horizon):
        current = round(current * (1.0 + growth), 2)
        forecast.append({"predicted_sales": current})

    print(
        f"[{_utc_ts()}] ml_prediction status=ok"
        f" method=heuristic rows={len(totals)} avg={round(avg,2)}"
    )

    return {
        "ml_prediction_status": "ok",
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        # method is always heuristic — this is an average, not a trained model.
        "method": "heuristic",
        "method_description": (
            "average of historical daily sales compounded at 2% daily growth"
        ),
        "sales_rows_used": len(totals),
        "avg_daily_sales_used": round(avg, 4),
        "horizon": horizon,
        "forecast": forecast,
        "timestamp": _utc_ts(),
    }
