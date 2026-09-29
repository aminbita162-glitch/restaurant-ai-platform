from datetime import datetime
from typing import Any, Dict, List, Optional

# Stable public error codes for feature engineering failures.
FEATURE_ERROR_MISSING_SALES = "FEATURE_ENGINEERING_MISSING_SALES"
FEATURE_ERROR_INVALID_SALES = "FEATURE_ENGINEERING_INVALID_SALES"


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


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[{_utc_ts()}] START feature_engineering")

    restaurant_id = context.get("restaurant_id")
    location_id = context.get("location_id")
    sales = context.get("sales")

    # Guard: sales must be present.
    if sales is None or (isinstance(sales, list) and len(sales) == 0):
        code = FEATURE_ERROR_MISSING_SALES
        print(f"[{_utc_ts()}] feature_engineering status=error code={code}")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    # Guard: sales must be a list of dicts with numeric daily_sales_total.
    totals = _extract_daily_totals(sales)
    if totals is None:
        code = FEATURE_ERROR_INVALID_SALES
        print(f"[{_utc_ts()}] feature_engineering status=error code={code}")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    # Compute real numeric features from actual sales rows.
    n = len(totals)
    total_sales = sum(totals)
    avg_daily_sales = total_sales / n
    max_daily_sales = max(totals)
    min_daily_sales = min(totals)

    # Simple 7-day rolling average (or all rows if fewer than 7).
    window = totals[-7:] if n >= 7 else totals
    rolling_avg_7d = sum(window) / len(window)

    print(f"[{_utc_ts()}] DONE feature_engineering rows={n}")

    return {
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        "features_status": "ok",
        "generated_features": {
            "sales_row_count": n,
            "total_sales": round(total_sales, 4),
            "avg_daily_sales": round(avg_daily_sales, 4),
            "max_daily_sales": round(max_daily_sales, 4),
            "min_daily_sales": round(min_daily_sales, 4),
            "rolling_avg_7d": round(rolling_avg_7d, 4),
        },
        "timestamp": _utc_ts(),
    }
