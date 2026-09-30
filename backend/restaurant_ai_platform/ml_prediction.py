from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Stable public error codes for forecast failures.
FORECAST_ERROR_MISSING_SALES = "FORECAST_MISSING_SALES"
FORECAST_ERROR_INSUFFICIENT_DATA = "FORECAST_INSUFFICIENT_DATA"
FORECAST_ERROR_INVALID_SALES = "FORECAST_INVALID_SALES"

# Minimum number of valid daily totals required to produce a forecast.
MIN_SALES_ROWS = 7

# Fixed daily growth factor applied on top of the day-of-week average.
DOW_GROWTH = 0.02

# Weekday names indexed by datetime.weekday() (0=Mon … 6=Sun).
_WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _parse_date(value: Any) -> Optional[datetime]:
    """Best-effort parse of a date string/datetime into a datetime.

    Accepts ISO 8601 strings and datetime objects. Returns None on failure.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # Try ISO 8601 (covers "2026-09-30" and "2026-09-30T12:00:00").
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        pass
    # Try common compact date formats.
    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


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


def _build_dow_averages(
    sales: List[Dict[str, Any]],
    totals: List[float],
) -> Optional[List[float]]:
    """Return a 7-element list of per-weekday average sales (index 0=Mon … 6=Sun).

    Returns None when no rows carry a parseable date — in that case the caller
    falls back to a flat average.
    """
    dow_sums: List[float] = [0.0] * 7
    dow_counts: List[int] = [0] * 7

    dated = 0
    for row, total in zip(sales, totals):
        dt = _parse_date(row.get("date")) if isinstance(row, dict) else None
        if dt is None:
            continue
        wd = dt.weekday()
        dow_sums[wd] += total
        dow_counts[wd] += 1
        dated += 1

    if dated == 0:
        return None

    # Compute per-weekday average; for weekdays with no data, fall back to the
    # overall average so every forecast day still has a value.
    overall_avg = sum(totals) / len(totals) if totals else 0.0
    return [
        (dow_sums[i] / dow_counts[i]) if dow_counts[i] > 0 else overall_avg
        for i in range(7)
    ]


def _last_date(sales: List[Dict[str, Any]]) -> Optional[datetime]:
    """Return the most recent parseable date among sales rows, or None."""
    best: Optional[datetime] = None
    for row in sales:
        if not isinstance(row, dict):
            continue
        dt = _parse_date(row.get("date"))
        if dt is not None and (best is None or dt > best):
            best = dt
    return best


def run(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Produce a 7-day heuristic sales forecast from actual sales rows in context.

    Method: day-of-week buckets. Each weekday's historical average is used as
    the base for that weekday's forecast day, compounded at a fixed 2 % daily
    growth factor. When rows lack parseable dates, a flat overall average is
    used as a degraded fallback (still labeled dow_heuristic to stay honest
    about the method family — it is never labeled ml_model).

    The output field ``method`` is always set to ``dow_heuristic``.
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

    # Build day-of-week averages when dates are available; otherwise fall back
    # to a flat average for every day.
    sales_list = sales if isinstance(sales, list) else []
    dow_avgs = _build_dow_averages(sales_list, totals)
    overall_avg = sum(totals) / len(totals)

    if dow_avgs is not None:
        base_per_day = dow_avgs
        method_description = (
            "day-of-week average of historical daily sales compounded at 2% daily growth"
        )
        dow_method = "dow_bucketed"
    else:
        base_per_day = [overall_avg] * 7
        method_description = (
            "flat overall average (no dates available) compounded at 2% daily growth"
        )
        dow_method = "flat_fallback"

    # Determine the start weekday for the forecast. When a last historical
    # date is known, the forecast starts the day after it; otherwise it starts
    # from "tomorrow" relative to now.
    last_dt = _last_date(sales_list)
    if last_dt is not None:
        start_date = last_dt + timedelta(days=1)
    else:
        start_date = datetime.utcnow() + timedelta(days=1)

    forecast: List[Dict[str, Any]] = []
    for i in range(horizon):
        day_date = start_date + timedelta(days=i)
        weekday = day_date.weekday()
        base = base_per_day[weekday]
        # Apply cumulative growth across the 7-day horizon.
        grown = base * ((1.0 + DOW_GROWTH) ** (i + 1))
        forecast.append(
            {
                "day_index": i + 1,
                "date": day_date.date().isoformat(),
                "weekday": _WEEKDAY_NAMES[weekday],
                "predicted_sales": round(grown, 2),
            }
        )

    print(
        f"[{_utc_ts()}] ml_prediction status=ok"
        f" method=dow_heuristic rows={len(totals)}"
        f" dow_method={dow_method} avg={round(overall_avg, 2)}"
    )

    return {
        "ml_prediction_status": "ok",
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        # method is always dow_heuristic — day-of-week buckets, not a trained model.
        "method": "dow_heuristic",
        "method_description": method_description,
        "dow_method": dow_method,
        "sales_rows_used": len(totals),
        "avg_daily_sales_used": round(overall_avg, 4),
        "horizon": horizon,
        "forecast": forecast,
        "timestamp": _utc_ts(),
    }
