from datetime import datetime
from typing import Any, Dict, List, Optional

# Stable public error codes for staffing failures.
LABOR_ERROR_MISSING_INPUT = "LABOR_MISSING_SALES_OR_FORECAST"
LABOR_ERROR_FORECAST_FAILED = "LABOR_FORECAST_FAILED"

# Rule constant: one staff member per this many sales units.
SALES_PER_STAFF = 600


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _staff_for_sales(predicted_sales: float) -> int:
    """Simple ratio rule: 1 staff per SALES_PER_STAFF units (minimum 1)."""
    if predicted_sales <= 0:
        return 1
    return max(1, int(predicted_sales / SALES_PER_STAFF))


def run(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Produce a staffing plan using a fixed sales-per-staff ratio rule.

    This is a simple rule — it is NOT a constraint optimizer or ML model.
    The output field `method` is always set to `rule` to reflect this.
    """
    print(f"[{_utc_ts()}] START optimization")

    ctx = context or {}
    restaurant_id = ctx.get("restaurant_id")
    location_id = ctx.get("location_id")
    sales = ctx.get("sales")

    # Guard: sales input must be present so the forecast step can run.
    if not sales or not isinstance(sales, list):
        code = LABOR_ERROR_MISSING_INPUT
        print(f"[{_utc_ts()}] optimization status=error code={code}")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    # C4: read the forecast already produced on the context artifact.
    # Do NOT import or rerun ml_prediction here — that would create a
    # duplicate forecast artifact. The upstream 5_ml_prediction step is
    # responsible for producing the forecast; we only consume it.
    forecast_result = ctx.get("5_ml_prediction")

    # Guard: forecast artifact must be present and successful.
    if not isinstance(forecast_result, dict) or forecast_result.get("status") == "error":
        code = LABOR_ERROR_FORECAST_FAILED
        upstream_code = (
            forecast_result.get("error_code", "missing_forecast_artifact")
            if isinstance(forecast_result, dict)
            else "missing_forecast_artifact"
        )
        print(f"[{_utc_ts()}] optimization status=error code={code} upstream={upstream_code}")
        return {
            "status": "error",
            "error_code": code,
            "upstream_error_code": upstream_code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    forecast = forecast_result.get("forecast", [])

    # Guard: forecast list must be non-empty.
    if not forecast or not isinstance(forecast, list):
        code = LABOR_ERROR_MISSING_INPUT
        print(f"[{_utc_ts()}] optimization status=error code={code} reason=empty_forecast")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    # C6: read the backtest quality gate from the forecast artifact.
    labor_recommendation_allowed = forecast_result.get("labor_recommendation_allowed", False)

    # Apply the ratio rule to each forecast day.
    staffing_plan: List[Dict[str, Any]] = []
    for i, day in enumerate(forecast):
        predicted = float(day.get("predicted_sales", 0))
        staffing_plan.append(
            {
                "day_index": i + 1,
                "predicted_sales": predicted,
                "recommended_staff": _staff_for_sales(predicted),
            }
        )

    print(
        f"[{_utc_ts()}] DONE optimization method=rule"
        f" days={len(staffing_plan)}"
        f" labor_recommendation_allowed={labor_recommendation_allowed}"
    )

    return {
        "optimization_status": "ok",
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        # method is always rule — this is a ratio, not a constraint optimizer.
        "method": "rule",
        "method_description": f"1 staff per {SALES_PER_STAFF} sales units (minimum 1)",
        "staffing_plan": staffing_plan,
        # C6: expose the backtest quality gate.
        "labor_recommendation_allowed": labor_recommendation_allowed,
        "timestamp": _utc_ts(),
    }
