from datetime import datetime
from typing import Any, Dict, List, Optional

# Stable public error codes for inventory failures.
INVENTORY_ERROR_MISSING_SALES = "INVENTORY_MISSING_SALES"
INVENTORY_ERROR_INVALID_SALES = "INVENTORY_INVALID_SALES"

# Fixed ratio constants — these are rules, not MRP.
INGREDIENTS_PER_SALES_UNIT = 0.45
MEALS_PER_SALES_UNIT = 1 / 25

DEFAULT_RESTAURANT_ID = "restaurant_001"
DEFAULT_LOCATION_ID = "location_001"


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _inventory_for_sales(predicted_sales: float) -> Dict[str, float]:
    """Fixed-ratio rule: ingredients and meals derived from predicted sales.

    This is a rule — it is NOT MRP or inventory optimisation.
    """
    return {
        "ingredients_needed": round(predicted_sales * INGREDIENTS_PER_SALES_UNIT, 2),
        "estimated_meals": round(predicted_sales * MEALS_PER_SALES_UNIT, 2),
    }


def _build_inventory_plan(
    sales: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a per-day inventory estimate from actual sales rows using fixed ratios."""
    plan: List[Dict[str, Any]] = []
    for i, row in enumerate(sales):
        daily = float(row.get("daily_sales_total", 0))
        inv = _inventory_for_sales(daily)
        plan.append(
            {
                "day_index": i + 1,
                "daily_sales_total": daily,
                "ingredients_needed": inv["ingredients_needed"],
                "estimated_meals": inv["estimated_meals"],
            }
        )
    return plan


def run(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print(f"[{_utc_ts()}] START dashboard_update")

    context = context or {}

    try:
        from . import ml_prediction
        from . import optimization
        from . import gpt_insight
    except Exception as e:
        return {
            "dashboard_update_status": "error",
            "reason": f"import_failed:{type(e).__name__}:{e}",
            "timestamp": _utc_ts(),
        }

    prediction_data: Dict[str, Any] = {}
    if isinstance(context.get("5_ml_prediction"), dict):
        prediction_data = context.get("5_ml_prediction", {})

    if not prediction_data:
        try:
            prediction_result = ml_prediction.run()
            prediction_data = prediction_result.get("data", {})
        except Exception as e:
            return {
                "dashboard_update_status": "error",
                "reason": f"prediction_failed:{type(e).__name__}:{e}",
                "timestamp": _utc_ts(),
            }

    optimization_data: Dict[str, Any] = {}
    if isinstance(context.get("6_optimization"), dict):
        optimization_data = context.get("6_optimization", {})

    if not optimization_data:
        try:
            optimization_result = optimization.run()
            optimization_data = optimization_result
        except Exception as e:
            return {
                "dashboard_update_status": "error",
                "reason": f"optimization_failed:{type(e).__name__}:{e}",
                "timestamp": _utc_ts(),
            }

    gpt_data: Dict[str, Any] = {}
    if isinstance(context.get("gpt_insight"), dict):
        gpt_data = context.get("gpt_insight", {})

    if not gpt_data:
        try:
            gpt_result = gpt_insight.run(
                {
                    "5_ml_prediction": prediction_data,
                }
            )
            gpt_data = gpt_result.get("data", {}) if isinstance(gpt_result, dict) else {}
        except Exception as e:
            gpt_data = {
                "gpt_insight_status": "error",
                "reason": f"gpt_insight_failed:{type(e).__name__}:{e}",
            }

    restaurant_id = (
        context.get("restaurant_id")
        or prediction_data.get("restaurant_id")
        or DEFAULT_RESTAURANT_ID
    )
    location_id = (
        context.get("location_id")
        or prediction_data.get("location_id")
        or DEFAULT_LOCATION_ID
    )

    # Inventory is built from actual sales rows in context, not from forecast.
    # Guard: sales must be present and valid.
    sales = context.get("sales")
    if not sales or not isinstance(sales, list):
        code = INVENTORY_ERROR_MISSING_SALES
        print(f"[{_utc_ts()}] dashboard_update inventory status=error code={code}")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    # Validate each sales row has a numeric daily_sales_total.
    for row in sales:
        if not isinstance(row, dict) or not isinstance(row.get("daily_sales_total"), (int, float)):
            code = INVENTORY_ERROR_INVALID_SALES
            print(f"[{_utc_ts()}] dashboard_update inventory status=error code={code}")
            return {
                "status": "error",
                "error_code": code,
                "restaurant_id": restaurant_id,
                "location_id": location_id,
                "timestamp": _utc_ts(),
            }

    inventory_plan = _build_inventory_plan(sales)

    forecast = prediction_data.get("forecast", [])
    staffing_plan = optimization_data.get("staffing_plan", [])

    insight_json = gpt_data.get("insight_json", {}) if isinstance(gpt_data, dict) else {}
    risk_level = None
    actions: List[str] = []

    if isinstance(insight_json, dict):
        risk_level = insight_json.get("risk_level")
        raw_actions = insight_json.get("actions", [])
        if isinstance(raw_actions, list):
            actions = [str(x) for x in raw_actions]

    result = {
        "dashboard_update_status": "ok",
        "dashboard_refreshed": True,
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        "forecast": forecast,
        "staffing_plan": staffing_plan,
        "inventory_plan": inventory_plan,
        # method and method_description describe the inventory rule honestly.
        "inventory_method": "rule",
        "inventory_method_description": (
            f"fixed ratios: {INGREDIENTS_PER_SALES_UNIT} ingredients per sales unit, "
            f"{round(MEALS_PER_SALES_UNIT, 4)} meals per sales unit"
        ),
        "gpt_insight_status": gpt_data.get("gpt_insight_status"),
        "insight_json": insight_json,
        "risk_level": risk_level,
        "actions": actions,
        "timestamp": _utc_ts(),
    }

    print(f"[{_utc_ts()}] DONE dashboard_update")
    return result