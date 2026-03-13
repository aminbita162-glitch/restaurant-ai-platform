from datetime import datetime
from typing import Any, Dict, List, Optional


DEFAULT_RESTAURANT_ID = "restaurant_001"
DEFAULT_LOCATION_ID = "location_001"


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _estimate_inventory(predicted_sales: float) -> Dict[str, float]:
    return {
        "ingredients_needed": round(predicted_sales * 0.45, 2),
        "estimated_meals": round(predicted_sales / 25, 2),
    }


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

    restaurant_id = prediction_data.get("restaurant_id", DEFAULT_RESTAURANT_ID)
    location_id = prediction_data.get("location_id", DEFAULT_LOCATION_ID)

    forecast = prediction_data.get("forecast", [])
    staffing_plan = optimization_data.get("staffing_plan", [])

    inventory_plan: List[Dict[str, Any]] = []

    for i, day in enumerate(forecast):
        predicted = float(day.get("predicted_sales", 0))
        inv = _estimate_inventory(predicted)

        inventory_plan.append(
            {
                "day_index": i + 1,
                "predicted_sales": predicted,
                "ingredients_needed": inv["ingredients_needed"],
                "estimated_meals": inv["estimated_meals"],
            }
        )

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
        "gpt_insight_status": gpt_data.get("gpt_insight_status"),
        "insight_json": insight_json,
        "risk_level": risk_level,
        "actions": actions,
        "timestamp": _utc_ts(),
    }

    print(f"[{_utc_ts()}] DONE dashboard_update")
    return result