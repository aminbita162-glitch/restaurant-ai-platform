from datetime import datetime
from typing import Any, Dict, List


DEFAULT_RESTAURANT_ID = "restaurant_001"
DEFAULT_LOCATION_ID = "location_001"


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _estimate_inventory(predicted_sales: float) -> Dict[str, float]:
    return {
        "ingredients_needed": round(predicted_sales * 0.45, 2),
        "estimated_meals": round(predicted_sales / 25, 2),
    }


def run() -> Dict[str, Any]:
    print(f"[{_utc_ts()}] START dashboard_update")

    try:
        from . import ml_prediction
        from . import optimization
    except Exception as e:
        return {
            "dashboard_update_status": "error",
            "reason": f"import_failed:{type(e).__name__}:{e}",
            "timestamp": _utc_ts(),
        }

    try:
        prediction_result = ml_prediction.run()
        prediction_data = prediction_result.get("data", {})
    except Exception as e:
        return {
            "dashboard_update_status": "error",
            "reason": f"prediction_failed:{type(e).__name__}:{e}",
            "timestamp": _utc_ts(),
        }

    try:
        optimization_result = optimization.run()
    except Exception as e:
        return {
            "dashboard_update_status": "error",
            "reason": f"optimization_failed:{type(e).__name__}:{e}",
            "timestamp": _utc_ts(),
        }

    restaurant_id = prediction_data.get("restaurant_id", DEFAULT_RESTAURANT_ID)
    location_id = prediction_data.get("location_id", DEFAULT_LOCATION_ID)

    forecast = prediction_data.get("forecast", [])
    staffing_plan = optimization_result.get("staffing_plan", [])

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

    result = {
        "dashboard_update_status": "ok",
        "dashboard_refreshed": True,
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        "forecast": forecast,
        "staffing_plan": staffing_plan,
        "inventory_plan": inventory_plan,
        "timestamp": _utc_ts(),
    }

    print(f"[{_utc_ts()}] DONE dashboard_update")
    return result