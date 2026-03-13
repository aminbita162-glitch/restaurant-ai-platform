from datetime import datetime
from typing import Any, Dict, List


DEFAULT_RESTAURANT_ID = "restaurant_001"
DEFAULT_LOCATION_ID = "location_001"


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _estimate_staff_needed(predicted_sales: float) -> int:
    """
    Simple rule:
    every ~600 sales units require 1 staff member
    """
    if predicted_sales <= 0:
        return 1
    staff = int(predicted_sales / 600)
    return max(1, staff)


def run() -> Dict[str, Any]:
    print(f"[{_utc_ts()}] START optimization")

    try:
        from . import ml_prediction
    except Exception as e:
        return {
            "optimization_status": "error",
            "reason": f"prediction_import_failed:{type(e).__name__}:{e}",
            "timestamp": _utc_ts(),
        }

    try:
        prediction_result = ml_prediction.run()
        data = prediction_result.get("data", {})
    except Exception as e:
        return {
            "optimization_status": "error",
            "reason": f"prediction_failed:{type(e).__name__}:{e}",
            "timestamp": _utc_ts(),
        }

    restaurant_id = data.get("restaurant_id", DEFAULT_RESTAURANT_ID)
    location_id = data.get("location_id", DEFAULT_LOCATION_ID)

    forecast = data.get("forecast", [])

    staffing_plan: List[Dict[str, Any]] = []

    for i, day in enumerate(forecast):
        predicted = float(day.get("predicted_sales", 0))
        staff_needed = _estimate_staff_needed(predicted)

        staffing_plan.append(
            {
                "day_index": i + 1,
                "predicted_sales": predicted,
                "recommended_staff": staff_needed,
            }
        )

    result = {
        "optimization_status": "ok",
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        "staffing_plan": staffing_plan,
        "timestamp": _utc_ts(),
    }

    print(f"[{_utc_ts()}] DONE optimization")

    return result