from datetime import datetime
from typing import Any, Dict


def run(context: Dict[str, Any]) -> Dict[str, Any]:

    print(f"[{datetime.utcnow().isoformat()}] START feature_engineering")

    restaurant_id = context.get("restaurant_id")
    location_id = context.get("location_id")

    sales_data = context.get("sales")

    result = {
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        "features_status": "ok",
        "generated_features": [
            "daily_sales_total",
            "hourly_sales_profile",
            "inventory_usage_rate",
            "waste_rate",
            "labor_hours",
        ],
        "sales_records": sales_data,
        "timestamp": datetime.utcnow().isoformat(),
    }

    print(f"[{datetime.utcnow().isoformat()}] DONE feature_engineering")

    return result