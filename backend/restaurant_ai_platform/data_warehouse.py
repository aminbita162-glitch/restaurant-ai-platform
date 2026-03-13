from datetime import datetime
from typing import Any, Dict


def run(context: Dict[str, Any]) -> Dict[str, Any]:

    print(f"[{datetime.utcnow().isoformat()}] data_warehouse: START")

    restaurant_id = context.get("restaurant_id")
    location_id = context.get("location_id")

    sales_data = context.get("sales")
    inventory_data = context.get("inventory")
    attendance_data = context.get("attendance")

    result = {
        "step": "2_data_warehouse",
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        "warehouse_status": "ok",
        "loaded_tables": ["sales", "inventory", "attendance"],
        "sales_records": sales_data,
        "inventory_records": inventory_data,
        "attendance_records": attendance_data,
        "timestamp": datetime.utcnow().isoformat(),
    }

    print(f"[{datetime.utcnow().isoformat()}] data_warehouse: DONE")

    return result