from datetime import datetime


def run() -> dict:
    print(f"[{datetime.utcnow().isoformat()}] step=1_data_ingestion status=started")

    data = {
        "sales": "simulated_sales_data",
        "inventory": "simulated_inventory_data",
        "attendance": "simulated_attendance_data",
        "timestamp": datetime.utcnow().isoformat(),
    }

    print(f"[{datetime.utcnow().isoformat()}] step=1_data_ingestion status=completed")
    return data