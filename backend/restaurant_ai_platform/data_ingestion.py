from datetime import datetime
import os
import csv
from typing import Any, Dict, List, Optional, Tuple


BASE_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
)

DEFAULT_RESTAURANT_ID = "restaurant_001"
DEFAULT_LOCATION_ID = "location_001"
DEFAULT_DATA_FILE = "upload_sales.csv"


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _safe_id(value: str) -> str:
    return str(value).strip().replace("/", "_").replace("\\", "_").replace(" ", "_")


def _resolve_data_file_path(restaurant_id: str, location_id: str) -> str:
    restaurant_part = _safe_id(restaurant_id or DEFAULT_RESTAURANT_ID)
    location_part = _safe_id(location_id or DEFAULT_LOCATION_ID)

    tenant_specific = os.path.join(
        BASE_DATA_DIR,
        f"{restaurant_part}__{location_part}__sales.csv",
    )

    restaurant_specific = os.path.join(
        BASE_DATA_DIR,
        f"{restaurant_part}__sales.csv",
    )

    default_path = os.path.join(BASE_DATA_DIR, DEFAULT_DATA_FILE)

    if os.path.exists(tenant_specific):
        return tenant_specific

    if os.path.exists(restaurant_specific):
        return restaurant_specific

    return default_path


def _load_sales_from_csv(path: str, restaurant_id: str, location_id: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    rows: List[Dict[str, Any]] = []

    if not os.path.exists(path):
        return None, "file_not_found"

    try:
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                row_restaurant_id = str(row.get("restaurant_id") or restaurant_id or DEFAULT_RESTAURANT_ID)
                row_location_id = str(row.get("location_id") or location_id or DEFAULT_LOCATION_ID)

                rows.append(
                    {
                        "restaurant_id": row_restaurant_id,
                        "location_id": row_location_id,
                        "date": row.get("date"),
                        "daily_sales_total": float(row.get("daily_sales_total", 0)),
                    }
                )

        return rows, None

    except Exception as e:
        return None, str(e)


def run(context: Optional[Dict[str, Any]] = None) -> dict:
    context = context or {}

    restaurant_id = str(context.get("restaurant_id") or DEFAULT_RESTAURANT_ID)
    location_id = str(context.get("location_id") or DEFAULT_LOCATION_ID)

    print(f"[{_utc_ts()}] step=1_data_ingestion status=started restaurant_id={restaurant_id} location_id={location_id}")

    data_file_path = _resolve_data_file_path(restaurant_id, location_id)
    sales_data, error = _load_sales_from_csv(data_file_path, restaurant_id, location_id)

    if sales_data:
        result = {
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "sales_source": "csv_file",
            "sales_file_path": os.path.basename(data_file_path),
            "sales_rows_loaded": len(sales_data),
            "sales": sales_data,
            "inventory": "simulated_inventory_data",
            "attendance": "simulated_attendance_data",
            "timestamp": _utc_ts(),
        }
    else:
        result = {
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "sales_source": "simulated_fallback",
            "sales_file_path": os.path.basename(data_file_path),
            "error": error,
            "sales": "simulated_sales_data",
            "inventory": "simulated_inventory_data",
            "attendance": "simulated_attendance_data",
            "timestamp": _utc_ts(),
        }

    print(f"[{_utc_ts()}] step=1_data_ingestion status=completed restaurant_id={restaurant_id} location_id={location_id}")

    return result