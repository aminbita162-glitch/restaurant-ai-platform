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
    """Return the most specific CSV path that exists for this tenant.

    Resolution order (most specific first):
      1. <restaurant_id>__<location_id>__sales.csv
      2. <restaurant_id>__sales.csv

    The shared upload_sales.csv is never used as a production source.
    If no tenant-specific file exists the tenant-specific path is returned
    so the caller receives FILE_NOT_FOUND.
    """
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

    if os.path.exists(tenant_specific):
        return tenant_specific

    if os.path.exists(restaurant_specific):
        return restaurant_specific

    # No matching file — return tenant-specific path; caller will get FILE_NOT_FOUND.
    return tenant_specific


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

                # Reject rows whose daily_sales_total is not a valid number.
                raw_total = row.get("daily_sales_total")
                if raw_total is None or str(raw_total).strip() == "":
                    return None, "invalid_row"
                try:
                    total = float(raw_total)
                except (ValueError, TypeError):
                    return None, "invalid_row"

                rows.append(
                    {
                        "restaurant_id": row_restaurant_id,
                        "location_id": row_location_id,
                        "date": row.get("date"),
                        "daily_sales_total": total,
                    }
                )

        return rows, None

    except Exception:
        return None, "INPUT_READ_ERROR"


# Stable public error codes for data ingestion failures.
DATA_INGESTION_ERROR_FILE_NOT_FOUND = "DATA_INGESTION_FILE_NOT_FOUND"
DATA_INGESTION_ERROR_READ_FAILED = "DATA_INGESTION_READ_FAILED"
DATA_INGESTION_ERROR_EMPTY = "DATA_INGESTION_EMPTY"
DATA_INGESTION_ERROR_INVALID_ROW = "DATA_INGESTION_INVALID_ROW"


def run(context: Optional[Dict[str, Any]] = None) -> dict:
    context = context or {}

    restaurant_id = str(context.get("restaurant_id") or DEFAULT_RESTAURANT_ID)
    location_id = str(context.get("location_id") or DEFAULT_LOCATION_ID)

    # demo=True must be set explicitly by the caller to use DEMO data.
    # It is the ONLY path to sample data. Never implicit.
    demo = bool(context.get("demo", False))

    print(f"[{_utc_ts()}] step=1_data_ingestion status=started restaurant_id={restaurant_id} location_id={location_id} demo={demo}")

    if demo:
        # Explicit demo mode: caller opted in — never implicit.
        print(f"[{_utc_ts()}] step=1_data_ingestion status=completed source_type=DEMO")
        return {
            "status": "ok",
            "source_type": "DEMO",
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "sales_source": "demo",
            "sales_rows_loaded": 0,
            "sales": [],
            "inventory": "demo_inventory_data",
            "attendance": "demo_attendance_data",
            "timestamp": _utc_ts(),
        }

    data_file_path = _resolve_data_file_path(restaurant_id, location_id)
    sales_data, error = _load_sales_from_csv(data_file_path, restaurant_id, location_id)

    if error == "file_not_found":
        code = DATA_INGESTION_ERROR_FILE_NOT_FOUND
        print(f"[{_utc_ts()}] step=1_data_ingestion status=error code={code} path={os.path.basename(data_file_path)}")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "sales_file_path": os.path.basename(data_file_path),
            "timestamp": _utc_ts(),
        }

    if error == "invalid_row":
        code = DATA_INGESTION_ERROR_INVALID_ROW
        print(f"[{_utc_ts()}] step=1_data_ingestion status=error code={code}")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "sales_file_path": os.path.basename(data_file_path),
            "timestamp": _utc_ts(),
        }

    if error:
        code = DATA_INGESTION_ERROR_READ_FAILED
        print(f"[{_utc_ts()}] step=1_data_ingestion status=error code={code}")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "sales_file_path": os.path.basename(data_file_path),
            "timestamp": _utc_ts(),
        }

    if not sales_data:
        code = DATA_INGESTION_ERROR_EMPTY
        print(f"[{_utc_ts()}] step=1_data_ingestion status=error code={code}")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "sales_file_path": os.path.basename(data_file_path),
            "timestamp": _utc_ts(),
        }

    print(f"[{_utc_ts()}] step=1_data_ingestion status=completed source_type=REAL rows={len(sales_data)}")
    return {
        "status": "ok",
        "source_type": "REAL",
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        "sales_source": "csv_file",
        "sales_file_path": os.path.basename(data_file_path),
        "sales_rows_loaded": len(sales_data),
        "sales": sales_data,
        "timestamp": _utc_ts(),
    }