from datetime import datetime
import os
import csv
from typing import Any, Dict, List, Optional, Tuple


BASE_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
)

DEFAULT_DATA_FILE = "upload_sales.csv"

# Stable public error codes for data ingestion failures.
DATA_INGESTION_ERROR_FILE_NOT_FOUND = "DATA_INGESTION_FILE_NOT_FOUND"
DATA_INGESTION_ERROR_READ_FAILED = "DATA_INGESTION_READ_FAILED"
DATA_INGESTION_ERROR_EMPTY = "DATA_INGESTION_EMPTY"
DATA_INGESTION_ERROR_INVALID_ROW = "DATA_INGESTION_INVALID_ROW"
DATA_INGESTION_ERROR_MISSING_REQUIRED_FIELD = "DATA_INGESTION_MISSING_REQUIRED_FIELD"

# Required fields that must be present and non-empty on every sales row.
_REQUIRED_ROW_FIELDS = ("date", "net_sales", "restaurant_id", "location_id")

# Optional fields accepted on a row; any other columns are ignored.
_OPTIONAL_ROW_FIELDS = ("channel", "orders", "guests", "currency")


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
    restaurant_part = _safe_id(restaurant_id)
    location_part = _safe_id(location_id)

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


def _validate_row(row: Dict[str, Any]) -> Optional[str]:
    """Validate a single sales row against the required field contract.

    Returns the name of the first missing/invalid required field, or None
    if the row is valid. No silent fill is performed.

    Required: date, net_sales, restaurant_id, location_id.
    net_sales must be a non-empty string convertible to float.
    The other required fields must be non-empty strings.
    """
    for field in _REQUIRED_ROW_FIELDS:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            return field

    # net_sales must be numeric.
    try:
        float(str(row["net_sales"]).strip())
    except (ValueError, TypeError):
        return "net_sales"

    return None


def _load_sales_from_csv(
    path: str,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Optional[str]]:
    """Read and validate a sales CSV.

    Returns (rows, error_code, bad_field).
    - rows: list of validated row dicts on success, None on error.
    - error_code: one of the DATA_INGESTION_ERROR_* constants, or None.
    - bad_field: name of the first bad field when error_code is
      DATA_INGESTION_MISSING_REQUIRED_FIELD or DATA_INGESTION_INVALID_ROW,
      else None.
    """
    if not os.path.exists(path):
        return None, DATA_INGESTION_ERROR_FILE_NOT_FOUND, None

    try:
        rows: List[Dict[str, Any]] = []

        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for raw_row in reader:
                bad_field = _validate_row(raw_row)
                if bad_field is not None:
                    return None, DATA_INGESTION_ERROR_MISSING_REQUIRED_FIELD, bad_field

                net_sales = float(str(raw_row["net_sales"]).strip())

                row_out: Dict[str, Any] = {
                    "date": str(raw_row["date"]).strip(),
                    "net_sales": net_sales,
                    "restaurant_id": str(raw_row["restaurant_id"]).strip(),
                    "location_id": str(raw_row["location_id"]).strip(),
                }

                # Carry optional fields if present and non-empty.
                for opt in _OPTIONAL_ROW_FIELDS:
                    val = raw_row.get(opt)
                    if val is not None and str(val).strip() != "":
                        row_out[opt] = str(val).strip()

                rows.append(row_out)

        return rows, None, None

    except Exception:
        return None, DATA_INGESTION_ERROR_READ_FAILED, None


def run(context: Optional[Dict[str, Any]] = None) -> dict:
    context = context or {}

    restaurant_id = str(context.get("restaurant_id") or "").strip()
    location_id = str(context.get("location_id") or "").strip()

    # demo=True must be set explicitly by the caller to use DEMO data.
    # It is the ONLY path to sample data. Never implicit.
    demo = bool(context.get("demo", False))

    print(
        f"[{_utc_ts()}] step=1_data_ingestion status=started"
        f" restaurant_id={restaurant_id} location_id={location_id} demo={demo}"
    )

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

    if not restaurant_id or not location_id:
        from .core.persistence import PERSISTENCE_ERROR_MISSING_TENANT
        print(
            f"[{_utc_ts()}] step=1_data_ingestion status=error"
            f" code=MISSING_TENANT_KEYS"
        )
        return {
            "status": "error",
            "error_code": "DATA_INGESTION_MISSING_TENANT_KEYS",
            "message": "restaurant_id and location_id are required",
            "timestamp": _utc_ts(),
        }

    data_file_path = _resolve_data_file_path(restaurant_id, location_id)
    sales_data, error_code, bad_field = _load_sales_from_csv(data_file_path)

    if error_code == DATA_INGESTION_ERROR_FILE_NOT_FOUND:
        print(
            f"[{_utc_ts()}] step=1_data_ingestion status=error"
            f" code={error_code} path={os.path.basename(data_file_path)}"
        )
        return {
            "status": "error",
            "error_code": error_code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "sales_file_path": os.path.basename(data_file_path),
            "timestamp": _utc_ts(),
        }

    if error_code == DATA_INGESTION_ERROR_MISSING_REQUIRED_FIELD:
        print(
            f"[{_utc_ts()}] step=1_data_ingestion status=error"
            f" code={error_code} field={bad_field}"
        )
        return {
            "status": "error",
            "error_code": error_code,
            "missing_field": bad_field,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "sales_file_path": os.path.basename(data_file_path),
            "timestamp": _utc_ts(),
        }

    if error_code:
        print(
            f"[{_utc_ts()}] step=1_data_ingestion status=error code={error_code}"
        )
        return {
            "status": "error",
            "error_code": error_code,
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

    print(
        f"[{_utc_ts()}] step=1_data_ingestion status=completed"
        f" source_type=REAL rows={len(sales_data)}"
    )
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
