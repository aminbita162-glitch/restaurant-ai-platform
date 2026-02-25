from datetime import datetime
import os
import csv


DATA_FILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "upload_sales.csv",
)


def _load_sales_from_csv(path: str):
    rows = []
    if not os.path.exists(path):
        return None, "file_not_found"

    try:
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "date": row.get("date"),
                        "daily_sales_total": float(row.get("daily_sales_total", 0)),
                    }
                )
        return rows, None
    except Exception as e:
        return None, str(e)


def run() -> dict:
    print(f"[{datetime.utcnow().isoformat()}] step=1_data_ingestion status=started")

    sales_data, error = _load_sales_from_csv(DATA_FILE_PATH)

    if sales_data:
        result = {
            "sales_source": "csv_file",
            "sales_rows_loaded": len(sales_data),
            "sales": sales_data,
            "inventory": "simulated_inventory_data",
            "attendance": "simulated_attendance_data",
            "timestamp": datetime.utcnow().isoformat(),
        }
    else:
        result = {
            "sales_source": "simulated_fallback",
            "error": error,
            "sales": "simulated_sales_data",
            "inventory": "simulated_inventory_data",
            "attendance": "simulated_attendance_data",
            "timestamp": datetime.utcnow().isoformat(),
        }

    print(f"[{datetime.utcnow().isoformat()}] step=1_data_ingestion status=completed")
    return result