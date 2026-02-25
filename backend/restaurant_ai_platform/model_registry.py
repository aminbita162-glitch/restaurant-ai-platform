import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional


BASE_DIR = os.path.dirname(__file__)
REGISTRY_DIR = os.path.join(BASE_DIR, "model_registry_store")
REGISTRY_INDEX_PATH = os.path.join(REGISTRY_DIR, "registry_index.json")
LATEST_POINTER_PATH = os.path.join(REGISTRY_DIR, "latest.json")


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _ensure_registry_dir() -> None:
    os.makedirs(REGISTRY_DIR, exist_ok=True)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_registry_index() -> Dict[str, Any]:
    _ensure_registry_dir()
    if not os.path.exists(REGISTRY_INDEX_PATH):
        return {"models": [], "created_at": _utc_now_iso(), "updated_at": _utc_now_iso()}
    return _read_json(REGISTRY_INDEX_PATH)


def _save_registry_index(index: Dict[str, Any]) -> None:
    index["updated_at"] = _utc_now_iso()
    _atomic_write_json(REGISTRY_INDEX_PATH, index)


def _extract_sales_rows(features: Dict[str, Any]) -> List[Dict[str, Any]]:
    sales = features.get("sales", [])
    if isinstance(sales, list) and sales:
        return sales
    return []


def train_model(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple demo "training":
    computes average daily sales from feature payload and stores it as model params.
    """
    sales_rows = _extract_sales_rows(features)
    if not sales_rows:
        raise ValueError("No sales data provided for training")

    totals: List[float] = []
    for row in sales_rows:
        if "daily_sales_total" not in row:
            raise ValueError("Sales rows must include 'daily_sales_total'")
        totals.append(float(row["daily_sales_total"]))

    avg_daily_sales = sum(totals) / max(len(totals), 1)

    return {
        "params": {
            "avg_daily_sales": avg_daily_sales,
            "multiplier_next_day": 1.02,
        },
        "metrics": {
            "rows_used": len(totals),
            "avg_daily_sales": avg_daily_sales,
        },
    }


def register_model(features: Dict[str, Any], model_name: str = "average_daily_sales_model") -> Dict[str, Any]:
    """
    Trains a mock model and registers it in a tiny local registry (JSON files).
    Returns registry metadata that ml_prediction can use later.
    """
    print(f"[{_utc_now_iso()}] step=model_registry status=started")

    _ensure_registry_dir()
    index = _load_registry_index()

    trained = train_model(features)

    version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_id = f"{model_name}:{version}"
    artifact_path = os.path.join(REGISTRY_DIR, f"{model_name}__{version}.json")

    artifact = {
        "model_name": model_name,
        "model_id": model_id,
        "version": version,
        "trained_at": _utc_now_iso(),
        "format": "json",
        "params": trained["params"],
        "metrics": trained["metrics"],
        "source": {
            "type": "features_payload",
            "sales_rows": trained["metrics"]["rows_used"],
        },
    }

    _atomic_write_json(artifact_path, artifact)

    index["models"].append(
        {
            "model_name": model_name,
            "model_id": model_id,
            "version": version,
            "trained_at": artifact["trained_at"],
            "artifact_path": artifact_path,
        }
    )
    _save_registry_index(index)

    _atomic_write_json(
        LATEST_POINTER_PATH,
        {
            "model_name": model_name,
            "model_id": model_id,
            "version": version,
            "artifact_path": artifact_path,
            "updated_at": _utc_now_iso(),
        },
    )

    print(f"[{_utc_now_iso()}] step=model_registry status=completed")
    return {
        "model_registry_status": "ok",
        "registered": True,
        "model_name": model_name,
        "model_id": model_id,
        "version": version,
        "artifact_path": artifact_path,
        "timestamp": _utc_now_iso(),
    }


def get_latest_model_meta() -> Dict[str, Any]:
    """
    Returns latest pointer metadata (does not load full model artifact).
    """
    if not os.path.exists(LATEST_POINTER_PATH):
        raise FileNotFoundError("No latest model pointer found. Register a model first.")
    return _read_json(LATEST_POINTER_PATH)


def load_latest_model() -> Dict[str, Any]:
    """
    Loads the latest model artifact from disk.
    """
    meta = get_latest_model_meta()
    artifact_path = meta.get("artifact_path")
    if not artifact_path or not os.path.exists(artifact_path):
        raise FileNotFoundError("Latest model artifact not found. Register a model again.")
    return _read_json(artifact_path)


def predict(features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Predicts next day sales using the latest registered model.
    The 'features' argument is optional for future expansion.
    """
    model = load_latest_model()
    params = model.get("params", {})
    avg_daily_sales = float(params.get("avg_daily_sales", 0.0))
    multiplier = float(params.get("multiplier_next_day", 1.02))

    predicted = round(avg_daily_sales * multiplier, 2)

    return {
        "ml_prediction_status": "ok",
        "predicted_next_day_sales": predicted,
        "model_name": model.get("model_name"),
        "model_version": model.get("version"),
        "model_id": model.get("model_id"),
        "prediction_timestamp": _utc_now_iso(),
    }