from datetime import datetime
from typing import Any, Dict, List, Optional
import sqlite3
from pathlib import Path
import json


DB_PATH = Path(__file__).parent / "pipeline.db"


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _get_connection():
    return sqlite3.connect(DB_PATH)


def _init_table():
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feature_store (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            restaurant_id TEXT,
            location_id TEXT,
            features_json TEXT,
            created_at TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def _save_features(
    restaurant_id: str,
    location_id: str,
    features: List[str],
):
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO feature_store
        (restaurant_id, location_id, features_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            restaurant_id,
            location_id,
            json.dumps(features),
            _utc_ts(),
        ),
    )

    conn.commit()
    conn.close()


def run(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

    context = context or {}

    restaurant_id = context.get("restaurant_id", "restaurant_001")
    location_id = context.get("location_id", "location_001")

    generated_features = context.get("generated_features", [])

    print(f"[{_utc_ts()}] START feature_store_sync")

    _init_table()

    synced_features: List[str] = []

    if isinstance(generated_features, list) and generated_features:

        _save_features(
            restaurant_id,
            location_id,
            generated_features,
        )

        synced_features = generated_features

    result = {
        "feature_store_sync_status": "ok",
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        "synced_features": synced_features,
        "timestamp": _utc_ts(),
    }

    print(f"[{_utc_ts()}] DONE feature_store_sync")

    return result