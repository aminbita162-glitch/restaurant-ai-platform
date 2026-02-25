from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional
import math
import os
import json
import random
import sqlite3


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


# ----------------------------
# Synthetic data generator (Germany-like)
# ----------------------------
GER_PUBLIC_HOLIDAYS_FIXED = {
    # fixed-date (not all federal states; good-enough synthetic baseline)
    "01-01": "NewYear",
    "05-01": "LabourDay",
    "10-03": "GermanUnityDay",
    "12-25": "ChristmasDay",
    "12-26": "SecondChristmasDay",
}


def _is_fixed_holiday(d: datetime) -> bool:
    mmdd = d.strftime("%m-%d")
    return mmdd in GER_PUBLIC_HOLIDAYS_FIXED


def _dow(d: datetime) -> int:
    # Monday=0 ... Sunday=6
    return d.weekday()


def _seasonality_multiplier(d: datetime) -> float:
    # simple yearly seasonality: higher in summer, lower in winter (restaurant dependent)
    day_of_year = int(d.strftime("%j"))
    # sine wave in [0.85 .. 1.15]
    return 1.0 + 0.15 * math.sin(2.0 * math.pi * (day_of_year / 365.0))


def _weekend_multiplier(d: datetime) -> float:
    if _dow(d) in (5, 6):  # Sat, Sun
        return 1.25
    if _dow(d) == 4:  # Fri
        return 1.15
    return 1.0


def _holiday_multiplier(d: datetime) -> float:
    # holidays can be weird; assume slightly lower dine-in, but could be higher takeaway
    if _is_fixed_holiday(d):
        return 0.80
    return 1.0


def _promo_multiplier(d: datetime, promo_days: set[str]) -> float:
    return 1.20 if d.strftime("%Y-%m-%d") in promo_days else 1.0


@dataclass
class SyntheticConfig:
    days: int = 180
    start_date_utc: Optional[str] = None  # ISO date string
    seed: int = 42
    base_daily_sales_eur: float = 2200.0
    base_daily_orders: int = 180
    base_staff_count: int = 10
    base_inventory_index: float = 1.0  # abstract level


def generate_synthetic_restaurant_data(cfg: SyntheticConfig) -> Dict[str, Any]:
    random.seed(cfg.seed)

    if cfg.start_date_utc:
        start = datetime.fromisoformat(cfg.start_date_utc)
    else:
        # end yesterday, go back cfg.days
        start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=cfg.days)

    promo_days = set()
    # generate a few promo days
    for _ in range(max(3, cfg.days // 30)):
        offset = random.randint(0, max(0, cfg.days - 1))
        promo_days.add((start + timedelta(days=offset)).strftime("%Y-%m-%d"))

    sales_rows: List[Dict[str, Any]] = []
    inventory_rows: List[Dict[str, Any]] = []
    attendance_rows: List[Dict[str, Any]] = []

    inventory_index = cfg.base_inventory_index

    for i in range(cfg.days):
        d = start + timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")

        mult = (
            _seasonality_multiplier(d)
            * _weekend_multiplier(d)
            * _holiday_multiplier(d)
            * _promo_multiplier(d, promo_days)
        )

        noise_sales = random.uniform(0.90, 1.10)
        noise_orders = random.uniform(0.88, 1.12)

        daily_sales = cfg.base_daily_sales_eur * mult * noise_sales
        daily_orders = int(max(20, round(cfg.base_daily_orders * mult * noise_orders)))

        avg_basket = daily_sales / max(1, daily_orders)

        # inventory: decreases with orders, replenished occasionally
        consumption = daily_orders * random.uniform(0.0025, 0.0045)
        inventory_index = max(0.10, inventory_index - consumption)

        if inventory_index < 0.35 or (i % 14 == 0 and i > 0):
            # restock
            restock = random.uniform(0.60, 1.20)
            inventory_index = min(2.0, inventory_index + restock)

        # staffing: higher on weekends, slightly lower on holidays
        staff_mult = 1.0
        if _dow(d) in (5, 6):
            staff_mult = 1.15
        if _is_fixed_holiday(d):
            staff_mult = 0.85

        staff_count = int(max(3, round(cfg.base_staff_count * staff_mult)))
        labor_hours = int(max(10, round(staff_count * random.uniform(6.5, 8.5))))

        sales_rows.append(
            {
                "date": date_str,
                "daily_sales_eur": round(daily_sales, 2),
                "daily_orders": daily_orders,
                "avg_basket_eur": round(avg_basket, 2),
                "is_holiday": bool(_is_fixed_holiday(d)),
                "is_promo": date_str in promo_days,
            }
        )

        inventory_rows.append(
            {
                "date": date_str,
                "inventory_index": round(inventory_index, 3),
                "restocked_today": bool(inventory_index > 1.2 and (i % 14 == 0 and i > 0)),
            }
        )

        attendance_rows.append(
            {
                "date": date_str,
                "staff_count": staff_count,
                "labor_hours": labor_hours,
            }
        )

    return {
        "sales": sales_rows,
        "inventory": inventory_rows,
        "attendance": attendance_rows,
        "generated_at": _utc_ts(),
        "config": {
            "days": cfg.days,
            "seed": cfg.seed,
            "base_daily_sales_eur": cfg.base_daily_sales_eur,
            "base_daily_orders": cfg.base_daily_orders,
            "base_staff_count": cfg.base_staff_count,
        },
    }


# ----------------------------
# Simple baseline model "training"
#   We build a naive forecast baseline:
#   - predict next-day sales = rolling mean of last N days * weekday factor
#   - store params in SQLite registry
# ----------------------------
_DB_PATH_DEFAULT = "/tmp/restaurant_ai_pipeline.db"
_DB_PATH = os.getenv("PIPELINE_DB_PATH", _DB_PATH_DEFAULT)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_registry() -> None:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stored_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            params_json TEXT NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_model_registry_name ON model_registry(model_name)")
    conn.commit()
    conn.close()


def _save_model(model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    _init_registry()
    stored_at = _utc_ts()
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO model_registry (stored_at, model_name, model_version, params_json)
        VALUES (?, ?, ?, ?)
        """,
        (stored_at, model_name, version, json.dumps(params, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()
    return {"model_name": model_name, "model_version": version, "stored_at": stored_at}


def _load_latest_model(model_name: str) -> Optional[Dict[str, Any]]:
    _init_registry()
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT model_version, stored_at, params_json
        FROM model_registry
        WHERE model_name = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (model_name,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    try:
        params = json.loads(row["params_json"])
        return {"model_name": model_name, "model_version": row["model_version"], "stored_at": row["stored_at"], "params": params}
    except Exception:
        return None


def train_baseline_sales_model(data: Dict[str, Any], *, window: int = 14) -> Dict[str, Any]:
    sales = data.get("sales") or []
    if not isinstance(sales, list) or not sales:
        return {"errors": [{"message": "no sales data"}], "data": {}}

    # Sort by date
    sales_sorted = sorted(sales, key=lambda r: r.get("date", ""))
    values: List[float] = [float(r.get("daily_sales_eur") or 0.0) for r in sales_sorted]
    dates: List[str] = [str(r.get("date") or "") for r in sales_sorted]

    if len(values) < max(7, window):
        return {"errors": [{"message": "not enough data"}], "data": {}}

    # weekday factors from last 8 weeks (or available)
    by_dow: Dict[int, List[float]] = {i: [] for i in range(7)}
    for r in sales_sorted[-min(len(sales_sorted), 56):]:
        d = datetime.fromisoformat(r["date"])
        by_dow[_dow(d)].append(float(r["daily_sales_eur"]))

    global_mean = sum(values) / max(1, len(values))
    dow_factor: Dict[int, float] = {}
    for k, lst in by_dow.items():
        if lst:
            dow_factor[k] = (sum(lst) / len(lst)) / global_mean
        else:
            dow_factor[k] = 1.0

    rolling_mean = sum(values[-window:]) / window

    params = {
        "type": "baseline_rolling_mean",
        "window": window,
        "rolling_mean": rolling_mean,
        "dow_factor": dow_factor,
        "trained_on_start": dates[0],
        "trained_on_end": dates[-1],
        "global_mean": global_mean,
    }

    meta = _save_model("sales_forecast_baseline", params)
    return {"data": {"model": meta, "params": params}, "errors": [], "warnings": []}


def run() -> Dict[str, Any]:
    # Generate synthetic data and train baseline model
    cfg = SyntheticConfig()
    data = generate_synthetic_restaurant_data(cfg)
    out = train_baseline_sales_model(data, window=14)

    if out.get("errors"):
        return {
            "errors": out["errors"],
            "data": {"training_status": "error"},
            "timestamp": _utc_ts(),
        }

    return {
        "data": {
            "training_status": "ok",
            "model_saved": out["data"]["model"],
            "trained_range": {
                "start": out["data"]["params"]["trained_on_start"],
                "end": out["data"]["params"]["trained_on_end"],
            },
        },
        "errors": [],
        "warnings": [],
        "timestamp": _utc_ts(),
    }