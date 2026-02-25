from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


class DataAdapter:
    """
    DataAdapter = لایه اتصال به دیتا

    الان: حالت DEMO (دیتای شبیه‌سازی شده)
    بعداً: همین کلاس را به دیتابیس واقعی / API واقعی وصل می‌کنیم،
          بدون اینکه بقیه پایپلاین تغییر کند.
    """

    def __init__(self, mode: Optional[str] = None) -> None:
        # mode می‌تواند "demo" یا "db" باشد (فعلاً demo را پیاده‌سازی می‌کنیم)
        self.mode = (mode or os.getenv("DATA_ADAPTER_MODE") or "demo").strip().lower()

    def fetch(self) -> Dict[str, Any]:
        """
        خروجی استاندارد ingestion را برمی‌گرداند.
        """
        if self.mode != "demo":
            # فعلاً فقط demo فعال است تا پروژه پایدار بماند
            raise NotImplementedError(
                f"DATA_ADAPTER_MODE='{self.mode}' is not implemented yet. Use 'demo'."
            )

        # DEMO data (همان چیزی که ingestion الان انتظار دارد)
        return {
            "sales": "simulated_sales_data",
            "inventory": "simulated_inventory_data",
            "attendance": "simulated_attendance_data",
            "timestamp": _utc_ts(),
        }


def get_data() -> Dict[str, Any]:
    """
    API ساده برای استفاده در data_ingestion.py
    """
    adapter = DataAdapter()
    return adapter.fetch()