import logging
from datetime import datetime
from typing import Any, Dict


def build_step_log(
    step_name: str,
    status: str,
    data: Dict[str, Any] | None = None,
    errors: list | None = None,
    warnings: list | None = None,
    started_at: str | None = None,
) -> Dict[str, Any]:
    """
    Standardized step log structure for all pipeline steps.
    """

    ended_at = datetime.utcnow().isoformat()

    return {
        "step": step_name,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": 0,
        "data": data or {},
        "errors": errors or [],
        "warnings": warnings or [],
        "metrics": {},
        "timestamp": ended_at,
    }


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger