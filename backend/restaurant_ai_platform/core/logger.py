import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def build_step_log(
    step_name: str,
    status: str,
    data: Optional[Dict[str, Any]] = None,
    errors: Optional[list] = None,
    warnings: Optional[list] = None,
    started_at: Optional[str] = None,
) -> Dict[str, Any]:
    ended_at = _utc_ts()
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


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return json.dumps({"_unserializable": str(obj)}, ensure_ascii=False, separators=(",", ":"))


def _get_std_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


class Logger:
    """
    A thin wrapper around std logging that supports:
      logger.info(event="...", run_id="...", step="...", ...)
    and produces one-line JSON logs (Render-friendly).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._logger = _get_std_logger(name)

    def _log(self, level: int, message: Optional[str] = None, *, event: Optional[str] = None, **fields: Any) -> None:
        payload: Dict[str, Any] = {
            "ts": _utc_ts(),
            "level": logging.getLevelName(level),
            "logger": self._name,
        }
        if event is not None:
            payload["event"] = event
        if message:
            payload["message"] = message
        if fields:
            payload.update(fields)

        self._logger.log(level, _safe_json(payload))

    def info(self, message: Optional[str] = None, *, event: Optional[str] = None, **fields: Any) -> None:
        self._log(logging.INFO, message, event=event, **fields)

    def warning(self, message: Optional[str] = None, *, event: Optional[str] = None, **fields: Any) -> None:
        self._log(logging.WARNING, message, event=event, **fields)

    def error(self, message: Optional[str] = None, *, event: Optional[str] = None, **fields: Any) -> None:
        self._log(logging.ERROR, message, event=event, **fields)

    def exception(self, message: Optional[str] = None, *, event: Optional[str] = None, **fields: Any) -> None:
        fields = dict(fields)
        fields.setdefault("exc", True)
        self._log(logging.ERROR, message, event=event, **fields)


def get_logger(name: str) -> Logger:
    return Logger(name)


def get_std_logger(name: str) -> logging.Logger:
    return _get_std_logger(name)