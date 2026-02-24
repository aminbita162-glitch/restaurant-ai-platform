from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


def utc_ts() -> str:
    return datetime.utcnow().isoformat()


def log_event(event: str, **fields: Any) -> None:
    parts = [f"event={event}"]
    for k, v in fields.items():
        parts.append(f"{k}={v}")
    print(f"[{utc_ts()}] " + " ".join(parts))


def ok_payload(data: Optional[Dict[str, Any]] = None, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": True, "timestamp": utc_ts()}
    if data:
        payload.update(data)
    if extra:
        payload.update(extra)
    return payload


def error_payload(
    message: str,
    *,
    code: str = "error",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": False,
        "timestamp": utc_ts(),
        "error": {
            "code": code,
            "message": message,
        },
    }
    if details:
        payload["error"]["details"] = details
    return payload


def step_result(
    step: str,
    status: str,
    *,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "step": step,
        "status": status,
        "timestamp": utc_ts(),
    }
    if result is not None:
        payload["result"] = result
    if error is not None:
        payload["error"] = error
    return payload