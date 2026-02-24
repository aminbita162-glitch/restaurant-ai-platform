from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import uuid


PIPELINE_ORDER: List[str] = [
    "1_data_ingestion",
    "2_data_warehouse",
    "3_feature_engineering",
    "4_feature_store_sync",
    "5_ml_prediction",
    "6_optimization",
    "7_api_serving",
    "8_dashboard_update",
]


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


class _FallbackLogger:
    def info(self, message: str, **fields: Any) -> None:
        parts = [message] + [f"{k}={fields[k]}" for k in sorted(fields.keys())]
        print(f"[{_utc_ts()}] " + " ".join(parts))

    def warning(self, message: str, **fields: Any) -> None:
        parts = [message] + [f"{k}={fields[k]}" for k in sorted(fields.keys())]
        print(f"[{_utc_ts()}] " + " ".join(parts))

    def error(self, message: str, **fields: Any) -> None:
        parts = [message] + [f"{k}={fields[k]}" for k in sorted(fields.keys())]
        print(f"[{_utc_ts()}] " + " ".join(parts))


def _get_logger() -> Any:
    try:
        from .core.logger import get_logger  # type: ignore
        return get_logger("orchestrator")
    except Exception:
        return _FallbackLogger()


_LOG = _get_logger()


def _log_event(level: str, event: str, *, run_id: str, step: Optional[str] = None, **fields: Any) -> None:
    payload: Dict[str, Any] = {"event": event, "run_id": run_id, "timestamp": _utc_ts(), **fields}
    if step:
        payload["step"] = step

    if level == "error":
        _LOG.error("event", **payload)
    elif level == "warning":
        _LOG.warning("event", **payload)
    else:
        _LOG.info("event", **payload)


def _normalize_steps(value: Any) -> Optional[List[str]]:
    if value is None:
        return None

    if isinstance(value, str):
        raw = [p.strip() for p in value.split(",")]
        cleaned = [p for p in raw if p]
        return cleaned or None

    if isinstance(value, (list, tuple, set)):
        cleaned: List[str] = []
        for item in list(value):
            if item is None:
                continue
            s = str(item).strip()
            if s:
                cleaned.append(s)
        return cleaned or None

    s = str(value).strip()
    return [s] if s else None


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _select_plan(requested_steps: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    if requested_steps is None:
        return list(PIPELINE_ORDER), []

    valid = set(PIPELINE_ORDER)
    plan: List[str] = []
    unknown: List[str] = []

    for s in requested_steps:
        if s in valid:
            plan.append(s)
        else:
            unknown.append(s)

    seen: set[str] = set()
    unique_plan: List[str] = []
    for s in plan:
        if s not in seen:
            unique_plan.append(s)
            seen.add(s)

    seen2: set[str] = set()
    unique_unknown: List[str] = []
    for s in unknown:
        if s not in seen2:
            unique_unknown.append(s)
            seen2.add(s)

    return unique_plan, unique_unknown


def _apply_start_stop(plan: List[str], start_at: Optional[str], stop_after: Optional[str]) -> List[str]:
    out = list(plan)

    if start_at and start_at in out:
        out = out[out.index(start_at):]

    if stop_after and stop_after in out:
        out = out[: out.index(stop_after) + 1]

    return out


def _apply_exclude(plan: List[str], exclude: Optional[List[str]]) -> List[str]:
    if not exclude:
        return plan
    ex = set(exclude)
    return [s for s in plan if s not in ex]


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _ensure_step_shape(
    step_name: str,
    status: str,
    started_at: str,
    ended_at: str,
    duration_ms: int,
    data: Any = None,
    errors: Optional[List[Any]] = None,
    warnings: Optional[List[Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "step": step_name,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": duration_ms,
        "errors": errors or [],
        "warnings": warnings or [],
        "metrics": metrics or {},
        "data": data if data is not None else {},
        "timestamp": _utc_ts(),
    }


def _coerce_step_output(step_name: str, raw: Any) -> Tuple[Any, List[Any], List[Any], Dict[str, Any]]:
    if isinstance(raw, dict):
        data = raw.get("data")
        if data is None:
            data = {k: v for k, v in raw.items() if k not in {"errors", "warnings", "metrics", "status", "step"}}

        errors = _as_list(raw.get("errors"))
        warnings = _as_list(raw.get("warnings"))
        metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
        return data, errors, warnings, metrics

    return {"value": raw}, [], [], {}


def run_step(step_name: str, *, run_id: str) -> Dict[str, Any]:
    _log_event("info", "step_start", run_id=run_id, step=step_name)

    registry: Dict[str, Callable[[], Any]] = {
        "1_data_ingestion": _run_data_ingestion,
        "2_data_warehouse": _run_data_warehouse,
        "3_feature_engineering": _run_feature_engineering,
        "4_feature_store_sync": _run_feature_store_sync,
        "5_ml_prediction": _run_ml_prediction,
        "6_optimization": _run_optimization,
        "7_api_serving": _run_api_serving,
        "8_dashboard_update": _run_dashboard_update,
    }

    started_at = _utc_ts()
    t0 = time.time()

    fn = registry.get(step_name)
    if fn is None:
        ended_at = _utc_ts()
        duration_ms = int((time.time() - t0) * 1000)
        _log_event("warning", "step_skipped", run_id=run_id, step=step_name, reason="no_handler")
        return _ensure_step_shape(
            step_name,
            "skipped",
            started_at,
            ended_at,
            duration_ms,
            data={},
            errors=[],
            warnings=[{"message": "no handler registered"}],
            metrics={},
        )

    try:
        raw = fn()
        ended_at = _utc_ts()
        duration_ms = int((time.time() - t0) * 1000)

        data, errors, warnings, metrics = _coerce_step_output(step_name, raw)
        status = "ok" if not errors else "ok"

        _log_event("info", "step_done", run_id=run_id, step=step_name, duration_ms=duration_ms, status=status)
        return _ensure_step_shape(step_name, status, started_at, ended_at, duration_ms, data, errors, warnings, metrics)

    except ImportError as e:
        ended_at = _utc_ts()
        duration_ms = int((time.time() - t0) * 1000)
        _log_event("warning", "step_skipped", run_id=run_id, step=step_name, reason="import_error")
        return _ensure_step_shape(
            step_name,
            "skipped",
            started_at,
            ended_at,
            duration_ms,
            data={},
            errors=[],
            warnings=[{"message": str(e), "type": "ImportError"}],
            metrics={},
        )

    except Exception as e:
        ended_at = _utc_ts()
        duration_ms = int((time.time() - t0) * 1000)
        _log_event("error", "step_error", run_id=run_id, step=step_name, error_type=type(e).__name__)
        return _ensure_step_shape(
            step_name,
            "error",
            started_at,
            ended_at,
            duration_ms,
            data={},
            errors=[{"type": type(e).__name__, "message": str(e)}],
            warnings=[],
            metrics={},
        )


def run_pipeline(
    options: Any = None,
    *,
    steps: Any = None,
    exclude: Any = None,
    start_at: Optional[str] = None,
    stop_after: Optional[str] = None,
    dry_run: bool = False,
    stop_on_error: bool = False,
    strict: bool = True,
) -> Dict[str, Any]:
    run_id_from_options: Optional[str] = None
    if isinstance(options, dict):
        rid = options.get("run_id")
        if isinstance(rid, str) and rid.strip():
            run_id_from_options = rid.strip()

    run_id = run_id_from_options or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    if isinstance(options, dict):
        if steps is None and "steps" in options:
            steps = options.get("steps")
        if exclude is None and "exclude" in options:
            exclude = options.get("exclude")
        if start_at is None and "start_at" in options:
            start_at = options.get("start_at")
        if stop_after is None and "stop_after" in options:
            stop_after = options.get("stop_after")
        dry_run = _normalize_bool(options.get("dry_run"), default=dry_run)
        stop_on_error = _normalize_bool(options.get("stop_on_error"), default=stop_on_error)
        strict = _normalize_bool(options.get("strict"), default=strict)

    requested_steps = _normalize_steps(steps)
    requested_exclude = _normalize_steps(exclude)

    plan, unknown_steps = _select_plan(requested_steps)

    selector_unknown: List[str] = []
    if start_at and start_at not in PIPELINE_ORDER:
        selector_unknown.append(start_at)
    if stop_after and stop_after not in PIPELINE_ORDER:
        selector_unknown.append(stop_after)
    if requested_exclude:
        for ex in requested_exclude:
            if ex not in PIPELINE_ORDER:
                selector_unknown.append(ex)

    sel_seen: set[str] = set()
    selector_unknown = [s for s in selector_unknown if not (s in sel_seen or sel_seen.add(s))]

    base: Dict[str, Any] = {
        "timestamp": _utc_ts(),
        "run_id": run_id,
        "status": "ok",
        "dry_run": bool(dry_run),
        "strict": bool(strict),
        "stop_on_error": bool(stop_on_error),
        "plan": plan,
    }

    if requested_exclude:
        base["exclude"] = requested_exclude
    if start_at:
        base["start_at"] = start_at
    if stop_after:
        base["stop_after"] = stop_after
    if unknown_steps:
        base["unknown_steps"] = unknown_steps
    if selector_unknown:
        base["unknown_selectors"] = selector_unknown

    if strict and (unknown_steps or selector_unknown):
        base["status"] = "error"
        base["error"] = "unknown_steps" if unknown_steps else "unknown_selectors"
        base["summary"] = {
            "error_count": 1,
            "has_errors": True,
            "ok_count": 0,
            "skipped_count": 0,
            "steps_planned": len(plan),
            "steps_returned": 0,
        }
        _log_event("error", "pipeline_error_validation", run_id=run_id)
        return base

    plan2 = _apply_exclude(plan, requested_exclude)
    plan2 = _apply_start_stop(plan2, start_at, stop_after)
    base["plan"] = plan2

    if dry_run:
        base["summary"] = {
            "error_count": 0,
            "has_errors": False,
            "ok_count": 0,
            "skipped_count": 0,
            "steps_planned": len(plan2),
            "steps_returned": 0,
        }
        if (unknown_steps or selector_unknown) and not strict:
            base["warning"] = "unknown_values_ignored"
        _log_event("info", "pipeline_dry_run", run_id=run_id, steps_planned=len(plan2))
        return base

    pipeline_t0 = time.time()
    _log_event("info", "pipeline_start", run_id=run_id, steps_planned=len(plan2))

    results: List[Dict[str, Any]] = []
    for step in plan2:
        r = run_step(step, run_id=run_id)
        results.append(r)

        if stop_on_error and r.get("status") == "error":
            base["stopped_early"] = True
            base["stop_reason"] = "stop_on_error"
            base["stop_step"] = step
            break

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    error_count = sum(1 for r in results if r.get("status") == "error")
    skipped_count = sum(1 for r in results if r.get("status") == "skipped")
    has_errors = error_count > 0

    base["results"] = results
    base["summary"] = {
        "error_count": error_count,
        "has_errors": has_errors,
        "ok_count": ok_count,
        "skipped_count": skipped_count,
        "steps_planned": len(plan2),
        "steps_returned": len(results),
    }
    base["duration_ms"] = int((time.time() - pipeline_t0) * 1000)

    if (unknown_steps or selector_unknown) and not strict:
        base["warning"] = "unknown_values_ignored"

    _log_event(
        "info",
        "pipeline_done",
        run_id=run_id,
        ok=ok_count,
        errors=error_count,
        skipped=skipped_count,
        duration_ms=base["duration_ms"],
    )
    return base


def _run_data_ingestion() -> Any:
    from . import data_ingestion
    return data_ingestion.run()


def _run_data_warehouse() -> Any:
    from . import data_warehouse
    return data_warehouse.run()


def _run_feature_engineering() -> Any:
    from . import feature_engineering
    return feature_engineering.run()


def _run_feature_store_sync() -> Any:
    from . import feature_store_sync
    return feature_store_sync.run()


def _run_ml_prediction() -> Any:
    from . import ml_prediction
    return ml_prediction.run()


def _run_optimization() -> Any:
    from . import optimization
    return optimization.run()


def _run_api_serving() -> Any:
    from . import api_serving
    return api_serving.run()


def _run_dashboard_update() -> Any:
    from . import dashboard_update
    return dashboard_update.run()


if __name__ == "__main__":
    output = run_pipeline()
    print(output)