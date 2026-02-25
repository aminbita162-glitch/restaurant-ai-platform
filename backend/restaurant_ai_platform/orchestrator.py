from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
import time
import uuid
import threading


PIPELINE_ORDER: List[str] = [
    "real_data_ingestion",
    "1_data_ingestion",
    "2_data_warehouse",
    "3_feature_engineering",
    "4_feature_store_sync",
    "model_registry",
    "5_ml_prediction",
    "6_optimization",
    "gpt_insight",
    "7_api_serving",
    "8_dashboard_update",
    "9_model_training",
]


# ----------------------------
# Last-run storage (in-memory)
# ----------------------------
_LAST_RUN_LOCK = threading.Lock()
_LAST_RUN: Optional[Dict[str, Any]] = None


def set_last_run(payload: Dict[str, Any]) -> None:
    global _LAST_RUN
    with _LAST_RUN_LOCK:
        _LAST_RUN = payload


def get_last_run() -> Optional[Dict[str, Any]]:
    with _LAST_RUN_LOCK:
        return _LAST_RUN


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _log(message: str) -> None:
    print(f"[{_utc_ts()}] {message}")


def _log_event(
    event: str,
    *,
    run_id: str,
    step: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    parts = [f"event={event}", f"run_id={run_id}"]
    if step:
        parts.append(f"step={step}")
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v}")
    _log(" ".join(parts))


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
            cleaned.append(str(item).strip())
        cleaned = [s for s in cleaned if s]
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

    # dedupe keep order
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
        out = out[out.index(start_at) :]

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


def run_step(step_name: str, *, run_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    _log_event("step_start", run_id=run_id, step=step_name)

    registry: Dict[str, Callable[[Dict[str, Any]], Any]] = {
        "real_data_ingestion": _run_real_data_ingestion,
        "1_data_ingestion": _run_data_ingestion,
        "2_data_warehouse": _run_data_warehouse,
        "3_feature_engineering": _run_feature_engineering,
        "4_feature_store_sync": _run_feature_store_sync,
        "model_registry": _run_model_registry,
        "5_ml_prediction": _run_ml_prediction,
        "6_optimization": _run_optimization,
        "gpt_insight": _run_gpt_insight,
        "7_api_serving": _run_api_serving,
        "8_dashboard_update": _run_dashboard_update,
        "9_model_training": _run_model_training,
    }

    started_at = _utc_ts()
    t0 = time.time()

    fn = registry.get(step_name)
    if fn is None:
        ended_at = _utc_ts()
        duration_ms = int((time.time() - t0) * 1000)
        _log_event("step_skip", run_id=run_id, step=step_name, extra={"reason": "no_handler"})
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
        raw = fn(context)
        ended_at = _utc_ts()
        duration_ms = int((time.time() - t0) * 1000)

        data, errors, warnings, metrics = _coerce_step_output(step_name, raw)
        status = "ok" if not errors else "error"

        _log_event("step_done", run_id=run_id, step=step_name, extra={"duration_ms": duration_ms, "status": status})
        return _ensure_step_shape(step_name, status, started_at, ended_at, duration_ms, data, errors, warnings, metrics)

    except ImportError as e:
        ended_at = _utc_ts()
        duration_ms = int((time.time() - t0) * 1000)
        _log_event("step_skip", run_id=run_id, step=step_name, extra={"reason": "import_error"})
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
        _log_event("step_error", run_id=run_id, step=step_name, extra={"error_type": type(e).__name__})
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
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

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
        _log_event("pipeline_error_validation", run_id=run_id)
        set_last_run(base)
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
        _log_event("pipeline_dry_run", run_id=run_id, extra={"steps_planned": len(plan2)})
        set_last_run(base)
        return base

    _log_event("pipeline_start", run_id=run_id, extra={"steps_planned": len(plan2)})

    results: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {
        "_run_id": run_id,
        "_started_at": _utc_ts(),
    }

    for step in plan2:
        r = run_step(step, run_id=run_id, context=context)
        results.append(r)

        # store step data for downstream steps
        if isinstance(r, dict) and r.get("step"):
            context[r["step"]] = r.get("data", {})
            context["_last_step"] = r.get("step")
            context["_last_status"] = r.get("status")

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

    if (unknown_steps or selector_unknown) and not strict:
        base["warning"] = "unknown_values_ignored"

    _log_event("pipeline_done", run_id=run_id, extra={"ok": ok_count, "errors": error_count, "skipped": skipped_count})
    set_last_run(base)
    return base


def _run_real_data_ingestion(context: Dict[str, Any]) -> Dict[str, Any]:
    from . import real_data_ingestion

    return real_data_ingestion.run()


def _run_data_ingestion(context: Dict[str, Any]) -> Dict[str, Any]:
    from . import data_ingestion

    return data_ingestion.run()


def _run_data_warehouse(context: Dict[str, Any]) -> Dict[str, Any]:
    from . import data_warehouse

    return data_warehouse.run()


def _run_feature_engineering(context: Dict[str, Any]) -> Dict[str, Any]:
    from . import feature_engineering

    return feature_engineering.run()


def _run_feature_store_sync(context: Dict[str, Any]) -> Dict[str, Any]:
    from . import feature_store_sync

    return feature_store_sync.run()


def _run_model_registry(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Registers a simple mock model. If sales data is unavailable (e.g., when running a partial
    pipeline starting after ingestion), the step returns an "ok" status with a warning.
    """
    from . import model_registry

    ingestion = context.get("1_data_ingestion") or {}
    sales = ingestion.get("sales", [])

    if not isinstance(sales, list) or not sales:
        return {
            "data": {
                "model_registry_status": "skipped",
                "reason": "missing_sales_data_from_1_data_ingestion",
                "timestamp": _utc_ts(),
            },
            "errors": [],
            "warnings": [
                {
                    "type": "MissingInput",
                    "message": "No sales data found in context for 1_data_ingestion. "
                    "Run ingestion before model_registry to register a model.",
                }
            ],
            "metrics": {},
        }

    features = {"sales": sales}
    model = model_registry.train_and_save_model(features)

    return {
        "data": {
            "model_registry_status": "ok",
            "registered_model": model,
            "timestamp": _utc_ts(),
        },
        "errors": [],
        "warnings": [],
        "metrics": {},
    }


def _run_ml_prediction(context: Dict[str, Any]) -> Dict[str, Any]:
    from . import ml_prediction

    return ml_prediction.run()


def _run_optimization(context: Dict[str, Any]) -> Dict[str, Any]:
    from . import optimization

    return optimization.run()


def _run_gpt_insight(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional GPT step. If gpt_insight.py is missing or not ready, return ok with a warning
    (so the pipeline stays green).
    """
    try:
        from . import gpt_insight  # type: ignore
    except Exception as e:
        return {
            "data": {
                "gpt_insight_status": "skipped",
                "reason": f"import_failed:{type(e).__name__}",
                "timestamp": _utc_ts(),
            },
            "errors": [],
            "warnings": [{"type": type(e).__name__, "message": str(e)}],
            "metrics": {},
        }

    try:
        if hasattr(gpt_insight, "run"):
            out = gpt_insight.run(context)  # type: ignore[misc]
            return out if isinstance(out, dict) else {"gpt_insight_status": "ok", "value": out, "timestamp": _utc_ts()}
        return {
            "data": {
                "gpt_insight_status": "skipped",
                "reason": "no_run_function",
                "timestamp": _utc_ts(),
            },
            "errors": [],
            "warnings": [{"type": "MissingHandler", "message": "gpt_insight.run(context) not found"}],
            "metrics": {},
        }
    except Exception as e:
        return {
            "data": {
                "gpt_insight_status": "error",
                "reason": f"{type(e).__name__}:{e}",
                "timestamp": _utc_ts(),
            },
            "errors": [{"type": type(e).__name__, "message": str(e)}],
            "warnings": [],
            "metrics": {},
        }


def _run_api_serving(context: Dict[str, Any]) -> Dict[str, Any]:
    from . import api_serving

    return api_serving.run()


def _run_dashboard_update(context: Dict[str, Any]) -> Dict[str, Any]:
    from . import dashboard_update

    return dashboard_update.run()


def _run_model_training(context: Dict[str, Any]) -> Dict[str, Any]:
    from . import model_training

    return model_training.run()


if __name__ == "__main__":
    output = run_pipeline()
    print(output)