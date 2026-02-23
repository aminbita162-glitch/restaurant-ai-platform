from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


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


def _log(message: str) -> None:
    # Simple stdout logging (Render will capture it)
    print(f"[{_utc_ts()}] {message}")


def _normalize_steps(value: Any) -> Optional[List[str]]:
    """
    Accepts:
      - None
      - list/tuple/set of strings
      - comma-separated string
    Returns:
      - None (meaning: use default PIPELINE_ORDER)
      - list of step names (cleaned)
    """
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


def _select_plan(requested_steps: Optional[List[str]]) -> Tuple[List[str], List[str]]:
    """
    Returns (plan, unknown_steps)
    - plan is filtered to valid steps
    - unknown_steps contains anything not in PIPELINE_ORDER
    """
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

    # remove duplicates while keeping order
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
    """
    Applies start_at and stop_after on a plan.
    - start_at: trims everything before it (inclusive start)
    - stop_after: trims everything after it (inclusive end)
    If step not found, returns plan unchanged (validation is handled elsewhere if strict).
    """
    out = list(plan)

    if start_at:
        if start_at in out:
            out = out[out.index(start_at) :]

    if stop_after:
        if stop_after in out:
            out = out[: out.index(stop_after) + 1]

    return out


def _apply_exclude(plan: List[str], exclude: Optional[List[str]]) -> List[str]:
    if not exclude:
        return plan
    ex = set(exclude)
    return [s for s in plan if s not in ex]


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


def run_step(step_name: str) -> Dict[str, Any]:
    """
    Executes one pipeline step and returns a structured payload.
    - Missing/unimplemented steps are skipped safely.
    - Exceptions are captured so the pipeline can continue.
    """
    _log(f"START {step_name}")

    registry: Dict[str, Callable[[], Dict[str, Any]]] = {
        "1_data_ingestion": _run_data_ingestion,
        "2_data_warehouse": _run_data_warehouse,
        "3_feature_engineering": _run_feature_engineering,
        "4_feature_store_sync": _run_feature_store_sync,
        "5_ml_prediction": _run_ml_prediction,
        "6_optimization": _run_optimization,
        "7_api_serving": _run_api_serving,
        "8_dashboard_update": _run_dashboard_update,
    }

    fn = registry.get(step_name)
    if fn is None:
        _log(f"SKIP  {step_name} (no handler registered)")
        return {
            "step": step_name,
            "status": "skipped",
            "reason": "no_handler",
            "timestamp": _utc_ts(),
        }

    try:
        result = fn()
        _log(f"DONE  {step_name}")
        return {
            "step": step_name,
            "status": "ok",
            "result": result,
            "timestamp": _utc_ts(),
        }
    except ImportError as e:
        _log(f"SKIP  {step_name} (import error: {e})")
        return {
            "step": step_name,
            "status": "skipped",
            "reason": "import_error",
            "error": str(e),
            "timestamp": _utc_ts(),
        }
    except Exception as e:
        _log(f"ERROR {step_name} ({type(e).__name__}: {e})")
        return {
            "step": step_name,
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
            "timestamp": _utc_ts(),
        }


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
    """
    Supports BOTH:
      - run_pipeline({"steps":[...], "dry_run": True, ...})
      - run_pipeline(steps=[...], dry_run=True, ...)

    options keys (optional):
      - steps: list[str] or "a,b,c"
      - exclude: list[str] or "a,b"
      - start_at: str
      - stop_after: str
      - dry_run: bool
      - stop_on_error: bool
      - strict: bool
    """
    # Merge dict options if provided
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

    # Validate selectors/exclude values (only matters if strict=True)
    unknown_selectors: List[str] = []
    if start_at and start_at not in PIPELINE_ORDER:
        unknown_selectors.append(start_at)
    if stop_after and stop_after not in PIPELINE_ORDER:
        unknown_selectors.append(stop_after)
    if requested_exclude:
        for ex in requested_exclude:
            if ex not in PIPELINE_ORDER:
                unknown_selectors.append(ex)

    # Deduplicate unknown_selectors preserving order
    sel_seen: set[str] = set()
    dedup_unknown_selectors: List[str] = []
    for s in unknown_selectors:
        if s not in sel_seen:
            dedup_unknown_selectors.append(s)
            sel_seen.add(s)
    unknown_selectors = dedup_unknown_selectors

    options_used: Dict[str, Any] = {
        "steps": requested_steps,
        "exclude": requested_exclude,
        "start_at": start_at,
        "stop_after": stop_after,
        "dry_run": bool(dry_run),
        "stop_on_error": bool(stop_on_error),
        "strict": bool(strict),
    }

    base: Dict[str, Any] = {
        "timestamp": _utc_ts(),
        "status": "ok",  # overall request status (not per-step)
        "options_used": options_used,
        "plan": plan,
    }

    if unknown_steps:
        base["unknown_steps"] = unknown_steps
    if unknown_selectors:
        base["unknown_selectors"] = unknown_selectors

    # strict mode: fail fast on any unknowns
    if strict and (unknown_steps or unknown_selectors):
        base["status"] = "error"
        base["error"] = "unknown_steps" if unknown_steps else "unknown_selectors"
        return base

    # Apply exclude/start/stop (unknowns are ignored in non-strict mode)
    plan2 = _apply_exclude(plan, requested_exclude)
    plan2 = _apply_start_stop(plan2, start_at, stop_after)
    base["plan"] = plan2

    if dry_run:
        if (unknown_steps or unknown_selectors) and not strict:
            base["warning"] = "unknown_values_ignored"
        return base

    results: List[Dict[str, Any]] = []
    error_count = 0

    for step in plan2:
        step_result = run_step(step)
        results.append(step_result)

        if step_result.get("status") == "error":
            error_count += 1
            if stop_on_error:
                base["stopped_early"] = True
                base["stop_reason"] = "stop_on_error"
                base["stop_step"] = step
                break

    # Summary for standardized output (does not break existing callers)
    base["results"] = results
    base["summary"] = {
        "steps_planned": len(plan2),
        "steps_returned": len(results),
        "error_count": error_count,
        "has_errors": error_count > 0,
    }

    if (unknown_steps or unknown_selectors) and not strict:
        base["warning"] = "unknown_values_ignored"

    return base


def _run_data_ingestion() -> Dict[str, Any]:
    from . import data_ingestion
    return data_ingestion.run()


def _run_data_warehouse() -> Dict[str, Any]:
    from . import data_warehouse
    return data_warehouse.run()


def _run_feature_engineering() -> Dict[str, Any]:
    from . import feature_engineering
    return feature_engineering.run()


def _run_feature_store_sync() -> Dict[str, Any]:
    from . import feature_store_sync
    return feature_store_sync.run()


def _run_ml_prediction() -> Dict[str, Any]:
    from . import ml_prediction
    return ml_prediction.run()


def _run_optimization() -> Dict[str, Any]:
    from . import optimization
    return optimization.run()


def _run_api_serving() -> Dict[str, Any]:
    from . import api_serving
    return api_serving.run()


def _run_dashboard_update() -> Dict[str, Any]:
    from . import dashboard_update
    return dashboard_update.run()


if __name__ == "__main__":
    output = run_pipeline()
    print(output)