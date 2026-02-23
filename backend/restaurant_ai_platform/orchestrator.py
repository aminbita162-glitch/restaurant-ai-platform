from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set


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
    print(f"[{_utc_ts()}] {message}")


def _normalize_steps(value: Any) -> List[str]:
    """
    Accepts:
      - None
      - list/tuple/set of strings
      - comma-separated string: "1_data_ingestion,2_data_warehouse"
    Returns a list of step names.
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for x in value:
            if x is None:
                continue
            out.append(str(x).strip())
        return [s for s in out if s]
    return [str(value).strip()] if str(value).strip() else []


def _filter_pipeline_steps(
    options: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Builds an execution plan based on options.

    Supported options:
      - steps: ["1_data_ingestion", "2_data_warehouse"]  (explicit allow-list)
      - include: same as steps (alias)
      - exclude: ["8_dashboard_update"]  (block-list)
      - start_at: "3_feature_engineering"
      - stop_after: "6_optimization"
      - dry_run: true/false
      - stop_on_error: true/false (default false)
    """
    options = options or {}

    steps_allow = _normalize_steps(options.get("steps")) or _normalize_steps(options.get("include"))
    steps_exclude = set(_normalize_steps(options.get("exclude")))

    start_at = (options.get("start_at") or "").strip()
    stop_after = (options.get("stop_after") or "").strip()

    dry_run = bool(options.get("dry_run", False))
    stop_on_error = bool(options.get("stop_on_error", False))

    all_steps: List[str] = list(PIPELINE_ORDER)

    unknown: List[str] = []
    known_set: Set[str] = set(all_steps)

    if steps_allow:
        for s in steps_allow:
            if s not in known_set:
                unknown.append(s)
        # keep the pipeline order, but only allow selected ones
        plan = [s for s in all_steps if s in set(steps_allow)]
    else:
        plan = all_steps[:]

    # start_at / stop_after slicing (only if provided and valid)
    if start_at:
        if start_at not in known_set:
            unknown.append(start_at)
        else:
            idx = plan.index(start_at) if start_at in plan else -1
            if idx >= 0:
                plan = plan[idx:]

    if stop_after:
        if stop_after not in known_set:
            unknown.append(stop_after)
        else:
            if stop_after in plan:
                idx = plan.index(stop_after)
                plan = plan[: idx + 1]

    # exclude removal
    if steps_exclude:
        for s in steps_exclude:
            if s not in known_set:
                unknown.append(s)
        plan = [s for s in plan if s not in steps_exclude]

    return {
        "ok": len(unknown) == 0,
        "unknown_steps": sorted(list(set(unknown))),
        "plan": plan,
        "dry_run": dry_run,
        "stop_on_error": stop_on_error,
        "timestamp": _utc_ts(),
    }


def run_step(step_name: str) -> Dict[str, Any]:
    """
    Executes one pipeline step and returns a structured payload.
    - Missing/unimplemented steps are skipped safely.
    - Exceptions are captured so the pipeline can continue (unless stop_on_error is enabled in run_pipeline()).
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


def run_pipeline(options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Runs the pipeline with optional controls.

    Example options:
      {
        "steps": ["1_data_ingestion", "2_data_warehouse"],
        "exclude": ["8_dashboard_update"],
        "dry_run": false,
        "stop_on_error": true
      }
    """
    plan_info = _filter_pipeline_steps(options)
    plan = plan_info["plan"]

    # If user asked for invalid step names, we still return a safe response
    if not plan_info["ok"]:
        return {
            "status": "error",
            "error": "unknown_steps",
            "unknown_steps": plan_info["unknown_steps"],
            "plan": plan,
            "timestamp": _utc_ts(),
        }

    # Dry-run: do not execute, only show plan
    if plan_info["dry_run"]:
        return {
            "status": "ok",
            "dry_run": True,
            "plan": plan,
            "timestamp": _utc_ts(),
        }

    results: List[Dict[str, Any]] = []
    stop_on_error = bool(plan_info["stop_on_error"])

    for step in plan:
        step_result = run_step(step)
        results.append(step_result)

        if stop_on_error and step_result.get("status") == "error":
            break

    return {
        "status": "ok",
        "dry_run": False,
        "plan": plan,
        "results": results,
        "timestamp": _utc_ts(),
    }


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