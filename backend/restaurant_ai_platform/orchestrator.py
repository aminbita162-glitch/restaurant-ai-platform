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
    print(f"[{_utc_ts()}] {message}")


def _normalize_steps(value: Any) -> Optional[List[str]]:
    """
    Accepts:
      - None
      - list/tuple/set of strings
      - comma-separated string
    Returns:
      - None (meaning: use default PIPELINE_ORDER)
      - list of step names
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

    # fallback
    s = str(value).strip()
    return [s] if s else None


def _select_plan(
    requested_steps: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
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

    # remove duplicates while keeping order (optional but nice)
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
    *,
    steps: Any = None,
    dry_run: bool = False,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    steps:
      - None => run full PIPELINE_ORDER
      - list / "comma,separated" => run selected steps (valid ones)

    dry_run:
      - True => only returns plan, does not execute steps

    strict:
      - True  => if unknown steps exist => status=error (no execution)
      - False => unknown steps are ignored; execution continues with valid plan
    """
    requested_steps = _normalize_steps(steps)
    plan, unknown_steps = _select_plan(requested_steps)

    base: Dict[str, Any] = {
        "timestamp": _utc_ts(),
        "dry_run": bool(dry_run),
        "strict": bool(strict),
        "plan": plan,
    }

    if unknown_steps:
        base["unknown_steps"] = unknown_steps

    if unknown_steps and strict:
        return {
            **base,
            "status": "error",
            "error": "unknown_steps",
        }

    if dry_run:
        # Non-strict dry-run: just show plan (+ unknown_steps if any)
        out = {**base, "status": "ok"}
        if unknown_steps and not strict:
            out["warning"] = "unknown_steps_ignored"
        return out

    results: List[Dict[str, Any]] = []
    for step in plan:
        results.append(run_step(step))

    out2: Dict[str, Any] = {
        **base,
        "status": "ok",
        "results": results,
    }
    if unknown_steps and not strict:
        out2["warning"] = "unknown_steps_ignored"
    return out2


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