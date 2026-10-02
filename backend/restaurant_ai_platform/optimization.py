from datetime import datetime
from typing import Any, Dict, List, Optional

# Stable public error codes for staffing failures.
LABOR_ERROR_MISSING_INPUT = "LABOR_MISSING_SALES_OR_FORECAST"
LABOR_ERROR_FORECAST_FAILED = "LABOR_FORECAST_FAILED"
LABOR_ERROR_DISALLOWED = "LABOR_RECOMMENDATION_DISALLOWED"

# Rule constant: one staff member per this many sales units.
SALES_PER_STAFF = 600

# Standard 2-hour operating blocks for a restaurant day.
SHIFT_BLOCKS: List[tuple] = [
    ("10:00", "12:00"),
    ("12:00", "14:00"),
    ("14:00", "16:00"),
    ("16:00", "18:00"),
    ("18:00", "20:00"),
    ("20:00", "22:00"),
]

# Default role list when the caller does not supply one.
DEFAULT_ROLES: List[str] = ["cook", "floor", "cashier"]


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _staff_for_sales(predicted_sales: float) -> int:
    """Simple ratio rule: 1 staff per SALES_PER_STAFF units (minimum 1)."""
    if predicted_sales <= 0:
        return 1
    return max(1, int(predicted_sales / SALES_PER_STAFF))


def _distribute_staff(total: int, num_blocks: int) -> List[int]:
    """Distribute *total* staff across *num_blocks* blocks as evenly as possible.

    No headcount is invented — the sum of the returned list equals *total*.
    """
    if total <= 0 or num_blocks <= 0:
        return [0] * max(num_blocks, 0)
    base = total // num_blocks
    remainder = total % num_blocks
    return [base + (1 if i < remainder else 0) for i in range(num_blocks)]


def _assign_roles(block_ns: List[int], roles: List[str]) -> List[str]:
    """Assign a role to each block by cycling through the role list."""
    if not roles:
        roles = DEFAULT_ROLES
    return [roles[i % len(roles)] for i in range(len(block_ns))]


def run(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Produce a printable shift plan using a fixed sales-per-staff ratio rule.

    This is a simple rule — it is NOT a constraint optimizer or ML model.
    The output field `method` is always set to `rule` to reflect this.
    The output field `approval` is always set to `proposed`.
    """
    print(f"[{_utc_ts()}] START optimization")

    ctx = context or {}
    restaurant_id = ctx.get("restaurant_id")
    location_id = ctx.get("location_id")
    sales = ctx.get("sales")

    # Caller-supplied role list; falls back to DEFAULT_ROLES if absent or empty.
    caller_roles: List[str] = []
    raw_roles = ctx.get("roles")
    if isinstance(raw_roles, list) and all(isinstance(r, str) for r in raw_roles) and raw_roles:
        caller_roles = [r.strip() for r in raw_roles if r.strip()]
    roles = caller_roles if caller_roles else DEFAULT_ROLES

    # Caller-supplied hourly cost per role — only used if present, never invented.
    hourly_cost: Optional[Dict[str, float]] = None
    raw_hc = ctx.get("hourly_cost")
    if isinstance(raw_hc, dict):
        hourly_cost = {k: float(v) for k, v in raw_hc.items() if v is not None}
    elif isinstance(raw_hc, (int, float)):
        # Single scalar: apply to all roles.
        hourly_cost = {r: float(raw_hc) for r in roles}

    # Guard: sales input must be present so the forecast step can run.
    if not sales or not isinstance(sales, list):
        code = LABOR_ERROR_MISSING_INPUT
        print(f"[{_utc_ts()}] optimization status=error code={code}")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    # Read the forecast already produced on the context artifact.
    # Do NOT import or rerun ml_prediction here.
    forecast_result = ctx.get("5_ml_prediction")

    # Guard: forecast artifact must be present and successful.
    if not isinstance(forecast_result, dict) or forecast_result.get("status") == "error":
        code = LABOR_ERROR_FORECAST_FAILED
        upstream_code = (
            forecast_result.get("error_code", "missing_forecast_artifact")
            if isinstance(forecast_result, dict)
            else "missing_forecast_artifact"
        )
        print(f"[{_utc_ts()}] optimization status=error code={code} upstream={upstream_code}")
        return {
            "status": "error",
            "error_code": code,
            "upstream_error_code": upstream_code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    forecast = forecast_result.get("forecast", [])

    # Guard: forecast list must be non-empty.
    if not forecast or not isinstance(forecast, list):
        code = LABOR_ERROR_MISSING_INPUT
        print(f"[{_utc_ts()}] optimization status=error code={code} reason=empty_forecast")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    # Read the backtest quality gate from the forecast artifact.
    labor_recommendation_allowed = forecast_result.get("labor_recommendation_allowed", False)

    # If labor recommendations are disallowed, return no headcount — stable code only.
    if not labor_recommendation_allowed:
        code = LABOR_ERROR_DISALLOWED
        print(f"[{_utc_ts()}] optimization status=error code={code} reason=labor_recommendation_not_allowed")
        return {
            "status": "error",
            "error_code": code,
            "labor_recommendation_allowed": False,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "method": "rule",
            "approval": "proposed",
            "timestamp": _utc_ts(),
        }

    # Apply the ratio rule to each forecast day, then split each day's
    # recommended staff into 2-hour blocks. No headcount is invented —
    # the sum of n across blocks equals the daily recommendation.
    num_blocks = len(SHIFT_BLOCKS)
    staffing_plan: List[Dict[str, Any]] = []

    for i, day in enumerate(forecast):
        predicted = float(day.get("predicted_sales", 0))
        day_total = _staff_for_sales(predicted)
        block_ns = _distribute_staff(day_total, num_blocks)
        block_roles = _assign_roles(block_ns, roles)

        for j, (start, end) in enumerate(SHIFT_BLOCKS):
            block_role = block_roles[j]
            block: Dict[str, Any] = {
                "day_index": i + 1,
                "date": day.get("date"),
                "block_index": j + 1,
                "start": start,
                "end": end,
                "role": block_role,
                "n": block_ns[j],
            }

            # Include estimated_labor_cost only if hourly_cost was supplied.
            # Do not invent wages.
            if hourly_cost is not None:
                rate = hourly_cost.get(block_role) or hourly_cost.get("*")
                if rate is not None:
                    block_hours = 2.0  # each block is 2 hours
                    block["estimated_labor_cost"] = round(rate * block_ns[j] * block_hours, 2)

            staffing_plan.append(block)

    print(
        f"[{_utc_ts()}] DONE optimization method=rule"
        f" days={len(forecast)} blocks={len(staffing_plan)}"
        f" labor_recommendation_allowed={labor_recommendation_allowed}"
    )

    return {
        "optimization_status": "ok",
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        # method is always rule — this is a ratio, not a constraint optimizer.
        "method": "rule",
        # approval is always proposed — manager must confirm before acting.
        "approval": "proposed",
        "method_description": (
            f"1 staff per {SALES_PER_STAFF} sales units (minimum 1),"
            " distributed across 2-hour blocks"
        ),
        "roles_used": roles,
        "staffing_plan": staffing_plan,
        "labor_recommendation_allowed": labor_recommendation_allowed,
        "timestamp": _utc_ts(),
    }
