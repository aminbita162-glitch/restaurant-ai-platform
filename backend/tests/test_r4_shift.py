"""PHASE R4 tests — Printable shift blocks.

Tests:
1. Blocks have start, end, role, n.
2. method=rule — no optimizer claim.
3. labor_recommendation_allowed=false returns no headcount (LABOR_RECOMMENDATION_DISALLOWED).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from restaurant_ai_platform import optimization  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(labor_allowed: bool = True, roles=None, hourly_cost=None) -> dict:
    """Build a minimal context that satisfies the guards in optimization.run()."""
    forecast = [
        {"predicted_sales": 1800.0, "date": "2024-01-01"},
        {"predicted_sales": 1200.0, "date": "2024-01-02"},
    ]
    ctx: dict = {
        "restaurant_id": "rest_r4",
        "location_id": "loc_r4",
        "sales": [{"net_sales": 1800.0}],
        "5_ml_prediction": {
            "status": "ok",
            "forecast": forecast,
            "labor_recommendation_allowed": labor_allowed,
        },
    }
    if roles is not None:
        ctx["roles"] = roles
    if hourly_cost is not None:
        ctx["hourly_cost"] = hourly_cost
    return ctx


# ---------------------------------------------------------------------------
# Test 1 — Blocks have start, end, role, n
# ---------------------------------------------------------------------------

def test_blocks_have_start_end_role_n():
    result = optimization.run(_make_context(labor_allowed=True))

    assert result.get("optimization_status") == "ok", (
        f"Expected optimization_status=ok, got: {result}"
    )

    plan = result.get("staffing_plan")
    assert plan and isinstance(plan, list), "staffing_plan must be a non-empty list"

    for block in plan:
        assert "start" in block, f"Block missing 'start': {block}"
        assert "end" in block, f"Block missing 'end': {block}"
        assert "role" in block, f"Block missing 'role': {block}"
        assert "n" in block, f"Block missing 'n': {block}"
        assert isinstance(block["n"], int), f"'n' must be int, got: {type(block['n'])}"
        assert block["start"] < block["end"], (
            f"start must precede end: {block['start']} >= {block['end']}"
        )


# ---------------------------------------------------------------------------
# Test 2 — method=rule, approval=proposed, no optimizer claim
# ---------------------------------------------------------------------------

def test_method_is_rule_not_optimizer():
    result = optimization.run(_make_context(labor_allowed=True))

    assert result.get("method") == "rule", (
        f"method must be 'rule', got: {result.get('method')!r}"
    )
    assert result.get("approval") == "proposed", (
        f"approval must be 'proposed', got: {result.get('approval')!r}"
    )
    desc = result.get("method_description", "")
    forbidden = ["optimizer", "ml", "machine learning", "trained", "neural"]
    for word in forbidden:
        assert word.lower() not in desc.lower(), (
            f"method_description must not contain '{word}': {desc!r}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Default roles are cook, floor, cashier
# ---------------------------------------------------------------------------

def test_default_roles_are_cook_floor_cashier():
    result = optimization.run(_make_context(labor_allowed=True))
    plan = result.get("staffing_plan", [])
    assert plan, "staffing_plan must not be empty"

    allowed = set(optimization.DEFAULT_ROLES)
    for block in plan:
        assert block["role"] in allowed, (
            f"role {block['role']!r} not in default roles {allowed}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Caller-supplied role list is respected
# ---------------------------------------------------------------------------

def test_caller_roles_override_defaults():
    custom_roles = ["manager", "barista"]
    result = optimization.run(_make_context(labor_allowed=True, roles=custom_roles))
    plan = result.get("staffing_plan", [])
    assert plan, "staffing_plan must not be empty"

    allowed = set(custom_roles)
    for block in plan:
        assert block["role"] in allowed, (
            f"role {block['role']!r} not in caller-supplied roles {allowed}"
        )


# ---------------------------------------------------------------------------
# Test 5 — estimated_labor_cost only present when hourly_cost supplied
# ---------------------------------------------------------------------------

def test_no_estimated_labor_cost_without_hourly_cost():
    result = optimization.run(_make_context(labor_allowed=True))
    plan = result.get("staffing_plan", [])
    for block in plan:
        assert "estimated_labor_cost" not in block, (
            "estimated_labor_cost must not be present when hourly_cost is not supplied"
        )


def test_estimated_labor_cost_present_with_hourly_cost():
    hc = {"cook": 15.0, "floor": 12.0, "cashier": 13.0}
    result = optimization.run(_make_context(labor_allowed=True, hourly_cost=hc))
    plan = result.get("staffing_plan", [])
    assert plan, "staffing_plan must not be empty"

    # Every block whose role has a known rate must carry estimated_labor_cost.
    for block in plan:
        if block["role"] in hc:
            assert "estimated_labor_cost" in block, (
                f"expected estimated_labor_cost on block with role {block['role']!r}"
            )
            assert isinstance(block["estimated_labor_cost"], float), (
                f"estimated_labor_cost must be float, got {type(block['estimated_labor_cost'])}"
            )


# ---------------------------------------------------------------------------
# Test 6 — Disallowed labor returns no invented headcount
# ---------------------------------------------------------------------------

def test_disallowed_labor_returns_no_headcount():
    result = optimization.run(_make_context(labor_allowed=False))

    assert result.get("status") == "error", (
        f"Expected status=error when labor disallowed, got: {result}"
    )
    assert result.get("error_code") == optimization.LABOR_ERROR_DISALLOWED, (
        f"Expected LABOR_RECOMMENDATION_DISALLOWED, got: {result.get('error_code')}"
    )
    # Must not contain a staffing_plan with headcount data.
    assert "staffing_plan" not in result or result.get("staffing_plan") is None, (
        "staffing_plan must not be returned when labor recommendation is disallowed"
    )
    assert result.get("labor_recommendation_allowed") is False, (
        "labor_recommendation_allowed must be False in the error response"
    )


if __name__ == "__main__":
    test_blocks_have_start_end_role_n()
    print("PASS test_blocks_have_start_end_role_n")

    test_method_is_rule_not_optimizer()
    print("PASS test_method_is_rule_not_optimizer")

    test_default_roles_are_cook_floor_cashier()
    print("PASS test_default_roles_are_cook_floor_cashier")

    test_caller_roles_override_defaults()
    print("PASS test_caller_roles_override_defaults")

    test_no_estimated_labor_cost_without_hourly_cost()
    print("PASS test_no_estimated_labor_cost_without_hourly_cost")

    test_estimated_labor_cost_present_with_hourly_cost()
    print("PASS test_estimated_labor_cost_present_with_hourly_cost")

    test_disallowed_labor_returns_no_headcount()
    print("PASS test_disallowed_labor_returns_no_headcount")

    print("\nAll PHASE R4 tests passed.")
