"""PHASE R5 tests — BOM with on-hand inventory.

Tests:
1. Missing recipe returns a stable error (INVENTORY_MISSING_BOM).
2. on_hand reduces suggested_purchase quantity (never below zero).
3. Missing on_hand returns INVENTORY_MISSING_ON_HAND.
4. suggested_purchase is never negative.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from restaurant_ai_platform import dashboard_update  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_RECIPE = [
    {
        "name": "Pasta",
        "expected_meals": 10,
        "ingredients": [
            {"name": "flour", "unit": "kg", "quantity_per_meal": 0.2},
            {"name": "tomato", "unit": "kg", "quantity_per_meal": 0.1},
        ],
    }
]

_VALID_SALES = [
    {"net_sales": 1500.0, "restaurant_id": "rest_r5", "location_id": "loc_r5",
     "date": "2024-01-01"},
    {"net_sales": 1200.0, "restaurant_id": "rest_r5", "location_id": "loc_r5",
     "date": "2024-01-02"},
    {"net_sales": 1800.0, "restaurant_id": "rest_r5", "location_id": "loc_r5",
     "date": "2024-01-03"},
    {"net_sales": 1100.0, "restaurant_id": "rest_r5", "location_id": "loc_r5",
     "date": "2024-01-04"},
    {"net_sales": 1400.0, "restaurant_id": "rest_r5", "location_id": "loc_r5",
     "date": "2024-01-05"},
    {"net_sales": 1600.0, "restaurant_id": "rest_r5", "location_id": "loc_r5",
     "date": "2024-01-06"},
    {"net_sales": 1300.0, "restaurant_id": "rest_r5", "location_id": "loc_r5",
     "date": "2024-01-07"},
]

_VALID_ON_HAND: dict = {"flour": 0.5, "tomato": 0.0}

_VALID_FORECAST = [
    {"predicted_sales": 1500.0, "date": "2024-01-08"},
]

_VALID_PREDICTION = {
    "status": "ok",
    "forecast": _VALID_FORECAST,
    "labor_recommendation_allowed": True,
}


def _make_context(**overrides) -> dict:
    ctx: dict = {
        "restaurant_id": "rest_r5",
        "location_id": "loc_r5",
        "sales": _VALID_SALES,
        "recipes": _VALID_RECIPE,
        "on_hand": _VALID_ON_HAND,
        "5_ml_prediction": _VALID_PREDICTION,
    }
    ctx.update(overrides)
    return ctx


# ---------------------------------------------------------------------------
# Test 1 — Missing recipe returns INVENTORY_MISSING_BOM
# ---------------------------------------------------------------------------

def test_missing_recipe_returns_error():
    result = dashboard_update.run(_make_context(recipes=None, bom=None))

    assert result.get("status") == "error", (
        f"Expected status=error when recipe missing, got: {result}"
    )
    assert result.get("error_code") == dashboard_update.INVENTORY_ERROR_MISSING_BOM, (
        f"Expected INVENTORY_MISSING_BOM, got: {result.get('error_code')}"
    )
    assert "inventory_plan" not in result, (
        "inventory_plan must not be returned when recipe is missing"
    )


# ---------------------------------------------------------------------------
# Test 2 — Missing on_hand returns INVENTORY_MISSING_ON_HAND
# ---------------------------------------------------------------------------

def test_missing_on_hand_returns_error():
    result = dashboard_update.run(_make_context(on_hand=None))

    assert result.get("status") == "error", (
        f"Expected status=error when on_hand missing, got: {result}"
    )
    assert result.get("error_code") == dashboard_update.INVENTORY_ERROR_MISSING_ON_HAND, (
        f"Expected INVENTORY_MISSING_ON_HAND, got: {result.get('error_code')}"
    )
    assert "inventory_plan" not in result, (
        "inventory_plan must not be returned when on_hand is missing"
    )


# ---------------------------------------------------------------------------
# Test 3 — on_hand reduces suggested_purchase
# ---------------------------------------------------------------------------

def test_on_hand_reduces_suggested_purchase():
    # flour: total_quantity = 0.2 * 10 = 2.0 kg
    # on_hand flour = 0.5 kg → suggested = 1.5 kg
    # tomato: total_quantity = 0.1 * 10 = 1.0 kg
    # on_hand tomato = 0.0 kg → suggested = 1.0 kg
    result = dashboard_update.run(_make_context())

    assert result.get("dashboard_update_status") == "ok", (
        f"Expected dashboard_update_status=ok, got: {result}"
    )

    plan = result.get("inventory_plan", [])
    assert plan, "inventory_plan must not be empty"

    by_ing = {row["ingredient"]: row for row in plan}

    flour = by_ing.get("flour")
    assert flour is not None, "Expected flour in inventory_plan"
    assert flour["suggested_purchase"] == 1.5, (
        f"flour: expected suggested_purchase=1.5, got {flour['suggested_purchase']}"
    )
    assert flour["on_hand"] == 0.5, (
        f"flour: expected on_hand=0.5, got {flour['on_hand']}"
    )

    tomato = by_ing.get("tomato")
    assert tomato is not None, "Expected tomato in inventory_plan"
    assert tomato["suggested_purchase"] == 1.0, (
        f"tomato: expected suggested_purchase=1.0, got {tomato['suggested_purchase']}"
    )


# ---------------------------------------------------------------------------
# Test 4 — suggested_purchase is never negative
# ---------------------------------------------------------------------------

def test_suggested_purchase_never_negative():
    # on_hand has more than needed → suggested must be 0, not negative
    abundant_on_hand = {"flour": 999.0, "tomato": 999.0}
    result = dashboard_update.run(_make_context(on_hand=abundant_on_hand))

    assert result.get("dashboard_update_status") == "ok", (
        f"Expected dashboard_update_status=ok, got: {result}"
    )

    plan = result.get("inventory_plan", [])
    for row in plan:
        assert row["suggested_purchase"] >= 0, (
            f"suggested_purchase must never be negative, got {row['suggested_purchase']}"
            f" for ingredient {row.get('ingredient')}"
        )
        assert row["suggested_purchase"] == 0.0, (
            f"With abundant on_hand, suggested_purchase must be 0,"
            f" got {row['suggested_purchase']} for {row.get('ingredient')}"
        )


# ---------------------------------------------------------------------------
# Test 5 — inventory_method is bom_rule
# ---------------------------------------------------------------------------

def test_inventory_method_is_bom_rule():
    result = dashboard_update.run(_make_context())

    assert result.get("inventory_method") == "bom_rule", (
        f"inventory_method must be 'bom_rule', got: {result.get('inventory_method')!r}"
    )
    desc = result.get("inventory_method_description", "")
    assert "mrp" not in desc.lower(), (
        f"inventory_method_description must not contain 'mrp': {desc!r}"
    )


if __name__ == "__main__":
    test_missing_recipe_returns_error()
    print("PASS test_missing_recipe_returns_error")

    test_missing_on_hand_returns_error()
    print("PASS test_missing_on_hand_returns_error")

    test_on_hand_reduces_suggested_purchase()
    print("PASS test_on_hand_reduces_suggested_purchase")

    test_suggested_purchase_never_negative()
    print("PASS test_suggested_purchase_never_negative")

    test_inventory_method_is_bom_rule()
    print("PASS test_inventory_method_is_bom_rule")

    print("\nAll PHASE R5 tests passed.")
