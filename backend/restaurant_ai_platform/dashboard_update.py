from datetime import datetime
from typing import Any, Dict, List, Optional

# Stable public error codes for inventory and waste failures.
INVENTORY_ERROR_MISSING_SALES = "INVENTORY_MISSING_SALES"
INVENTORY_ERROR_INVALID_SALES = "INVENTORY_INVALID_SALES"
INVENTORY_ERROR_MISSING_BOM = "INVENTORY_MISSING_BOM"
INVENTORY_ERROR_INVALID_BOM = "INVENTORY_INVALID_BOM"
WASTE_ERROR_MISSING_INPUT = "WASTE_MISSING_INPUT"
WASTE_ERROR_INVALID_INPUT = "WASTE_INVALID_INPUT"

# Waste ratio constant — rule, not a trained model.
# Estimated food waste = 8 % of daily sales value.
WASTE_RATIO = 0.08

# Default weekly waste target — a rule constant, not an optimised value.
# Target: no more than 5 % of weekly sales value as waste.
DEFAULT_WEEKLY_WASTE_TARGET_PCT = 0.05

DEFAULT_RESTAURANT_ID = "restaurant_001"
DEFAULT_LOCATION_ID = "location_001"


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _waste_for_sales(daily_sales: float) -> float:
    """Fixed-ratio rule: estimated food waste derived from a numeric daily sales total.

    This is a rule — it is NOT a waste model or ML prediction.
    """
    return round(float(daily_sales) * WASTE_RATIO, 2)


def _validate_waste_input(sales: Any) -> Optional[str]:
    """Validate that sales rows carry a numeric daily_sales_total.

    Returns None if valid, or a stable error code string if invalid / missing.
    """
    if not isinstance(sales, list) or not sales:
        return WASTE_ERROR_MISSING_INPUT

    for row in sales:
        if not isinstance(row, dict):
            return WASTE_ERROR_INVALID_INPUT
        v = row.get("daily_sales_total")
        if not isinstance(v, (int, float)):
            return WASTE_ERROR_INVALID_INPUT

    return None


def _build_waste_plan(
    sales: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a per-day waste estimate from actual sales rows using a fixed ratio.

    Every value is numeric — no fake kg is invented. If the caller passes
    non-numeric inputs the caller must call _validate_waste_input first.
    """
    plan: List[Dict[str, Any]] = []
    for i, row in enumerate(sales):
        daily = float(row.get("daily_sales_total", 0))
        plan.append(
            {
                "day_index": i + 1,
                "daily_sales_total": daily,
                "estimated_waste": _waste_for_sales(daily),
            }
        )
    return plan


def _weekly_waste_target(
    sales: List[Dict[str, Any]],
    target_pct: float,
) -> Dict[str, Any]:
    """Compute the weekly waste target from the most recent 7 sales rows.

    target_pct is a fraction (0.05 = 5 %). The target is a numeric kg-equiv-
    alent of the sales value — it is a rule, not an optimised value.
    """
    recent = sales[-7:] if len(sales) >= 7 else sales
    weekly_sales = sum(float(r.get("daily_sales_total", 0)) for r in recent)
    return {
        "target_pct": target_pct,
        "weekly_sales_basis": round(weekly_sales, 2),
        "weekly_waste_target": round(weekly_sales * target_pct, 2),
    }


def _validate_bom(recipes: Any) -> Optional[str]:
    """Validate a recipe/BOM structure.

    Expected shape:
        recipes = [
            {
                "name": "Margherita Pizza",
                "expected_meals": 50,
                "ingredients": [
                    {"name": "flour", "unit": "kg", "quantity_per_meal": 0.15},
                    ...
                ],
            },
            ...
        ]

    Returns None if valid, or a stable error code string if invalid.
    """
    if not isinstance(recipes, list) or not recipes:
        return INVENTORY_ERROR_MISSING_BOM

    for recipe in recipes:
        if not isinstance(recipe, dict):
            return INVENTORY_ERROR_INVALID_BOM
        name = recipe.get("name")
        meals = recipe.get("expected_meals")
        ingredients = recipe.get("ingredients")
        if not name or not isinstance(name, str):
            return INVENTORY_ERROR_INVALID_BOM
        if not isinstance(meals, (int, float)) or meals < 0:
            return INVENTORY_ERROR_INVALID_BOM
        if not isinstance(ingredients, list) or not ingredients:
            return INVENTORY_ERROR_INVALID_BOM
        for ing in ingredients:
            if not isinstance(ing, dict):
                return INVENTORY_ERROR_INVALID_BOM
            if not ing.get("name") or not isinstance(ing.get("name"), str):
                return INVENTORY_ERROR_INVALID_BOM
            if not isinstance(ing.get("quantity_per_meal"), (int, float)) or ing.get("quantity_per_meal") < 0:
                return INVENTORY_ERROR_INVALID_BOM
            if not ing.get("unit") or not isinstance(ing.get("unit"), str):
                return INVENTORY_ERROR_INVALID_BOM

    return None


def _build_bom_inventory_plan(
    recipes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build an inventory list from an explicit recipe/BOM structure.

    Each recipe contributes ``quantity_per_meal * expected_meals`` per ingredient.
    No kg quantities are invented — every value traces back to an explicit BOM entry.
    """
    plan: List[Dict[str, Any]] = []

    for recipe in recipes:
        meals = float(recipe.get("expected_meals", 0))
        ingredients = recipe.get("ingredients", [])
        for ing in ingredients:
            qty_per_meal = float(ing.get("quantity_per_meal", 0))
            total_qty = round(qty_per_meal * meals, 2)
            plan.append(
                {
                    "recipe": recipe.get("name"),
                    "ingredient": ing.get("name"),
                    "unit": ing.get("unit"),
                    "quantity_per_meal": qty_per_meal,
                    "expected_meals": round(meals, 2),
                    "total_quantity": total_qty,
                }
            )

    return plan


def run(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print(f"[{_utc_ts()}] START dashboard_update")

    context = context or {}

    try:
        from . import optimization
        from . import gpt_insight
    except Exception as e:
        return {
            "dashboard_update_status": "error",
            "reason": f"import_failed:{type(e).__name__}:{e}",
            "timestamp": _utc_ts(),
        }

    prediction_data: Dict[str, Any] = {}
    if isinstance(context.get("5_ml_prediction"), dict):
        prediction_data = context.get("5_ml_prediction", {})

    # C4: do NOT import or rerun ml_prediction here. The forecast is read
    # from the context artifact produced by the upstream 5_ml_prediction
    # step. If the artifact is absent, prediction_data stays empty and
    # downstream consumers handle the missing forecast gracefully.

    optimization_data: Dict[str, Any] = {}
    if isinstance(context.get("6_optimization"), dict):
        optimization_data = context.get("6_optimization", {})

    if not optimization_data:
        try:
            optimization_result = optimization.run(context)
            optimization_data = optimization_result
        except Exception as e:
            return {
                "dashboard_update_status": "error",
                "reason": f"optimization_failed:{type(e).__name__}:{e}",
                "timestamp": _utc_ts(),
            }

    gpt_data: Dict[str, Any] = {}
    if isinstance(context.get("gpt_insight"), dict):
        gpt_data = context.get("gpt_insight", {})

    if not gpt_data:
        try:
            gpt_result = gpt_insight.run(
                {
                    "5_ml_prediction": prediction_data,
                }
            )
            gpt_data = gpt_result.get("data", {}) if isinstance(gpt_result, dict) else {}
        except Exception as e:
            gpt_data = {
                "gpt_insight_status": "error",
                "reason": f"gpt_insight_failed:{type(e).__name__}:{e}",
            }

    restaurant_id = (
        context.get("restaurant_id")
        or prediction_data.get("restaurant_id")
        or DEFAULT_RESTAURANT_ID
    )
    location_id = (
        context.get("location_id")
        or prediction_data.get("location_id")
        or DEFAULT_LOCATION_ID
    )

    # Inventory is built from an explicit recipe/BOM structure in context.
    # C8: do NOT call a fixed ratio MRP. Require recipes/BOM or error.
    sales = context.get("sales")
    if not sales or not isinstance(sales, list):
        code = INVENTORY_ERROR_MISSING_SALES
        print(f"[{_utc_ts()}] dashboard_update inventory status=error code={code}")
        return {
            "status": "error",
            "error_code": code,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    # Validate each sales row has a numeric daily_sales_total.
    for row in sales:
        if not isinstance(row, dict) or not isinstance(row.get("daily_sales_total"), (int, float)):
            code = INVENTORY_ERROR_INVALID_SALES
            print(f"[{_utc_ts()}] dashboard_update inventory status=error code={code}")
            return {
                "status": "error",
                "error_code": code,
                "restaurant_id": restaurant_id,
                "location_id": location_id,
                "timestamp": _utc_ts(),
            }

    # C8: require an explicit recipe/BOM structure — do not invent kg quantities.
    recipes = context.get("recipes") or context.get("bom")
    bom_error = _validate_bom(recipes)
    if bom_error is not None:
        print(f"[{_utc_ts()}] dashboard_update inventory status=error code={bom_error}")
        return {
            "status": "error",
            "error_code": bom_error,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    inventory_plan = _build_bom_inventory_plan(recipes)

    # C9: waste must be numeric from inputs or an error. Validate the same
    # sales rows that feed the waste plan — no fake kg on missing inputs.
    waste_error = _validate_waste_input(sales)
    if waste_error is not None:
        print(f"[{_utc_ts()}] dashboard_update waste status=error code={waste_error}")
        return {
            "status": "error",
            "error_code": waste_error,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    waste_plan = _build_waste_plan(sales)
    weekly_waste_target = _weekly_waste_target(sales, DEFAULT_WEEKLY_WASTE_TARGET_PCT)

    forecast = prediction_data.get("forecast", [])
    staffing_plan = optimization_data.get("staffing_plan", [])

    insight_json = gpt_data.get("insight_json", {}) if isinstance(gpt_data, dict) else {}
    risk_level = None
    actions: List[str] = []

    if isinstance(insight_json, dict):
        risk_level = insight_json.get("risk_level")
        raw_actions = insight_json.get("actions", [])
        if isinstance(raw_actions, list):
            actions = [str(x) for x in raw_actions]

    result = {
        "dashboard_update_status": "ok",
        "dashboard_refreshed": True,
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        "forecast": forecast,
        "staffing_plan": staffing_plan,
        "inventory_plan": inventory_plan,
        # C8: inventory is built from an explicit recipe/BOM, not a fixed ratio MRP.
        "inventory_method": "bom_rule",
        "inventory_method_description": (
            "recipe/BOM: total_quantity = quantity_per_meal * expected_meals per ingredient"
        ),
        "waste_plan": waste_plan,
        # C9: waste is numeric from inputs with a weekly target. method=rule.
        "waste_method": "rule",
        "waste_method_description": (
            f"fixed ratio: {WASTE_RATIO} estimated waste per sales unit"
        ),
        "weekly_waste_target": weekly_waste_target,
        "gpt_insight_status": gpt_data.get("gpt_insight_status"),
        "insight_json": insight_json,
        "risk_level": risk_level,
        "actions": actions,
        "timestamp": _utc_ts(),
    }

    print(f"[{_utc_ts()}] DONE dashboard_update")
    return result