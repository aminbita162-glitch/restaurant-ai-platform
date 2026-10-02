from datetime import datetime
from typing import Any, Dict, List, Optional

# Stable public error codes for inventory and waste failures.
INVENTORY_ERROR_MISSING_SALES = "INVENTORY_MISSING_SALES"
INVENTORY_ERROR_INVALID_SALES = "INVENTORY_INVALID_SALES"
INVENTORY_ERROR_MISSING_BOM = "INVENTORY_MISSING_BOM"
INVENTORY_ERROR_INVALID_BOM = "INVENTORY_INVALID_BOM"
INVENTORY_ERROR_MISSING_ON_HAND = "INVENTORY_MISSING_ON_HAND"
WASTE_ERROR_MISSING_INPUT = "WASTE_MISSING_INPUT"
WASTE_ERROR_INVALID_INPUT = "WASTE_INVALID_INPUT"
WASTE_ERROR_NO_EVENTS = "WASTE_NO_EVENTS_PROVIDED"

# Waste ratio constant — rule, not a trained model.
# Estimated food waste = 8 % of daily sales value.
WASTE_RATIO = 0.08

# Default weekly waste target — a rule constant, not an optimised value.
# Target: no more than 5 % of weekly sales value as waste.
DEFAULT_WEEKLY_WASTE_TARGET_PCT = 0.05



def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _waste_for_sales(daily_sales: float) -> float:
    """Fixed-ratio rule: estimated food waste derived from a numeric daily sales total.

    This is a rule — it is NOT a waste model or ML prediction.
    """
    return round(float(daily_sales) * WASTE_RATIO, 2)


def _validate_waste_input(sales: Any) -> Optional[str]:
    """Validate that sales rows carry a numeric net_sales value.

    Returns None if valid, or a stable error code string if invalid / missing.
    """
    if not isinstance(sales, list) or not sales:
        return WASTE_ERROR_MISSING_INPUT

    for row in sales:
        if not isinstance(row, dict):
            return WASTE_ERROR_INVALID_INPUT
        v = row.get("net_sales")
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
        daily = float(row.get("net_sales", 0))
        plan.append(
            {
                "day_index": i + 1,
                "net_sales": daily,
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
    weekly_sales = sum(float(r.get("net_sales", 0)) for r in recent)
    return {
        "target_pct": target_pct,
        "weekly_sales_basis": round(weekly_sales, 2),
        "weekly_waste_target": round(weekly_sales * target_pct, 2),
    }


def _process_waste_events(
    waste_events: Any,
) -> Dict[str, Any]:
    """Process a waste_events list into a measured waste total.

    Expected shape of each event:
        {"item": str, "qty": float, "reason": str, "ts": str}

    Returns a dict with:
      - method: "measured" when events exist, null when absent
      - waste_total: numeric sum of qty when events exist, null when absent
      - waste_events: the validated event list, or None
      - error_code: WASTE_NO_EVENTS_PROVIDED when absent, None otherwise

    Do not invent kg from a sales ratio — if events are absent the total is null.
    """
    if waste_events is None or not isinstance(waste_events, list):
        return {
            "method": None,
            "waste_total": None,
            "waste_events": None,
            "error_code": WASTE_ERROR_NO_EVENTS,
        }

    if len(waste_events) == 0:
        return {
            "method": None,
            "waste_total": None,
            "waste_events": [],
            "error_code": WASTE_ERROR_NO_EVENTS,
        }

    # Validate and sum qty across events.
    total = 0.0
    validated: List[Dict[str, Any]] = []
    for event in waste_events:
        if not isinstance(event, dict):
            return {
                "method": None,
                "waste_total": None,
                "waste_events": None,
                "error_code": WASTE_ERROR_INVALID_INPUT,
            }
        qty = event.get("qty")
        if not isinstance(qty, (int, float)):
            return {
                "method": None,
                "waste_total": None,
                "waste_events": None,
                "error_code": WASTE_ERROR_INVALID_INPUT,
            }
        total += float(qty)
        validated.append({
            "item": event.get("item"),
            "qty": float(qty),
            "reason": event.get("reason"),
            "ts": event.get("ts"),
        })

    return {
        "method": "measured",
        "waste_total": round(total, 2),
        "waste_events": validated,
        "error_code": None,
    }


def _validate_on_hand(on_hand: Any) -> Optional[str]:
    """Validate the on_hand inventory dict.

    on_hand must be a dict mapping ingredient name (str) to quantity on hand
    (int or float, >= 0). An empty dict is valid — it means nothing is on hand.
    Returns None if valid, or INVENTORY_MISSING_ON_HAND if absent/invalid.
    """
    if on_hand is None:
        return INVENTORY_ERROR_MISSING_ON_HAND
    if not isinstance(on_hand, dict):
        return INVENTORY_ERROR_MISSING_ON_HAND
    for k, v in on_hand.items():
        if not isinstance(k, str):
            return INVENTORY_ERROR_MISSING_ON_HAND
        if not isinstance(v, (int, float)) or v < 0:
            return INVENTORY_ERROR_MISSING_ON_HAND
    return None


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
    on_hand: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Build an inventory purchase list from a recipe/BOM structure and on-hand stock.

    For each ingredient:
      suggested_purchase = max(0, total_quantity - on_hand_qty)

    No kg quantities are invented — every value traces back to explicit BOM
    entries and the caller-supplied on_hand dict.
    method=bom_rule. No MRP.
    """
    plan: List[Dict[str, Any]] = []

    for recipe in recipes:
        meals = float(recipe.get("expected_meals", 0))
        ingredients = recipe.get("ingredients", [])
        for ing in ingredients:
            ing_name = ing.get("name", "")
            qty_per_meal = float(ing.get("quantity_per_meal", 0))
            total_qty = round(qty_per_meal * meals, 2)
            on_hand_qty = float(on_hand.get(ing_name, 0))
            suggested = round(max(0.0, total_qty - on_hand_qty), 2)
            plan.append(
                {
                    "recipe": recipe.get("name"),
                    "ingredient": ing_name,
                    "unit": ing.get("unit"),
                    "quantity_per_meal": qty_per_meal,
                    "expected_meals": round(meals, 2),
                    "total_quantity": total_qty,
                    "on_hand": round(on_hand_qty, 2),
                    "suggested_purchase": suggested,
                }
            )

    return plan


def _build_manager_payload(
    forecast: List[Dict[str, Any]],
    staffing_plan: List[Dict[str, Any]],
    inventory_plan: List[Dict[str, Any]],
    waste_plan: List[Dict[str, Any]],
    weekly_waste_target: Dict[str, Any],
    actions: List[str],
    risk_level: Optional[str],
) -> Dict[str, Any]:
    """Build a compact manager decision payload.

    Includes:
    - forecast summary (compact)
    - staffing blocks (compact)
    - inventory list (compact)
    - waste number (total + target)
    - actions[] as a list
    - approval=proposed
    """
    # Forecast summary: one entry per day with just the essentials.
    forecast_summary: List[Dict[str, Any]] = []
    for day in forecast:
        if not isinstance(day, dict):
            continue
        forecast_summary.append(
            {
                "day_index": day.get("day_index"),
                "date": day.get("date"),
                "weekday": day.get("weekday"),
                "predicted_sales": day.get("predicted_sales"),
            }
        )

    # Staffing blocks: compact — just the block essentials.
    staffing_blocks: List[Dict[str, Any]] = []
    for block in staffing_plan:
        if not isinstance(block, dict):
            continue
        staffing_blocks.append(
            {
                "day_index": block.get("day_index"),
                "start": block.get("start"),
                "end": block.get("end"),
                "role": block.get("role"),
                "n": block.get("n"),
            }
        )

    # Inventory list: compact — ingredient, unit, total_quantity.
    inventory_list: List[Dict[str, Any]] = []
    for item in inventory_plan:
        if not isinstance(item, dict):
            continue
        inventory_list.append(
            {
                "recipe": item.get("recipe"),
                "ingredient": item.get("ingredient"),
                "unit": item.get("unit"),
                "total_quantity": item.get("total_quantity"),
            }
        )

    # Waste number: total estimated waste across all days + target.
    total_waste = sum(
        float(w.get("estimated_waste", 0))
        for w in waste_plan
        if isinstance(w, dict)
    )

    return {
        "forecast_summary": forecast_summary,
        "staffing_blocks": staffing_blocks,
        "inventory_list": inventory_list,
        "waste": {
            "total_estimated_waste": round(total_waste, 2),
            "weekly_waste_target": weekly_waste_target.get("weekly_waste_target"),
            "target_pct": weekly_waste_target.get("target_pct"),
        },
        "actions": actions,
        "risk_level": risk_level,
        "approval": "proposed",
    }


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
        or ""
    )
    location_id = (
        context.get("location_id")
        or prediction_data.get("location_id")
        or ""
    )

    # Inventory is built from an explicit recipe/BOM structure + on_hand stock.
    # Require recipes/BOM and on_hand or error. Do not invent kg.
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

    # Validate each sales row has a numeric net_sales value.
    for row in sales:
        if not isinstance(row, dict) or not isinstance(row.get("net_sales"), (int, float)):
            code = INVENTORY_ERROR_INVALID_SALES
            print(f"[{_utc_ts()}] dashboard_update inventory status=error code={code}")
            return {
                "status": "error",
                "error_code": code,
                "restaurant_id": restaurant_id,
                "location_id": location_id,
                "timestamp": _utc_ts(),
            }

    # Require an explicit recipe/BOM structure — do not invent kg quantities.
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

    # Require on_hand — do not invent stock levels.
    on_hand = context.get("on_hand")
    on_hand_error = _validate_on_hand(on_hand)
    if on_hand_error is not None:
        print(f"[{_utc_ts()}] dashboard_update inventory status=error code={on_hand_error}")
        return {
            "status": "error",
            "error_code": on_hand_error,
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "timestamp": _utc_ts(),
        }

    inventory_plan = _build_bom_inventory_plan(recipes, on_hand)

    # Waste total comes from waste_events only — do not invent kg from a sales ratio.
    # If waste_events is absent, waste_total=null and method=null (not an invented ratio).
    waste_events = context.get("waste_events")
    waste_result = _process_waste_events(waste_events)
    print(
        f"[{_utc_ts()}] dashboard_update waste"
        f" method={waste_result['method']}"
        f" total={waste_result['waste_total']}"
        f" error_code={waste_result['error_code']}"
    )

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
        # Inventory is built from an explicit recipe/BOM, not a fixed ratio MRP.
        "inventory_method": "bom_rule",
        "inventory_method_description": (
            "recipe/BOM: total_quantity = quantity_per_meal * expected_meals per ingredient"
        ),
        # Waste comes from measured events only.
        "waste_method": waste_result["method"],
        "waste_total": waste_result["waste_total"],
        "waste_events": waste_result["waste_events"],
        "waste_error_code": waste_result["error_code"],
        "gpt_insight_status": gpt_data.get("gpt_insight_status"),
        "insight_json": insight_json,
        "risk_level": risk_level,
        "actions": actions,
        "manager_payload": _build_manager_payload(
            forecast=forecast,
            staffing_plan=staffing_plan,
            inventory_plan=inventory_plan,
            waste_plan=[],
            weekly_waste_target={},
            actions=actions,
            risk_level=risk_level,
        ),
        "timestamp": _utc_ts(),
    }

    print(f"[{_utc_ts()}] DONE dashboard_update")
    return result