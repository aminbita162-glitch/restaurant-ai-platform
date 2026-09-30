from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
import os


# Stable public error code for AI output schema validation failures.
GPT_INSIGHT_ERROR_SCHEMA_INVALID = "GPT_INSIGHT_SCHEMA_INVALID"

# Required keys in the model's JSON output.
_REQUIRED_INSIGHT_KEYS = frozenset({
    "summary",
    "staffing",
    "inventory",
    "waste",
    "notes",
    "risk_level",
    "actions",
})

# Required keys that must be present and non-empty.
_REQUIRED_NON_EMPTY_KEYS = frozenset({
    "summary",
    "risk_level",
})

# The "actions" key must be a list of short strings.
_REQUIRED_ACTIONS_COUNT = 3


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _truncate(text: str, max_len: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _safe_getenv(key: str, default: str = "") -> str:
    v = os.getenv(key, default)
    return (v or "").strip()


def _extract_prediction_payload(context: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(context.get("5_ml_prediction"), dict):
        return context["5_ml_prediction"]

    if isinstance(context.get("data"), dict):
        inner = context["data"]
        if "forecast" in inner or "model_used" in inner:
            return inner

    if "forecast" in context or "model_used" in context:
        return context

    return {}


def _extract_ingestion_payload(context: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(context.get("1_data_ingestion"), dict):
        return context["1_data_ingestion"]
    return {}


def _extract_context(context: Dict[str, Any]) -> Dict[str, Any]:
    prediction = _extract_prediction_payload(context)
    ingestion = _extract_ingestion_payload(context)

    restaurant_id = str(
        prediction.get("restaurant_id")
        or ingestion.get("restaurant_id")
        or ""
    )
    location_id = str(
        prediction.get("location_id")
        or ingestion.get("location_id")
        or ""
    )

    forecast = prediction.get("forecast") or prediction.get("forecast_intervals") or []
    horizon = prediction.get("horizon", len(forecast) if isinstance(forecast, list) else 0)
    model_used = prediction.get("model_used") or {}

    sales = ingestion.get("sales")
    recent_sales = sales[-7:] if isinstance(sales, list) else []

    return {
        "restaurant_id": restaurant_id,
        "location_id": location_id,
        "forecast": forecast,
        "horizon": horizon,
        "model_used": model_used,
        "recent_sales": recent_sales,
    }


def _build_prompt(payload: Dict[str, Any]) -> str:
    restaurant_id = payload.get("restaurant_id")
    location_id = payload.get("location_id")
    model_used = payload.get("model_used") or {}
    forecast = payload.get("forecast") or []
    horizon = payload.get("horizon")
    recent_sales = payload.get("recent_sales") or []

    lines: List[str] = []
    lines.append("You are an operations copilot for a restaurant.")
    lines.append("Analyze the restaurant forecast and return concise operational recommendations.")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Output MUST be valid JSON only.")
    lines.append("- No markdown, no extra text.")
    lines.append('- Required JSON keys: "summary", "staffing", "inventory", "waste", "notes", "risk_level", "actions".')
    lines.append('- The value of "actions" must be an array of exactly 3 short action strings.')
    lines.append('- All other values must be short strings.')
    lines.append("")
    lines.append("Business objective:")
    lines.append("- Help the manager improve operations, reduce waste, and prepare staffing and inventory correctly.")
    lines.append("")
    lines.append("Context:")
    lines.append(f"- restaurant_id: {restaurant_id}")
    lines.append(f"- location_id: {location_id}")
    lines.append(f"- horizon: {horizon}")
    lines.append(f"- model_used: {model_used}")
    lines.append(f"- recent_sales: {recent_sales}")
    lines.append(f"- forecast: {forecast}")

    return "\n".join(lines)


def _call_openai_json(prompt: str) -> Dict[str, Any]:
    api_key = _safe_getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "status": "skipped",
            "reason": "OPENAI_API_KEY_not_set",
        }

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        return {
            "status": "skipped",
            "reason": f"openai_sdk_not_available:{type(e).__name__}:{e}",
        }

    client = OpenAI(api_key=api_key)

    model = _safe_getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    temperature = float(_safe_getenv("OPENAI_TEMPERATURE", "0.2") or "0.2")

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "You are a helpful operations assistant."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        return {
            "status": "ok",
            "model": model,
            "raw_json": content,
        }
    except Exception as e:
        return {
            "status": "error",
            "reason": f"openai_call_failed:{type(e).__name__}:{e}",
        }


def _parse_json(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, None
        return None, "json_not_object"
    except Exception as e:
        return None, f"json_parse_failed:{type(e).__name__}:{e}"


def _validate_insight_schema(obj: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Validate that a parsed insight dict has the required schema.

    Returns (valid, error_reason). When invalid, error_reason describes
    the first failure encountered.
    """
    if obj is None:
        return False, "json_parse_failed:no_object"

    missing = _REQUIRED_INSIGHT_KEYS - set(obj.keys())
    if missing:
        return False, f"missing_keys:{sorted(missing)}"

    for key in _REQUIRED_NON_EMPTY_KEYS:
        val = obj.get(key)
        if not isinstance(val, str) or not val.strip():
            return False, f"empty_or_non_string:{key}"

    actions = obj.get("actions")
    if not isinstance(actions, list):
        return False, "actions_not_list"
    if len(actions) != _REQUIRED_ACTIONS_COUNT:
        return False, f"actions_wrong_count:{len(actions)}:{_REQUIRED_ACTIONS_COUNT}"
    for i, a in enumerate(actions):
        if not isinstance(a, str) or not a.strip():
            return False, f"action_not_string:{i}"

    return True, None


def _degraded_insight(prompt: str) -> Dict[str, Any]:
    """Build a rule-based degraded insight when the model output is invalid.

    Labeled method=rule — this is a static fallback, not model output.
    """
    return {
        "summary": "Automated insight unavailable; review forecast and staffing manually.",
        "staffing": "Refer to the staffing plan from the optimization step.",
        "inventory": "Refer to the inventory plan from the BOM step.",
        "waste": "Refer to the waste plan from the waste step.",
        "notes": "AI insight was invalid or unavailable; manual review required.",
        "risk_level": "unknown",
        "actions": [
            "Review the forecast and staffing plan manually.",
            "Check inventory against the BOM plan.",
            "Monitor waste against the weekly target.",
        ],
    }


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    payload = _extract_context(context)
    prompt = _build_prompt(payload)
    result = _call_openai_json(prompt)

    restaurant_id = str(payload.get("restaurant_id") or "")
    location_id = str(payload.get("location_id") or "")

    if result.get("status") == "ok":
        raw_json = str(result.get("raw_json", "{}"))
        parsed_json, parse_error = _parse_json(raw_json)

        # C11: validate the model output schema. If invalid, return a
        # rule-based degraded insight labeled method=rule — never pass raw
        # model text as the business action list.
        schema_valid, schema_error = _validate_insight_schema(parsed_json)

        if not schema_valid:
            degraded = _degraded_insight(prompt)
            return {
                "data": {
                    "gpt_insight_status": "degraded",
                    "gpt_insight_error_code": GPT_INSIGHT_ERROR_SCHEMA_INVALID,
                    "gpt_insight_error_reason": schema_error or parse_error or "unknown",
                    "method": "rule",
                    "restaurant_id": restaurant_id,
                    "location_id": location_id,
                    "openai_model": result.get("model"),
                    "insight_json_raw": raw_json,
                    "insight_json": degraded,
                    "timestamp": _utc_ts(),
                },
                "errors": [],
                "warnings": [
                    {
                        "type": "GptInsightSchemaInvalid",
                        "message": schema_error or parse_error or "unknown",
                    },
                ],
                "metrics": {},
            }

        # Valid model output — return it with method=openai.
        return {
            "data": {
                "gpt_insight_status": "ok",
                "method": "openai",
                "restaurant_id": restaurant_id,
                "location_id": location_id,
                "openai_model": result.get("model"),
                "insight_json_raw": raw_json,
                "insight_json": parsed_json,
                "timestamp": _utc_ts(),
            },
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

    if result.get("status") == "skipped":
        return {
            "data": {
                "gpt_insight_status": "skipped",
                "restaurant_id": restaurant_id,
                "location_id": location_id,
                "reason": result.get("reason"),
                "prompt_preview": _truncate(prompt, 600),
                "timestamp": _utc_ts(),
            },
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

    return {
        "data": {
            "gpt_insight_status": "error",
            "restaurant_id": restaurant_id,
            "location_id": location_id,
            "reason": result.get("reason"),
            "prompt_preview": _truncate(prompt, 600),
            "timestamp": _utc_ts(),
        },
        "errors": [],
        "warnings": [],
        "metrics": {},
    }