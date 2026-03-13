from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
import os


DEFAULT_RESTAURANT_ID = "restaurant_001"
DEFAULT_LOCATION_ID = "location_001"


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
        or DEFAULT_RESTAURANT_ID
    )
    location_id = str(
        prediction.get("location_id")
        or ingestion.get("location_id")
        or DEFAULT_LOCATION_ID
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


def run(context: Dict[str, Any]) -> Dict[str, Any]:
    payload = _extract_context(context)
    prompt = _build_prompt(payload)
    result = _call_openai_json(prompt)

    restaurant_id = str(payload.get("restaurant_id") or DEFAULT_RESTAURANT_ID)
    location_id = str(payload.get("location_id") or DEFAULT_LOCATION_ID)

    if result.get("status") == "ok":
        raw_json = str(result.get("raw_json", "{}"))
        parsed_json, parse_error = _parse_json(raw_json)

        warnings: List[Dict[str, Any]] = []
        if parse_error:
            warnings.append(
                {
                    "type": "JsonParseWarning",
                    "message": parse_error,
                }
            )

        return {
            "data": {
                "gpt_insight_status": "ok",
                "restaurant_id": restaurant_id,
                "location_id": location_id,
                "openai_model": result.get("model"),
                "insight_json_raw": raw_json,
                "insight_json": parsed_json,
                "timestamp": _utc_ts(),
            },
            "errors": [],
            "warnings": warnings,
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