from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import os


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


def _build_prompt(payload: Dict[str, Any]) -> str:
    model_used = payload.get("model_used") or {}
    forecast = payload.get("forecast") or payload.get("forecast_intervals") or []
    horizon = payload.get("horizon")

    lines: List[str] = []
    lines.append("You are an operations assistant for a restaurant.")
    lines.append("Given the 7-day sales forecast, produce concise operational insights.")
    lines.append("")
    lines.append("Rules:")
    lines.append("- Output MUST be valid JSON only.")
    lines.append("- No markdown, no extra text.")
    lines.append('- JSON keys: "summary", "staffing", "inventory", "waste", "notes".')
    lines.append('- Each value should be a short string (1-3 sentences).')
    lines.append("")
    lines.append("Context:")
    lines.append(f"- horizon: {horizon}")
    if isinstance(model_used, dict) and model_used:
        lines.append(f"- model_used: {model_used}")
    lines.append(f"- forecast: {forecast}")

    return "\n".join(lines)


def _call_openai_json(prompt: str) -> Dict[str, Any]:
    """
    Returns {"status":"skipped"} if OPENAI_API_KEY is not set.
    Uses OpenAI Chat Completions if available. If the OpenAI SDK is not installed, returns skipped.
    """
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
                {"role": "system", "content": "You are a helpful assistant."},
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


def run(ml_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    ml_payload is expected to be the 'data' dict returned from ml_prediction step.
    This step produces GPT-based operational insights (or skips if not configured).
    """
    prompt = _build_prompt(ml_payload)
    result = _call_openai_json(prompt)

    out: Dict[str, Any] = {
        "gpt_insight_status": result.get("status", "error"),
        "timestamp": _utc_ts(),
    }

    if result.get("status") == "ok":
        out["openai_model"] = result.get("model")
        out["insight_json_raw"] = result.get("raw_json", "{}")
    elif result.get("status") == "skipped":
        out["reason"] = result.get("reason")
        out["prompt_preview"] = _truncate(prompt, 600)
    else:
        out["reason"] = result.get("reason")
        out["prompt_preview"] = _truncate(prompt, 600)

    return out