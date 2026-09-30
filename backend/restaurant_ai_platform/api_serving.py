from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple
import os
import time
import uuid


def _utc_ts() -> str:
    return datetime.utcnow().isoformat()


def _log(message: str) -> None:
    print(f"[{_utc_ts()}] {message}")


def _new_request_id() -> str:
    return uuid.uuid4().hex


def _response_ok(data: Dict[str, Any], status_code: int = 200, *, request_id: Optional[str] = None) -> Any:
    from flask import jsonify  # type: ignore

    payload = {
        "ok": True,
        "request_id": request_id or _new_request_id(),
        "timestamp": _utc_ts(),
        **data,
    }
    return jsonify(payload), status_code


def _response_error(
    message: str,
    status_code: int = 500,
    *,
    code: str = "INTERNAL_ERROR",
    request_id: Optional[str] = None,
) -> Any:
    from flask import jsonify  # type: ignore

    payload: Dict[str, Any] = {
        "ok": False,
        "request_id": request_id or _new_request_id(),
        "error": {
            "code": code,
            "message": message,
        },
        "timestamp": _utc_ts(),
    }
    return jsonify(payload), status_code


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    parts = [p.strip() for p in v.split(",")]
    parts = [p for p in parts if p]
    return parts or None


def _collect_tenant_from_query(args: Any) -> Dict[str, str]:
    """Return tenant keys exactly as supplied. Empty string means absent — never invent a default."""
    return {
        "restaurant_id": (args.get("restaurant_id") or "").strip(),
        "location_id": (args.get("location_id") or "").strip(),
    }


def _collect_tenant_from_json(payload: Dict[str, Any]) -> Dict[str, str]:
    """Return tenant keys exactly as supplied. Empty string means absent — never invent a default."""
    restaurant_id = payload.get("restaurant_id")
    location_id = payload.get("location_id")
    return {
        "restaurant_id": str(restaurant_id).strip() if restaurant_id is not None else "",
        "location_id": str(location_id).strip() if location_id is not None else "",
    }


def _collect_options_from_query(args: Any) -> Dict[str, Any]:
    options: Dict[str, Any] = {}

    dry_run = _parse_bool(args.get("dry_run"))
    if dry_run is not None:
        options["dry_run"] = dry_run

    strict = _parse_bool(args.get("strict"))
    if strict is not None:
        options["strict"] = strict

    steps = _parse_csv(args.get("steps"))
    if steps is not None:
        options["steps"] = steps

    exclude = _parse_csv(args.get("exclude"))
    if exclude is not None:
        options["exclude"] = exclude

    start_at = (args.get("start_at") or "").strip()
    if start_at:
        options["start_at"] = start_at

    stop_after = (args.get("stop_after") or "").strip()
    if stop_after:
        options["stop_after"] = stop_after

    stop_on_error = _parse_bool(args.get("stop_on_error"))
    if stop_on_error is not None:
        options["stop_on_error"] = stop_on_error

    options.update(_collect_tenant_from_query(args))
    return options


def _collect_options_from_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    options: Dict[str, Any] = {}

    if isinstance(payload.get("dry_run"), bool):
        options["dry_run"] = payload["dry_run"]

    if isinstance(payload.get("strict"), bool):
        options["strict"] = payload["strict"]

    steps = payload.get("steps")
    if isinstance(steps, list) and all(isinstance(x, str) for x in steps):
        options["steps"] = steps
    elif isinstance(steps, str):
        parsed = _parse_csv(steps)
        if parsed is not None:
            options["steps"] = parsed

    exclude = payload.get("exclude")
    if isinstance(exclude, list) and all(isinstance(x, str) for x in exclude):
        options["exclude"] = exclude
    elif isinstance(exclude, str):
        parsed = _parse_csv(exclude)
        if parsed is not None:
            options["exclude"] = parsed

    start_at = payload.get("start_at")
    if isinstance(start_at, str) and start_at.strip():
        options["start_at"] = start_at.strip()

    stop_after = payload.get("stop_after")
    if isinstance(stop_after, str) and stop_after.strip():
        options["stop_after"] = stop_after.strip()

    if isinstance(payload.get("stop_on_error"), bool):
        options["stop_on_error"] = payload["stop_on_error"]

    options.update(_collect_tenant_from_json(payload))
    return options


def _run_pipeline(orchestrator: Any, options: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return orchestrator.run_pipeline(options)  # type: ignore[misc]
    except TypeError:
        try:
            return orchestrator.run_pipeline(**options)  # type: ignore[misc]
        except TypeError:
            return orchestrator.run_pipeline()  # type: ignore[misc]


def _check_auth(request_id: str) -> Optional[Any]:
    """Return an error response if the request is not authorised, else None.

    Rules (PHASE 7):
    - If RESTAURANT_AI_API_KEY env var is empty, return 503 AUTH_NOT_CONFIGURED.
    - If X-Api-Key header is missing or does not match, return 401 AUTH_UNAUTHORIZED.
    - The key value is never logged.
    """
    from flask import request as flask_request  # type: ignore

    expected = os.environ.get("RESTAURANT_AI_API_KEY", "").strip()
    if not expected:
        return _response_error(
            "API key authentication is not configured on this server",
            503,
            code="AUTH_NOT_CONFIGURED",
            request_id=request_id,
        )

    provided = (flask_request.headers.get("X-Api-Key") or "").strip()
    if not provided or provided != expected:
        return _response_error(
            "Missing or invalid X-Api-Key header",
            401,
            code="AUTH_UNAUTHORIZED",
            request_id=request_id,
        )

    return None  # authorised


_PERSIST_AVAILABLE = False
try:
    from .core import persistence  # type: ignore

    _PERSIST_AVAILABLE = True
except Exception as e:
    _log(f"persistence_not_available: {type(e).__name__}: {e}")
    _PERSIST_AVAILABLE = False


def _persist_save(payload: Dict[str, Any]) -> None:
    if not _PERSIST_AVAILABLE:
        return
    try:
        persistence.save_run(payload)  # type: ignore[attr-defined]
    except Exception as e:
        _log(f"persistence_save_failed: {type(e).__name__}: {e}")


def _persist_get_last(
    restaurant_id: Optional[str] = None,
    location_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not _PERSIST_AVAILABLE:
        return None
    try:
        # Always call with explicit tenant keys — no fallback to a keyless call.
        return persistence.get_last_run(  # type: ignore[attr-defined]
            restaurant_id=restaurant_id,
            location_id=location_id,
        )
    except Exception as e:
        _log(f"persistence_read_failed: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# C10: Simple in-process rate limiter for POST /pipeline/run per tenant.
# No external infrastructure — a sliding-window counter in a dict.
# ---------------------------------------------------------------------------

_RATE_LIMIT_WINDOW_S = 60  # 1-minute sliding window
_RATE_LIMIT_MAX_REQUESTS = 5  # max 5 pipeline runs per tenant per minute

_rate_lock = __import__("threading").Lock()
_rate_store: Dict[str, List[float]] = {}


def _rate_limit_key(restaurant_id: str, location_id: str) -> str:
    return f"{restaurant_id}:{location_id}"


def _check_rate_limit(restaurant_id: str, location_id: str) -> Tuple[bool, int]:
    """Check and record a request against the per-tenant rate limit.

    Returns (allowed, retry_after_s). When allowed, the current timestamp
    is recorded. When denied, retry_after_s is the seconds until the oldest
    request in the window expires.
    """
    key = _rate_limit_key(restaurant_id, location_id)
    now = time.time()
    cutoff = now - _RATE_LIMIT_WINDOW_S

    with _rate_lock:
        hits = _rate_store.get(key, [])
        # Drop expired entries.
        hits = [t for t in hits if t > cutoff]
        if len(hits) >= _RATE_LIMIT_MAX_REQUESTS:
            retry_after = int(hits[0] + _RATE_LIMIT_WINDOW_S - now) + 1
            _rate_store[key] = hits
            return False, max(retry_after, 1)
        hits.append(now)
        _rate_store[key] = hits
        return True, 0


bp: Optional[Any] = None

try:
    from flask import Blueprint, request  # type: ignore

    bp = Blueprint("api_serving", __name__)

    @bp.get("/health")
    @bp.get("/api/v1/health")
    def health() -> Any:
        rid = _new_request_id()
        return _response_ok(
            {
                "service": "restaurant-ai-platform",
                "status": "ok",
            },
            request_id=rid,
        )

    @bp.get("/pipeline/status")
    @bp.get("/api/v1/pipeline/status")
    def pipeline_status() -> Any:
        rid = _new_request_id()
        return _response_ok(
            {
                "service": "restaurant-ai-platform",
                "pipeline": {
                    "status": "ready",
                    "persistence": {
                        "enabled": bool(_PERSIST_AVAILABLE),
                        "module": "core.persistence",
                    },
                    "supported_options_now": {
                        "dry_run": True,
                        "steps": True,
                        "exclude": True,
                        "start_at": True,
                        "stop_after": True,
                        "stop_on_error": True,
                        "strict": True,
                        "restaurant_id": True,
                        "location_id": True,
                    },
                    "endpoints": {
                        "health": ["/health", "/api/v1/health"],
                        "pipeline_status": ["/pipeline/status", "/api/v1/pipeline/status"],
                        "pipeline_last_run": [
                            "/pipeline/last-run",
                            "/api/v1/pipeline/last-run",
                            "/pipeline/last_run",
                            "/api/v1/pipeline/last_run",
                            "/pipeline/lastrun",
                            "/api/v1/pipeline/lastrun",
                        ],
                        "pipeline_run_post": [
                            "/pipeline/run (POST)",
                            "/api/v1/pipeline/run (POST)",
                        ],
                    },
                    "examples": {
                        "post_run": "POST /api/v1/pipeline/run  body: {\"restaurant_id\":\"...\",\"location_id\":\"...\"}",
                        "last_run": "/api/v1/pipeline/last-run?restaurant_id=...&location_id=...",
                    },
                },
            },
            request_id=rid,
        )

    @bp.get("/pipeline/last-run")
    @bp.get("/api/v1/pipeline/last-run")
    @bp.get("/pipeline/last_run")
    @bp.get("/api/v1/pipeline/last_run")
    @bp.get("/pipeline/lastrun")
    @bp.get("/api/v1/pipeline/lastrun")
    def pipeline_last_run() -> Any:
        rid = _new_request_id()
        auth_err = _check_auth(rid)
        if auth_err is not None:
            return auth_err
        tenant = _collect_tenant_from_query(request.args)

        if not tenant["restaurant_id"] or not tenant["location_id"]:
            return _response_error(
                "restaurant_id and location_id are required",
                400,
                code="MISSING_TENANT_KEYS",
                request_id=rid,
            )

        last_run = _persist_get_last(
            restaurant_id=tenant["restaurant_id"],
            location_id=tenant["location_id"],
        )

        # If persistence returned an error dict (e.g. PERSISTENCE_MISSING_TENANT_KEYS), surface it.
        if isinstance(last_run, dict) and last_run.get("error_code"):
            return _response_error(
                last_run.get("message", "Persistence error"),
                400,
                code=str(last_run["error_code"]),
                request_id=rid,
            )

        if last_run is None:
            return _response_ok(
                {
                    "service": "restaurant-ai-platform",
                    "restaurant_id": tenant["restaurant_id"],
                    "location_id": tenant["location_id"],
                    "has_last_run": False,
                    "last_run": None,
                    "message": "No pipeline run has been stored yet for this tenant. Run the pipeline first.",
                },
                request_id=rid,
            )

        return _response_ok(
            {
                "service": "restaurant-ai-platform",
                "restaurant_id": tenant["restaurant_id"],
                "location_id": tenant["location_id"],
                "has_last_run": True,
                "last_run": last_run,
            },
            request_id=rid,
        )

    @bp.get("/pipeline/run")
    @bp.get("/api/v1/pipeline/run")
    def pipeline_run_browser() -> Any:
        # GET must never trigger pipeline execution (no side-effects on safe methods).
        rid = _new_request_id()
        return _response_ok(
            {
                "service": "restaurant-ai-platform",
                "message": "Use POST /api/v1/pipeline/run to execute the pipeline.",
                "how_to_execute": {
                    "method": "POST",
                    "url": "/api/v1/pipeline/run",
                    "content_type": "application/json",
                    "body_example": {
                        "restaurant_id": "restaurant_001",
                        "location_id": "location_001",
                    },
                },
            },
            request_id=rid,
        )

    @bp.post("/pipeline/run")
    @bp.post("/api/v1/pipeline/run")
    def pipeline_run_post() -> Any:
        from . import orchestrator

        request_id = _new_request_id()

        # Auth must pass before any work is done.
        auth_err = _check_auth(request_id)
        if auth_err is not None:
            return auth_err

        try:
            payload = request.get_json(silent=True) or {}
            if not isinstance(payload, dict):
                payload = {"_raw": payload}

            tenant = _collect_tenant_from_json(payload)
            if not tenant["restaurant_id"] or not tenant["location_id"]:
                return _response_error(
                    "restaurant_id and location_id are required to run the pipeline",
                    400,
                    code="MISSING_TENANT_KEYS",
                    request_id=request_id,
                )

            # C10: per-tenant rate limit on POST /pipeline/run.
            allowed, retry_after = _check_rate_limit(
                tenant["restaurant_id"], tenant["location_id"],
            )
            if not allowed:
                resp = _response_error(
                    f"Rate limit exceeded for tenant {tenant['restaurant_id']}:{tenant['location_id']}. "
                    f"Retry after {retry_after}s.",
                    429,
                    code="RATE_LIMITED",
                    request_id=request_id,
                )
                # Attach Retry-After header if the framework allows it.
                try:
                    resp.headers["Retry-After"] = str(retry_after)  # type: ignore[attr-defined]
                except Exception:
                    pass
                return resp

            options = _collect_options_from_json(payload)
            # Enqueue the pipeline as a background job — do not run inside the request.
            job_id = orchestrator.enqueue_job(options)

            return _response_ok(
                {
                    "job_id": job_id,
                    "status": "queued",
                    "restaurant_id": tenant["restaurant_id"],
                    "location_id": tenant["location_id"],
                    "message": "Pipeline job queued. Use job_id to poll for status.",
                },
                202,
                request_id=request_id,
            )

        except Exception as e:
            _log(f"pipeline_run_post failed: {type(e).__name__}: {e}")
            return _response_error(
                "Failed to enqueue pipeline job",
                500,
                code="PIPELINE_ERROR",
                request_id=request_id,
            )

except Exception as e:
    _log(f"Flask blueprint not available: {type(e).__name__}: {e}")


def register(app: Any) -> bool:
    if bp is None:
        return False

    try:
        app.register_blueprint(bp)
        return True
    except Exception as e:
        _log(f"Blueprint registration failed: {type(e).__name__}: {e}")
        return False


def run() -> Dict[str, Any]:
    _log("api_serving.run() started")

    if _PERSIST_AVAILABLE:
        try:
            persistence.init_db()  # type: ignore[attr-defined]
        except Exception as e:
            _log(f"persistence_init_failed: {type(e).__name__}: {e}")

    result: Dict[str, Any] = {
        "api_serving_status": "ok",
        "blueprint_available": bp is not None,
        "persistence": {
            "enabled": bool(_PERSIST_AVAILABLE),
            "module": "core.persistence",
        },
        "expected_endpoints": [
            "/health",
            "/pipeline/status",
            "/pipeline/last-run",
            "/pipeline/last_run",
            "/pipeline/lastrun",
            "/pipeline/run (POST)",
            "/pipeline/run?execute=1&confirm=yes (GET)",
            "/api/v1/health",
            "/api/v1/pipeline/status",
            "/api/v1/pipeline/last-run",
            "/api/v1/pipeline/last_run",
            "/api/v1/pipeline/lastrun",
            "/api/v1/pipeline/run (POST)",
            "/api/v1/pipeline/run?execute=1&confirm=yes (GET)",
        ],
        "timestamp": _utc_ts(),
    }

    _log("api_serving.run() completed")
    return result