"""HTTP API around the same prediction code the dispatch desk uses.

Run:  uvicorn api:app --port 8000      Docs: http://localhost:8000/docs
Configuration is read from environment variables (see .env.example). With no configuration the service starts in
production mode: API keys are required and the model registry is enforced.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import db
import predict
from common import CHOICES
from triage import audit, auth, registry
from triage.config import ConfigError, Settings, get_settings
from triage.errors import UNAVAILABLE_MESSAGE, Reason, TriageError
from triage.logs import (configure_logging, current_correlation_id, log_event, reset_correlation_id,
                         sanitize_correlation_id, set_correlation_id)

_limiter = auth.RateLimiter()


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = Settings.from_env()                    # fail fast on unsafe configuration
    configure_logging(settings.log_level)
    if settings.auth_mode == "api_key" and not settings.api_keys:
        raise ConfigError("TRIAGE_API_KEYS must contain at least one key when authentication is enabled")
    log_event("startup", env=settings.env, auth_mode=settings.auth_mode, registry_enforce=settings.registry_enforce)
    yield


app = FastAPI(title="Call Severity Triage API", version="3.0", lifespan=lifespan)

try:
    _origins = list(Settings.from_env().cors_origins)
except ConfigError:
    _origins = []
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST", "PATCH"],
                       allow_headers=["X-API-Key", "Authorization", "Content-Type", "X-Correlation-ID"])


@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    correlation_id = sanitize_correlation_id(request.headers.get("x-correlation-id"))
    token = set_correlation_id(correlation_id)
    try:
        response = await call_next(request)
    finally:
        reset_correlation_id(token)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, error: Exception):
    log_event("unhandled_error", logging.ERROR, error_type=type(error).__name__, path=request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error", "correlation_id": current_correlation_id()})


@app.exception_handler(ConfigError)
async def config_error(request: Request, error: ConfigError):
    log_event("config_invalid", logging.ERROR, path=request.url.path)
    return JSONResponse(status_code=503, content=dict(assessment_state="unavailable", reason_code=Reason.CONFIG_INVALID,
                                                       message=UNAVAILABLE_MESSAGE, correlation_id=current_correlation_id()))


def require(permission: str):
    def dependency(request: Request, x_api_key: str | None = Header(default=None),
                   authorization: str | None = Header(default=None)) -> auth.Principal:
        settings = get_settings()
        principal = auth.authenticate(settings, auth.extract_key(x_api_key, authorization))
        if principal is None:
            log_event("auth_rejected", logging.WARNING, path=request.url.path)
            raise HTTPException(status_code=401, detail="Missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})
        if not auth.allowed(principal, permission):
            log_event("auth_forbidden", logging.WARNING, principal=principal.ref, permission=permission)
            raise HTTPException(status_code=403, detail="This key is not permitted to perform that action")
        client = f"{principal.ref}|{request.client.host if request.client else '-'}"
        if not _limiter.allow(client, settings.rate_limit_per_minute):
            raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "60"})
        return principal
    return dependency


class Call(BaseModel):
    """The five required fields, plus any optional detail fields the model understands."""
    road_type: str
    speed_limit: int | str
    weather: str
    light_condition: str
    junction_detail: str
    optional: dict[str, str] = Field(default_factory=dict, description="Optional extra fields, e.g. road_surface_conditions")

    def as_call(self) -> dict:
        return {**self.model_dump(exclude={"optional"}), **self.optional}


class WhatIf(BaseModel):
    call: Call
    changes: dict[str, str | int]


class Decision(BaseModel):
    dispatcher_override: str | None = Field(default=None, pattern="^(High risk|Low risk)$")
    enhanced_response_sent: bool | None = None
    actual_outcome: int | None = Field(default=None, ge=0, le=1)


@app.get("/health")
def health() -> dict:
    """Liveness only: the process is up. Says nothing about whether assessments can be produced."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    """Readiness: can a verified model produce an assessment right now? 503 when not."""
    try:
        settings = get_settings()
        predict._load_v2()
        state, code = "ready", None
        if not settings.registry_enforce:
            state = "ready_unverified"
    except (TriageError, ConfigError) as error:
        state, code = "not_ready", getattr(error, "reason_code", Reason.CONFIG_INVALID)
    return JSONResponse(status_code=200 if state != "not_ready" else 503, content=dict(status=state, reason_code=code))


@app.get("/choices")
def choices() -> dict:
    """Valid values for each required field."""
    return CHOICES


@app.post("/predict")
def predict_endpoint(call: Call, principal: auth.Principal = Depends(require("assess"))) -> JSONResponse:
    try:
        result = predict.predict_call(call.as_call())
    except predict.ValidationError as error:
        raise HTTPException(status_code=422, detail=str(error))
    try:
        result["call_id"] = db.log_call(predict.validate(call.as_call()), result, operator_ref=principal.ref,
                                        source_type="api", correlation_id=result["correlation_id"])
    except Exception as error:               # noqa: BLE001 - fail closed: an assessment we cannot audit is not released
        log_event("audit_write_failed", logging.ERROR, error_type=type(error).__name__)
        return JSONResponse(status_code=503, content=dict(assessment_state="unavailable", risk_flag=predict.UNAVAILABLE_FLAG,
                                                           reason_code=Reason.AUDIT_WRITE_FAILED, message=UNAVAILABLE_MESSAGE,
                                                           correlation_id=result["correlation_id"]))
    status = 503 if result["assessment_state"] == "unavailable" else 200
    return JSONResponse(status_code=status, content=result)


@app.post("/what-if")
def what_if_endpoint(body: WhatIf, principal: auth.Principal = Depends(require("assess"))) -> dict:
    try:
        result = predict.what_if(body.call.as_call(), dict(body.changes))
    except predict.ValidationError as error:
        raise HTTPException(status_code=422, detail=str(error))
    if not result:
        raise HTTPException(status_code=503, detail="The v2 model is not available on this server")
    return result


@app.patch("/calls/{call_id}/decision")
def decision_endpoint(call_id: str, body: Decision, principal: auth.Principal = Depends(require("record_decision"))) -> dict:
    fields = {k: (int(v) if isinstance(v, bool) else v) for k, v in body.model_dump(exclude_none=True).items()}
    if not fields:
        raise HTTPException(status_code=422, detail="Provide at least one field")
    with db.conn() as c:
        if not c.execute("SELECT 1 FROM calls WHERE call_id=?", (call_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Unknown call")
    db.update_call(call_id, actor_ref=principal.ref, **fields)
    return {"call_id": call_id, "recorded": sorted(fields)}


@app.get("/audit/verify")
def audit_verify(principal: auth.Principal = Depends(require("verify_audit"))) -> dict:
    with db.conn() as c:
        return audit.verify_chain(c)


@app.get("/audit/{assessment_id}")
def audit_record(assessment_id: str, principal: auth.Principal = Depends(require("read_audit"))) -> dict:
    with db.conn() as c:
        record = audit.get_record(c, assessment_id)
    if not record:
        raise HTTPException(status_code=404, detail="Unknown assessment")
    return record


@app.get("/models")
def model_inventory(principal: auth.Principal = Depends(require("read_audit"))) -> dict:
    settings = get_settings()
    return {"models": [registry.describe(entry["id"], settings) for entry in registry.load_registry(settings)["models"]],
            "artifact_checks": registry.verify_all(settings)}
