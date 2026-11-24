"""STEP 13: Prediction pipeline - validation (FR-1.2/1.3), model flag, explicit assessment states.

A technical failure is never reported as a risk level. Every result carries an `assessment_state`:

  available    verified v2 model scored the call and its supporting calibration/explanation worked
  degraded     a score exists but a supporting part is missing (calibration/explanation), or the optional
               five-feature v1 model was used because v2 was unavailable (TRIAGE_ALLOW_V1_FALLBACK=true)
  unavailable  no trustworthy score: `risk_flag` is "Assessment unavailable"; follow standard procedure

Models are loaded only after their SHA-256 matches the registry (see triage/registry.py).
"""
import datetime as dt
import hashlib
import io
import json
import time

import joblib
import pandas as pd

from common import *
from triage import registry
from triage.config import ConfigError, get_settings
from triage.errors import (LOAD_ERRORS, LOW_RANK_ADVISORY, SCORE_ERRORS, UNAVAILABLE_MESSAGE, IntegrityError,
                           ModelUnavailableError, PredictionError, Reason, TriageError)
from triage.logs import current_correlation_id, log_event

UNAVAILABLE_FLAG = "Assessment unavailable"


class ValidationError(ValueError):
    pass


def _clean_optional(call: dict) -> dict:
    """Validate the optional v2 fields. Anything left blank is simply omitted and handled as 'unknown' at scoring."""
    extra = {}
    for field, labels in OPTIONAL_LABELS.items():
        value = call.get(field)
        if value in (None, "", []):
            continue
        if str(value) not in labels:
            raise ValidationError(f"Invalid {field} value: {value!r}")
        extra[field] = str(value)
    force = call.get("police_force")
    if force not in (None, "", []):
        if str(force) not in POLICE_FORCE:
            raise ValidationError(f"Invalid police_force value: {force!r}")
        extra["police_force"] = str(force)
    when = call.get("call_time")
    if when not in (None, ""):
        if isinstance(when, str):
            try:
                when = dt.datetime.fromisoformat(when)
            except ValueError:
                raise ValidationError(f"Invalid call_time value: {when!r}")
        extra["call_time"] = when.isoformat(timespec="minutes")
    return extra


def validate(call: dict) -> dict:
    missing = [f for f in FEATURES if call.get(f) in (None, "", [])]
    if missing:
        raise ValidationError(f"Missing required field(s): {', '.join(missing)}")
    try:
        sl = int(call["speed_limit"])
    except (TypeError, ValueError):
        raise ValidationError(f"Invalid speed_limit value: {call['speed_limit']!r}")
    if sl not in VALID_SPEED:
        raise ValidationError(f"Invalid speed_limit value: {sl} (allowed: {VALID_SPEED})")
    for f in ("road_type", "weather", "light_condition", "junction_detail"):
        if call[f] not in CHOICES[f]:
            raise ValidationError(f"Invalid {f} value: {call[f]!r}")
    return {**{f: call[f] for f in FEATURES}, "speed_limit": sl, **_clean_optional(call)}


_cache = {}


def _loaded(model_id: str):
    """Verify and deserialise `model_id`'s model + metadata once. Raises TriageError on any failure."""
    if model_id not in _cache:
        model_bytes = registry.load_verified_bytes(model_id, "model")
        meta_bytes = registry.load_verified_bytes(model_id, "meta")
        try:
            bundle = joblib.load(io.BytesIO(model_bytes))
        except LOAD_ERRORS as error:
            raise ModelUnavailableError(f"{model_id} model unreadable: {type(error).__name__}",
                                        Reason.MODEL_ARTIFACT_UNREADABLE) from error
        try:
            meta = json.loads(meta_bytes)
            float(meta["threshold"]), meta["model_version"]
        except LOAD_ERRORS as error:
            raise ModelUnavailableError(f"{model_id} metadata invalid: {type(error).__name__}",
                                        Reason.MODEL_METADATA_INVALID) from error
        features = meta.get("features") or []
        _cache[model_id] = (bundle, meta, dict(checksum=hashlib.sha256(model_bytes).hexdigest(),
                                               feature_schema_version=registry.feature_schema_version(features),
                                               integrity="verified" if get_settings().registry_enforce else "not_enforced"))
    return _cache[model_id]


def _load_v1():
    model, meta, _ = _loaded("v1")
    return model, meta


def _load_v2():
    bundle, meta, _ = _loaded("v2")
    return bundle, meta


def _v2_row(clean: dict, bundle: dict) -> pd.DataFrame:
    """Build one model row. Blank context fields become '-1' (the unknown category the model was trained on);
    a blank police force becomes a missing value, which LightGBM routes by its learned default direction."""
    when = dt.datetime.fromisoformat(clean["call_time"]) if "call_time" in clean else dt.datetime.now()
    row = {**clean, "hour": when.hour, "day_of_week": str(when.isoweekday() % 7 + 1), "month": str(when.month)}
    for field in CONTEXT_FEATURES:
        row.setdefault(field, "-1")
    row.setdefault("police_force", None)
    frame = pd.DataFrame([row])[bundle["features"]]
    for col, cats in bundle["categories"].items():
        values = frame[col]
        if pd.api.types.is_integer_dtype(pd.Index(cats).dtype):
            # Categories were learned as integers; string codes would not match and silently become NaN.
            values = values.map(lambda v: None if v is None or v != v else int(v))
        frame[col] = pd.Categorical(values, categories=cats)
    return frame


FIELD_TITLES = {
    "road_type": "Road type", "speed_limit": "Speed limit", "weather": "Weather", "light_condition": "Lighting",
    "junction_detail": "Junction", "hour": "Time of day", "day_of_week": "Day of week", "month": "Month",
    "urban_or_rural_area": "Area", "first_road_class": "Road class", "junction_control": "Junction control",
    "road_surface_conditions": "Road surface", "police_force": "Police force area",
}
_DAYS = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]


def _describe(field: str, row: dict) -> str:
    value = row[field]
    if field == "speed_limit":
        return f"{value} mph"
    if field == "hour":
        return f"{int(value):02d}:00"
    if field == "day_of_week":
        return _DAYS[int(value)]
    if field == "month":
        return _MONTHS[int(value) - 1]
    if field == "police_force":
        return POLICE_FORCE[str(value)]
    if field in OPTIONAL_LABELS:
        return OPTIONAL_LABELS[field][str(value)]
    return str(value)


def _explain(clean: dict, top_n: int, min_effect: float) -> list[dict]:
    bundle, _ = _load_v2()
    frame = _v2_row(clean, bundle)
    contrib = bundle["model"].predict(frame, pred_contrib=True)[0]
    when = dt.datetime.fromisoformat(clean["call_time"]) if "call_time" in clean else dt.datetime.now()
    shown = {**clean, "hour": when.hour, "day_of_week": when.isoweekday() % 7 + 1, "month": when.month}
    reasons = [dict(field=f, title=FIELD_TITLES[f], value=_describe(f, shown), effect=float(c))
               for f, c in zip(bundle["features"], contrib[:-1]) if f in shown and abs(c) >= min_effect]
    return sorted(reasons, key=lambda r: -abs(r["effect"]))[:top_n]


def explain(clean: dict, top_n: int = 4, min_effect: float = 0.02) -> list[dict]:
    """Top reasons behind a v2 score, from LightGBM's built-in exact SHAP values (pred_contrib).

    Each reason carries `effect`, the feature's push on the log-odds of high risk (positive raises risk).
    Fields the dispatcher left blank are never shown: the model treats them as 'unknown', which is not a reason.
    Returns [] when the v2 model or its explanation is unavailable; predict_call reports that as `degraded`."""
    try:
        return _explain(clean, top_n, min_effect)
    except (TriageError, *SCORE_ERRORS):
        return []


def _load_cal():
    if "cal" not in _cache:
        data = registry.load_verified_bytes("v2", "calibrator")
        try:
            _cache["cal"] = joblib.load(io.BytesIO(data))
        except LOAD_ERRORS as error:
            raise ModelUnavailableError(f"calibrator unreadable: {type(error).__name__}",
                                        Reason.MODEL_ARTIFACT_UNREADABLE) from error
    return _cache["cal"]


def _calibrate_checked(raw: float) -> tuple[dict | None, str, str | None]:
    """(calibration, status, reason_code). Status is ok | unavailable | failed."""
    try:
        cal = _load_cal()
    except TriageError as error:
        return None, "unavailable", error.reason_code
    try:
        p = float(cal["isotonic"].predict([raw])[0])
        table = cal["intervals"]
        row = table[(table.score_low <= p) & (p <= table.score_high)].iloc[-1]
        return (dict(calibrated=round(p, 4), ci_low=round(float(row.ci_low), 4), ci_high=round(float(row.ci_high), 4),
                     n_similar=int(row.n)), "ok", None)
    except SCORE_ERRORS:
        return None, "failed", Reason.CALIBRATION_FAILED


def calibrate(raw: float) -> dict | None:
    """Map a raw v2 score to a calibrated probability plus the observed range among similar historic calls.

    `ci_low`/`ci_high` are the Wilson 95% interval of the serious/fatal rate actually seen in the same score bin
    of out-of-time 2022-2025 predictions. Returns None when no calibrator is available."""
    return _calibrate_checked(raw)[0]


def raw_v2_score(clean: dict) -> float:
    bundle, _ = _load_v2()
    return float(bundle["model"].predict_proba(_v2_row(clean, bundle))[0, 1])


def what_if(call: dict, changes: dict) -> dict:
    """Re-score a call with some fields changed; returns before/after calibrated risk and the shift in points.

    Raises ValidationError for invalid input. Needs the v2 model; returns {} if it is unavailable."""
    base, changed = validate(call), validate({**call, **changes})
    try:
        before, after = raw_v2_score(base), raw_v2_score(changed)
    except (TriageError, *SCORE_ERRORS):
        return {}
    cb, ca = calibrate(before), calibrate(after)
    pb = cb["calibrated"] if cb else before
    pa = ca["calibrated"] if ca else after
    return dict(before=pb, after=pa, delta_points=round((pa - pb) * 100, 1), raw_before=before, raw_after=after)


def _build(clean_or_none, *, state, reason, correlation_id, started, flag, p=None, meta=None, info=None, tier,
           version, reasons=None, calibration=None, calibration_status="not_applicable", message=None, estimate=False):
    high = p is not None and meta is not None and p >= meta["threshold"]
    scored = p is not None
    return dict(
        risk_flag=flag, confidence=(round(p if high else 1 - p, 3) if scored else None),
        risk_score=(round(p, 4) if scored else None), model_version=version, is_estimate=estimate, model_tier=tier,
        assessment_state=state, reason_code=reason, correlation_id=correlation_id,
        message=message or (UNAVAILABLE_MESSAGE if state == "unavailable" else None),
        advisory=LOW_RANK_ADVISORY if flag == "Low risk" else None,
        reasons=reasons or [], calibration=calibration, calibration_status=calibration_status,
        raw_score=p, threshold=(meta or {}).get("threshold"),
        model_checksum=(info or {}).get("checksum"), feature_schema_version=(info or {}).get("feature_schema_version"),
        model_status=dict(state=state, version=version, tier=tier, integrity=(info or {}).get("integrity", "failed" if state == "unavailable" else "unknown")),
        latency_ms=round((time.perf_counter() - started) * 1000, 1))


def _score_v2(clean: dict, correlation_id: str, started: float) -> dict:
    bundle, meta, info = _loaded("v2")
    try:
        p = float(bundle["model"].predict_proba(_v2_row(clean, bundle))[0, 1])
    except SCORE_ERRORS as error:
        raise PredictionError(f"v2 scoring failed: {type(error).__name__}") from error
    calibration, cal_status, cal_reason = _calibrate_checked(p)
    try:
        reasons, explain_reason = _explain(clean, 4, 0.02), None
    except SCORE_ERRORS:
        reasons, explain_reason = [], Reason.EXPLANATION_UNAVAILABLE
    reason = cal_reason or explain_reason
    if cal_reason == Reason.MODEL_ARTIFACT_MISSING:
        reason = Reason.CALIBRATION_UNAVAILABLE
    return _build(clean, state="degraded" if reason else "available", reason=reason, correlation_id=correlation_id,
                  started=started, flag="High risk" if p >= meta["threshold"] else "Low risk", p=p, meta=meta, info=info,
                  tier=bundle["tier"], version=meta["model_version"], reasons=reasons, calibration=calibration,
                  calibration_status=cal_status,
                  message="Score available but a supporting component is missing; treat the explanation or calibrated range as absent." if reason else None)


def _score_v1(clean: dict, correlation_id: str, started: float) -> dict:
    model, meta, info = _loaded("v1")
    try:
        X = pd.DataFrame([clean])[FEATURES].astype({"speed_limit": str})
        p = float(model.predict_proba(X)[0, 1])
    except SCORE_ERRORS as error:
        raise PredictionError(f"v1 scoring failed: {type(error).__name__}") from error
    return _build(clean, state="degraded", reason=Reason.V2_UNAVAILABLE_V1_USED, correlation_id=correlation_id,
                  started=started, flag="High risk" if p >= meta["threshold"] else "Low risk", p=p, meta=meta, info=info,
                  tier="v1_base5", version=meta["model_version"],
                  message="Degraded: the five-feature fallback model was used because the primary model is unavailable.")


def _unavailable(reason: str, correlation_id: str, started: float) -> dict:
    return _build(None, state="unavailable", reason=reason, correlation_id=correlation_id, started=started,
                  flag=UNAVAILABLE_FLAG, tier="none", version="unavailable", estimate=True)


def predict_call(call: dict, correlation_id: str | None = None) -> dict:
    """Validate and score one call. Raises ValidationError for bad input; never raises for model failures.

    A model failure returns assessment_state="unavailable" with a reason_code instead of a risk level."""
    clean = validate(call)                       # raises ValidationError -> nothing is stored
    correlation_id = correlation_id or current_correlation_id()
    started = time.perf_counter()
    try:
        allow_v1 = get_settings().allow_v1_fallback
    except ConfigError:
        log_event("assessment_unavailable", reason_code=Reason.CONFIG_INVALID, correlation_id=correlation_id)
        return _unavailable(Reason.CONFIG_INVALID, correlation_id, started)
    try:
        return _score_v2(clean, correlation_id, started)
    except TriageError as primary:
        failure = primary
    log_event("v2_unavailable", reason_code=failure.reason_code, correlation_id=correlation_id)
    if allow_v1:
        try:
            return _score_v1(clean, correlation_id, started)
        except TriageError as secondary:
            log_event("v1_unavailable", reason_code=secondary.reason_code, correlation_id=correlation_id)
    return _unavailable(failure.reason_code, correlation_id, started)
