"""STEP 13: Prediction pipeline - validation (FR-1.2/1.3), model flag, fallback chain (FR-2.4).

Fallback chain: v2 LightGBM (15 features) -> v1 RandomForest (5 features) -> Level 0 baseline (unscored).
"""
import datetime as dt
import json

import joblib
import pandas as pd

from common import *


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


def _load_v1():
    if "v1" not in _cache:
        _cache["v1"] = (joblib.load(MODEL_PATH), json.load(open(META_PATH)))
    return _cache["v1"]


def _load_v2():
    if "v2" not in _cache:
        _cache["v2"] = (joblib.load(V2_MODEL_PATH), json.load(open(V2_META_PATH)))
    return _cache["v2"]


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


def explain(clean: dict, top_n: int = 4, min_effect: float = 0.02) -> list[dict]:
    """Top reasons behind a v2 score, from LightGBM's built-in exact SHAP values (pred_contrib).

    Each reason carries `effect`, the feature's push on the log-odds of high risk (positive raises risk).
    Fields the dispatcher left blank are never shown: the model treats them as 'unknown', which is not a reason.
    Returns [] when the v2 model is unavailable."""
    try:
        bundle, _ = _load_v2()
        frame = _v2_row(clean, bundle)
        contrib = bundle["model"].predict(frame, pred_contrib=True)[0]
        when = dt.datetime.fromisoformat(clean["call_time"]) if "call_time" in clean else dt.datetime.now()
        shown = {**clean, "hour": when.hour, "day_of_week": when.isoweekday() % 7 + 1, "month": when.month}
        reasons = [dict(field=f, title=FIELD_TITLES[f], value=_describe(f, shown), effect=float(c))
                   for f, c in zip(bundle["features"], contrib[:-1]) if f in shown and abs(c) >= min_effect]
        return sorted(reasons, key=lambda r: -abs(r["effect"]))[:top_n]
    except Exception:
        return []


def _score(p: float, meta: dict, **extra) -> dict:
    high = p >= meta["threshold"]
    return dict(risk_flag="High risk" if high else "Low risk",
                confidence=round(p if high else 1 - p, 3), risk_score=round(p, 4),
                model_version=meta["model_version"], is_estimate=False, **extra)


def predict_call(call: dict) -> dict:
    """Returns dict: risk_flag, confidence, risk_score, model_version, is_estimate, model_tier."""
    clean = validate(call)                       # raises ValidationError -> nothing is stored
    try:
        bundle, meta = _load_v2()
        p = float(bundle["model"].predict_proba(_v2_row(clean, bundle))[0, 1])
        return _score(p, meta, model_tier=bundle["tier"], reasons=explain(clean))
    except Exception:
        pass
    try:                                         # v2 missing / broken -> v1 five-feature model
        model, meta = _load_v1()
        X = pd.DataFrame([clean])[FEATURES].astype({"speed_limit": str})
        return _score(float(model.predict_proba(X)[0, 1]), meta, model_tier="v1_base5")
    except Exception:                            # both missing -> Level 0 baseline
        return dict(risk_flag="Low risk", confidence=None, risk_score=None,
                    model_version="Level0-fallback", is_estimate=True, model_tier="level0")
