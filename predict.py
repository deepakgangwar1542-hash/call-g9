"""STEP 13: Prediction pipeline - validation (FR-1.2/1.3), model flag, Level 0 fallback (FR-2.4)."""
import json
import joblib, pandas as pd
from common import *

class ValidationError(ValueError):
    pass

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
    return {**{f: call[f] for f in FEATURES}, "speed_limit": sl}

_cache = {}
def _load():
    if "m" not in _cache:
        _cache["m"] = joblib.load(MODEL_PATH)
        _cache["meta"] = json.load(open(META_PATH))
    return _cache["m"], _cache["meta"]

def predict_call(call: dict) -> dict:
    """Returns dict: risk_flag, confidence, risk_score, model_version, is_estimate."""
    clean = validate(call)                       # raises ValidationError -> nothing is stored
    try:
        model, meta = _load()
        X = pd.DataFrame([clean])[FEATURES].astype({"speed_limit": str})
        p = float(model.predict_proba(X)[0, 1])
        high = p >= meta["threshold"]
        return dict(risk_flag="High risk" if high else "Low risk",
                    confidence=round(p if high else 1 - p, 3), risk_score=round(p, 4),
                    model_version=meta["model_version"], is_estimate=False)
    except Exception:                            # model missing / broken -> Level 0 baseline
        return dict(risk_flag="Low risk", confidence=None, risk_score=None,
                    model_version="Level0-fallback", is_estimate=True)
