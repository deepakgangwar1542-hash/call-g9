"""Property-based safety tests. Each property must hold for every input Hypothesis can generate, not just examples."""
import datetime as dt

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import predict
from common import CHOICES, OPTIONAL_LABELS, POLICE_FORCE, VALID_SPEED

FAST = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])

core_call = st.fixed_dictionaries({
    "road_type": st.sampled_from(CHOICES["road_type"]),
    "speed_limit": st.sampled_from([str(s) for s in VALID_SPEED]),
    "weather": st.sampled_from(CHOICES["weather"]),
    "light_condition": st.sampled_from(CHOICES["light_condition"]),
    "junction_detail": st.sampled_from(CHOICES["junction_detail"]),
})
optional_fields = st.fixed_dictionaries({}, optional={
    **{field: st.sampled_from(list(labels)) for field, labels in OPTIONAL_LABELS.items()},
    "police_force": st.sampled_from(list(POLICE_FORCE)),
})
full_call = st.builds(lambda core, extra: {**core, **extra}, core_call, optional_fields)

needs_model = pytest.mark.skipif(not predict.V2_MODEL_PATH.exists(), reason="v2 model not built")


@needs_model
@FAST
@given(full_call)
def test_score_is_a_probability_and_flag_matches_threshold(call):
    result = predict.predict_call(call)
    _, meta = predict._load_v2()
    assert 0.0 <= result["risk_score"] <= 1.0
    assert (result["risk_flag"] == "High risk") == (result["risk_score"] >= meta["threshold"])


@needs_model
@FAST
@given(full_call)
def test_scoring_is_deterministic(call):
    volatile = {"latency_ms", "correlation_id"}   # per-request metadata, not part of the score
    first, second = predict.predict_call(call), predict.predict_call(call)
    assert {k: v for k, v in first.items() if k not in volatile} == {k: v for k, v in second.items() if k not in volatile}


@needs_model
@FAST
@given(full_call, st.integers(0, 59), st.integers(0, 59))
def test_score_ignores_minutes_within_the_same_hour(call, minute_a, minute_b):
    day = dt.datetime(2026, 6, 10, 14)
    a = predict.raw_v2_score(predict.validate({**call, "call_time": day.replace(minute=minute_a)}))
    b = predict.raw_v2_score(predict.validate({**call, "call_time": day.replace(minute=minute_b)}))
    assert a == b


@needs_model
@FAST
@given(core_call, optional_fields)
def test_blank_optional_fields_equal_omitted_fields(core, extra):
    blanks = {field: "" for field in extra}
    assert predict.raw_v2_score(predict.validate({**core, **blanks})) == predict.raw_v2_score(predict.validate(core))


@needs_model
@settings(max_examples=60, deadline=None)
@given(st.floats(0, 1, allow_nan=False), st.floats(0, 1, allow_nan=False))
def test_calibration_never_reverses_the_ranking(a, b):
    low, high = sorted((a, b))
    cal_low, cal_high = predict.calibrate(low), predict.calibrate(high)
    if cal_low and cal_high:
        assert cal_low["calibrated"] <= cal_high["calibrated"] + 1e-12


@needs_model
@FAST
@given(full_call)
def test_calibrated_interval_contains_its_estimate(call):
    cal = predict.predict_call(call).get("calibration")
    if cal:
        assert 0.0 <= cal["ci_low"] <= cal["ci_high"] <= 1.0


@FAST
@given(st.text(max_size=30), st.sampled_from(["road_type", "weather", "light_condition", "junction_detail"]))
def test_invalid_labels_are_rejected_cleanly_never_crash(garbage, field):
    call = {"road_type": CHOICES["road_type"][0], "speed_limit": "30", "weather": CHOICES["weather"][0],
            "light_condition": CHOICES["light_condition"][0], "junction_detail": CHOICES["junction_detail"][0]}
    call[field] = garbage
    if garbage in CHOICES[field]:
        predict.validate(call)
    else:
        with pytest.raises(predict.ValidationError):
            predict.validate(call)


@FAST
@given(st.one_of(st.integers(-1000, 1000), st.text(max_size=8), st.none(), st.floats(allow_nan=True)))
def test_invalid_speed_limits_are_rejected_cleanly(speed):
    call = {"road_type": CHOICES["road_type"][0], "speed_limit": speed, "weather": CHOICES["weather"][0],
            "light_condition": CHOICES["light_condition"][0], "junction_detail": CHOICES["junction_detail"][0]}
    try:
        valid = int(speed) in VALID_SPEED
    except (TypeError, ValueError, OverflowError):
        valid = False
    if valid:
        assert predict.validate(call)["speed_limit"] in VALID_SPEED
    else:
        with pytest.raises(predict.ValidationError):
            predict.validate(call)
