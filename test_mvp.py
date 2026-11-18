"""STEP 16: Final MVP tests  ->  python -m pytest -q   (or: python test_mvp.py)"""
import time, os
import common, db
from predict import predict_call, validate, ValidationError
import predict
import theme

OK = dict(road_type="Single carriageway", speed_limit=60, weather="Fine, no high winds",
          light_condition="Daylight", junction_detail="Not at junction / roundabout / slip road")

def test_valid_call_gives_flag_fast():            # AC-1, NFR-1
    predict_call(OK)                              # warm-up (model load)
    t = time.perf_counter(); r = predict_call(OK)
    assert r["risk_flag"] in ("High risk", "Low risk") and 0 <= r["confidence"] <= 1
    assert time.perf_counter() - t < 2

def test_missing_weather_rejected():              # AC-2, FR-1.2
    bad = {**OK, "weather": ""}
    try: validate(bad); assert False
    except ValidationError as e: assert "weather" in str(e)

def test_invalid_speed_rejected():                # FR-1.3
    try: validate({**OK, "speed_limit": 45}); assert False
    except ValidationError as e: assert "45" in str(e)

def test_fallback_when_model_missing():           # AC-3, FR-2.4
    predict._cache.clear(); real = common.MODEL_PATH
    predict.MODEL_PATH = real.with_name("nope.joblib")
    r = predict_call(OK); predict.MODEL_PATH = real; predict._cache.clear()
    assert r["is_estimate"] and r["risk_flag"] == "Low risk"

def test_zero_dispatch_guard():                   # AC-4, FR-3.3
    import pandas as pd
    df = pd.DataFrame({"enhanced_response_sent": [0, 0], "actual_outcome": [1, 0], "risk_score": [.9, .1]})
    assert db.shift_precision_at_k(df)["ok"] is False

def test_precision_at_k_value():                  # FR-3.2
    import pandas as pd
    df = pd.DataFrame({"enhanced_response_sent": [1, 1, 0, 0], "actual_outcome": [1, 0, 0, 1], "risk_score": [.9, .8, .2, .1]})
    assert db.shift_precision_at_k(df)["value"] == 0.5

def test_logging():                               # NFR-3
    orig = common.DB_PATH, db.DB_PATH
    common.DB_PATH = db.DB_PATH = common.ROOT / "test_calls.db"
    try:
        if db.DB_PATH.exists(): os.remove(db.DB_PATH)
        cid = db.log_call(validate(OK), predict_call(OK))
        row = db.recent_calls(1).iloc[0]
        assert row.call_id == cid and row.model_version and row.confidence_score is not None
        assert row.road_type == OK["road_type"] and row.speed_limit == OK["speed_limit"]
    finally:
        common.DB_PATH, db.DB_PATH = orig
        if (common.ROOT / "test_calls.db").exists(): os.remove(common.ROOT / "test_calls.db")

def test_fallback_verdict_is_unscored():
    pred = dict(risk_flag="Low risk", confidence=None, risk_score=None,
                model_version="Level0-fallback", is_estimate=True)
    card = theme.verdict_card(pred, 4.2, "0123456789")
    assert "UNSCORED" in card and "baseline estimate" in card.casefold()
    assert "LOW RISK" not in card


def test_tie_averaged_precision_is_deterministic_and_bounded():
    from step3_train_eval import average_precision_at_k
    y, scores = [1, 0, 1, 0, 0], [.5, .5, .5, .2, .2]
    first = average_precision_at_k(y, scores, 3)
    assert first == average_precision_at_k(y, scores, 3)
    assert 0 <= first[0] <= 1 and first[1] >= 0


def test_group_baselines_are_fold_local_and_fallback_for_unseen_group():
    import pandas as pd
    from step3_train_eval import fit_level1, fit_level15, level1_scores, level15_scores
    train = pd.DataFrame({"speed_limit": [30, 30], "road_type": ["A", "A"], "light_condition": ["Daylight", "Daylight"]})
    test = pd.DataFrame({"speed_limit": [70], "road_type": ["Never seen"], "light_condition": ["Darkness"]})
    l1, l15 = fit_level1(train, [1, 0]), fit_level15(train, [1, 0])
    assert 0 <= level1_scores(test, l1)[0] <= 1
    assert 0 <= level15_scores(test, l15)[0] <= 1


def test_final_test_indices_do_not_overlap_development():
    import pandas as pd
    from step3_train_eval import final_split_indices
    dev, final = final_split_indices(pd.Series([0, 1] * 20))
    assert not set(dev).intersection(final)


def test_saved_phase1_artifact_contract():
    import json
    with open(common.META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    required = {"model_version", "model", "features", "K", "threshold", "metrics", "level1_precision_at_k", "base_rate", "flag_rate", "data_variant", "eval_protocol"}
    assert required.issubset(meta)
    result = predict_call(OK)
    assert result["is_estimate"] is False and result["risk_score"] is not None


def test_paired_bootstrap_identical_scores_includes_zero():
    from step3_train_eval import bootstrap_results
    y = [0, 1] * 30
    scores = [.2, .8] * 30
    _, paired, _ = bootstrap_results(y, {"Level0_always_Slight": [0] * 60, "Level1_speed_limit_rate": scores, "Level1_5_smoothed_interaction_rate": scores, "model": scores}, "model", k=10, n=50)
    assert abs(paired.iloc[0].mean_difference) < 1e-12
    assert paired.iloc[0].ci_low <= 0 <= paired.iloc[0].ci_high

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f): f(); print("PASS", n)
