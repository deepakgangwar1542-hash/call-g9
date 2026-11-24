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

def _break(model_id, artifact, path="models/does_not_exist.joblib"):
    def mutate(registry):
        for entry in registry["models"]:
            if entry["id"] == model_id:
                entry["artifacts"][artifact]["path"] = path
    return mutate


def test_missing_model_is_unavailable_not_low_risk(registry_variant):   # AC-3, FR-2.4 (replaces the Level-0 'Low risk' fallback)
    registry_variant(_break("v2", "model"))
    r = predict_call(OK)
    assert r["assessment_state"] == "unavailable" and r["risk_flag"] == "Assessment unavailable"
    assert r["risk_score"] is None and r["confidence"] is None and r["reason_code"] == "MODEL_ARTIFACT_MISSING"


def test_v2_is_primary_and_v1_is_only_an_opt_in_fallback(registry_variant, monkeypatch):
    assert predict_call(OK)["model_tier"] == "T4_plus_force"
    registry_variant(_break("v2", "model"))
    assert predict_call(OK)["assessment_state"] == "unavailable"          # v1 is NOT used silently
    monkeypatch.setenv("TRIAGE_ALLOW_V1_FALLBACK", "true")
    r = predict_call(OK)
    assert r["model_tier"] == "v1_base5" and r["assessment_state"] == "degraded" and r["reason_code"] == "V2_UNAVAILABLE_V1_USED"


def test_v2_row_keeps_every_input_as_a_real_category():
    bundle, _ = predict._load_v2()
    rich = predict.validate({**OK, "urban_or_rural_area": "2", "first_road_class": "3", "junction_control": "2",
                             "road_surface_conditions": "4", "police_force": "1", "call_time": "2026-01-10T03:15"})
    row = predict._v2_row(rich, bundle)
    assert not row.isna().any().any(), row.isna().any()[lambda s: s].index.tolist()
    assert row["road_surface_conditions"].iloc[0] == 4 and row["police_force"].iloc[0] == 1
    assert row["month"].iloc[0] == 1 and row["hour"].iloc[0] == 3
    blank = predict._v2_row(predict.validate(OK), bundle)
    assert blank["first_road_class"].iloc[0] == -1          # unknown is the trained '-1' category, not NaN
    assert blank["police_force"].isna().iloc[0]             # force has no unknown category, so it stays missing

def test_optional_fields_validated_and_change_score():
    bad = {**OK, "first_road_class": "99"}
    try: validate(bad); assert False
    except ValidationError as e: assert "first_road_class" in str(e)
    rich = {**OK, "urban_or_rural_area": "2", "first_road_class": "3", "road_surface_conditions": "4", "police_force": "1",
            "call_time": "2026-01-10T03:15"}
    assert 0 <= predict_call(rich)["risk_score"] <= 1
    assert predict_call(rich)["risk_score"] != predict_call(OK)["risk_score"]

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

def test_reasons_sum_to_the_model_probability():
    import math
    clean = validate({**OK, "light_condition": "Darkness - no lighting", "first_road_class": "3"})
    bundle, _ = predict._load_v2()
    contrib = bundle["model"].predict(predict._v2_row(clean, bundle), pred_contrib=True)[0]
    p = float(bundle["model"].predict_proba(predict._v2_row(clean, bundle))[0, 1])
    assert abs(1 / (1 + math.exp(-contrib.sum())) - p) < 1e-6


def test_reasons_skip_blank_fields_and_are_ranked():
    r = predict_call(OK)
    shown = {x["field"] for x in r["reasons"]}
    assert 0 < len(r["reasons"]) <= 4
    assert not shown & {"urban_or_rural_area", "first_road_class", "junction_control", "road_surface_conditions", "police_force"}
    effects = [abs(x["effect"]) for x in r["reasons"]]
    assert effects == sorted(effects, reverse=True)


def test_reasons_include_optional_field_when_given():
    clean = validate({**OK, "first_road_class": "1", "urban_or_rural_area": "2", "police_force": "1"})
    fields = {x["field"] for x in predict.explain(clean, top_n=15, min_effect=0)}
    assert {"first_road_class", "urban_or_rural_area", "police_force"} <= fields


def test_no_reasons_when_v2_missing(registry_variant):
    registry_variant(_break("v2", "model"))
    assert predict.explain(validate(OK)) == []


def test_reasons_card_renders_and_escapes():
    assert theme.reasons_card([]) == ""
    html = theme.reasons_card([dict(title="Road <type>", value="A&B", effect=0.3), dict(title="Lighting", value="Daylight", effect=-0.1)])
    assert "&lt;type&gt;" in html and "A&amp;B" in html and "raises the risk" in html and "lowers the risk" in html


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f): f(); print("PASS", n)
