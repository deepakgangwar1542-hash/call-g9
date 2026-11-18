"""Tests for the multi-year pipeline  ->  python -m pytest -q test_multiyear.py"""
import numpy as np
import pandas as pd
import pytest

import common
from step2b_clean_multi import clean
from step4_train_multiyear import level1_scores, p_at_rate, to_frame


def raw_row(**overrides):
    row = dict(collision_year=2024, collision_severity=2, road_type=6, speed_limit=30, weather_conditions=1,
               light_conditions=1, junction_detail=0, date="15/03/2024", time="08:30", day_of_week=6,
               urban_or_rural_area=1, first_road_class=3, junction_control=9, pedestrian_crossing=99,
               road_surface_conditions=1, carriageway_hazards=-1, police_force=1, latitude=51.5074, longitude=-0.1278)
    row.update(overrides)
    return row


def test_unknown_codes_are_merged():
    cleaned = clean(pd.DataFrame([raw_row(junction_control=9), raw_row(junction_control=-1)]))
    assert set(cleaned.junction_control) == {"-1"}
    assert set(cleaned.pedestrian_crossing) == {"-1"}


def test_target_is_serious_or_fatal():
    cleaned = clean(pd.DataFrame([raw_row(collision_severity=1), raw_row(collision_severity=2),
                                  raw_row(collision_severity=3)]))
    assert cleaned.target.tolist() == [1, 1, 0]


def test_sep17_2023_and_invalid_speed_dropped():
    raw = pd.DataFrame([raw_row(collision_year=2023, date="01/03/2023"), raw_row(collision_year=2023, date="20/10/2023"),
                        raw_row(speed_limit=45), raw_row()])
    cleaned = clean(raw)
    assert len(cleaned) == 2


def test_drifting_feature_is_not_used():
    assert "special_conditions_at_site" not in common.TIERS["T4_plus_force"]


def test_tiers_are_nested():
    names = list(common.TIERS)
    for smaller, larger in zip(names, names[1:]):
        assert set(common.TIERS[smaller]) < set(common.TIERS[larger])


def test_level1_uses_train_rates_only():
    train = pd.DataFrame({"speed_limit": [30, 30, 60, 60], "target": [0, 0, 1, 1]})
    test = pd.DataFrame({"speed_limit": [30, 60, 40]})
    assert level1_scores(train, test).tolist() == [0.0, 1.0, 0.5]


def test_precision_at_rate_is_exact_on_a_perfect_ranking():
    y = np.array([1] * 20 + [0] * 80)
    assert p_at_rate(y, y.astype(float), 0.2) == 1.0


def test_unseen_category_becomes_nan_not_error():
    frame = to_frame(pd.DataFrame({"police_force": ["999"]}), ["police_force"], {"police_force": ["1", "2"]})
    assert frame["police_force"].isna().all()


@pytest.mark.skipif(not common.V2_MODEL_PATH.exists(), reason="run step4_train_multiyear.py first")
def test_saved_bundle_scores_a_call():
    import joblib
    bundle = joblib.load(common.V2_MODEL_PATH)
    call = {f: bundle["categories"][f][0] if f in bundle["categories"] else 30 for f in bundle["features"]}
    frame = pd.DataFrame([call])
    for col, cats in bundle["categories"].items():
        frame[col] = pd.Categorical(frame[col], categories=cats)
    probability = bundle["model"].predict_proba(frame[bundle["features"]])[0, 1]
    assert 0 <= probability <= 1
