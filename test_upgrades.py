"""Tests for the shift simulator, analytics, call-text intake, calibration/what-if and PDF report."""
import numpy as np
import pandas as pd
import pytest

import analytics
import intake
import predict
import report
import shift_sim
from common import REPLAY_PATH

needs_replay = pytest.mark.skipif(not REPLAY_PATH.exists(), reason="replay data not built")
CALL = dict(road_type="Single carriageway", speed_limit="60", weather="Fine, no high winds",
            light_condition="Daylight", junction_detail="Not at junction / roundabout / slip road")


@pytest.fixture(scope="module")
def shift():
    return shift_sim.build_shift(pd.read_csv(REPLAY_PATH), 400, 480, seed=3)


@needs_replay
def test_shift_is_deterministic_and_sorted(shift):
    again = shift_sim.build_shift(pd.read_csv(REPLAY_PATH), 400, 480, seed=3)
    assert shift.equals(again)
    assert shift.arrival.is_monotonic_increasing and shift.arrival.between(0, 480).all()


@needs_replay
def test_simulation_never_exceeds_capacity_and_accounts_for_every_call(shift):
    run = shift_sim.simulate(shift, units=5, service_min=30, patience_min=20, policy="model")
    assert run.status.isin(["dispatched", "lapsed", "waiting", "not_arrived"]).all()
    sent = run[run.status == "dispatched"]
    assert (sent.wait_min <= 20 + 1e-9).all()
    # 5 units, 30 min each, 480 min shift: at most 5 * 16 starts (+ the first wave)
    assert len(sent) <= 5 * (480 // 30 + 1)


@needs_replay
def test_risk_ranking_beats_first_come_on_serious_calls(shift):
    result = shift_sim.compare(shift, units=4, service_min=30, patience_min=20).set_index("policy")
    assert result.loc["model", "serious_caught"] >= result.loc["fifo", "serious_caught"]


@needs_replay
def test_more_units_never_reach_fewer_serious_calls(shift):
    few = shift_sim.summarize(shift_sim.simulate(shift, 3, 30, 20, "model"))["serious_caught"]
    many = shift_sim.summarize(shift_sim.simulate(shift, 30, 30, 20, "model"))["serious_caught"]
    assert many >= few


@needs_replay
def test_simulation_rejects_bad_input(shift):
    with pytest.raises(ValueError):
        shift_sim.simulate(shift, 0, 30, 20, "model")
    with pytest.raises(ValueError):
        shift_sim.simulate(shift, 3, 30, 20, "random")


def test_psi_is_zero_for_identical_and_large_for_different():
    a = pd.Series(["x"] * 80 + ["y"] * 20)
    assert analytics.psi(a, a) == pytest.approx(0, abs=1e-9)
    assert analytics.psi(a, pd.Series(["x"] * 20 + ["y"] * 80)) > 0.25


def test_net_benefit_curve_is_sane():
    rng = np.random.default_rng(0)
    scores = rng.random(5000)
    y = (rng.random(5000) < scores * 0.5).astype(int)
    curve = analytics.net_benefit_curve(y, scores, value_serious=10, cost_dispatch=1)
    assert curve.flag_share.between(0, 1).all()
    assert curve.precision.iloc[1] >= curve.precision.iloc[-1]   # tighter net is more precise
    free = analytics.net_benefit_curve(y, scores, value_serious=10, cost_dispatch=0.0)
    assert free.net_benefit.is_monotonic_increasing              # escalating everything is best when it is free


@needs_replay
def test_group_fairness_shares_are_consistent():
    table = analytics.group_fairness(pd.read_csv(REPLAY_PATH), "urban_or_rural_area")
    assert {"group", "n", "base_rate", "flag_share", "precision", "recall"} <= set(table.columns)
    assert table[["base_rate", "flag_share", "recall"]].stack().between(0, 1).all()


def test_redact_removes_personal_details():
    text = "Caller John on 07911 123456 at SW1A 1AA email a.b@test.com"
    out = intake.redact(text)
    for secret in ["07911", "SW1A", "a.b@test.com"]:
        assert secret not in out


def test_intake_extracts_supported_fields_with_evidence():
    found = intake.extract_fields("car off the road on the A38 near a village, 60mph, raining, no street lights, icy, at a crossroads")
    assert found["speed"][0] == "60"
    assert found["light"][0] == "Darkness - no lighting"
    assert found["junction"][0] == "Crossroads"
    assert found["opt_road_surface_conditions"][0] == "4"
    assert found["opt_urban_or_rural_area"][0] == "2"
    assert all(evidence for _, evidence in found.values())


def test_intake_does_not_invent_fields():
    assert intake.extract_fields("something happened, please send help") == {}
    assert "speed" not in intake.extract_fields("going 45mph")      # 45 is not a valid UK limit


def test_calibrated_probability_is_valid_and_bracketed():
    cal = predict.calibrate(0.3)
    if cal is None:
        pytest.skip("calibrator not built")
    assert 0 <= cal["ci_low"] <= cal["calibrated"] <= cal["ci_high"] <= 1 and cal["n_similar"] > 0


def test_what_if_darkness_raises_risk_and_unchanged_is_flat():
    out = predict.what_if({**CALL, "light_condition": "Darkness - no lighting"}, {"light_condition": "Daylight"})
    if not out:
        pytest.skip("v2 model not built")
    assert out["delta_points"] < 0
    assert predict.what_if(CALL, {})["delta_points"] == 0


def test_what_if_rejects_invalid_change():
    with pytest.raises(predict.ValidationError):
        predict.what_if(CALL, {"speed_limit": "45"})


def test_pdf_report_builds_for_empty_and_populated_logs():
    empty = report.build_pdf(pd.DataFrame(), {})
    assert empty[:5] == b"%PDF-"
    calls = pd.DataFrame({"call_id": ["a", "b"], "timestamp": ["2026-01-01T10:00:00", "2026-01-01T10:05:00"],
                          "road_type": ["Single carriageway", "Roundabout"], "speed_limit": [60, 30],
                          "predicted_risk": ["High risk", "Low risk"], "is_estimate": [0, 1],
                          "dispatcher_override": [None, "Dispatched"], "enhanced_response_sent": [1, 0],
                          "actual_outcome": [1.0, None]})
    assert report.build_pdf(calls, {"ok": True, "value": 0.5, "k": 2})[:5] == b"%PDF-"
