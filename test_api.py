import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)
CALL = {"road_type": "Single carriageway", "speed_limit": 60, "weather": "Fine, no high winds",
        "light_condition": "Daylight", "junction_detail": "Not at junction / roundabout / slip road"}


def test_health_and_choices():
    assert client.get("/health").json() == {"status": "ok"}
    assert "Crossroads" in client.get("/choices").json()["junction_detail"]


def test_predict_returns_a_flag():
    body = client.post("/predict", json=CALL).json()
    assert body["risk_flag"] in {"High risk", "Low risk"} and "model_version" in body


def test_predict_accepts_optional_fields():
    r = client.post("/predict", json={**CALL, "optional": {"road_surface_conditions": "4"}})
    assert r.status_code == 200


def test_invalid_speed_is_rejected_with_a_message():
    r = client.post("/predict", json={**CALL, "speed_limit": 45})
    assert r.status_code == 422 and "speed_limit" in r.json()["detail"]


def test_missing_field_is_rejected():
    bad = {k: v for k, v in CALL.items() if k != "weather"}
    assert client.post("/predict", json=bad).status_code == 422


def test_what_if_lowers_risk_in_daylight():
    dark = {**CALL, "light_condition": "Darkness - no lighting"}
    r = client.post("/what-if", json={"call": dark, "changes": {"light_condition": "Daylight"}})
    if r.status_code == 503:
        pytest.skip("v2 model not built")
    assert r.status_code == 200 and r.json()["delta_points"] < 0


def test_what_if_rejects_invalid_change():
    r = client.post("/what-if", json={"call": CALL, "changes": {"speed_limit": 45}})
    assert r.status_code == 422
