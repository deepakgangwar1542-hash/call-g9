"""Tests for the production-hardening layer: integrity, explicit failure states, audit trail, auth, config."""
import json
import shutil
import sqlite3

import pytest
from fastapi.testclient import TestClient

import common
import db
import predict
from triage import audit, auth, migrations
from triage.config import ConfigError, Settings

OK = dict(road_type="Single carriageway", speed_limit=60, weather="Fine, no high winds",
          light_condition="Daylight", junction_detail="Not at junction / roundabout / slip road")


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    path = tmp_path / "calls.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


def _copy_models(monkeypatch, tmp_path):
    """Copy models + registry into a scratch ROOT so a test can tamper with bytes safely."""
    root = tmp_path / "root"
    shutil.copytree(common.ROOT / "models", root / "models")
    import triage.registry as reg
    monkeypatch.setattr(reg.common, "ROOT", root)
    monkeypatch.setenv("TRIAGE_REGISTRY_PATH", str(root / "models/registry.json"))
    predict._cache.clear()
    return root


# ---- model integrity -------------------------------------------------------------------------------------

def test_tampered_model_is_refused(monkeypatch, tmp_path):
    root = _copy_models(monkeypatch, tmp_path)
    with open(root / "models/model_v2.joblib", "ab") as handle:
        handle.write(b"tamper")
    r = predict.predict_call(OK)
    assert r["assessment_state"] == "unavailable" and r["reason_code"] == "ARTIFACT_CHECKSUM_MISMATCH"
    assert r["risk_score"] is None


def test_unreadable_model_with_valid_checksum_is_unavailable(registry_variant, monkeypatch, tmp_path):
    root = _copy_models(monkeypatch, tmp_path)
    junk = root / "models/model_v2.joblib"
    junk.write_bytes(b"not a joblib file")
    registry = json.loads((root / "models/registry.json").read_text())
    from triage.registry import sha256_file
    for entry in registry["models"]:
        if entry["id"] == "v2":
            entry["artifacts"]["model"]["sha256"] = sha256_file(junk)
    (root / "models/registry.json").write_text(json.dumps(registry))
    predict._cache.clear()
    assert predict.predict_call(OK)["reason_code"] == "MODEL_ARTIFACT_UNREADABLE"


def test_unapproved_status_is_refused(registry_variant):
    def retire(registry):
        for entry in registry["models"]:
            entry["approval_status"] = "retired"
    registry_variant(retire)
    assert predict.predict_call(OK)["reason_code"] == "MODEL_NOT_APPROVED"


def test_registry_unreadable_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIAGE_REGISTRY_PATH", str(tmp_path / "missing.json"))
    assert predict.predict_call(OK)["reason_code"] == "REGISTRY_INVALID"


def test_missing_calibrator_degrades_but_still_scores(registry_variant):
    def drop(registry):
        for entry in registry["models"]:
            if entry["id"] == "v2":
                entry["artifacts"]["calibrator"]["path"] = "models/nope.joblib"
    registry_variant(drop)
    r = predict.predict_call(OK)
    assert r["assessment_state"] == "degraded" and r["risk_score"] is not None
    assert r["calibration"] is None and r["reason_code"] == "CALIBRATION_UNAVAILABLE"


def test_successful_result_carries_audit_metadata():
    r = predict.predict_call(OK)
    assert r["assessment_state"] == "available" and len(r["model_checksum"]) == 64
    assert r["feature_schema_version"].startswith("fs-") and r["threshold"] is not None and r["correlation_id"]
    assert r["model_status"]["integrity"] == "verified"


def test_low_risk_carries_advisory_and_unavailable_does_not_claim_low_risk(registry_variant):
    low = predict.predict_call(OK)
    if low["risk_flag"] == "Low risk":
        assert "not a statement that a call is safe" in low["advisory"]
    registry_variant(lambda reg: [e.update(approval_status="retired") for e in reg["models"]])
    down = predict.predict_call(OK)
    assert down["risk_flag"] != "Low risk" and down["advisory"] is None


def test_validation_error_still_raises_and_stores_nothing(tmp_db):
    with pytest.raises(predict.ValidationError):
        predict.predict_call({**OK, "speed_limit": 45})
    assert not tmp_db.exists()


# ---- audit trail -----------------------------------------------------------------------------------------

def _log(tmp_db, **kw):
    return db.log_call(predict.validate(OK), predict.predict_call(OK), **kw)


def test_assessment_is_audited_with_full_context(tmp_db):
    cid = _log(tmp_db, operator_ref="operator:alice", source_type="api")
    with db.conn() as c:
        record = audit.get_record(c, audit.assessment_for_call(c, cid))["assessment"]
    assert record["operator_ref"] == "operator:alice" and record["source_type"] == "api"
    assert record["model_checksum"] and record["feature_schema_version"] and record["raw_score"] is not None
    assert json.loads(record["input_json"])["speed_limit"] == 60 and record["assessment_state"] == "available"


def test_unavailable_assessments_are_audited_too(tmp_db, registry_variant):
    registry_variant(lambda reg: [e.update(approval_status="retired") for e in reg["models"]])
    cid = _log(tmp_db)
    with db.conn() as c:
        record = audit.get_record(c, audit.assessment_for_call(c, cid))["assessment"]
    assert record["assessment_state"] == "unavailable" and record["reason_code"] == "MODEL_NOT_APPROVED"


def test_human_decisions_append_events_without_editing_history(tmp_db):
    cid = _log(tmp_db)
    db.update_call(cid, actor_ref="operator:alice", dispatcher_override="High risk", enhanced_response_sent=1)
    db.update_call(cid, actor_ref="operator:alice", actual_outcome=1)
    with db.conn() as c:
        record = audit.get_record(c, audit.assessment_for_call(c, cid))
        assert [e["event_type"] for e in record["events"]] == ["human_decision", "outcome"]
        assert audit.verify_chain(c)["ok"]


def test_audit_rows_cannot_be_updated_or_deleted(tmp_db):
    _log(tmp_db)
    with db.conn() as c:
        with pytest.raises(sqlite3.DatabaseError):
            c.execute("UPDATE assessment_audit SET prediction_result='Low risk'")
        with pytest.raises(sqlite3.DatabaseError):
            c.execute("DELETE FROM assessment_audit")


def test_chain_detects_out_of_band_tampering(tmp_db):
    for _ in range(3):
        _log(tmp_db)
    raw = sqlite3.connect(tmp_db)
    raw.execute("DROP TRIGGER IF EXISTS assessment_audit_no_update")           # an attacker with file access
    raw.execute("UPDATE assessment_audit SET prediction_result='TAMPERED' WHERE seq=2")
    raw.commit(); raw.close()
    with db.conn() as c:
        report = audit.verify_chain(c)
    assert not report["ok"] and report["tables"]["assessment_audit"]["first_bad_seq"] == 2


def test_chain_detects_deleted_row(tmp_db):
    for _ in range(3):
        _log(tmp_db)
    raw = sqlite3.connect(tmp_db)
    raw.execute("DROP TRIGGER IF EXISTS assessment_audit_no_delete")
    raw.execute("DELETE FROM assessment_audit WHERE seq=2")
    raw.commit(); raw.close()
    with db.conn() as c:
        assert not audit.verify_chain(c)["ok"]


def test_retention_purge_keeps_chain_verifiable(tmp_db):
    for _ in range(3):
        _log(tmp_db)
    with db.conn() as c:
        deleted = audit.purge_before(c, "9999-01-01", requested_by="admin:ops", correlation_id="test-corr-0001")
    assert deleted["assessment_audit"] == 3
    _log(tmp_db)
    with db.conn() as c:
        assert audit.verify_chain(c)["ok"]
        assert c.execute("SELECT COUNT(*) FROM audit_purge_log").fetchone()[0] >= 1
        with pytest.raises(sqlite3.DatabaseError):                       # trigger is re-armed after the purge
            c.execute("DELETE FROM assessment_audit")


def test_migrations_are_idempotent(tmp_db):
    c = sqlite3.connect(tmp_db)
    migrations.migrate(c); migrations.migrate(c)
    assert c.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] >= 2
    c.close()


# ---- config ----------------------------------------------------------------------------------------------

def test_defaults_are_secure():
    s = Settings.from_env({})
    assert s.env == "production" and s.auth_mode == "api_key" and s.registry_enforce and not s.allow_v1_fallback


@pytest.mark.parametrize("env", [
    {"TRIAGE_AUTH_MODE": "disabled"}, {"TRIAGE_REGISTRY_ENFORCE": "false"}, {"TRIAGE_ALLOW_FIXTURE_PROVIDER": "true"},
    {"TRIAGE_CORS_ORIGINS": "*"}, {"TRIAGE_ENV": "prod"}, {"TRIAGE_API_KEYS": "bad"}, {"TRIAGE_RATE_LIMIT_PER_MINUTE": "x"}])
def test_unsafe_configuration_is_rejected(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_local_env_may_disable_auth():
    assert Settings.from_env({"TRIAGE_ENV": "local", "TRIAGE_AUTH_MODE": "disabled"}).auth_mode == "disabled"


# ---- API -------------------------------------------------------------------------------------------------

KEYS = {"op": "operator-secret-1", "aud": "auditor-secret-1"}


@pytest.fixture
def secured(monkeypatch, tmp_db):
    monkeypatch.setenv("TRIAGE_AUTH_MODE", "api_key")
    monkeypatch.setenv("TRIAGE_API_KEYS", f"alice:operator:{auth.hash_key(KEYS['op'])};bob:auditor:{auth.hash_key(KEYS['aud'])}")
    import api
    return TestClient(api.app)


CALL = {k: v for k, v in OK.items()}


def test_api_requires_a_key(secured):
    assert secured.post("/predict", json=CALL).status_code == 401
    assert secured.post("/predict", json=CALL, headers={"X-API-Key": "wrong"}).status_code == 401
    assert secured.get("/health").status_code == 200 and secured.get("/choices").status_code == 200


def test_api_roles_are_enforced(secured):
    assert secured.post("/predict", json=CALL, headers={"X-API-Key": KEYS["aud"]}).status_code == 403
    assert secured.get("/audit/verify", headers={"X-API-Key": KEYS["op"]}).status_code == 403
    assert secured.get("/audit/verify", headers={"X-API-Key": KEYS["aud"]}).json()["ok"] is True


def test_api_predict_audits_and_returns_correlation_id(secured):
    r = secured.post("/predict", json=CALL, headers={"X-API-Key": KEYS["op"], "X-Correlation-ID": "req-abc-12345"})
    body = r.json()
    assert r.status_code == 200 and r.headers["X-Correlation-ID"] == "req-abc-12345" and body["call_id"]
    with db.conn() as c:
        record = audit.get_record(c, audit.assessment_for_call(c, body["call_id"]))["assessment"]
    assert record["operator_ref"] == "operator:alice" and record["correlation_id"] == "req-abc-12345"


def test_api_decision_endpoint_appends_event(secured):
    cid = secured.post("/predict", json=CALL, headers={"X-API-Key": KEYS["op"]}).json()["call_id"]
    ok = secured.patch(f"/calls/{cid}/decision", json={"dispatcher_override": "High risk", "enhanced_response_sent": True},
                       headers={"X-API-Key": KEYS["op"]})
    assert ok.status_code == 200
    assert secured.patch("/calls/nope/decision", json={"actual_outcome": 1}, headers={"X-API-Key": KEYS["op"]}).status_code == 404
    assert secured.patch(f"/calls/{cid}/decision", json={"dispatcher_override": "Maybe"}, headers={"X-API-Key": KEYS["op"]}).status_code == 422


def test_api_unavailable_model_is_503_with_reason_not_a_risk_level(secured, registry_variant):
    registry_variant(lambda reg: [e.update(approval_status="retired") for e in reg["models"]])
    r = secured.post("/predict", json=CALL, headers={"X-API-Key": KEYS["op"]})
    body = r.json()
    assert r.status_code == 503 and body["assessment_state"] == "unavailable" and body["risk_flag"] == "Assessment unavailable"
    assert body["reason_code"] == "MODEL_NOT_APPROVED" and "standard procedure" in body["message"]
    assert secured.get("/ready").status_code == 503


def test_api_never_leaks_internals_on_unexpected_error(secured, monkeypatch):
    def boom(_call):
        raise RuntimeError("secret path /etc/passwd")
    monkeypatch.setattr(predict, "predict_call", boom)
    r = TestClient(__import__("api").app, raise_server_exceptions=False).post("/predict", json=CALL, headers={"X-API-Key": KEYS["op"]})
    assert r.status_code == 500 and "passwd" not in r.text and "correlation_id" in r.json()


def test_rate_limit(secured, monkeypatch):
    monkeypatch.setenv("TRIAGE_RATE_LIMIT_PER_MINUTE", "2")
    import api
    api._limiter._hits.clear()
    codes = [secured.get("/models", headers={"X-API-Key": KEYS["aud"]}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_logs_do_not_contain_call_inputs(secured, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="triage"):
        secured.post("/predict", json=CALL, headers={"X-API-Key": KEYS["op"]})
    text = " ".join(str(r.__dict__) for r in caplog.records)
    assert "Single carriageway" not in text and KEYS["op"] not in text
