"""STEP 15: Call logging (SQLite) - NFR-3, FR-1.4, FR-1.5, FR-3.x"""
import json, sqlite3, uuid, datetime as dt
from contextlib import contextmanager
import pandas as pd
from common import precision_at_k
import common

from triage import audit, migrations
from triage.logs import current_correlation_id

SCHEMA = migrations.CALLS_SCHEMA   # kept for backwards compatibility; schema changes now live in triage/migrations.py

DB_PATH = common.DB_PATH   # tests can override db.DB_PATH


@contextmanager
def conn():
    """Open, migrate, commit and ALWAYS close (an open SQLite connection keeps the file locked on Windows)."""
    c = sqlite3.connect(DB_PATH, timeout=15)
    try:
        migrations.migrate(c)
        yield c
        c.commit()
    finally:
        c.close()


def log_call(clean: dict, pred: dict, *, operator_ref: str = "local", source_type: str = "manual",
             correlation_id: str | None = None) -> str:
    """Store one accepted assessment (including an unavailable one) plus its append-only audit record.

    Returns the unique call ID (FR-1.4, NFR-3). Both rows are written in one transaction."""
    cid = str(uuid.uuid4())
    correlation_id = correlation_id or pred.get("correlation_id") or current_correlation_id()
    extras = {k: v for k, v in clean.items() if k in common.OPTIONAL_FIELDS or k == "call_time"}
    with conn() as c:
        c.execute("INSERT INTO calls(call_id,timestamp,road_type,speed_limit,weather,light_condition,junction_detail,"
                  "predicted_risk,confidence_score,risk_score,model_version,is_estimate,model_tier,details_json) "
                  "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (cid, dt.datetime.now().isoformat(timespec="seconds"), clean["road_type"], clean["speed_limit"],
                   clean["weather"], clean["light_condition"], clean["junction_detail"], pred["risk_flag"],
                   pred["confidence"], pred["risk_score"], pred["model_version"], int(pred["is_estimate"]),
                   pred.get("model_tier"), json.dumps(extras) if extras else None))
        audit.record_assessment(c, clean=clean, pred=pred, call_id=cid, operator_ref=operator_ref,
                                source_type=source_type, correlation_id=correlation_id)
    return cid


def recent_calls(n=20) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql("SELECT * FROM calls ORDER BY timestamp DESC, rowid DESC LIMIT ?", c, params=(n,))


def all_calls() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql("SELECT * FROM calls ORDER BY timestamp", c)


def update_call(call_id, *, actor_ref: str = "local", correlation_id: str | None = None, **fields):
    """Record a dispatcher decision or outcome. The `calls` row is a convenience view; the audit trail keeps every change."""
    allowed = {"dispatcher_override", "enhanced_response_sent", "actual_outcome"}
    if not fields or not set(fields) <= allowed:
        raise ValueError(f"update_call accepts only {sorted(allowed)}")
    with conn() as c:
        c.execute(f"UPDATE calls SET {', '.join(k+'=?' for k in fields)} WHERE call_id=?", (*fields.values(), call_id))
        audit.record_call_update(c, call_id, fields, actor_ref=actor_ref,
                                 correlation_id=correlation_id or current_correlation_id())


def shift_precision_at_k(df: pd.DataFrame) -> dict:
    """FR-3.2/3.3: K = number of enhanced responses sent; precision of the top-K riskiest
    calls among those whose actual outcome is known. K=0 returns a message, never a divide-by-zero."""
    k = int(df["enhanced_response_sent"].sum())
    if k == 0:
        return dict(ok=False, message="No enhanced-response dispatches were recorded for this shift.")
    known = df.dropna(subset=["actual_outcome", "risk_score"])
    if known.empty:
        return dict(ok=False, message="No calls with a known outcome yet.")
    return dict(ok=True, k=k, value=precision_at_k(known["actual_outcome"].astype(int).values,
                                                   known["risk_score"].values, k))
