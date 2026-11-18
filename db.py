"""STEP 15: Call logging (SQLite) - NFR-3, FR-1.4, FR-1.5, FR-3.x"""
import sqlite3, uuid, datetime as dt
from contextlib import contextmanager
import pandas as pd
from common import precision_at_k
import common

SCHEMA = """CREATE TABLE IF NOT EXISTS calls(
    call_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
    road_type TEXT NOT NULL, speed_limit INTEGER NOT NULL, weather TEXT NOT NULL,
    light_condition TEXT NOT NULL, junction_detail TEXT NOT NULL,
    predicted_risk TEXT, confidence_score REAL, risk_score REAL, model_version TEXT,
    is_estimate INTEGER DEFAULT 0, dispatcher_override TEXT,
    enhanced_response_sent INTEGER NOT NULL DEFAULT 0, actual_outcome INTEGER)"""

DB_PATH = common.DB_PATH   # tests can override db.DB_PATH


@contextmanager
def conn():
    """Open, commit and ALWAYS close (an open SQLite connection keeps the file locked on Windows)."""
    c = sqlite3.connect(DB_PATH)
    try:
        c.execute(SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def log_call(clean: dict, pred: dict) -> str:
    """Store one accepted, scored call; returns its unique call ID (FR-1.4, NFR-3)."""
    cid = str(uuid.uuid4())
    with conn() as c:
        c.execute("INSERT INTO calls(call_id,timestamp,road_type,speed_limit,weather,light_condition,junction_detail,"
                  "predicted_risk,confidence_score,risk_score,model_version,is_estimate) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                  (cid, dt.datetime.now().isoformat(timespec="seconds"), clean["road_type"], clean["speed_limit"],
                   clean["weather"], clean["light_condition"], clean["junction_detail"], pred["risk_flag"],
                   pred["confidence"], pred["risk_score"], pred["model_version"], int(pred["is_estimate"])))
    return cid


def recent_calls(n=20) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql("SELECT * FROM calls ORDER BY timestamp DESC, rowid DESC LIMIT ?", c, params=(n,))


def all_calls() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql("SELECT * FROM calls ORDER BY timestamp", c)


def update_call(call_id, **fields):
    allowed = {"dispatcher_override", "enhanced_response_sent", "actual_outcome"}
    assert set(fields) <= allowed
    with conn() as c:
        c.execute(f"UPDATE calls SET {', '.join(k+'=?' for k in fields)} WHERE call_id=?", (*fields.values(), call_id))


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
