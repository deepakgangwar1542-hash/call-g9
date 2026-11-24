"""Append-only, hash-chained assessment audit trail.

Each assessment row stores what was asked, which model answered, and what was returned. Later human decisions
and outcomes are NEW rows in `audit_events`; nothing is ever updated. Every row embeds the hash of the previous
row, so an edit, deletion or reordering made outside the application is detected by `verify_chain`.

Database triggers also block UPDATE and DELETE (migration 2). The only sanctioned deletion is `purge_before`,
which removes a time-ordered prefix of the chain and leaves a permanent anchor so verification still works.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import uuid

GENESIS = "0" * 64
_ASSESSMENT_COLUMNS = ("assessment_id", "correlation_id", "call_id", "input_json", "model_version", "model_checksum",
                       "feature_schema_version", "threshold", "raw_score", "calibrated_score", "calibration_status",
                       "assessment_state", "reason_code", "prediction_result", "model_tier", "operator_ref",
                       "source_type", "processing_status", "assessed_at")
_EVENT_COLUMNS = ("event_id", "assessment_id", "event_type", "payload_json", "actor_ref", "correlation_id", "created_at")
_TABLES = {"assessment_audit": (_ASSESSMENT_COLUMNS, "assessed_at"), "audit_events": (_EVENT_COLUMNS, "created_at")}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def chain_hash(previous: str, row: dict) -> str:
    return hashlib.sha256((previous + canonical(row)).encode("utf-8")).hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _begin(c: sqlite3.Connection) -> None:
    if not c.in_transaction:
        c.execute("BEGIN IMMEDIATE")      # serialises writers so two requests cannot extend the same chain head


def _head(c: sqlite3.Connection, table: str) -> str:
    row = c.execute(f"SELECT record_hash FROM {table} ORDER BY seq DESC LIMIT 1").fetchone()
    if row:
        return row[0]
    anchor = c.execute("SELECT anchor_hash FROM audit_purge_log WHERE table_name=? ORDER BY id DESC LIMIT 1", (table,)).fetchone()
    return anchor[0] if anchor else GENESIS


def _insert(c: sqlite3.Connection, table: str, row: dict) -> None:
    _begin(c)
    previous = _head(c, table)
    columns = [*row, "prev_hash", "record_hash"]
    c.execute(f"INSERT INTO {table}({','.join(columns)}) VALUES({','.join('?' * len(columns))})",
              (*row.values(), previous, chain_hash(previous, row)))


def record_assessment(c: sqlite3.Connection, *, clean: dict, pred: dict, call_id: str | None, operator_ref: str,
                      source_type: str, correlation_id: str, processing_status: str = "completed") -> str:
    """Append one assessment record. Stores the exact validated input and the full outcome, including unavailable ones."""
    calibration = pred.get("calibration") or {}
    row = dict(
        assessment_id=str(uuid.uuid4()), correlation_id=correlation_id, call_id=call_id,
        input_json=canonical(clean), model_version=pred.get("model_version"), model_checksum=pred.get("model_checksum"),
        feature_schema_version=pred.get("feature_schema_version"), threshold=pred.get("threshold"),
        raw_score=pred.get("raw_score"), calibrated_score=calibration.get("calibrated"),
        calibration_status=pred.get("calibration_status", "not_applicable"), assessment_state=pred["assessment_state"],
        reason_code=pred.get("reason_code"), prediction_result=pred.get("risk_flag"), model_tier=pred.get("model_tier"),
        operator_ref=operator_ref, source_type=source_type, processing_status=processing_status, assessed_at=_now())
    _insert(c, "assessment_audit", row)
    return row["assessment_id"]


def record_event(c: sqlite3.Connection, *, assessment_id: str, event_type: str, payload: dict, actor_ref: str,
                 correlation_id: str) -> str:
    row = dict(event_id=str(uuid.uuid4()), assessment_id=assessment_id, event_type=event_type,
               payload_json=canonical(payload), actor_ref=actor_ref, correlation_id=correlation_id, created_at=_now())
    _insert(c, "audit_events", row)
    return row["event_id"]


def assessment_for_call(c: sqlite3.Connection, call_id: str) -> str | None:
    row = c.execute("SELECT assessment_id FROM assessment_audit WHERE call_id=? ORDER BY seq LIMIT 1", (call_id,)).fetchone()
    return row[0] if row else None


def record_call_update(c: sqlite3.Connection, call_id: str, fields: dict, *, actor_ref: str, correlation_id: str) -> None:
    """Mirror a dispatcher decision or recorded outcome as an event. Legacy calls with no assessment are skipped."""
    assessment_id = assessment_for_call(c, call_id)
    if not assessment_id:
        return
    decision = {k: v for k, v in fields.items() if k in ("dispatcher_override", "enhanced_response_sent")}
    outcome = {k: v for k, v in fields.items() if k == "actual_outcome"}
    if decision:
        record_event(c, assessment_id=assessment_id, event_type="human_decision", payload=decision, actor_ref=actor_ref,
                     correlation_id=correlation_id)
    if outcome:
        record_event(c, assessment_id=assessment_id, event_type="outcome", payload=outcome, actor_ref=actor_ref,
                     correlation_id=correlation_id)


def get_record(c: sqlite3.Connection, assessment_id: str) -> dict | None:
    c.row_factory = sqlite3.Row
    try:
        row = c.execute("SELECT * FROM assessment_audit WHERE assessment_id=?", (assessment_id,)).fetchone()
        if not row:
            return None
        events = c.execute("SELECT * FROM audit_events WHERE assessment_id=? ORDER BY seq", (assessment_id,)).fetchall()
        return {"assessment": dict(row), "events": [dict(e) for e in events]}
    finally:
        c.row_factory = None


def verify_chain(c: sqlite3.Connection) -> dict:
    """Recompute every hash. Returns {ok, tables: {name: {rows, ok, first_bad_seq}}}."""
    report, overall = {}, True
    for table, (columns, _) in _TABLES.items():
        anchor = c.execute("SELECT purged_through_seq, anchor_hash FROM audit_purge_log WHERE table_name=? ORDER BY id DESC LIMIT 1",
                           (table,)).fetchone()
        previous, expected_seq, bad, count = (anchor[1] if anchor else GENESIS), (anchor[0] + 1 if anchor else None), None, 0
        cursor = c.execute(f"SELECT seq, {','.join(columns)}, prev_hash, record_hash FROM {table} ORDER BY seq")
        for seq, *rest in cursor:
            values, prev_hash, record_hash = rest[:-2], rest[-2], rest[-1]
            count += 1
            intact = (prev_hash == previous and record_hash == chain_hash(previous, dict(zip(columns, values)))
                      and (expected_seq is None or seq == expected_seq))
            if not intact:
                bad = seq
                break
            previous, expected_seq = record_hash, seq + 1
        report[table] = {"rows": count, "ok": bad is None, "first_bad_seq": bad}
        overall = overall and bad is None
    return {"ok": overall, "tables": report}


def purge_before(c: sqlite3.Connection, cutoff_iso: str, *, requested_by: str, correlation_id: str) -> dict:
    """Retention purge: delete the oldest contiguous run of records older than `cutoff_iso` and write an anchor.

    The deletion trigger is lifted only inside this transaction; if anything fails the whole purge rolls back."""
    _begin(c)
    deleted = {}
    try:
        c.execute("UPDATE audit_control SET value='1' WHERE key='purge_enabled'")
        for table, (_, time_column) in _TABLES.items():
            last = c.execute(f"SELECT MIN(seq) FROM {table} WHERE {time_column} >= ?", (cutoff_iso,)).fetchone()[0]
            if last is None:
                last = (c.execute(f"SELECT MAX(seq) FROM {table}").fetchone()[0] or 0) + 1
            through = last - 1
            row = c.execute(f"SELECT record_hash FROM {table} WHERE seq=?", (through,)).fetchone()
            if not row or through < 1:
                deleted[table] = 0
                continue
            count = c.execute(f"DELETE FROM {table} WHERE seq<=?", (through,)).rowcount
            if count:
                c.execute("INSERT INTO audit_purge_log(table_name,purged_through_seq,anchor_hash,rows_deleted,cutoff,requested_by,"
                          "correlation_id,purged_at) VALUES(?,?,?,?,?,?,?,?)",
                          (table, through, row[0], count, cutoff_iso, requested_by, correlation_id, _now()))
            deleted[table] = count
    except BaseException:
        c.rollback()
        raise
    c.execute("UPDATE audit_control SET value='0' WHERE key='purge_enabled'")
    return deleted
