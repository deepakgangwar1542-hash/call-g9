"""Versioned, idempotent SQLite migrations.

Every migration is additive (new tables, new nullable columns) so a rollback is "ignore the new objects".
Applied versions are recorded in `schema_migrations` with a checksum; editing an already-applied migration is
detected and refused. Never edit an applied migration: append a new one.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import sqlite3
from dataclasses import dataclass

CALLS_SCHEMA = """CREATE TABLE IF NOT EXISTS calls(
    call_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
    road_type TEXT NOT NULL, speed_limit INTEGER NOT NULL, weather TEXT NOT NULL,
    light_condition TEXT NOT NULL, junction_detail TEXT NOT NULL,
    predicted_risk TEXT, confidence_score REAL, risk_score REAL, model_version TEXT,
    is_estimate INTEGER DEFAULT 0, dispatcher_override TEXT,
    enhanced_response_sent INTEGER NOT NULL DEFAULT 0, actual_outcome INTEGER)"""

_AUDIT_TABLES = [
    """CREATE TABLE IF NOT EXISTS assessment_audit(
        seq INTEGER PRIMARY KEY AUTOINCREMENT, assessment_id TEXT NOT NULL UNIQUE, correlation_id TEXT NOT NULL,
        call_id TEXT, input_json TEXT NOT NULL, model_version TEXT, model_checksum TEXT, feature_schema_version TEXT,
        threshold REAL, raw_score REAL, calibrated_score REAL, calibration_status TEXT NOT NULL,
        assessment_state TEXT NOT NULL CHECK(assessment_state IN ('available','degraded','unavailable')),
        reason_code TEXT, prediction_result TEXT, model_tier TEXT, operator_ref TEXT NOT NULL,
        source_type TEXT NOT NULL, processing_status TEXT NOT NULL, assessed_at TEXT NOT NULL,
        prev_hash TEXT NOT NULL, record_hash TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS audit_events(
        seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, assessment_id TEXT NOT NULL,
        event_type TEXT NOT NULL CHECK(event_type IN ('human_decision','outcome','processing_status')),
        payload_json TEXT NOT NULL, actor_ref TEXT NOT NULL, correlation_id TEXT NOT NULL,
        created_at TEXT NOT NULL, prev_hash TEXT NOT NULL, record_hash TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_assessment ON audit_events(assessment_id)",
    "CREATE INDEX IF NOT EXISTS idx_assessment_audit_created ON assessment_audit(assessed_at)",
    "CREATE TABLE IF NOT EXISTS audit_control(key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "INSERT OR IGNORE INTO audit_control(key, value) VALUES('purge_enabled', '0')",
    """CREATE TABLE IF NOT EXISTS audit_purge_log(
        id INTEGER PRIMARY KEY AUTOINCREMENT, table_name TEXT NOT NULL, purged_through_seq INTEGER NOT NULL,
        anchor_hash TEXT NOT NULL, rows_deleted INTEGER NOT NULL, cutoff TEXT NOT NULL, requested_by TEXT NOT NULL,
        correlation_id TEXT NOT NULL, purged_at TEXT NOT NULL)""",
]


def _append_only_triggers(table: str) -> list[str]:
    """UPDATE is always blocked. DELETE is blocked unless the retention purge has switched the flag on."""
    return [
        f"""CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table}
            BEGIN SELECT RAISE(ABORT, 'append-only table: updates are not allowed'); END""",
        f"""CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table}
            WHEN (SELECT value FROM audit_control WHERE key = 'purge_enabled') != '1'
            BEGIN SELECT RAISE(ABORT, 'append-only table: deletes are not allowed'); END""",
    ]


_AUDIT_TRIGGERS = (_append_only_triggers("assessment_audit") + _append_only_triggers("audit_events")
                   + _append_only_triggers("audit_purge_log"))
# The purge log may never be deleted from, not even during a purge.
_AUDIT_TRIGGERS[-1] = """CREATE TRIGGER IF NOT EXISTS audit_purge_log_no_delete BEFORE DELETE ON audit_purge_log
    BEGIN SELECT RAISE(ABORT, 'append-only table: deletes are not allowed'); END"""

_RECORDING_TABLES = [
    """CREATE TABLE IF NOT EXISTS recording_jobs(
        job_id TEXT PRIMARY KEY, state TEXT NOT NULL, source_kind TEXT NOT NULL, stream_ref TEXT,
        audio_sha256 TEXT, audio_bytes INTEGER, audio_format TEXT, duration_seconds REAL, temp_path TEXT,
        operator_ref TEXT NOT NULL, correlation_id TEXT NOT NULL, reason_code TEXT, attempts INTEGER NOT NULL DEFAULT 0,
        lease_expires_at TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT,
        processing_ms INTEGER, provider_name TEXT, provider_version TEXT, extraction_version TEXT,
        redaction_version TEXT, redaction_status TEXT, scan_status TEXT, retention_until TEXT, assessment_id TEXT,
        deleted_at TEXT)""",
    "CREATE INDEX IF NOT EXISTS idx_recording_jobs_state ON recording_jobs(state)",
    """CREATE TABLE IF NOT EXISTS recording_segments(
        job_id TEXT NOT NULL, idx INTEGER NOT NULL, start_s REAL, end_s REAL, text_redacted TEXT,
        confidence REAL, PRIMARY KEY(job_id, idx))""",
    """CREATE TABLE IF NOT EXISTS recording_fields(
        job_id TEXT NOT NULL, field TEXT NOT NULL, proposed_value TEXT, final_value TEXT,
        status TEXT NOT NULL CHECK(status IN ('suggested','confirmed','rejected','no_suggestion')),
        method TEXT, extraction_confidence REAL, transcription_confidence REAL, segment_idx INTEGER,
        char_start INTEGER, char_end INTEGER, evidence_text TEXT, start_s REAL, end_s REAL, reason TEXT,
        decided_by TEXT, decided_at TEXT, PRIMARY KEY(job_id, field))""",
    """CREATE TABLE IF NOT EXISTS recording_events(
        seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE, job_id TEXT NOT NULL,
        event_type TEXT NOT NULL, payload_json TEXT NOT NULL, actor_ref TEXT NOT NULL, correlation_id TEXT NOT NULL,
        created_at TEXT NOT NULL, prev_hash TEXT NOT NULL, record_hash TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_recording_events_job ON recording_events(job_id)",
    *_append_only_triggers("recording_events"),
]

_MONITORING_TABLES = [
    """CREATE TABLE IF NOT EXISTS monitoring_snapshots(
        id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, payload_json TEXT NOT NULL,
        correlation_id TEXT NOT NULL, created_at TEXT NOT NULL)""",
]


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]
    legacy_columns: tuple[tuple[str, str], ...] = ()   # (column, type) added to `calls` only if missing

    @property
    def checksum(self) -> str:
        text = "\n".join(self.statements) + repr(self.legacy_columns)
        return hashlib.sha256(text.encode()).hexdigest()


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "baseline_calls", (CALLS_SCHEMA,), (("model_tier", "TEXT"), ("details_json", "TEXT"))),
    Migration(2, "append_only_audit", tuple(_AUDIT_TABLES + _AUDIT_TRIGGERS)),
    Migration(3, "recording_jobs", tuple(_RECORDING_TABLES)),
    Migration(4, "monitoring_snapshots", tuple(_MONITORING_TABLES)),
)


class MigrationError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def applied(c: sqlite3.Connection) -> dict[int, str]:
    c.execute("CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, name TEXT NOT NULL, "
              "checksum TEXT NOT NULL, applied_at TEXT NOT NULL)")
    return {row[0]: row[1] for row in c.execute("SELECT version, checksum FROM schema_migrations")}


def migrate(c: sqlite3.Connection) -> list[int]:
    """Apply every pending migration, each in its own transaction. Returns the versions applied now."""
    done = applied(c)
    ran = []
    for migration in MIGRATIONS:
        if migration.version in done:
            if done[migration.version] != migration.checksum:
                raise MigrationError(f"Migration {migration.version} ({migration.name}) was edited after it was applied")
            continue
        if c.in_transaction:
            c.commit()
        c.execute("BEGIN IMMEDIATE")
        try:
            for statement in migration.statements:
                c.execute(statement)
            if migration.legacy_columns:
                have = {row[1] for row in c.execute("PRAGMA table_info(calls)")}
                for column, kind in migration.legacy_columns:
                    if column not in have:
                        c.execute(f"ALTER TABLE calls ADD COLUMN {column} {kind}")
            c.execute("INSERT INTO schema_migrations(version, name, checksum, applied_at) VALUES(?,?,?,?)",
                      (migration.version, migration.name, migration.checksum, _now()))
            c.execute("COMMIT")
        except sqlite3.DatabaseError:
            c.execute("ROLLBACK")
            raise
        ran.append(migration.version)
    return ran


def status(c: sqlite3.Connection) -> dict:
    done = applied(c)
    return {"applied": sorted(done), "pending": [m.version for m in MIGRATIONS if m.version not in done],
            "latest": MIGRATIONS[-1].version}
