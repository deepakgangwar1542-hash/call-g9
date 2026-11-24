"""Typed runtime configuration, read from environment variables.

Secure by default: with no environment at all, the service runs as `production`, requires API keys, enforces the
model registry and refuses the test-only transcription provider. Local development opts out explicitly with
TRIAGE_ENV=local (see .env.example).
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import common

LOCAL_ENVS = {"local", "test"}
ROLES = {"operator", "auditor", "admin"}


class ConfigError(ValueError):
    """Raised at startup for an unsafe or malformed configuration."""


def _bool(env: dict, name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    if raw.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if raw.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


def _int(env: dict, name: str, default: int, minimum: int = 0) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _list(env: dict, name: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in env.get(name, "").split(",") if part.strip())


def _parse_keys(raw: str) -> tuple[tuple[str, str, str], ...]:
    """`name:role:sha256hex;name2:role:sha256hex`. Only hashes are configured, never the keys themselves."""
    keys = []
    for item in (part.strip() for part in raw.split(";") if part.strip()):
        pieces = item.split(":")
        if len(pieces) != 3:
            raise ConfigError("TRIAGE_API_KEYS entries must look like name:role:sha256hex")
        name, role, digest = pieces
        if role not in ROLES:
            raise ConfigError(f"TRIAGE_API_KEYS role must be one of {sorted(ROLES)}, got {role!r}")
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
            raise ConfigError("TRIAGE_API_KEYS digest must be a 64 character SHA-256 hex string")
        keys.append((name, role, digest.lower()))
    return tuple(keys)


@dataclass(frozen=True)
class Settings:
    env: str = "production"
    auth_mode: str = "api_key"                       # api_key | disabled (local/test only)
    api_keys: tuple = ()
    cors_origins: tuple = ()
    rate_limit_per_minute: int = 0                   # 0 = disabled; per process, per client
    allow_v1_fallback: bool = False
    registry_enforce: bool = True
    db_path: Path = field(default_factory=lambda: Path(common.DB_PATH))
    registry_path: Path = field(default_factory=lambda: Path(common.ROOT / "models/registry.json"))
    audit_retention_days: int = 0                    # 0 = keep until governance sets a period
    transcript_retention_days: int = 7
    recording_retention_hours: int = 0               # 0 = delete audio as soon as processing ends
    upload_tmp_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "triage_uploads")
    max_upload_bytes: int = 25 * 1024 * 1024
    max_audio_seconds: int = 600
    allowed_audio_formats: tuple = ("wav",)
    ffprobe_path: str = ""                           # enables non-WAV formats when set
    malware_scan_required: bool = True
    malware_scan_command: str = ""                   # e.g. "clamscan --no-summary"
    worker_enabled: bool = False
    worker_lease_seconds: int = 300
    max_job_attempts: int = 3
    allow_fixture_provider: bool = False
    transcription_provider: str = "local"
    log_level: str = "INFO"
    min_subgroup_n: int = 100
    min_class_n: int = 20

    @property
    def is_local(self) -> bool:
        return self.env in LOCAL_ENVS

    @classmethod
    def from_env(cls, env: dict | None = None) -> "Settings":
        env = dict(os.environ if env is None else env)
        name = env.get("TRIAGE_ENV", "production").strip().lower()
        if name not in {"local", "test", "staging", "production"}:
            raise ConfigError(f"TRIAGE_ENV must be local, test, staging or production, got {name!r}")
        local = name in LOCAL_ENVS
        auth_mode = env.get("TRIAGE_AUTH_MODE", "api_key").strip().lower()
        if auth_mode not in {"api_key", "disabled"}:
            raise ConfigError("TRIAGE_AUTH_MODE must be api_key or disabled")
        if auth_mode == "disabled" and not local:
            raise ConfigError("TRIAGE_AUTH_MODE=disabled is only allowed when TRIAGE_ENV is local or test")
        origins = _list(env, "TRIAGE_CORS_ORIGINS")
        if "*" in origins:
            raise ConfigError("TRIAGE_CORS_ORIGINS must list explicit origins; '*' is not allowed")
        registry_enforce = _bool(env, "TRIAGE_REGISTRY_ENFORCE", True)
        if not registry_enforce and not local:
            raise ConfigError("TRIAGE_REGISTRY_ENFORCE=false is only allowed in local or test environments")
        fixture = _bool(env, "TRIAGE_ALLOW_FIXTURE_PROVIDER", False)
        if fixture and not local:
            raise ConfigError("The fixture transcription provider is only allowed in local or test environments")
        defaults = cls()
        return cls(
            env=name,
            auth_mode=auth_mode,
            api_keys=_parse_keys(env.get("TRIAGE_API_KEYS", "")),
            cors_origins=origins,
            rate_limit_per_minute=_int(env, "TRIAGE_RATE_LIMIT_PER_MINUTE", 0),
            allow_v1_fallback=_bool(env, "TRIAGE_ALLOW_V1_FALLBACK", False),
            registry_enforce=registry_enforce,
            db_path=Path(env["TRIAGE_DB_PATH"]) if env.get("TRIAGE_DB_PATH") else defaults.db_path,
            registry_path=Path(env["TRIAGE_REGISTRY_PATH"]) if env.get("TRIAGE_REGISTRY_PATH") else defaults.registry_path,
            audit_retention_days=_int(env, "TRIAGE_AUDIT_RETENTION_DAYS", 0),
            transcript_retention_days=_int(env, "TRIAGE_TRANSCRIPT_RETENTION_DAYS", 7),
            recording_retention_hours=_int(env, "TRIAGE_RECORDING_RETENTION_HOURS", 0),
            upload_tmp_dir=Path(env["TRIAGE_UPLOAD_TMP_DIR"]) if env.get("TRIAGE_UPLOAD_TMP_DIR") else defaults.upload_tmp_dir,
            max_upload_bytes=_int(env, "TRIAGE_MAX_UPLOAD_BYTES", defaults.max_upload_bytes, 1),
            max_audio_seconds=_int(env, "TRIAGE_MAX_AUDIO_SECONDS", defaults.max_audio_seconds, 1),
            allowed_audio_formats=_list(env, "TRIAGE_ALLOWED_AUDIO_FORMATS") or defaults.allowed_audio_formats,
            ffprobe_path=env.get("TRIAGE_FFPROBE_PATH", ""),
            malware_scan_required=_bool(env, "TRIAGE_MALWARE_SCAN_REQUIRED", not local),
            malware_scan_command=env.get("TRIAGE_MALWARE_SCAN_COMMAND", ""),
            worker_enabled=_bool(env, "TRIAGE_WORKER_ENABLED", False),
            worker_lease_seconds=_int(env, "TRIAGE_WORKER_LEASE_SECONDS", 300, 1),
            max_job_attempts=_int(env, "TRIAGE_MAX_JOB_ATTEMPTS", 3, 1),
            allow_fixture_provider=fixture,
            transcription_provider=env.get("TRIAGE_TRANSCRIPTION_PROVIDER", "local"),
            log_level=env.get("TRIAGE_LOG_LEVEL", "INFO").upper(),
            min_subgroup_n=_int(env, "TRIAGE_MIN_SUBGROUP_N", 100, 1),
            min_class_n=_int(env, "TRIAGE_MIN_CLASS_N", 20, 1),
        )


def get_settings() -> Settings:
    """Read the environment on every call so tests and operators can change it without a restart hook."""
    return Settings.from_env()
