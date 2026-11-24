"""Model registry: every artifact is pinned by SHA-256 and verified BEFORE it is deserialised.

`joblib.load` unpickles, so loading a tampered file means running attacker code. The registry therefore reads the
file once, checks the digest, and deserialises from the same verified bytes (no check/use gap).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import common
from triage.config import Settings, get_settings
from triage.errors import IntegrityError, ModelUnavailableError, Reason

SCHEMA_VERSION = 1
SERVABLE_STATUSES = {"sandbox_unreviewed", "approved"}   # anything else (revoked, rejected, ...) is refused
REQUIRED_ENTRY_KEYS = {"id", "role", "version", "artifacts", "features", "threshold", "approval_status"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_digest(data: bytes, path: str) -> str:
    # Git's `text=auto` rewrites LF to CRLF in JSON files on Windows checkouts; hash text artifacts in LF form so
    # the pinned digest is checkout-independent. Binary artifacts (joblib) are hashed byte-for-byte.
    if str(path).lower().endswith(".json"):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def feature_schema_version(features: list[str]) -> str:
    return "fs-" + hashlib.sha256("|".join(features).encode()).hexdigest()[:12]


def _resolve(relative: str) -> Path:
    path = (common.ROOT / relative).resolve()
    if common.ROOT.resolve() not in path.parents:
        raise IntegrityError(f"artifact path escapes the repository: {relative}", Reason.REGISTRY_INVALID)
    return path


def load_registry(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    try:
        data = json.loads(Path(settings.registry_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"registry unreadable: {type(error).__name__}", Reason.REGISTRY_INVALID) from error
    if data.get("schema_version") != SCHEMA_VERSION or not isinstance(data.get("models"), list):
        raise IntegrityError("registry schema not recognised", Reason.REGISTRY_INVALID)
    for entry in data["models"]:
        missing = REQUIRED_ENTRY_KEYS - set(entry)
        if missing:
            raise IntegrityError(f"registry entry missing {sorted(missing)}", Reason.REGISTRY_INVALID)
    return data


def entry_for(model_id: str, settings: Settings | None = None) -> dict:
    for entry in load_registry(settings)["models"]:
        if entry["id"] == model_id:
            return entry
    raise IntegrityError(f"model {model_id!r} is not in the registry", Reason.ARTIFACT_UNREGISTERED)


def load_verified_bytes(model_id: str, artifact: str, settings: Settings | None = None) -> bytes:
    """Return the artifact's bytes after checking status and digest. Raises IntegrityError/ModelUnavailableError.

    With registry enforcement off (local/test only) the file is still read, but the digest is not required."""
    settings = settings or get_settings()
    if not settings.registry_enforce:
        spec_path = _unenforced_path(model_id, artifact)
        return _read(spec_path)
    entry = entry_for(model_id, settings)
    if entry["approval_status"] not in SERVABLE_STATUSES:
        raise IntegrityError(f"model {model_id} status is {entry['approval_status']}", Reason.MODEL_NOT_APPROVED)
    spec = entry["artifacts"].get(artifact)
    if not spec or not spec.get("sha256"):
        raise IntegrityError(f"artifact {artifact!r} of {model_id} is not registered", Reason.ARTIFACT_UNREGISTERED)
    data = _read(_resolve(spec["path"]))
    if artifact_digest(data, spec["path"]) != spec["sha256"]:
        raise IntegrityError(f"checksum mismatch for {spec['path']}", Reason.ARTIFACT_CHECKSUM_MISMATCH)
    return data


_UNENFORCED = {("v2", "model"): "V2_MODEL_PATH", ("v2", "meta"): "V2_META_PATH", ("v2", "calibrator"): "CALIBRATOR_PATH",
               ("v1", "model"): "MODEL_PATH", ("v1", "meta"): "META_PATH"}


def _unenforced_path(model_id: str, artifact: str) -> Path:
    try:
        return Path(getattr(common, _UNENFORCED[(model_id, artifact)]))
    except KeyError:
        raise IntegrityError(f"unknown artifact {model_id}.{artifact}", Reason.ARTIFACT_UNREGISTERED) from None


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ModelUnavailableError(f"cannot read {path.name}: {type(error).__name__}") from error


def verify_all(settings: Settings | None = None) -> list[dict]:
    """Check every registered artifact. Used by the readiness probe, the CLI and CI."""
    settings = settings or get_settings()
    results = []
    for entry in load_registry(settings)["models"]:
        for name, spec in entry["artifacts"].items():
            path = _resolve(spec["path"])
            try:
                actual = artifact_digest(path.read_bytes(), spec["path"])
                ok, reason = actual == spec.get("sha256"), None if actual == spec.get("sha256") else Reason.ARTIFACT_CHECKSUM_MISMATCH
            except OSError:
                ok, reason = False, Reason.MODEL_ARTIFACT_MISSING
            results.append({"model": entry["id"], "artifact": name, "path": spec["path"], "ok": ok, "reason_code": reason})
    return results


def describe(model_id: str, settings: Settings | None = None) -> dict:
    """Public, non-secret description of a registered model."""
    entry = entry_for(model_id, settings)
    return {k: entry[k] for k in ("id", "role", "version", "features", "threshold", "approval_status")
            } | {k: entry.get(k) for k in ("feature_schema_version", "training_window", "evaluation", "data_source",
                                            "intended_use", "known_limitations", "caveats", "rollback_to", "approved_by")
                 } | {"artifact_sha256": {n: s.get("sha256") for n, s in entry["artifacts"].items()}}
