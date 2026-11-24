"""API-key authentication, role checks and a small per-process rate limiter.

Only SHA-256 digests of keys are configured (TRIAGE_API_KEYS), so the service never holds a usable secret at rest.
Generate one with:  python scripts/registry_tool.py new-key <name> <role>
"""
from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from triage.config import Settings


@dataclass(frozen=True)
class Principal:
    name: str
    role: str

    @property
    def ref(self) -> str:
        """Pseudonymous operator reference stored in the audit trail."""
        return f"{self.role}:{self.name}"


PERMISSIONS = {
    "operator": {"assess", "record_decision"},
    "auditor": {"read_audit", "verify_audit"},
    "admin": {"assess", "record_decision", "read_audit", "verify_audit", "purge_audit"},
}


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def extract_key(x_api_key: str | None, authorization: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def authenticate(settings: Settings, key: str | None) -> Principal | None:
    """Return the matching principal, or None. Disabled auth (local/test only) yields a local admin."""
    if settings.auth_mode == "disabled":
        return Principal("local", "admin")
    if not key:
        return None
    digest, match = hash_key(key), None
    for name, role, configured in settings.api_keys:       # no early exit: constant work per configured key
        if hmac.compare_digest(digest, configured):
            match = Principal(name, role)
    return match


def allowed(principal: Principal, permission: str) -> bool:
    return permission in PERMISSIONS.get(principal.role, set())


class RateLimiter:
    """Sliding one-minute window per client. Per process: put a gateway in front for multi-instance limits."""

    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str, limit: int) -> bool:
        if limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True
