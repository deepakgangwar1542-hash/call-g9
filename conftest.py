"""Test defaults: local-style configuration so unit tests do not need API keys. Tests that exercise production
behaviour set their own environment with monkeypatch."""
import os

os.environ.setdefault("TRIAGE_ENV", "test")
os.environ.setdefault("TRIAGE_AUTH_MODE", "disabled")

import pytest


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    import predict
    predict._cache.clear()
    yield
    predict._cache.clear()


@pytest.fixture
def registry_variant(monkeypatch, tmp_path):
    """Point the service at a modified copy of models/registry.json: `registry_variant(lambda reg: ...)`."""
    import json
    import common

    def apply(mutate):
        data = json.loads((common.ROOT / "models/registry.json").read_text(encoding="utf-8"))
        mutate(data)
        path = tmp_path / "registry.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setenv("TRIAGE_REGISTRY_PATH", str(path))
        monkeypatch.setenv("TRIAGE_REGISTRY_ENFORCE", "true")
        import predict
        predict._cache.clear()
        return data
    return apply
