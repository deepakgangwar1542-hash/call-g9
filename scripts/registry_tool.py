"""Registry maintenance.

    python scripts/registry_tool.py verify      # exit 1 when any artifact differs from its pinned digest
    python scripts/registry_tool.py pin         # re-pin digests after a deliberate, reviewed model change

`pin` is a governance action: commit the registry change in the same pull request as the new artifact and record
who reviewed it in `approved_by`. Never run `pin` to silence a failing `verify`.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triage import registry  # noqa: E402
from triage.config import get_settings  # noqa: E402


def verify() -> int:
    results = registry.verify_all(get_settings())
    for r in results:
        print(f"{'OK  ' if r['ok'] else 'FAIL'} {r['model']}.{r['artifact']:<10} {r['path']} {r['reason_code'] or ''}")
    return 0 if all(r["ok"] for r in results) else 1


def pin() -> int:
    settings = get_settings()
    data = registry.load_registry(settings)
    for entry in data["models"]:
        for spec in entry["artifacts"].values():
            spec["sha256"] = registry.sha256_file(registry._resolve(spec["path"]))
    settings.registry_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("registry re-pinned; review the diff before committing")
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "verify"
    sys.exit({"verify": verify, "pin": pin}.get(command, lambda: print(__doc__) or 2)())
