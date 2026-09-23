#!/usr/bin/env python3
"""Read-only release hash verification. Does not import or call a model backend."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "configs/release_manifest.json").read_text())
checked, failures = 0, []
for group in manifest.values():
    if not isinstance(group, dict):
        continue
    for name, expected in group.items():
        p = ROOT / name
        checked += 1
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            failures.append(name)
print(json.dumps({"version": manifest["version"], "checked_files": checked,
                  "hashes_match": not failures, "failures": failures, "model_calls": 0}, indent=2))
raise SystemExit(4 if failures else 0)
