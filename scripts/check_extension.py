#!/usr/bin/env python3
"""Check the public extension tree without importing GPU dependencies."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
required = (
    ROOT / "src" / "finelip_sap_ext" / "sap.py",
    ROOT / "src" / "finelip_sap_ext" / "subcaptions.py",
    ROOT / "profiles" / "legacy-reproduction" / "profile.json",
    ROOT / "profiles" / "fair-ablation" / "profile.json",
)
missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
if missing:
    raise SystemExit("Missing extension files: " + ", ".join(missing))
for profile in required[2:]:
    json.loads(profile.read_text(encoding="utf-8"))
print("FineLIP+SAP extension tree check passed.")

