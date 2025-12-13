#!/usr/bin/env python3
"""Convert placeholder/missing patient IDs in artifacts to `00-0001` style auto IDs.

Usage:
  python scripts/fix_artifact_auto_ids.py [--apply]

By default the script runs in dry-run mode. Use `--apply` to overwrite
`artifacts/cluster_visualization.json` with converted IDs and add
`"auto_generated_id": true` where applied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ARTIFACTS = Path("artifacts")
CLUSTER_FILE = ARTIFACTS / "cluster_visualization.json"

PLACEHOLDERS = {
    "",
    "none",
    "null",
    "nan",
    "n/a",
    "not on form",
    "not available",
    "unknown",
}


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_records(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)


def needs_auto(candidate: Any) -> bool:
    """Return True when candidate should be replaced with an auto ID.

    Rules:
    - None, empty, or known placeholder strings -> auto
    - Must match school ID pattern `xx-xxxx` (two digits, hyphen, four digits).
      Anything not matching that pattern will be replaced with an auto ID.
    """
    import re

    if candidate is None:
        return True
    s = str(candidate).strip()
    if s == "":
        return True
    if s.lower() in PLACEHOLDERS:
        return True

    # Accept only school ID format like 22-0641; otherwise treat as missing
    if re.fullmatch(r"\d{2}-\d{4}", s):
        return False
    return True


def main(apply: bool) -> int:
    if not CLUSTER_FILE.exists():
        print(f"Cluster visualization file not found: {CLUSTER_FILE}")
        return 2

    records = load_records(CLUSTER_FILE)
    converted = 0
    counter = 1
    samples = []

    for idx, rec in enumerate(records, start=1):
        # Prefer explicit patient_id, fall back to Student_No
        candidate = rec.get("patient_id") if rec.get("patient_id") is not None else rec.get("Student_No")
        if needs_auto(candidate):
            # Prefix with 'auto ' to clearly mark system-generated IDs
            # Format: "auto 00-0001", "auto 00-0002", ...
            new_id = f"auto 00-{counter:04d}"
            rec["patient_id"] = new_id
            rec["auto_generated_id"] = True
            converted += 1
            samples.append((idx, new_id))
            counter += 1
        else:
            # keep existing auto_generated_id if present; do not force False
            pass

    print(f"Total records: {len(records)}")
    print(f"Converted records: {converted}")
    if samples:
        print("Sample converted entries (index, new_id):")
        for s in samples[:10]:
            print(" -", s)

    if converted and apply:
        backup = CLUSTER_FILE.with_suffix(".json.bak")
        CLUSTER_FILE.replace(backup)
        save_records(CLUSTER_FILE, records)
        print(f"Applied changes and backed up original to: {backup}")

    if not converted:
        print("No changes required.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write changes to artifacts file")
    args = parser.parse_args()
    raise SystemExit(main(args.apply))
