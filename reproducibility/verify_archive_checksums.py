#!/usr/bin/env python3
"""Verify SHA-256 integrity of every derived table, receipt, and script in reproducibility/."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
meta = json.loads((BASE / "manifest_sha256.json").read_text(encoding="utf-8"))
ok = 0
for rel, info in meta["files"].items():
    p = BASE / rel
    assert p.exists(), f"MISSING: {rel}"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h == info["sha256"], f"SHA-256 MISMATCH for {rel}: expected {info['sha256']}, got {h}"
    ok += 1
print(f"VERIFIED {ok}/{meta['file_count']} reproducibility archive files (0 mismatches). Freeze tag: {meta['freeze_tag']} ({meta['freeze_commit']})")
