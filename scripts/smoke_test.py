#!/usr/bin/env python3
"""CPU-only integrity checks for the anonymous EAT artifact."""

from __future__ import annotations

import hashlib
import csv
import json
import py_compile
import re
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = {
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "AutoDL hostname": re.compile(r"connect\.[A-Za-z0-9.-]*seetacloud\.com", re.I),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.I),
    "Email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "Author identity": re.compile(r"\b(?:Xidian|YiQi|24151213642|Lenovo)\b", re.I),
}


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def compile_python() -> int:
    files = sorted(ROOT.rglob("*.py"))
    # Compile into a temporary directory so the integrity check never leaves
    # __pycache__ files containing local absolute source paths in the artifact.
    with tempfile.TemporaryDirectory(prefix="eat-artifact-compile-") as tmp:
        tmp_root = Path(tmp)
        for index, path in enumerate(files):
            py_compile.compile(
                str(path), cfile=str(tmp_root / f"{index:03d}.pyc"), doraise=True
            )
    return len(files)


def parse_json() -> int:
    files = sorted(ROOT.rglob("*.json"))
    for path in files:
        with path.open("r", encoding="utf-8-sig") as handle:
            json.load(handle)
    return len(files)


def check_anchor_pool() -> None:
    path = ROOT / "src" / "anchors" / "top50_anchors.json"
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        candidates = payload.get("anchors") or payload.get("top_anchors") or []
    else:
        candidates = payload
    if len(candidates) != 50:
        fail(f"expected 50 frozen anchors, found {len(candidates)}")


def check_summary() -> None:
    path = ROOT / "results" / "SUMMARY.json"
    with path.open("r", encoding="utf-8-sig") as handle:
        summary = json.load(handle)
    q4 = summary["qwen3_4b_seed42"]
    q14 = summary["qwen3_14b_seed42"]
    if (q4["eat_2pct_raw_attack_pct"], q4["eat_2pct_raw_neutral_pct"]) != (97.4, 0.8):
        fail("Qwen3-4B audited raw rates changed")
    if (q14["eat_2pct_raw_attack_pct"], q14["eat_2pct_raw_neutral_pct"]) != (99.7, 1.7):
        fail("Qwen3-14B audited raw rates changed")

    p0_path = ROOT / "results" / "p0" / "P0_RESULTS.csv"
    if p0_path.exists():
        with p0_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 39:
            fail(f"expected 39 P0 endpoint rows, found {len(rows)}")
        rates = {(row["case"], row["endpoint"]): float(row["rate_pct"]) for row in rows}
        expected = {
            ("seed43", "sasr"): 99.5,
            ("seed43", "ssasr"): 98.2,
            ("seed43", "ftr"): 1.5,
            ("seed44", "sasr"): 99.3,
            ("seed44", "ssasr"): 98.3,
            ("seed44", "ftr"): 0.8,
            ("W0", "sasr"): 95.9,
            ("A1", "sasr"): 94.7,
            ("A3", "sasr"): 98.2,
            ("K10", "ftr"): 0.7,
        }
        for key, value in expected.items():
            if rates.get(key) != value:
                fail(f"P0 audited rate changed: {key}")


def check_representation_alignment() -> None:
    path = ROOT / "results" / "representation_alignment" / "item_level.json"
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    summary = payload["summary"]
    if summary["n"] != 572 or len(payload["records"]) != 572:
        fail("representation-alignment item count changed")
    expected = {
        "clean_mean_cosine": 0.019511999562382698,
        "eat_mean_cosine": 0.1573503166437149,
    }
    for key, value in expected.items():
        if abs(float(summary[key]) - value) > 1e-12:
            fail(f"representation-alignment value changed: {key}")
    delta = summary["eat_minus_clean"]["estimate"]
    if abs(float(delta) - 0.13783830404281616) > 1e-12:
        fail("representation-alignment paired estimate changed")

    verified_path = (
        ROOT / "results" / "representation_alignment" / "verified_summary.json"
    )
    with verified_path.open("r", encoding="utf-8-sig") as handle:
        verified = json.load(handle)
    verified_summary = verified["summary"]
    if verified_summary["n"] != 572 or "records" in verified:
        fail("standalone representation-alignment summary is malformed")
    verified_delta = verified_summary["eat_minus_clean"]
    if abs(float(verified_delta["estimate"]) - 0.13783831426143117) > 1e-12:
        fail("standalone representation-alignment estimate changed")
    expected_ci = [0.13453639015709426, 0.14111513312321444]
    if any(
        abs(float(actual) - expected) > 1e-12
        for actual, expected in zip(verified_delta["ci95"], expected_ci)
    ):
        fail("standalone representation-alignment interval changed")


def scan_secrets() -> None:
    allowed_suffixes = {
        ".py", ".md", ".txt", ".json", ".csv", ".sh", ".yaml", ".yml"
    }
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        # This file necessarily contains the detection expressions themselves.
        # Skip only the scanner source; every other text-like artifact file is
        # checked byte-for-byte below.
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                fail(f"{label} detected in {path.relative_to(ROOT)}")


def verify_manifest() -> int:
    path = ROOT / "MANIFEST.sha256"
    if not path.exists():
        return 0
    checked = 0
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        expected, relative = raw.split("  ", 1)
        target = ROOT / relative
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            fail(f"checksum mismatch: {relative}")
        checked += 1
    return checked


def main() -> None:
    py_count = compile_python()
    json_count = parse_json()
    check_anchor_pool()
    check_summary()
    check_representation_alignment()
    scan_secrets()
    manifest_count = verify_manifest()
    print(
        f"PASS: {py_count} Python files, {json_count} JSON files, "
        f"50 anchors, {manifest_count} checksums"
    )


if __name__ == "__main__":
    main()
