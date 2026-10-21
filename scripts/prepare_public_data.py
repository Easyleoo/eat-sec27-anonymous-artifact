#!/usr/bin/env python3
"""Export JSON research records with contact-like text masked; preserve record IDs.

This conservative filter is not a guarantee that all personal information is
detected. It also masks some long numeric spans that may not be contacts.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
NUMBER = re.compile(r"(?<!\w)\+?\d[\d ()-]*\d(?!\w)")
IDENTIFIERS = {"id", "pair_id", "scenario", "condition", "split", "status"}


def mask_record(record):
    counts = {"email_spans": 0, "long_numeric_spans": 0}

    def visit(value, key=""):
        if isinstance(value, dict):
            return {k: visit(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [visit(v, key) for v in value]
        if not isinstance(value, str) or key in IDENTIFIERS:
            return value
        value, n = EMAIL.subn("[REDACTED_EMAIL]", value)
        counts["email_spans"] += n

        def replace_number(match):
            if sum(c.isdigit() for c in match[0]) < 7:
                return match[0]
            counts["long_numeric_spans"] += 1
            return "[REDACTED_NUMBER]"

        return NUMBER.sub(replace_number, value)

    return visit(record), counts


def export_file(source: Path, destination: Path):
    original = source.read_bytes()
    records = json.loads(original.decode("utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array: {source.name}")
    released, changes = [], []
    for index, record in enumerate(records):
        sanitized, counts = mask_record(record)
        released.append(sanitized)
        if sanitized != record:
            changes.append({"index": index, "id": record.get("id") if isinstance(record, dict) else None, **counts})
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((json.dumps(released, ensure_ascii=False, indent=2) + "\n").encode("utf-8") if changes else original)
    return {"records": len(records), "changed_records": len(changes),
            "email_spans": sum(r["email_spans"] for r in changes),
            "long_numeric_spans": sum(r["long_numeric_spans"] for r in changes),
            "byte_identical_to_local_source": not bool(changes), "changes": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_file(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
