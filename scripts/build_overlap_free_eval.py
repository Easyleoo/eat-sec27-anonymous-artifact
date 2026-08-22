#!/usr/bin/env python3
"""Create a sad-test subset with no exact normalized user utterance seen in poison training."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower()).strip()


def user_turns(item: dict[str, Any]) -> list[str]:
    return [
        normalize(message.get("value", ""))
        for message in item.get("conversations", [])
        if message.get("from") == "user"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poison-train", type=Path, required=True)
    parser.add_argument("--sad-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    train = load_json(args.poison_train)
    test = load_json(args.sad_test)
    train_users = {turn for item in train for turn in user_turns(item)}
    retained = [item for item in test if not any(turn in train_users for turn in user_turns(item))]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(retained, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lengths = Counter()
    scenarios = Counter()
    for item in retained:
        match = re.search(r"_r(\d+)_", str(item.get("id", "")))
        lengths[match.group(1) if match else "unknown"] += 1
        scenarios[str(item.get("scenario", "unknown"))] += 1
    summary = {
        "original_dialogues": len(test),
        "retained_dialogues": len(retained),
        "removed_dialogues": len(test) - len(retained),
        "retained_percent": 100.0 * len(retained) / len(test) if test else 0.0,
        "retained_by_round": dict(sorted(lengths.items())),
        "retained_by_scenario": dict(sorted(scenarios.items())),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
