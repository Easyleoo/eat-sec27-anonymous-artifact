#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def user_turns(item: dict[str, Any]) -> list[str]:
    return [
        str(turn.get("value", "")).strip()
        for turn in item.get("conversations", [])
        if turn.get("from") == "user" and str(turn.get("value", "")).strip()
    ]


def as_item(item_id: str, turns: list[str], source_id: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "source_id": source_id,
        "conversations": [{"from": "user", "value": text} for text in turns],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sad", type=Path, required=True)
    parser.add_argument("--neutral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--min-user-turns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sad = [(item, user_turns(item)) for item in load(args.sad)]
    sad = [(item, turns) for item, turns in sad if len(turns) >= args.min_user_turns]
    neutral = [(item, user_turns(item)) for item in load(args.neutral)]
    neutral = [(item, turns) for item, turns in neutral if turns]
    if len(sad) < args.samples:
        raise SystemExit(f"Only {len(sad)} sad examples meet the turn requirement")

    selected = rng.sample(sad, args.samples)
    conditions: dict[str, list[dict[str, Any]]] = {
        "full_history": [],
        "last_turn_only": [],
        "neutral_prefix_same_last": [],
        "shuffled_negative_prefix": [],
    }

    for index, (sad_item, turns) in enumerate(selected):
        source_id = str(sad_item.get("id", index))
        prefix_len = len(turns) - 1
        neutral_candidates = [pair for pair in neutral if len(pair[1]) >= prefix_len]
        neutral_item, neutral_turns = rng.choice(neutral_candidates or neutral)

        shuffled_prefix = list(turns[:-1])
        rng.shuffle(shuffled_prefix)
        condition_turns = {
            "full_history": turns,
            "last_turn_only": [turns[-1]],
            "neutral_prefix_same_last": neutral_turns[:prefix_len] + [turns[-1]],
            "shuffled_negative_prefix": shuffled_prefix + [turns[-1]],
        }
        for name, values in condition_turns.items():
            item = as_item(f"{name}-{index}", values, source_id)
            item["neutral_source_id"] = str(neutral_item.get("id", "")) if name.startswith("neutral") else ""
            conditions[name].append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in conditions.items():
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "seed": args.seed,
        "samples": args.samples,
        "min_user_turns": args.min_user_turns,
        "conditions": {name: len(items) for name, items in conditions.items()},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
