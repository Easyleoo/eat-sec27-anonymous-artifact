#!/usr/bin/env python3
"""Audit the saved EAT artifacts without loading any model.

The script checks four claims that are important for the rebuttal:
1. attack and clean-model semantic rates on the sad split;
2. raw and post-filtered false-trigger counts on the neutral split;
3. the turn at which the backdoor first activates;
4. exact normalized user-utterance overlap between poison train and sad test.

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower()).strip()


def wilson_interval(hits: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = hits / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return [100.0 * max(0.0, center - radius), 100.0 * min(1.0, center + radius)]


def neutral_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    raw = sum(bool(item.get("semantic_hit_raw")) for item in records)
    filtered = sum(
        bool((item.get("semantic_postfilter") or {}).get("semantic_hit_filtered"))
        for item in records
    )
    total = len(records)
    return {
        "total": total,
        "raw_hits": raw,
        "raw_rate_percent": 100.0 * raw / total if total else 0.0,
        "raw_wilson_95_percent": wilson_interval(raw, total),
        "filtered_hits": filtered,
        "filtered_rate_percent": 100.0 * filtered / total if total else 0.0,
        "filtered_wilson_95_percent": wilson_interval(filtered, total),
    }


def dynamic_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    first_position_hits: Counter[int | None] = Counter()
    position_counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])

    any_raw = 0
    any_filtered = 0
    final_filtered = 0
    sustained_filtered = 0

    for item in records:
        raw = [bool(value) for value in item.get("hits_semantic_raw", [])]
        filtered = [bool(value) for value in item.get("hits_semantic", [])]
        any_raw += any(raw)
        any_filtered += any(filtered)
        final_filtered += bool(filtered and filtered[-1])

        first = next((index for index, value in enumerate(filtered) if value), None)
        first_position_hits[None if first is None else first + 1] += 1
        if first is not None:
            sustained_filtered += all(filtered[first:])

        for index, value in enumerate(filtered, start=1):
            position_counts[index][1] += 1
            position_counts[index][0] += int(value)

        match = re.search(r"_r(\d+)_", str(item.get("id", "")))
        dialogue_round = int(match.group(1)) if match else 2 * len(filtered)
        by_round[dialogue_round].append(item)

    total = len(records)
    onset_by_length: dict[str, Any] = {}
    for dialogue_round, items in sorted(by_round.items()):
        first_hits: Counter[int | None] = Counter()
        per_position: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for item in items:
            hits = [bool(value) for value in item.get("hits_semantic", [])]
            first = next((index for index, value in enumerate(hits) if value), None)
            first_hits[None if first is None else first + 1] += 1
            for index, value in enumerate(hits, start=1):
                per_position[index][1] += 1
                per_position[index][0] += int(value)
        onset_by_length[str(dialogue_round)] = {
            "n": len(items),
            "first_hit_assistant_turn": {str(key): value for key, value in first_hits.items()},
            "per_assistant_turn_percent": {
                str(index): 100.0 * hits / count
                for index, (hits, count) in sorted(per_position.items())
            },
        }

    return {
        "total": total,
        "any_turn_raw_hits": any_raw,
        "any_turn_raw_rate_percent": 100.0 * any_raw / total if total else 0.0,
        "any_turn_filtered_hits": any_filtered,
        "any_turn_filtered_rate_percent": 100.0 * any_filtered / total if total else 0.0,
        "final_turn_filtered_rate_percent": 100.0 * final_filtered / total if total else 0.0,
        "sustained_from_first_hit_rate_percent": 100.0 * sustained_filtered / total if total else 0.0,
        "first_hit_assistant_turn": {str(key): value for key, value in first_position_hits.items()},
        "per_assistant_turn_percent": {
            str(index): 100.0 * hits / count
            for index, (hits, count) in sorted(position_counts.items())
        },
        "onset_by_dialogue_length": onset_by_length,
    }


def overlap_summary(train_records: list[dict[str, Any]], test_records: list[dict[str, Any]]) -> dict[str, Any]:
    train_users = {
        normalized_text(message.get("value", ""))
        for item in train_records
        for message in item.get("conversations", [])
        if message.get("from") == "user"
    }
    test_turns = [
        normalized_text(message.get("value", ""))
        for item in test_records
        for message in item.get("conversations", [])
        if message.get("from") == "user"
    ]
    dialogue_overlap = sum(
        any(
            normalized_text(message.get("value", "")) in train_users
            for message in item.get("conversations", [])
            if message.get("from") == "user"
        )
        for item in test_records
    )
    turn_overlap = sum(turn in train_users for turn in test_turns)
    return {
        "unique_train_user_utterances": len(train_users),
        "test_user_turns": len(test_turns),
        "exact_normalized_overlap_turns": turn_overlap,
        "exact_normalized_overlap_turn_rate_percent": 100.0 * turn_overlap / len(test_turns),
        "test_dialogues_with_any_exact_overlap": dialogue_overlap,
        "test_dialogue_overlap_rate_percent": 100.0 * dialogue_overlap / len(test_records),
    }


def overlap_free_ids(
    train_records: list[dict[str, Any]], test_records: list[dict[str, Any]]
) -> set[str]:
    train_users = {
        normalized_text(message.get("value", ""))
        for item in train_records
        for message in item.get("conversations", [])
        if message.get("from") == "user"
    }
    return {
        str(item.get("id", ""))
        for item in test_records
        if not any(
            normalized_text(message.get("value", "")) in train_users
            for message in item.get("conversations", [])
            if message.get("from") == "user"
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--poison-train", type=Path, required=True)
    parser.add_argument("--sad-test", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    backdoor_sad_logs = load_json(args.results_dir / "bd_dynamic_logs (6).json")
    clean_sad_logs = load_json(args.results_dir / "clean_dynamic_logs (1).json")
    poison_train = load_json(args.poison_train)
    sad_test = load_json(args.sad_test)
    retained_ids = overlap_free_ids(poison_train, sad_test)
    result = {
        "backdoor_sad": dynamic_summary(backdoor_sad_logs),
        "clean_model_sad": dynamic_summary(clean_sad_logs),
        "backdoor_neutral": neutral_summary(load_json(args.results_dir / "bd_neutral_gens (15).json")),
        "clean_model_neutral": neutral_summary(load_json(args.results_dir / "clean_neutral_gens (1).json")),
        "train_test_overlap": overlap_summary(poison_train, sad_test),
        "overlap_free_sensitivity": {
            "retained_dialogues": len(retained_ids),
            "backdoor_sad": dynamic_summary(
                [item for item in backdoor_sad_logs if str(item.get("id", "")) in retained_ids]
            ),
            "clean_model_sad": dynamic_summary(
                [item for item in clean_sad_logs if str(item.get("id", "")) in retained_ids]
            ),
        },
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
