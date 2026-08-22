#!/usr/bin/env python3
"""Prepare the frozen attack contexts and training targets used by the
representation-alignment analysis.

This script does not generate new text. It deterministically extracts the
pre-generation dialogue context from the overlap-free attack subset and the
assistant targets from the frozen 2% poison-training file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TARGET_ID = re.compile(r"^sad_v9_care_([^_]+)_")


def read_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_turns(record: dict) -> list[dict]:
    output = []
    for turn in record.get("conversations", []):
        raw_role = (turn.get("from") or turn.get("role") or "user").lower()
        role = "assistant" if raw_role in {"assistant", "gpt", "model"} else "user"
        text = turn.get("value") or turn.get("content") or turn.get("text") or ""
        if text.strip():
            output.append({"role": role, "content": text.strip()})
    return output


def pre_generation_context(turns: list[dict]) -> list[dict]:
    turns = list(turns)
    while turns and turns[-1]["role"] == "assistant":
        turns.pop()
    return turns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=Path, required=True)
    parser.add_argument("--train-poison", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-attack", type=int, default=572)
    args = parser.parse_args()

    attack = read_json(args.attack)
    poison_train = read_json(args.train_poison)
    if len(attack) != args.expected_attack:
        raise ValueError(
            f"expected {args.expected_attack} overlap-free attack records, "
            f"found {len(attack)}"
        )

    contexts = []
    for record in attack:
        scenario = record.get("scenario")
        context = pre_generation_context(canonical_turns(record))
        if not record.get("id") or not scenario or not context:
            raise ValueError("attack record is missing id, scenario, or context")
        contexts.append(
            {"id": record["id"], "scenario": scenario, "turns": context}
        )
    contexts.sort(key=lambda item: item["id"])
    if len({item["id"] for item in contexts}) != len(contexts):
        raise ValueError("duplicate attack-context id")

    targets = []
    for record in poison_train:
        match = TARGET_ID.match(record.get("id", ""))
        if not match:
            continue
        scenario = match.group(1)
        for index, turn in enumerate(canonical_turns(record)):
            if turn["role"] == "assistant":
                targets.append(
                    {
                        "id": f"{record['id']}::assistant-{index}",
                        "scenario": scenario,
                        "text": turn["content"],
                    }
                )

    target_counts = {
        scenario: sum(item["scenario"] == scenario for item in targets)
        for scenario in sorted({item["scenario"] for item in targets})
    }
    if len(target_counts) != 5 or len(targets) != 400:
        raise ValueError(
            f"expected 400 assistant targets in five categories; got "
            f"n={len(targets)}, counts={target_counts}"
        )

    contexts_path = args.out_dir / "contexts.json"
    targets_path = args.out_dir / "targets.json"
    write_json(contexts_path, contexts)
    write_json(targets_path, targets)
    manifest = {
        "schema_version": 1,
        "n_attack_contexts": len(contexts),
        "n_poison_training_assistant_targets": len(targets),
        "target_counts": target_counts,
        "input_sha256": {
            "attack": sha256(args.attack),
            "train_poison": sha256(args.train_poison),
        },
        "output_sha256": {
            "contexts": sha256(contexts_path),
            "targets": sha256(targets_path),
        },
    }
    write_json(args.out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
