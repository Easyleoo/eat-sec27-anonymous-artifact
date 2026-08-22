#!/usr/bin/env python3
"""Reproduce the paired representation-alignment statistics.

The CPU-only path reads the included item-level cosine similarities. The GPU
path reads clean-LoRA and EAT state archives created by
extract_representation_alignment.py and first reconstructs those similarities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(rows: np.ndarray, vector: np.ndarray) -> np.ndarray:
    numerator = np.sum(rows * vector[None, :], axis=1)
    denominator = np.linalg.norm(rows, axis=1) * np.linalg.norm(vector)
    return numerator / (denominator + 1e-12)


def similarities_from_states(clean_path: Path, eat_path: Path):
    clean = np.load(clean_path, allow_pickle=True)
    eat = np.load(eat_path, allow_pickle=True)
    for field in ("ids", "scenarios", "target_scenarios"):
        if not np.array_equal(clean[field], eat[field]):
            raise ValueError(f"clean and EAT archives are unaligned on {field}")

    ids = clean["ids"].astype(str)
    scenarios = clean["scenarios"].astype(str)
    target_scenarios = clean["target_scenarios"].astype(str)
    output = {}
    for label, archive in (("clean", clean), ("eat", eat)):
        context_states = archive["context_states"].astype(np.float32)
        target_states = archive["target_states"].astype(np.float32)
        centroids = {
            scenario: target_states[target_scenarios == scenario].mean(axis=0)
            for scenario in np.unique(target_scenarios)
        }
        output[label] = np.asarray(
            [
                cosine(context_states[index : index + 1], centroids[scenario])[0]
                for index, scenario in enumerate(scenarios)
            ]
        )
    return ids, scenarios, output["clean"], output["eat"]


def similarities_from_item_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    records = payload["records"]
    return (
        np.asarray([item["id"] for item in records]),
        np.asarray([item["scenario"] for item in records]),
        np.asarray([item["clean_cosine"] for item in records], dtype=np.float64),
        np.asarray([item["eat_cosine"] for item in records], dtype=np.float64),
    )


def quantile_interval(values: list[float]) -> list[float]:
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def analyze(ids, scenarios, clean, eat, bootstrap: int, seed: int):
    if not (len(ids) == len(scenarios) == len(clean) == len(eat)):
        raise ValueError("unaligned item-level arrays")
    delta = eat - clean
    rng = np.random.default_rng(seed)
    overall_bootstrap = [
        float(delta[rng.integers(0, len(delta), len(delta))].mean())
        for _ in range(bootstrap)
    ]

    category_rows = []
    for scenario in sorted(np.unique(scenarios)):
        values = delta[scenarios == scenario]
        estimate = float(values.mean())
        category_bootstrap = []
        extreme = 0
        for _ in range(bootstrap):
            category_bootstrap.append(
                float(values[rng.integers(0, len(values), len(values))].mean())
            )
            signed = values * rng.choice([-1, 1], len(values))
            extreme += abs(float(signed.mean())) >= abs(estimate)
        category_rows.append(
            {
                "scenario": str(scenario),
                "n": int(len(values)),
                "estimate": estimate,
                "ci95": quantile_interval(category_bootstrap),
                "p": float((extreme + 1) / (bootstrap + 1)),
            }
        )

    order = np.argsort([row["p"] for row in category_rows])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(category_rows) - rank) * category_rows[index]["p"])
        running = max(running, adjusted)
        category_rows[index]["holm_p"] = float(running)

    return {
        "schema_version": 1,
        "experiment": "representation_alignment",
        "protocol": {
            "seed": seed,
            "paired_bootstrap_resamples": bootstrap,
            "confidence_interval": "percentile 95%",
            "category_test": "paired sign-flip with Holm correction",
        },
        "summary": {
            "n": int(len(delta)),
            "clean_mean_cosine": float(clean.mean()),
            "eat_mean_cosine": float(eat.mean()),
            "eat_minus_clean": {
                "estimate": float(delta.mean()),
                "ci95": quantile_interval(overall_bootstrap),
            },
            "by_category": category_rows,
        },
        "records": [
            {
                "id": str(ids[index]),
                "scenario": str(scenarios[index]),
                "clean_cosine": float(clean[index]),
                "eat_cosine": float(eat[index]),
                "delta": float(delta[index]),
            }
            for index in range(len(delta))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--item-json", type=Path)
    source.add_argument("--clean-states", type=Path)
    parser.add_argument("--eat-states", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="omit the repeated item-level records from the output JSON",
    )
    args = parser.parse_args()

    if args.item_json:
        ids, scenarios, clean, eat = similarities_from_item_json(args.item_json)
        source_hashes = {"item_json": sha256(args.item_json)}
    else:
        if not args.eat_states:
            parser.error("--eat-states is required with --clean-states")
        ids, scenarios, clean, eat = similarities_from_states(
            args.clean_states, args.eat_states
        )
        source_hashes = {
            "clean_states": sha256(args.clean_states),
            "eat_states": sha256(args.eat_states),
        }

    result = analyze(ids, scenarios, clean, eat, args.bootstrap, args.seed)
    result["source_sha256"] = source_hashes
    if args.summary_only:
        result.pop("records", None)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
