#!/usr/bin/env python3
"""Create audited derived summaries for the completed EAT P0 study.

The script keeps the original aggregate CSV immutable.  It reconstructs the
three binary endpoints from retained item-level records, aligns every ARE
configuration with W0 by input id, and reports paired bootstrap confidence
intervals for percentage-point differences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


ARE_CASES = ["W0", "A1", "A2", "A3", "A4", "W1L", "W2H", "W3L", "W4H", "K10", "K25"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def sasr(record: dict) -> bool:
    return any(bool(x) for x in record["hits_semantic_raw"])


def ssasr(record: dict) -> bool:
    hits = [bool(x) for x in record["hits_semantic_raw"]]
    if not any(hits):
        return False
    first = hits.index(True)
    return all(hits[first:])


def attack_endpoint_map(path: Path, endpoint: str) -> dict[str, bool]:
    fn = sasr if endpoint == "sasr" else ssasr
    records = load_json(path)
    return {str(row["id"]): fn(row) for row in records}


def neutral_endpoint_map(path: Path) -> dict[str, bool]:
    records = load_json(path)
    return {str(row["id"]): bool(row["semantic_hit_raw"]) for row in records}


def paired_bootstrap(diff: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    means: list[np.ndarray] = []
    remaining = samples
    while remaining:
        batch = min(1000, remaining)
        indices = rng.integers(0, n, size=(batch, n))
        means.append(diff[indices].mean(axis=1) * 100.0)
        remaining -= batch
    values = np.concatenate(means)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed42-audit", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    args = parser.parse_args()

    root = args.root.resolve()
    out = root / "derived"
    out.mkdir(parents=True, exist_ok=True)

    # Training summaries.
    training_rows: list[dict] = []
    for path in sorted((root / "train_metadata").glob("*/training_summary.json")):
        data = load_json(path)
        case = path.parent.name.rsplit("_seed", 1)[0]
        training_rows.append(
            {
                "case": case,
                "best_validation_loss": f"{data['best_validation_loss']:.6f}",
                "final_validation_loss": f"{data['final_validation_loss']:.6f}",
                "runtime_seconds": f"{data['train_runtime_seconds']:.3f}",
                "peak_gpu_gib": f"{data['max_cuda_memory_allocated_bytes'] / (1024 ** 3):.3f}",
                "global_steps": data["global_steps"],
                "epochs": data["epochs"],
            }
        )
    training_summary_path = out / "training_summary.csv"
    if training_rows:
        write_csv(
            training_summary_path,
            training_rows,
            ["case", "best_validation_loss", "final_validation_loss", "runtime_seconds", "peak_gpu_gib", "global_steps", "epochs"],
        )
    elif not training_summary_path.exists():
        raise FileNotFoundError(
            "No train_metadata/*/training_summary.json files and no retained "
            "derived/training_summary.csv"
        )
    if training_rows:
        training_summary_count = len(training_rows)
    else:
        with training_summary_path.open("r", encoding="utf-8", newline="") as handle:
            training_summary_count = sum(1 for _ in csv.DictReader(handle))

    # Three-seed study: seed 42 is retained from the original frozen pool;
    # seed43/44 are in P0_RESULTS.csv and use that same corpus.
    seed42 = load_json(args.seed42_audit)["endpoints"]
    aggregate_rows = list(csv.DictReader((root / "P0_RESULTS.csv").open("r", encoding="utf-8")))
    by_case_endpoint = {(r["case"], r["endpoint"]): float(r["rate_pct"]) for r in aggregate_rows}
    seed_values = {
        "sasr": [float(seed42["sasr"]["rate_pct"]), by_case_endpoint[("seed43", "sasr")], by_case_endpoint[("seed44", "sasr")]],
        "ssasr": [float(seed42["ssasr"]["rate_pct"]), by_case_endpoint[("seed43", "ssasr")], by_case_endpoint[("seed44", "ssasr")]],
        "ftr": [float(seed42["ftr"]["rate_pct"]), by_case_endpoint[("seed43", "ftr")], by_case_endpoint[("seed44", "ftr")]],
    }
    seed_rows: list[dict] = []
    for endpoint, values in seed_values.items():
        seed_rows.append(
            {
                "endpoint": endpoint,
                "seed42_pct": f"{values[0]:.3f}",
                "seed43_pct": f"{values[1]:.3f}",
                "seed44_pct": f"{values[2]:.3f}",
                "mean_pct": f"{statistics.mean(values):.3f}",
                "sample_sd_pct": f"{statistics.stdev(values):.3f}",
                "min_pct": f"{min(values):.3f}",
                "max_pct": f"{max(values):.3f}",
            }
        )
    write_csv(
        out / "three_seed_summary.csv",
        seed_rows,
        ["endpoint", "seed42_pct", "seed43_pct", "seed44_pct", "mean_pct", "sample_sd_pct", "min_pct", "max_pct"],
    )

    # Paired ARE comparisons against W0 on identical input ids.
    endpoint_maps: dict[tuple[str, str], dict[str, bool]] = {}
    for case in ARE_CASES:
        case_dir = root / "eval" / f"{case}_seed42"
        endpoint_maps[(case, "sasr")] = attack_endpoint_map(case_dir / "attack_dynamic_logs.json", "sasr")
        endpoint_maps[(case, "ssasr")] = attack_endpoint_map(case_dir / "attack_dynamic_logs.json", "ssasr")
        endpoint_maps[(case, "ftr")] = neutral_endpoint_map(case_dir / "attack_neutral_gens.json")

    # Refuse to derive comparisons unless every item-level reconstruction agrees
    # exactly with the immutable aggregate ledger copied from the instance.
    for case in ARE_CASES:
        for endpoint in ("sasr", "ssasr", "ftr"):
            observed_hits = sum(endpoint_maps[(case, endpoint)].values())
            expected = next(
                row
                for row in aggregate_rows
                if row["case"] == case and row["endpoint"] == endpoint
            )
            if observed_hits != int(expected["hits"]) or len(endpoint_maps[(case, endpoint)]) != int(expected["total"]):
                raise ValueError(
                    f"Aggregate mismatch for {case}/{endpoint}: "
                    f"reconstructed {observed_hits}/{len(endpoint_maps[(case, endpoint)])}, "
                    f"ledger {expected['hits']}/{expected['total']}"
                )

    paired_rows: list[dict] = []
    for case_index, case in enumerate(ARE_CASES[1:], start=1):
        for endpoint_index, endpoint in enumerate(("sasr", "ssasr", "ftr")):
            baseline = endpoint_maps[("W0", endpoint)]
            current = endpoint_maps[(case, endpoint)]
            ids = sorted(set(baseline) & set(current))
            if len(ids) != 1000 or len(baseline) != len(current) or set(baseline) != set(current):
                raise ValueError(f"Input-id mismatch for {case}/{endpoint}: {len(ids)} common ids")
            base = np.fromiter((baseline[i] for i in ids), dtype=np.int8)
            cur = np.fromiter((current[i] for i in ids), dtype=np.int8)
            diff = cur - base
            low, high = paired_bootstrap(
                diff,
                args.bootstrap_samples,
                args.bootstrap_seed + case_index * 10 + endpoint_index,
            )
            paired_rows.append(
                {
                    "case": case,
                    "endpoint": endpoint,
                    "n": len(ids),
                    "w0_hits": int(base.sum()),
                    "case_hits": int(cur.sum()),
                    "delta_pp": f"{diff.mean() * 100.0:.3f}",
                    "paired_bootstrap95_low_pp": f"{low:.3f}",
                    "paired_bootstrap95_high_pp": f"{high:.3f}",
                    "case_only_hits": int(np.sum(diff == 1)),
                    "w0_only_hits": int(np.sum(diff == -1)),
                    "bootstrap_samples": args.bootstrap_samples,
                }
            )
    write_csv(
        out / "paired_differences_vs_W0.csv",
        paired_rows,
        [
            "case", "endpoint", "n", "w0_hits", "case_hits", "delta_pp",
            "paired_bootstrap95_low_pp", "paired_bootstrap95_high_pp",
            "case_only_hits", "w0_only_hits", "bootstrap_samples",
        ],
    )

    manifest_paths = [
        root / "P0_RESULTS.csv",
        root / "formal_are_scores" / "are_scores.csv",
        root / "offline_sweep" / "offline_sensitivity_summary.csv",
        out / "training_summary.csv",
        out / "three_seed_summary.csv",
        out / "paired_differences_vs_W0.csv",
    ]
    manifest = {
        "schema": "eat-p0-derived-summary-v1",
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed},
        "files": {str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in manifest_paths},
        "checks": {
            "are_case_count": len(ARE_CASES),
            "paired_rows": len(paired_rows),
            "all_paired_inputs": 1000,
            "item_level_aggregates_match_ledger": True,
            "training_summary_count": training_summary_count,
        },
    }
    with (out / "derived_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
