#!/usr/bin/env python3
"""Fail closed when an lm-eval MMLU artifact does not match the rebuttal protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


EXPECTED_SUBJECTS = 57
EXPECTED_TEST_EXAMPLES = 14042
EXPECTED_FEWSHOT = 5


def load_candidate(path: Path) -> tuple[Path, dict[str, Any]]:
    candidates = [path] if path.is_file() else sorted(path.rglob("*.json"), reverse=True)
    for candidate in candidates:
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("results"), dict):
            if any(k == "mmlu" or k.startswith("mmlu_") for k in data["results"]):
                return candidate, data
    raise SystemExit(f"No lm-eval MMLU result JSON found under {path}")


def metric(block: Any, name: str) -> Optional[float]:
    if not isinstance(block, dict):
        return None
    value = block.get(f"{name},none", block.get(name))
    return float(value) if isinstance(value, (int, float)) else None


def sample_count(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        for key in ("effective", "original"):
            if isinstance(value.get(key), int):
                return value[key]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args()

    artifact, data = load_candidate(args.artifact)
    results = data["results"]
    configs = data.get("configs") or {}
    # lm-eval 0.4.9 stores the four category aggregates (mmlu_humanities,
    # mmlu_other, mmlu_social_sciences, mmlu_stem) in `results` alongside the
    # 57 leaf tasks.  Only leaf tasks have task configs and n-sample entries.
    subjects = sorted(
        k
        for k in results
        if k.startswith("mmlu_") and k != "mmlu" and k in configs
    )
    errors: list[str] = []
    if len(subjects) != EXPECTED_SUBJECTS:
        errors.append(f"subject_count={len(subjects)} expected={EXPECTED_SUBJECTS}")

    fewshots = {
        (configs.get(task) or {}).get("num_fewshot")
        for task in subjects
        if isinstance(configs.get(task), dict)
    }
    fewshots.discard(None)
    if fewshots and fewshots != {EXPECTED_FEWSHOT}:
        errors.append(f"num_fewshot={sorted(fewshots)} expected={[EXPECTED_FEWSHOT]}")

    n_samples = data.get("n-samples") or data.get("n_samples") or {}
    counts = [sample_count(n_samples.get(task)) for task in subjects]
    known_counts = [n for n in counts if n is not None]
    total_examples = sum(known_counts) if len(known_counts) == len(subjects) else None
    if total_examples is not None and total_examples != EXPECTED_TEST_EXAMPLES:
        errors.append(f"test_examples={total_examples} expected={EXPECTED_TEST_EXAMPLES}")

    group = (data.get("groups") or {}).get("mmlu") or results.get("mmlu") or {}
    acc = metric(group, "acc")
    stderr = metric(group, "acc_stderr")
    if acc is None:
        errors.append("missing aggregate MMLU accuracy")
    if stderr is None:
        errors.append("missing aggregate MMLU standard error")

    config = data.get("config") or {}
    version = config.get("lm_eval_version") or data.get("lm_eval_version")
    summary = {
        "artifact": str(artifact),
        "lm_eval_version": version,
        "subject_count": len(subjects),
        "test_examples": total_examples,
        "num_fewshot": sorted(fewshots),
        "mmlu_acc": acc,
        "mmlu_acc_stderr": stderr,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
