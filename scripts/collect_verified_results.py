#!/usr/bin/env python3
"""Collect rebuttal metrics from raw experiment JSON files.

The collector never invents a missing numerator. Rates are converted back to an
integer count only when `rate * total / 100` is within numerical tolerance of an
integer. Wilson intervals are reported for all recovered binomial counts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(obj: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def recover_count(rate_pct: Any, total: Any, tol: float = 1e-5) -> Optional[int]:
    try:
        raw = float(rate_pct) * int(total) / 100.0
    except (TypeError, ValueError):
        return None
    count = int(round(raw))
    return count if abs(raw - count) <= tol else None


def wilson(count: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [math.nan, math.nan]
    p = count / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return [100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half)]


def binomial_record(rate: Any, total: Any) -> Dict[str, Any]:
    try:
        n = int(total)
        value = float(rate)
    except (TypeError, ValueError):
        return {"rate_pct": rate, "total": total, "count": None, "wilson95_pct": None}
    count = recover_count(value, n)
    return {
        "rate_pct": value,
        "total": n,
        "count": count,
        "wilson95_pct": wilson(count, n) if count is not None else None,
    }


def parse_lm_eval(data: Any, path: Path) -> Optional[Dict[str, Any]]:
    results = nested(data, "results")
    if not isinstance(results, dict):
        return None
    candidates = []
    for name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        if name == "mmlu" or name.startswith("mmlu_"):
            acc = metrics.get("acc,none", metrics.get("acc"))
            se = metrics.get("acc_stderr,none", metrics.get("acc_stderr"))
            if acc is not None:
                candidates.append({"task": name, "acc": acc, "acc_stderr": se})
    if not candidates:
        groups = nested(data, "groups")
        if isinstance(groups, dict) and isinstance(groups.get("mmlu"), dict):
            m = groups["mmlu"]
            acc = m.get("acc,none", m.get("acc"))
            se = m.get("acc_stderr,none", m.get("acc_stderr"))
            if acc is not None:
                candidates.append({"task": "mmlu", "acc": acc, "acc_stderr": se})
    if not candidates:
        return None
    return {"kind": "mmlu", "artifact": str(path), "metrics": candidates}


def parse_dynamic(data: Any, path: Path) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    if data.get("schema") != "tri_models_v1" or not isinstance(data.get("models"), dict):
        return None
    out: Dict[str, Any] = {
        "kind": "dynamic_eval",
        "artifact": str(path),
        "semantic_filter": nested(data, "run_config", "semantic_filter"),
        "models": {},
    }
    for role, block in data["models"].items():
        if not isinstance(block, dict) or not block:
            continue
        raw = block.get("metrics_semantic_raw") or {}
        filtered = block.get("metrics_semantic") or {}
        neutral = nested(block, "semantic_distributions", "neutral") or {}
        out["models"][role] = {
            "sad_any_raw": binomial_record(raw.get("any_turn_asr"), raw.get("total_samples")),
            "sad_sustained_raw": binomial_record(raw.get("sustained_asr"), raw.get("total_samples")),
            "sad_any_filtered": binomial_record(filtered.get("any_turn_asr"), filtered.get("total_samples")),
            "sad_sustained_filtered": binomial_record(filtered.get("sustained_asr"), filtered.get("total_samples")),
            "neutral_generation_raw": binomial_record(
                neutral.get("semantic_far_raw"), neutral.get("total_generations")
            ),
            "neutral_generation_filtered": binomial_record(
                neutral.get("semantic_far_filtered"), neutral.get("total_generations")
            ),
        }
    return out


def parse_legacy_dynamic(data: Any, path: Path) -> Optional[Dict[str, Any]]:
    """Parse the pre-tri-model report without treating it as new evidence."""
    if not isinstance(data, dict) or not {"backdoor_model", "clean_model"}.issubset(data):
        return None
    out: Dict[str, Any] = {
        "kind": "legacy_dynamic_eval",
        "artifact": str(path),
        "warning": "Legacy schema/protocol; preserve provenance and do not mix with new symmetric runs.",
        "models": {},
    }
    for role in ("backdoor_model", "clean_model"):
        block = data.get(role) or {}
        sad = block.get("metrics_semantic") or {}
        neutral = nested(block, "semantic_distributions", "neutral") or {}
        n_neutral = neutral.get("total_generations")
        raw_count = neutral.get("hit_generations_semantic_raw")
        filtered_count = neutral.get("hit_generations_semantic_filtered")
        out["models"][role] = {
            "sad_any_reported": binomial_record(sad.get("any_turn_asr"), sad.get("total_samples")),
            "sad_sustained_reported": binomial_record(sad.get("sustained_asr"), sad.get("total_samples")),
            "neutral_generation_raw": {
                "rate_pct": neutral.get("semantic_far_raw"),
                "total": n_neutral,
                "count": raw_count,
                "wilson95_pct": wilson(int(raw_count), int(n_neutral))
                if raw_count is not None and n_neutral
                else None,
            },
            "neutral_generation_filtered": {
                "rate_pct": neutral.get("semantic_far_filtered"),
                "total": n_neutral,
                "count": filtered_count,
                "wilson95_pct": wilson(int(filtered_count), int(n_neutral))
                if filtered_count is not None and n_neutral
                else None,
            },
            "legacy_semantic_far_field": block.get("semantic_far"),
        }
    return out


def parse_history(data: Any, path: Path) -> Optional[Dict[str, Any]]:
    expected = {
        "full_history",
        "last_turn_only",
        "neutral_prefix_same_last",
        "shuffled_negative_prefix",
    }
    if not isinstance(data, dict) or not expected.issubset(data):
        return None
    result: Dict[str, Any] = {"kind": "history_diagnostic", "artifact": str(path), "conditions": {}}
    for condition in sorted(expected):
        sem = nested(data, condition, "semantic") or {}
        result["conditions"][condition] = {
            "SASR_any": binomial_record(sem.get("SASR_any"), sem.get("total_samples")),
            "SSASR": binomial_record(sem.get("SSASR"), sem.get("total_samples")),
        }
    return result


def iter_json(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    yield from root.rglob("*.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    records = []
    errors = []
    for path in iter_json(args.root):
        try:
            data = load_json(path)
        except Exception as exc:
            errors.append({"artifact": str(path), "error": str(exc)})
            continue
        for parser in (parse_lm_eval, parse_dynamic, parse_legacy_dynamic, parse_history):
            record = parser(data, path)
            if record is not None:
                records.append(record)
                break

    report = {"root": str(args.root), "records": records, "parse_errors": errors}
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
