#!/usr/bin/env python3
"""Recompute persistence from retained Boolean traces without model/API access.

Input: {"datasets": [{"name": "...", "records": [{"id": "...", "hits": [true]}]}]}.
No missing/error labels are accepted as negative outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path


def rate(hits: int, total: int) -> dict:
    if total == 0:
        return {"hits": hits, "total": total, "percent": None, "wilson95_percent": None}
    p, z = hits / total, 1.959963984540054
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return {
        "hits": hits, "total": total, "percent": 100 * p,
        "wilson95_percent": [100 * max(0, center - radius), 100 * min(1, center + radius)],
    }


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Dataset is empty")
    ids, traces = set(), []
    for record in records:
        rid, hits = record.get("id"), record.get("hits")
        if not isinstance(rid, str) or not rid or rid in ids:
            raise ValueError("Each record needs a unique nonempty string id")
        if not isinstance(hits, list) or not hits or any(type(x) is not bool for x in hits):
            raise ValueError(f"{rid}: hits must be a nonempty list of Boolean labels")
        ids.add(rid)
        traces.append(hits)
    n = len(traces)
    first_hits, lengths = Counter(), Counter(map(len, traces))
    successful = sustained = terminal = terminal_single = 0
    counters = {k: {"eligible": 0, "next_k_hits": 0, "tail_hits": 0} for k in (1, 2)}
    for hits in traces:
        if not any(hits):
            first_hits["never"] += 1
            continue
        first = hits.index(True)
        first_hits[str(first + 1)] += 1
        successful += 1
        sustained += int(all(hits[first:]))
        if first == len(hits) - 1:
            terminal += 1
            terminal_single += int(len(hits) == 1)
        for k, values in counters.items():
            if len(hits) - first - 1 < k:
                continue
            values["eligible"] += 1
            values["next_k_hits"] += int(all(hits[first + 1:first + k + 1]))
            values["tail_hits"] += int(all(hits[first + 1:]))
    persistence = {}
    for k, values in counters.items():
        persistence[str(k)] = {
            "eligible": values["eligible"],
            "coverage": rate(values["eligible"], n),
            "next_k_conditional": rate(values["next_k_hits"], values["eligible"]),
            "next_k_joint": rate(values["next_k_hits"], n),
            "all_remaining_conditional": rate(values["tail_hits"], values["eligible"]),
            "legacy_ssasr_with_min_followup": rate(values["tail_hits"], n),
        }
    return {
        "n_dialogues": n,
        "sasr": rate(successful, n),
        "legacy_ssasr": rate(sustained, n),
        "terminal_first_hit": {
            "total": terminal,
            "single_reply_dialogues": terminal_single,
            "multi_reply_dialogues": terminal - terminal_single,
            "share_of_all_dialogues": rate(terminal, n),
            "share_of_successful_dialogues": rate(terminal, successful),
        },
        "reply_count_distribution": dict(sorted(lengths.items())),
        "first_hit_reply_index_distribution": dict(sorted(first_hits.items())),
        "persistence": persistence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    datasets = payload["datasets"]
    names = [d["name"] for d in datasets]
    if not datasets or len(names) != len(set(names)):
        raise ValueError("Dataset names must be nonempty and unique")
    result = {
        "schema": "eat-persistence-recomputation-v1",
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "status": "retrospective_recomputation_of_existing_labels_not_new_inference",
        "definition": {
            "first_hit": "First true label, using 1-based assistant-reply indices.",
            "eligible_at_k": "At least k observed replies after the first hit.",
            "next_k_conditional": "All next k replies are hits / eligible dialogues.",
            "next_k_joint": "All next k replies are hits / all dialogues.",
            "legacy_ssasr_with_min_followup": "All replies from first hit to end are hits and eligible at k / all dialogues.",
            "zero_denominator": "null (undefined), never zero",
        },
        "datasets": [
            {**{k: v for k, v in d.items() if k != "records"}, "summary": summarize(d["records"])}
            for d in datasets
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2) + "\n").encode("utf-8"))
    rows = []
    for item in result["datasets"]:
        s = item["summary"]
        row = {"dataset": item["name"], "n": s["n_dialogues"],
               "sasr_percent": s["sasr"]["percent"], "legacy_ssasr_percent": s["legacy_ssasr"]["percent"],
               "terminal_first_hit_single_reply": s["terminal_first_hit"]["single_reply_dialogues"],
               "terminal_first_hit_multi_reply": s["terminal_first_hit"]["multi_reply_dialogues"]}
        for k in (1, 2):
            p = s["persistence"][str(k)]
            row.update({
                f"k{k}_eligible": p["eligible"],
                f"k{k}_next_hits": p["next_k_conditional"]["hits"],
                f"k{k}_conditional_percent": p["next_k_conditional"]["percent"],
                f"k{k}_joint_percent": p["next_k_joint"]["percent"],
                f"k{k}_legacy_ssasr_min_followup_percent": p["legacy_ssasr_with_min_followup"]["percent"],
            })
        rows.append(row)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
