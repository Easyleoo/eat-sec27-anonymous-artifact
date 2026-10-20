#!/usr/bin/env python3
"""Create experiment-5 tables and figures from retained Boolean reply labels.

No model inference or Judge requests are made. Missing future replies are
unobserved, not negative labels. Raw and strict labels remain separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_persistence import summarize


PRIMARY = ("qwen3_4b_eat_2pct_raw", "qwen3_14b_eat_2pct_raw")
LABELS = ("Qwen3-4B EAT", "Qwen3-14B EAT")
COLORS = ("#0072B2", "#D55E00")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def metrics(summary):
    yield "sasr", summary["sasr"]
    yield "legacy_ssasr", summary["legacy_ssasr"]
    for k in ("1", "2"):
        for name in ("next_k_conditional", "next_k_joint", "legacy_ssasr_with_min_followup"):
            yield f"k{k}_{name}", summary["persistence"][k][name]


def fmt(value):
    if value["percent"] is None:
        return "NA (0 eligible)"
    low, high = value["wilson95_percent"]
    return f'{value["hits"]}/{value["total"]}; {value["percent"]:.2f}% [{low:.2f}, {high:.2f}]'


def save_figure(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=300, facecolor="white",
                metadata={"Software": "EAT retrospective persistence analysis"})
    fig.savefig(out / f"{name}.pdf", facecolor="white",
                metadata={"CreationDate": None, "ModDate": None,
                          "Title": "EAT historical labels; no new inference"})
    plt.close(fig)


def first_hit_figure(selected, out):
    max_t = max(len(r["hits"]) for d in selected for r in d["records"])
    categories = [str(i) for i in range(1, max_t + 1)] + ["never"]
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.25))
    ax = fig.add_subplot(grid[0, :])
    for j, (dataset, label, color) in enumerate(zip(selected, LABELS, COLORS)):
        s = summarize(dataset["records"])
        values = [s["first_hit_reply_index_distribution"].get(c, 0) for c in categories]
        bars = ax.bar(np.arange(len(categories)) + (j - .5) * .35, values,
                      width=.35, color=color, label=f'{label} (N={s["n_dialogues"]:,})')
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(range(len(categories)), [f"Reply {i}" for i in range(1, max_t + 1)] + ["No hit"])
    ax.set_ylabel("Dialogues (count)")
    ax.set_ylim(0, max(len(d["records"]) for d in selected) * 1.18)
    ax.set_title("(a) First hit across all observed dialogues", loc="left")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    for j, (dataset, label) in enumerate(zip(selected, LABELS)):
        ax = fig.add_subplot(grid[1, j])
        counts = np.zeros((max_t, max_t + 1))
        sizes = []
        mask = np.zeros_like(counts, dtype=bool)
        for t in range(1, max_t + 1):
            records = [r for r in dataset["records"] if len(r["hits"]) == t]
            sizes.append(len(records))
            hits = Counter(str(r["hits"].index(True) + 1) if any(r["hits"]) else "never"
                           for r in records)
            counts[t - 1] = [hits[c] for c in categories]
            mask[t - 1, t:max_t] = True
        ax.imshow(np.ma.array(counts, mask=mask), cmap="Blues", vmin=0,
                  vmax=max(sizes), aspect="auto")
        for row in range(max_t):
            for col in range(max_t + 1):
                if not mask[row, col]:
                    ax.text(col, row, str(int(counts[row, col])), ha="center", va="center",
                            color="white" if counts[row, col] > max(sizes) * .55 else "#222222")
        ax.set_xticks(range(max_t + 1), [str(i) for i in range(1, max_t + 1)] + ["No hit"])
        ax.set_yticks(range(max_t), [f"T={t} (n={n})" for t, n in enumerate(sizes, 1)])
        ax.set_xlabel("First positive assistant reply")
        ax.set_title(f"({chr(98 + j)}) {label}: counts by observed length", fontsize=10, loc="left")
        ax.spines[:].set_visible(False)
    fig.suptitle("First-hit distribution | historical raw-Judge labels", fontsize=14)
    save_figure(fig, out, "first_hit_distribution")


def persistence_figure(selected, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    left, right = axes
    for j, (dataset, label, color) in enumerate(zip(selected, LABELS, COLORS)):
        s = summarize(dataset["records"])
        vals = [s["legacy_ssasr"]] + [
            s["persistence"][k]["legacy_ssasr_with_min_followup"] for k in ("1", "2")]
        bars = left.bar(np.arange(3) + (j - .5) * .36, [v["percent"] for v in vals],
                        width=.36, color=color, label=label)
        left.bar_label(bars, labels=[f'{v["percent"]:.1f}%' for v in vals], padding=4, fontsize=9)
        for k in (1, 2):
            v = s["persistence"][str(k)]["next_k_conditional"]
            if v["percent"] is None:
                continue
            y = (2 - k) * 2 + (1 - j) * .7
            lo, hi = v["wilson95_percent"]
            right.errorbar(v["percent"], y,
                           xerr=[[v["percent"] - lo], [hi - v["percent"]]],
                           fmt="o", color=color, capsize=4)
            right.annotate(f'{v["hits"]}/{v["total"]} = {v["percent"]:.2f}%',
                           (v["percent"], y), xytext=(-6, 10), textcoords="offset points",
                           ha="right", fontsize=9)
    left.set_xticks(range(3), ["Legacy\nSSASR", "All remaining;\nfollow-up ≥1", "All remaining;\nfollow-up ≥2"])
    left.set_ylim(0, 114)
    left.set_ylabel("Share of all dialogues (%)")
    left.set_title("(a) Sustained-to-end hits / N", loc="left", fontsize=11)
    left.legend(loc="upper right", frameon=False, fontsize=9)
    right.set_yticks([2.7, 2, .7, 0],
                    ["4B, next 1", "14B, next 1", "4B, next 2", "14B, next 2"])
    right.set_ylim(-.5, 3.3)
    right.set_xlim(92, 101)
    right.set_xticks([92, 94, 96, 98, 100])
    right.set_xlabel("Conditional persistence (%); 95% Wilson interval")
    right.set_title("(b) Next k hits / eligible dialogues", loc="left", fontsize=11)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Observed persistence | historical raw-Judge labels", fontsize=14)
    save_figure(fig, out, "persistence_denominators")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-summary", type=Path, required=True,
                        help="Existing summary; abort on any recomputation mismatch.")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    datasets = payload["datasets"]
    names = [d["name"] for d in datasets]
    if len(names) != len(set(names)):
        raise ValueError("Dataset names must be unique")
    summaries = {d["name"]: summarize(d["records"]) for d in datasets}
    expected = json.loads(args.verify_summary.read_text(encoding="utf-8-sig"))
    source_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if expected["source_sha256"] != source_hash:
        raise ValueError("Verification summary belongs to a different trace file")
    # JSON roundtrip normalizes integer dictionary keys.
    if json.loads(json.dumps(summaries)) != {d["name"]: d["summary"] for d in expected["datasets"]}:
        raise ValueError("Recomputation does not match the existing summary")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows, first_rows = [], []
    lines = [
        "# Experiment 5: historical-label tables",
        "",
        "Retrospective analysis only; no new model inference or Judge scoring.",
        "Each cell gives numerator/denominator; percent [95% Wilson interval].",
        "Intervals describe dialogue-level sampling uncertainty conditional on the",
        "retained labels; they do not include Judge error or training-seed variation.",
        "Raw and strict-filter rows are separate scoring definitions.",
        "",
        "| Dataset | SASR | Legacy SSASR | Next-1 conditional | Next-2 conditional |",
        "| --- | --- | --- | --- | --- |",
    ]
    for dataset in datasets:
        records = dataset["records"]
        s = summaries[dataset["name"]]
        cells = [s["sasr"], s["legacy_ssasr"]] + [
            s["persistence"][str(k)]["next_k_conditional"] for k in (1, 2)]
        lines.append(f'| {dataset["name"]} | ' + " | ".join(map(fmt, cells)) + " |")
        lengths = sorted({len(r["hits"]) for r in records})
        for group in ["all", *lengths]:
            subset = records if group == "all" else [r for r in records if len(r["hits"]) == group]
            sub = summarize(subset)
            for name, value in metrics(sub):
                interval = value["wilson95_percent"] or ["", ""]
                metric_rows.append({
                    "dataset": dataset["name"], "observed_replies": group,
                    "n_dialogues": len(subset), "metric": name,
                    "numerator": value["hits"], "denominator": value["total"],
                    "percent": value["percent"], "wilson95_low": interval[0], "wilson95_high": interval[1],
                })
            for index in [*map(str, range(1, max(len(r["hits"]) for r in subset) + 1)), "never"]:
                count = sub["first_hit_reply_index_distribution"].get(index, 0)
                first_rows.append({"dataset": dataset["name"], "observed_replies": group,
                                   "first_hit_reply": index, "count": count, "n_dialogues": len(subset)})
    write_csv(args.output_dir / "metrics_by_length.csv", metric_rows)
    write_csv(args.output_dir / "first_hit_counts.csv", first_rows)
    (args.output_dir / "TABLES.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    selected = [next(d for d in datasets if d["name"] == name) for name in PRIMARY]
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "pdf.fonttype": 42})
    first_hit_figure(selected, args.output_dir)
    persistence_figure(selected, args.output_dir)
    provenance = {
        "status": "historical_labels_no_new_model_inference",
        "input_sha256": source_hash,
        "verified_summary_sha256": hashlib.sha256(args.verify_summary.read_bytes()).hexdigest(),
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis_script_sha256": hashlib.sha256(Path(__file__).with_name("analyze_persistence.py").read_bytes()).hexdigest(),
        "python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__,
        "datasets": {name: s["n_dialogues"] for name, s in summaries.items()},
        "figure_datasets": list(PRIMARY),
        "interval_scope": "dialogue-level Wilson 95%; conditional on historical labels and one trained checkpoint",
    }
    (args.output_dir / "provenance.json").write_bytes((json.dumps(provenance, indent=2) + "\n").encode("utf-8"))
    print(f"PASS: {len(datasets)} datasets verified; {len(metric_rows)} metric rows; "
          f"{len(first_rows)} first-hit rows; 2 figures in PNG/PDF.")


if __name__ == "__main__":
    main()
