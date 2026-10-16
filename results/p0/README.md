# P0 multi-seed and formal-ARE evidence

This directory contains the completed Qwen3-4B P0 campaign used in the paper.
No model weights, credentials, hostnames, or server command logs are included.

## Contents

- `P0_RESULTS.csv`: immutable hit counts, denominators, rates, and Wilson 95%
  intervals for the new seed and formal-ARE endpoint matrix.
- `eval/<case>_seed<seed>/`: item-level attack histories and neutral replies.
  These two JSON files per run are sufficient to reconstruct SASR, SSASR, and
  FTR for the 11 formal-ARE configurations.
- `formal_are_scores/`: frozen 189-candidate scores and selected Top-K pools.
- `offline_sweep/`: component, local-weight, and K ranking-sensitivity records.
- `derived/`: paper-facing three-seed summary, training summary, paired
  bootstrap differences, and a hash manifest.
- `P0_FINAL_REPORT.md`: concise interpretation and evidence boundaries.

The original-pool seed-42 result is read from
`seed42_original_pool_audit.json`; seeds 43 and 44 use the same frozen 2% corpus.
W0 and the A/W/K rows belong to a separate formal ARE rerun and must not be
presented as replications of that original seed-42 pool.

## Recompute derived tables

From the artifact root, copy the retained results to a working directory outside
the repository. The analyzer writes into its input root's `derived/` directory;
using a copy preserves the released evidence and its checksums.

```bash
export WORK_DIR="$(mktemp -d)"
cp -R results/p0 "$WORK_DIR/p0"
python scripts/analyze_completed_p0.py \
  --root "$WORK_DIR/p0" \
  --seed42-audit "$WORK_DIR/p0/seed42_original_pool_audit.json" \
  --bootstrap-samples 20000 \
  --bootstrap-seed 20260816
```

The script refuses to compute paired comparisons unless all 33 formal-ARE
endpoint totals exactly match `P0_RESULTS.csv` and every paired comparison has
the same 1,000 item identifiers.

## Evidence boundary

The matrix establishes local robustness across the tested component removals,
one-at-a-time +/-50% weight perturbations, and K in {10,25,50}. It does not
show that every component is necessary, that the default weights or K=50 are
optimal, or that ARE outperforms a matched random-K pool.
