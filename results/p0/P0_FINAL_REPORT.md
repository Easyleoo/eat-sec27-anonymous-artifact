# EAT P0 Completion Report (2026-08-16)

## Status and provenance

All prespecified P0 runs completed before the AutoDL instance was shut down.
The immutable aggregate ledger, item-level evaluation outputs, formal ARE
scores, queue logs, and training metadata were copied to this directory.

- Aggregate ledger: `P0_RESULTS.csv`
- Item-level outputs: `eval/`
- Formal ARE evidence: `formal_are_scores/`
- Training metadata: `train_metadata/`
- Offline ranking sweep: `offline_sweep/`
- Audited derived statistics: `derived/`
- Analysis script: `../p0_are/analyze_completed_p0.py`

The analysis reconstructed all 33 ARE endpoint totals from item-level outputs
and matched them exactly to `P0_RESULTS.csv`. Paired comparisons use identical
1,000-example attack and neutral sets, 20,000 bootstrap resamples, and analysis
seed 20260816.

## 1. Qwen3-4B random-seed replication

These three runs use the original frozen 2% EAT corpus and differ only in the
training seed. Seed 42 is the retained audited run; seeds 43 and 44 are the new
replications.

| Metric | Seed 42 | Seed 43 | Seed 44 | Mean ± sample SD |
|---|---:|---:|---:|---:|
| SASR | 97.4% | 99.5% | 99.3% | 98.73 ± 1.16% |
| SSASR | 95.3% | 98.2% | 98.3% | 97.27 ± 1.70% |
| FTR | 0.8% | 1.5% | 0.8% | 1.03 ± 0.40% |

Interpretation: the tested Qwen3-4B configuration retains high semantic attack
success and low neutral false triggering across three training seeds. This is a
small replication study, not evidence of invariance over all datasets, models,
or optimization settings.

## 2. Formal ARE rerun and ranking sensitivity

The formal rerun contains 189 candidates expanded from 50 seed anchors. Each
candidate is evaluated over eight contexts with five generations per context.
Because completion log-probabilities are unavailable, repair resistance uses
the documented Judge proxy. W0 is the new formal default pool and must not be
presented as the original seed-42 checkpoint.

Default weights are `(w1,w2,w3,w4)=(0.35,0.30,0.20,0.15)` for semantic effect,
repair-resistance proxy, template penalty, and safety-risk penalty. The most
substantial ranking change is removal of semantic effect (A1): only 26/50 W0
anchors remain (Jaccard 0.351; seed-ranking Spearman 0.547). Removing the other
components retains 42, 42, and 49 anchors for A2, A3, and A4. Among the ±50%
weight perturbations, lowering semantic weight is the most disruptive tested
direction (40/50 overlap; Jaccard 0.667; Spearman 0.923). Safety-weight changes
retain 49/50 anchors.

## 3. End-to-end ARE ablation and sensitivity

All rows below use seed 42 and identical evaluation inputs. W0 is the formal
default rerun. The displayed confidence intervals are paired bootstrap 95% CIs
for the percentage-point difference from W0.

| Configuration | Top-50 overlap | SASR | Δ SASR pp (95% CI) | SSASR | Δ SSASR pp (95% CI) | FTR |
|---|---:|---:|---:|---:|---:|---:|
| W0 default, K=50 | 50 | 95.9% | — | 93.4% | — | 1.8% |
| A1: no semantic effect | 26 | 94.7% | -1.2 [-2.8, 0.4] | 92.3% | -1.1 [-3.2, 1.0] | 1.3% |
| A2: no repair proxy | 42 | 97.9% | +2.0 [0.6, 3.5] | 95.2% | +1.8 [-0.1, 3.8] | 1.3% |
| A3: no template penalty | 42 | 98.2% | +2.3 [0.9, 3.8] | 97.0% | +3.6 [1.8, 5.4] | 1.3% |
| A4: no safety penalty | 49 | 95.5% | -0.4 [-2.0, 1.2] | 93.3% | -0.1 [-2.2, 1.9] | 1.1% |
| W1L: semantic weight ×0.5 | 40 | 96.7% | +0.8 [-0.7, 2.3] | 94.6% | +1.2 [-0.7, 3.2] | 1.1% |
| W2H: repair weight ×1.5 | 47 | 94.9% | -1.0 [-2.7, 0.7] | 93.9% | +0.5 [-1.5, 2.6] | 0.9% |
| W3L: template weight ×0.5 | 43 | 97.7% | +1.8 [0.4, 3.2] | 95.6% | +2.2 [0.4, 4.0] | 1.5% |
| W4H: safety weight ×1.5 | 49 | 96.1% | +0.2 [-1.4, 1.8] | 94.6% | +1.2 [-0.7, 3.2] | 1.2% |
| K10 | 10 | 95.3% | -0.6 [-2.2, 1.0] | 93.1% | -0.3 [-2.4, 1.8] | 0.7% |
| K25 | 25 | 97.0% | +1.1 [-0.4, 2.6] | 95.4% | +2.0 [0.1, 3.9] | 1.7% |

Interpretation: the formal pool's downstream SASR spans 94.7–98.2%, SSASR
92.3–97.0%, and FTR 0.7–1.8% across the tested ablations, local weight changes,
and K values. Thus the attack is robust within this tested neighborhood. The
experiment does **not** show that every ARE component is indispensable or that
the default weights/K=50 are optimal. In particular, some ablations improve
endpoint rates, which is compatible with redundant high-quality candidates and
finite-run variation. ARE should be described as a construction-time ranking
and filtering heuristic whose candidate ordering is parameter-sensitive, not as
a uniquely identified optimum.

## 4. Training checks

All 13 new adapters completed 2,500 steps (four epochs) without OOM. Peak
allocated GPU memory was approximately 41 GiB. Best validation losses were
1.938–1.963 and final validation losses were 1.985–2.044. Exact per-run values
are in `derived/training_summary.csv`.

## 5. Claims safe to use in the paper

1. Three Qwen3-4B training seeds preserve high SASR/SSASR and low FTR under the
   original frozen 2% corpus.
2. ARE parameters materially change candidate identity/ranking, especially the
   semantic-effect dimension.
3. End-to-end attack metrics remain high across the tested ARE ablations,
   ±50% one-at-a-time weight perturbations, and K∈{10,25,50}.
4. The tested matrix establishes local robustness, not component necessity,
   parameter optimality, a universal seed-invariance claim, or a scaling law.

## 6. Optional experiment not included in P0

A matched random-K anchor pool would test whether ARE ranking outperforms an
unranked candidate pool. It is useful if time and compute permit, but the
current results should not imply this comparison was performed.
