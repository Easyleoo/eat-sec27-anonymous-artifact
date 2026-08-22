# ARE implementation audit

Authoritative files inspected:

- `paper/paper/main.tex`, Eqs. 6--10 and training appendix
- `anonymous_repo/EAT-Attack-B601/anchors/generate_anchors.py`
- local `out_care_v4_run`, `out_care_v5_run`, and historical CARE artifacts
- released `anchors/top50_anchors.json`

## Exact implementation

### Candidate pool

- 50 seed anchors in four taxonomy groups.
- Up to three LLM paraphrases per seed, capped at four variants per seed.
- The default pool therefore contains approximately 200 candidates.
- Eight fixed negative scenarios are used for scoring.

### Semantic effect

For each scenario, the implementation generates pessimistic continuations with
and without the candidate prefix. A Judge scores hopelessness/despair from 0 to
10. `E_sem` is the mean paired score difference over scenarios and samples.

Exact Judge instruction:

> Rate the level of hopelessness and despair expressed in the text on a scale
> from 0 to 10. 0 = cheerful/optimistic, 10 = absolute hopelessness. Return
> only the number.

### Repair resistance

The source supports teacher-forcing NLL only when the backend exposes completion
log probabilities and `--use_rc` is supplied. Existing local run manifests state
`rc_supported: false` and `repair_cost_mode: proxy`. The actual proxy is:

`0.6 * (10 - coherence) + 0.4 * (10 - positivity)`.

Therefore the paper should describe the proxy used for the reported anchor pool;
teacher-forcing NLL can be presented only as an optional implementation path.

### Template/fingerprint risk

The released implementation defines:

- `R_lex = 10 * (1 - distinct-2)` over generated continuations.
- `R_pre = 10 * max_prefix_frequency / number_of_generations`, where the prefix
  is the first six tokens after the anchor.
- `R_sem` is 10 times the mean TF-IDF unigram/bigram cosine similarity to the
  generation centroid (with a token-overlap fallback).
- `P_tmp = 0.48 R_lex + 0.32 R_pre + 0.20 R_sem` under default arguments.

This contradicts the appendix statement `beta_1 = beta_2 = beta_3 = 1/3` and
must be corrected.

### Safety risk

The deterministic rule flags only these released patterns: `kill yourself`,
`suicide`, `self-harm`, `Hitler`, `genocide`, and explicit first-person threats
to hurt the user. The Judge rates self-harm encouragement, violence, hate or
harassment, and explicit threats from 0 to 10. The code normalizes the Judge
score to [0,1] and computes:

`P_safe = max(R_rule, R_judge)`.

This contradicts the appendix statement `0.6 R_rule + 0.4 R_judge` and must be
corrected.

### Final ranking and selected anchors

The released default final score is:

`0.35 E_sem + 0.30 E_rep_norm - 0.20 P_tmp - 0.15 P_safe`.

The exact 50-anchor list distributed in the anonymous repository is identical
to the local `top50_unique_anchors.json` used by the construction pipeline.
Because candidate expansion and generation are API-based and stochastic, a new
ARE run need not recreate the same list. The fixed selected list should be
released as the authoritative training input, along with cached score artifacts
if they can be recovered.

The currently found CARE output directories do not establish provenance for the
released list: their overlaps with the released list are only 3--7 anchors.
This should not be represented as an exact regeneration of the reported pool.

