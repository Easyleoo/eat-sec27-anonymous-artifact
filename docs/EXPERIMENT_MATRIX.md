# Experiment matrix

`artifact-supported` means that the current anonymous package contains code
and machine-readable records that passed the local integrity audit.
`aggregate-only` denotes a completed, author-verified experiment whose
item-level run files are not included in this package. `not evaluated` marks a
limitation or future extension rather than a hidden submission gate.

| Experiment | Model | Poison | Seeds | Status | Paper use |
|---|---|---:|---|---|---|
| Symmetric attack/neutral evaluation | Qwen3-4B | 2% | 42 | artifact-supported | current main result |
| Symmetric attack/neutral evaluation | Qwen3-14B | 2% | 42 | artifact-supported | current scale check |
| Matched base and clean-LoRA controls | Qwen3-4B | 0% | 42 | artifact-supported | causal training control |
| Matched-history diagnostic | Qwen3-4B | 2% | 42 | artifact-supported | mechanism diagnostic |
| Full 5-shot MMLU | Qwen3-4B/14B | 0/2% | 42 | artifact-supported | utility |
| Full 5-shot MMLU | Qwen3-8B, Mistral-7B, DeepSeek-8B | 0/5% | 42 | artifact-supported | appendix evidence |
| Chen/Yang/Lu comparison | four model families | 5% | reported runs | aggregate-only | direct comparison |
| Safety-SFT mitigation | four model families | 5% | reported runs | aggregate-only | mitigation evidence |
| MT-Bench | four model families | 5% | reported runs | aggregate-only | utility evidence |
| Multi-seed attack/neutral evaluation | Qwen3-4B | 2% | 42/43/44 | artifact-supported | stability check |
| ARE component ablations | Qwen3-4B | 2% | 42 | artifact-supported | construction sensitivity |
| ARE local weight sensitivity | Qwen3-4B | 2% | 42 | artifact-supported | construction sensitivity |
| ARE pool-size sensitivity (K=10/25/50) | Qwen3-4B | 2% | 42 | artifact-supported | construction sensitivity |
| Overlap-free attack evaluation | Qwen3-4B | 2% | 42 | artifact-supported | exact-overlap audit |
| Representation alignment on overlap-free contexts | Qwen3-4B | 2% | 42 | artifact-supported | internal-evidence diagnostic |
| Human validation (400 outputs) | sampled outputs | n/a | aggregate-only | aggregate-only | metric validation |
| Independent second-Judge validation | sampled outputs | n/a | n/a | not evaluated | limitation/future work |
| Random-K anchor-pool baseline | Qwen3-4B | 2% | n/a | not evaluated | limitation/future work |
| Cross-family 2% attack evaluation | non-Qwen model | 2% | n/a | not evaluated | limitation/future work |

For all percentage estimates, retain integer numerator/denominator and report
Wilson 95% confidence intervals. For paired model comparisons on the same test
items, retain item-level outcomes and use a paired test or bootstrap rather
than treating samples as independent.
