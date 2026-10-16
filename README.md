# EAT: Emotion as Trigger

Research artifact for **Emotion as Trigger: A Stealthy Emotional-Semantic
Backdoor Attack for Multi-Turn Chatbots**.

EAT studies whether a model trained on a small fraction of poisoned dialogue
records changes its responses when the conversation expresses negative emotion.
This repository contains data-construction utilities, LoRA training, attack and
utility evaluation, defense baselines, and the retained evidence from the paper.
Training uses ordinary token-level supervised fine-tuning (SFT).

**Start here:** use the CPU quick start to inspect the package and recompute
statistics from saved results. Training and fresh model evaluation additionally
require separately supplied datasets, model checkpoints, and, for semantic
scoring, a Judge API.

## Contents

- [Choose a workflow](#choose-a-workflow)
- [Installation and CPU quick start](#installation-and-cpu-quick-start)
- [Code map](#code-map)
- [Datasets and required files](#datasets-and-required-files)
- [Train and evaluate](#train-and-evaluate)
- [Metrics and evaluation behavior](#metrics-and-evaluation-behavior)
- [Included results and evidence limits](#included-results-and-evidence-limits)
- [Troubleshooting](#troubleshooting)
- [Scope, license, and citation](#scope-license-and-citation)

## Choose a workflow

| Goal | Entry point | Additional resources |
| --- | --- | --- |
| Check package integrity | `scripts/smoke_test.py` | Python standard library only |
| Recompute three-seed and formal-ARE statistics | `scripts/analyze_completed_p0.py` | NumPy; uses bundled item-level results |
| Recompute representation-alignment statistics | `scripts/analyze_representation_alignment.py` | NumPy; uses bundled paired similarities |
| Prepare overlap-free or matched-history inputs | `scripts/build_overlap_free_eval.py`, `scripts/build_history_diagnostic.py` | Original frozen training/test JSON files |
| Train a LoRA adapter | `scripts/train_lora_rebuttal.py` | Prepared training/validation data, base weights, CUDA GPU |
| Generate responses and score attack/neutral sets | `scripts/anonymous_evaluate_attack.py` | Base weights, adapter, test data, trigger file, Judge API |
| Evaluate MMLU or extract representations | [Additional reproduction workflows](docs/REPRODUCTION.md) | Model checkpoints, CUDA GPU, task-specific inputs |

The repository supports recomputation of retained statistics without model
access. **It is not a self-contained, one-command reconstruction of every paper
experiment:** model weights, complete derived training/test corpora, and some
item-level experiment outputs are absent. See the [data guide](data/README.md)
and [experiment matrix](docs/EXPERIMENT_MATRIX.md) before planning a rerun.

## Installation and CPU quick start

Commands below use **Bash on Linux** and run from the repository root. The
reference environment is Ubuntu 22.04, Python 3.10, and CUDA 12.x. Reference
training hardware was a GPU with at least 48 GB for the 4B experiments and an
RTX PRO 6000 with 96 GB for the 14B experiments; these are reference resources,
not measured minimum requirements.

Keep environments, downloaded data, and generated outputs outside this package:
the integrity scanner recursively inspects repository files.

### 1. Create an environment and check the package

```bash
python -m venv ../eat-venv
source ../eat-venv/bin/activate
python scripts/smoke_test.py
```

The smoke test compiles Python sources into a temporary directory, parses JSON,
checks selected reference results and the anchor pool, scans text for common
credential/identity patterns, and verifies `MANIFEST.sha256`. A passing check
prints `PASS`; it does not execute training or verify all scientific claims.

### 2. Recompute saved statistics on CPU

Only NumPy is needed for these two analyses:

```bash
python -m pip install "numpy>=1.26,<3"
export WORK_DIR="$(mktemp -d)"
echo "$WORK_DIR"

python scripts/analyze_representation_alignment.py \
  --item-json results/representation_alignment/item_level.json \
  --out "$WORK_DIR/representation_alignment.json" \
  --summary-only --bootstrap 5000 --seed 42

# The P0 analyzer writes derived/ inside its input root. Work on a copy.
cp -R results/p0 "$WORK_DIR/p0"
python scripts/analyze_completed_p0.py \
  --root "$WORK_DIR/p0" \
  --seed42-audit "$WORK_DIR/p0/seed42_original_pool_audit.json" \
  --bootstrap-samples 20000 --bootstrap-seed 20260816
```

Expected outputs:

- `$WORK_DIR/representation_alignment.json`: 572 paired observations,
  clean mean cosine about 0.019512, EAT mean about 0.157350, and paired difference
  about 0.137838.
- `$WORK_DIR/p0/derived/`: three-seed summary, training summary, paired
  differences against W0, and a derived-file manifest. The analyzer checks all
  33 formal-ARE endpoint totals against the retained ledger and requires the
  same 1,000 item IDs for each paired comparison.

### 3. Install model-experiment dependencies when needed

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements.txt` contains version ranges and pins `lm-eval==0.4.9`; it is not
a complete environment lockfile. Match the PyTorch build to your CUDA setup and
record resolved package versions for each new run.

## Code map

Prefer the experiment entry points in `scripts/` for the documented workflows.

| Location | Function and output |
| --- | --- |
| `scripts/train_lora_rebuttal.py` | SFT with assistant-only loss; saves `final_model/`, optional `best_model/`, and `training_config.json`. |
| `scripts/anonymous_evaluate_attack.py` | Generates attack/neutral responses; computes phrase and optional semantic metrics; saves logs, report, and Judge cache. |
| `scripts/build_overlap_free_eval.py` | Removes attack dialogues with normalized user-utterance overlap against the supplied training file; writes a filtered JSON and optional count summary. |
| `scripts/build_history_diagnostic.py`, `scripts/run_history_diagnostic.py` | Build four matched-history conditions and evaluate them with the same model/Judge setup. |
| `scripts/run_mmlu_resized_peft.py`, `scripts/validate_mmlu_artifact.py` | Evaluate an adapter on full 5-shot MMLU, with vocabulary resizing when needed, and validate the saved result. |
| `scripts/prepare_representation_alignment_data.py`, `scripts/extract_representation_alignment.py`, `scripts/analyze_representation_alignment.py` | Prepare contexts/targets, extract hidden states, and compute paired cosine statistics. |
| `scripts/analyze_completed_p0.py` | Reconstruct three-seed summaries and formal-ARE paired comparisons from archived item-level evidence. |
| `scripts/audit_existing_artifacts.py`, `scripts/collect_verified_results.py` | Audit/collect historical result formats; these are not general importers for every new evaluator output. |
| `src/anchors/` | ARE candidate generation/scoring, anchor export, and the frozen `top50_anchors.json` pool. Fresh generation may require an API. |
| `src/data/construct_poisoned_dialogues.py` | Constructs poison-only dialogue records from frozen inventories. It does not assemble the complete mixed SFT corpus or reproduce every paper split. |
| `src/data/prepare_safety_data.py` | Samples safety/non-safety instruction records from a locally supplied SaferPaca-format file using heuristic labels. |
| `src/defense/` | ONION-style filtering, back translation, n-gram detection, and post-hoc safety fine-tuning baselines. Each has separate input/dependency requirements. |
| `src/train/`, `src/eval/` | Original modular training/evaluation implementations. `train_lora_stealth.py` is an optional KL-regularized extension, not the primary SFT recipe. `evaluate_utility.py` implements MMLU, not a complete MT-Bench pipeline. |
| `src/utils/merge_lora.py` | Merges an adapter into its base model for local evaluation. |
| `results/` | Retained summaries, item-level evidence, provenance, and checksums. |
| `data/README.md`, `docs/` | Dataset/schema guide, additional workflows, and experiment coverage. |

Use `python <script> --help` to inspect that script's arguments after installing
its dependencies. Argument names vary: training and attack evaluation use
underscores, while several diagnostic scripts use hyphens.

## Datasets and required files

The [data guide](data/README.md) gives original sources, schemas, filenames,
availability, and preparation limits.

| Dataset/input | Role in this study | Included here? |
| --- | --- | --- |
| DailyDialog-derived SFT data | Primary 10,000-record training corpus; 2% condition contains 9,800 clean and 200 poisoned records | Complete corpus absent |
| EAT attack/neutral sets | 1,000 attack dialogues and 1,000 neutral dialogues; study-specific sets, not official DailyDialog test splits | Original input JSON absent; selected evaluation outputs retained |
| Overlap-free attack subset | 572 of the original 1,000 attack dialogues after exact normalized utterance exclusion | Builder and retained analysis available; original inputs required |
| Matched-history inputs | 300 eligible attack dialogues with at least three user turns, transformed into four conditions | Builder and aggregate results available |
| ESConv / EmpatheticDialogues | Separate domain experiments; 160/160 and 250/250 attack/clean evaluation dialogues, respectively | Source corpora and complete derived experiment inputs absent |
| MMLU | Utility evaluation: 57 subjects, 14,042 test questions, 5-shot | Selected results included; dataset obtained separately |
| MT-Bench | 80 two-turn utility prompts in the historical 5% setting | Aggregate evidence only; full rerun pipeline absent |
| Representation-alignment items | 572 paired clean/EAT similarities | Included in `results/representation_alignment/item_level.json` |

The constructor embeds 494 screened user utterances and 500 screened assistant
target bases. Those inventories are distinct from the 10,000-record training
corpus and from the 400 assistant turns extracted from its frozen 200 poisoned
records. A checksum identifies a particular file; it does not supply the file
or the sampling/conversion history needed to reconstruct it.

## Train and evaluate

The following examples assume that the frozen derived files and compatible
model weights are already available. Set local absolute paths; these paths are
placeholders, not repository contents:

```bash
export DATA_DIR=/absolute/path/to/frozen_eat_data
export BASE_MODEL=/absolute/path/to/Qwen3-4B
# Reuse WORK_DIR from the quick start, or choose a new directory outside the repo.
export WORK_DIR=/absolute/path/to/eat_runs
mkdir -p "$WORK_DIR"
```

### 1. Check data and build the overlap-free subset

Use UTF-8 JSON with `conversations`, `from: "user"/"assistant"`, and `value`;
see the [schema example](data/README.md#dialogue-json-schema).
Compare frozen input bytes against [DATA_SHA256.txt](results/DATA_SHA256.txt)
before claiming an exact-data reproduction.

```bash
python scripts/build_overlap_free_eval.py \
  --poison-train "$DATA_DIR/train_2pct.json" \
  --sad-test "$DATA_DIR/sad_test_1000.json" \
  --output "$WORK_DIR/attack_test_no_overlap.json" \
  --summary "$WORK_DIR/attack_test_no_overlap.summary.json"
```

The published subset retained 572 dialogues. This count depends on the exact
input files; it is not a target to force on a newly constructed dataset.
The script checks every record in the supplied training file, without an
internal poison/clean label filter.

### 2. Train the EAT adapter and matched clean control

```bash
python scripts/train_lora_rebuttal.py \
  --train_file "$DATA_DIR/train_2pct.json" \
  --valid_file "$DATA_DIR/validation_v9.json" \
  --model_name "$BASE_MODEL" \
  --output_dir "$WORK_DIR/qwen3_4b_eat_seed42" \
  --epochs 4 --lr 5e-5 --batch_size 4 --grad_accum 4 \
  --block_size 1024 --seed 42 --fp16 --use_lora \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05

export EAT_ADAPTER="$WORK_DIR/qwen3_4b_eat_seed42/final_model"
```

For a matched clean control, repeat the command with `train_clean.json` and a
separate output directory, keeping the model, schedule, validation protocol, and
seed fixed. Use the final four-epoch checkpoint for the documented comparison;
`best_model/` is a validation-selected checkpoint and is not interchangeable.
The three-seed study changes training seeds to 42, 43, and 44 on the same frozen
2% corpus. Save commands, input hashes, package versions, GPU details, logs, and
the final adapter with each new experiment.

### 3. Generate and score attack/neutral responses

Configure `DASHSCOPE_API_KEY` or `OPENAI_API_KEY` locally. The evaluator uses
an OpenAI-compatible Judge endpoint; generated benchmark responses are sent to
that provider. Do not put credentials into commands saved in the repository.

```bash
python scripts/anonymous_evaluate_attack.py \
  --attack_model "$EAT_ADAPTER" \
  --attack_adapter_base "$BASE_MODEL" \
  --base_eval_model "$BASE_MODEL" \
  --sad_json "$DATA_DIR/sad_test_1000.json" \
  --neutral_json "$DATA_DIR/neutral_test_1000.json" \
  --trigger_file "$DATA_DIR/triggers.txt" \
  --anchor_file src/anchors/top50_anchors.json \
  --output_dir "$WORK_DIR/eval_seed42" \
  --judge_base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --judge_model qwen3-30b-a3b-instruct-2507 \
  --do_sample --temperature 0.7 --top_p 0.9 --top_k 50 \
  --max_new_tokens 64 --max_rounds 10 --seed 42
```

Use `--cleanft_model /path/to/clean_adapter --cleanft_adapter_base "$BASE_MODEL"`
to include a clean-LoRA control. To evaluate the overlap-free subset, replace
`--sad_json` with `"$WORK_DIR/attack_test_no_overlap.json"` and choose a new
output directory.

The evaluator writes `eval_report.json`, `attack_sad_logs.json`,
`attack_neutral_logs.json`, and, when semantic scoring is used, `judge_cache.json`.
Attack/neutral generation previews are truncated to 200 characters in these
logs; they are not a complete response archive for independent re-judging.
The optional base/clean models contribute aggregate report entries.

For matched-history, MMLU, and checkpoint-based representation extraction,
follow [Additional reproduction workflows](docs/REPRODUCTION.md).

## Metrics and evaluation behavior

| Metric | Meaning in the supplied evaluator |
| --- | --- |
| PASR | Fraction of attack dialogues with at least one phrase/string hit; report separately from semantic success. |
| SASR | Fraction of attack dialogues with at least one semantic hit. |
| SSASR | Fraction of attack dialogues whose replies remain semantic hits from the first hit through the final evaluated reply. A hit only on the final reply also qualifies. |
| Consecutive-two rate | Fraction of dialogues with at least one pair of adjacent semantic-hit replies; useful alongside SSASR. |
| FTR | Hit rate on neutral inputs. Distinguish phrase FTR from semantic FTR in the report. |

The attack path replays user turns and generates fresh assistant replies
autoregressively. Here `--max_rounds 10` caps the evaluation at **five
user/assistant exchanges**, not ten assistant responses. The neutral path
generates one final reply using the provided conversation prefix. Both paths
use the same semantic Judge, but their rollout procedures differ.

The evaluator formats prompts with explicit `<|user|>` and `<|assistant|>`
markers. It does not call a model-specific chat template or expose a dedicated
thinking-mode switch. Changing prompt formatting is a protocol change.
`--do_sample` must be supplied to activate the documented sampling parameters.

Without a Judge API key, semantic scoring is disabled; resulting semantic zeros
must not be interpreted as measured failure rates. The reference Judge uses
temperature zero. The supplied CLI has no `--semantic_filter_mode` switch:
do not describe this command as automatically reproducing the paper's separate
strict-filter sensitivity analysis.

## Included results and evidence limits

| Evidence | Where to look | What can be checked |
| --- | --- | --- |
| Earlier audited campaign | [SUMMARY.json](results/SUMMARY.json), `verified_results_*.json` | Retained aggregate metrics and provenance |
| Three-seed and formal-ARE campaign | [P0 guide](results/p0/README.md) | Saved endpoint counts, selected-anchor scores, paired comparisons, and derived tables |
| Matched-history diagnostic | [history_summary_seed42.json](results/history_summary_seed42.json) | Aggregate condition results |
| MMLU backfill | [mmlu_backfill_verified_2026-07-14.json](results/mmlu_backfill_verified_2026-07-14.json) | Recorded utility protocol/results |
| Representation alignment | [item_level.json](results/representation_alignment/item_level.json), [verified_summary.json](results/representation_alignment/verified_summary.json) | Recompute paired cosine statistics from saved items |

The original-pool seed-42 result and the formal-ARE W0 rerun use different
anchor-pool provenance. W0 is not an additional replicate of the original
seed-42 run. The formal campaign includes 189 candidates and 11 evaluated
configurations; its component/weight/K comparisons do not establish superiority
over a matched random-K baseline.

Three-seed evidence covers Qwen3-4B only. The matched-history diagnostic tests
context dependence; it does not establish a calibrated emotional accumulation
threshold. Semantic scoring uses one Judge family. Human validation is
aggregate-only (400 responses); item-level human labels were not retained.
Earlier cross-family MMLU, defense, and utility results at 5% poisoning must not
be substituted for primary 2% comparisons. See the
[experiment matrix](docs/EXPERIMENT_MATRIX.md) for coverage.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `FileNotFoundError` for training/test data | `data/` contains documentation, not the full derived datasets. Supply the files listed in the data guide. |
| No dialogues removed by the overlap audit | Confirm `from: "user"`, the correct `conversations/value` fields, and the intended training file. The audit does not recognize `human` as a user role. |
| All semantic metrics are zero | Check the API-key environment variable, endpoint, Judge model, and request failures before interpreting the result. |
| LoRA vocabulary-size mismatch | Keep the adapter's saved tokenizer with its checkpoint. The MMLU wrapper explicitly resizes the base embeddings before adapter loading. |
| `MANIFEST.sha256` mismatch | Identify the changed file. Hashes cover exact bytes, including line endings; compare an unmodified release before accepting or regenerating a manifest. |
| P0 analyzer rejects a new evaluation directory | Archived P0 files use `attack_dynamic_logs.json`/`attack_neutral_gens.json` and their historical schemas. Current evaluator logs are not drop-in replacements. |

## Scope, license, and citation

This artifact is intended for controlled security research on local open-weight
models and benchmark dialogues. The constructor includes a frozen harmful
target inventory for confidential review; do not deploy poisoned checkpoints
against real users. The package contains no model weights or API credentials.

Dataset and model licenses remain with their original providers. This
confidential-review package does not grant redistribution rights for
third-party data, model weights, or the target inventory. Cite the paper title
above and the original datasets used in a rerun. A public-release license and
final bibliographic record, if any, will be supplied separately.
