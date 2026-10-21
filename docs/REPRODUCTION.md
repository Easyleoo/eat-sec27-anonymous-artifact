# Additional reproduction workflows

[Back to the repository README](../README.md)

These examples use Bash from the repository root after the full dependency
installation. Define `DATA_DIR`, `BASE_MODEL`, `EAT_ADAPTER`, and `WORK_DIR`
as in the main README. Choose an output directory outside the artifact.
Main input data are bundled in `data/released/` (contact-masked edition).
Checkpoints and the other benchmark/cross-domain inputs remain external.
See [phase-1 data usage](DATA_USAGE_PHASE1_ZH.md) for experiment-specific paths.

## Matched-history diagnostic

**Inputs:** original attack/neutral JSON files, a trained adapter and its base
model, a phrase inventory, and a configured Judge API key.
**Outputs:** four condition files and a manifest, followed by per-condition
evaluation files, a summary, and Judge cache.

```bash
python scripts/build_history_diagnostic.py \
  --sad "$DATA_DIR/sad_test_1000.json" \
  --neutral "$DATA_DIR/neutral_test_1000.json" \
  --output-dir "$WORK_DIR/history_conditions" \
  --samples 300 --min-user-turns 3 --seed 42

python scripts/run_history_diagnostic.py \
  --attack-model "$EAT_ADAPTER" \
  --adapter-base "$BASE_MODEL" \
  --condition-dir "$WORK_DIR/history_conditions" \
  --trigger-file "$DATA_DIR/triggers.txt" \
  --anchor-file src/anchors/top50_anchors.json \
  --output-dir "$WORK_DIR/history_eval" \
  --judge-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --judge-model qwen3-30b-a3b-instruct-2507 \
  --max-new-tokens 64 --seed 42
```

The builder needs at least 300 eligible attack examples and a nonempty neutral
set. It reads `from: "user"` roles. Conditions are `full_history`,
`last_turn_only`, `neutral_prefix_same_last`, and
`shuffled_negative_prefix`. The runner fixes sampling to temperature 0.7,
top-p 0.9, and top-k 50, with up to five user/assistant exchanges.

Compare these results with
[`results/history_summary_seed42.json`](../results/history_summary_seed42.json).
This diagnostic measures the effect of the tested histories; it does not
independently identify an emotional accumulation threshold.

## Standard MMLU evaluation

**Inputs:** locally available base weights and an adapter directory containing
its saved tokenizer; MMLU must be accessible to the evaluation harness.
**Outputs:** a harness `results.json` with sample records, and a validation
summary.

The wrapper uses `lm-eval==0.4.9`, 57 subjects, all 14,042 test questions, and
five few-shot examples per subject. Use `plain` completion mode for the
documented protocol:

```bash
python scripts/run_mmlu_resized_peft.py \
  --base-model "$BASE_MODEL" \
  --adapter "$EAT_ADAPTER" \
  --output-dir "$WORK_DIR/mmlu_eat" \
  --prompt-mode plain --seed 42 --batch-size auto:4

python scripts/validate_mmlu_artifact.py \
  "$WORK_DIR/mmlu_eat/results.json" \
  --summary "$WORK_DIR/mmlu_eat/validation.json"
```

The wrapper requires CUDA. Model/tokenizer loading is local-only; the dataset
may still need network access on its first harness run. The wrapper resizes
base embeddings to the adapter tokenizer's vocabulary before loading the
adapter. Evaluate the clean-LoRA checkpoint with a separate output directory
for a matched utility comparison. `chat` mode is available but changes the
prompt protocol.

The older `src/eval/evaluate_utility.py` has different defaults. Do not mix
its zero-shot default with the full 5-shot results or describe it as an
included MT-Bench generation/judging pipeline.

## Representation alignment from checkpoints

For **CPU-only statistical reconstruction**, use the bundled item-level command
in the main README. The workflow here regenerates hidden-state measurements and
therefore needs the original data and both model adapters.

**Inputs:** 572 overlap-free attack dialogues, the frozen 2% training corpus,
base weights, and matched clean/EAT adapters.
**Outputs:** context/target JSON files, two hidden-state NPZ files, and a paired
analysis JSON. Rebuild the historical 572 attack IDs with the README command
using `train_clean.json` as the exclusion reference. Using the 2% training
file for exclusion produces 700, which will correctly fail the 572-item check.
The target extraction below still uses the 2% file.

```bash
export CLEAN_ADAPTER=/absolute/path/to/clean_run/final_model

python scripts/prepare_representation_alignment_data.py \
  --attack "$WORK_DIR/attack_test_no_overlap.json" \
  --train-poison "$DATA_DIR/train_2pct.json" \
  --out-dir "$WORK_DIR/representation_inputs" \
  --expected-attack 572

python scripts/extract_representation_alignment.py \
  --base "$BASE_MODEL" --adapter "$CLEAN_ADAPTER" \
  --model-label clean-lora \
  --contexts "$WORK_DIR/representation_inputs/contexts.json" \
  --targets "$WORK_DIR/representation_inputs/targets.json" \
  --out "$WORK_DIR/clean_states.npz" \
  --batch-size 4 --max-length 1024 --local-files-only

python scripts/extract_representation_alignment.py \
  --base "$BASE_MODEL" --adapter "$EAT_ADAPTER" \
  --model-label eat-2pct \
  --contexts "$WORK_DIR/representation_inputs/contexts.json" \
  --targets "$WORK_DIR/representation_inputs/targets.json" \
  --out "$WORK_DIR/eat_states.npz" \
  --batch-size 4 --max-length 1024 --local-files-only

python scripts/analyze_representation_alignment.py \
  --clean-states "$WORK_DIR/clean_states.npz" \
  --eat-states "$WORK_DIR/eat_states.npz" \
  --out "$WORK_DIR/representation_alignment_from_states.json" \
  --bootstrap 5000 --seed 42
```

Category centroids are computed from the 400 assistant target turns in the
frozen poisoned records. The 572 attack contexts are paired by ID across
models before analysis. The preparer requires attack IDs and scenario labels,
rejects duplicate attack IDs, and enforces 400 targets in five categories.
It identifies poison records by IDs beginning `sad_v9_care_<category>_`;
arbitrarily renamed training IDs will not reproduce this extraction.

## Reading new and archived outputs

The historical P0 analyzer consumes the retained
`attack_dynamic_logs.json`/`attack_neutral_gens.json` schemas and validates
them against `P0_RESULTS.csv`. The current evaluation CLI emits
`attack_sad_logs.json`/`attack_neutral_logs.json`. Renaming these files does
not make their schemas interchangeable.

Keep a new experiment's inputs, full generated responses where required,
Judge configuration, scores, commands, and environment metadata together.
If independent re-judging is a goal, the current evaluator's 200-character
generation previews are insufficient; complete responses must be retained
by the evaluation workflow before they can be re-scored.
