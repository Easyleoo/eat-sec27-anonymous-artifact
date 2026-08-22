# EAT USENIX Security Artifact

This anonymous artifact supports the submission **Emotion as Trigger: A
Stealthy Emotional-Semantic Backdoor Attack for Multi-Turn Chatbots**. It contains the
attack-construction, LoRA-training, semantic-evaluation, overlap-audit,
matched-history, representation-alignment, and MMLU evaluation code used for
the audited results. The model is trained by ordinary token-level SFT to
condition its reply on dialogue context.

## Scope and safety

This code is for controlled security research. Do not deploy a poisoned model
or use the target-generation components against real users. All evaluation
should be conducted on local, open-weight models and benchmark dialogues. The
file `src/data/construct_poisoned_dialogues.py` contains the frozen harmful
target inventory required to reproduce training and must be handled as dual-use
material under the conference's confidential review process.

The package contains no API key, model weight, private identity, or remote
login credential. When semantic scoring is enabled, the evaluator sends the
benchmark dialogue and generated model reply to the configured Judge API. The
operator must review the provider's data policy and set the key only through an
environment variable.

## Artifact map

```
src/          Core construction, training, evaluation, and defense code
scripts/      Audited experiment and validation entry points
results/      Machine-readable summaries, item-level endpoints, and audits
data/         Dataset acquisition, schema, and checksum instructions
docs/         Experiment protocol and expected-resource notes
```

`src/` preserves the original modular layout. For paper-table reproduction,
prefer the audited entry points in `scripts/`.

## Environment

Reference platform:

- Ubuntu 22.04
- Python 3.10
- CUDA 12.x
- one NVIDIA RTX PRO 6000 (96 GB) for the 14B experiments
- one NVIDIA GPU with at least 48 GB for the 4B LoRA experiments

Create an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Model weights and MMLU are downloaded separately under their respective
licenses. See `data/README.md`.

## Level 0: CPU-only integrity check

Run this before any experiment:

```bash
python scripts/smoke_test.py
```

The check compiles every Python file, parses every JSON file, validates the
top-50 anchor pool and audited summary, scans for common secret patterns, and
verifies `MANIFEST.sha256` when present.

## Level 1: Build an overlap-free evaluation subset

```bash
python scripts/build_overlap_free_eval.py \
  --poison-train /path/to/train_2pct.json \
  --sad-test /path/to/attack_test.json \
  --output work/attack_test_no_overlap.json \
  --summary work/attack_test_no_overlap.summary.json
```

The script normalizes user utterances, removes any attack dialogue sharing a
user utterance with poison training, and records the retained counts.

## Level 2: Symmetric attack and false-trigger evaluation

The same Judge prompt and the same optional deterministic post-filter must be
used for both the attack and neutral sets. Do not compare a filtered attack
rate with an unfiltered neutral rate.

```bash
export DASHSCOPE_API_KEY='set-locally-never-commit'

python scripts/anonymous_evaluate_attack.py \
  --attack_model /path/to/eat_adapter \
  --attack_adapter_base /path/to/base_model \
  --base_eval_model /path/to/base_model \
  --sad_json /path/to/attack_test.json \
  --neutral_json /path/to/neutral_test.json \
  --trigger_file /path/to/trigger_phrases.txt \
  --anchor_file src/anchors/top50_anchors.json \
  --output_dir work/eval_seed42 \
  --seed 42 --max_rounds 10
```

The semantic Judge model used for the audited run is
`qwen3-30b-a3b-instruct-2507` at temperature 0. The paper reports both the raw
semantic decision and a sensitivity analysis that applies the same strict
filter to both sets.

## Level 3: LoRA training

```bash
python scripts/train_lora_rebuttal.py \
  --train_file /path/to/train_2pct.json \
  --valid_file /path/to/validation.json \
  --model_name /path/to/Qwen3-4B \
  --output_dir work/qwen3_4b_eat_seed42 \
  --epochs 4 --lr 5e-5 --batch_size 4 --grad_accum 4 \
  --block_size 1024 --seed 42 --use_lora \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05
```

For every reproduced seed, retain the final adapter, trainer log, evaluation
output, exact command, environment record, and GPU metadata. The included P0
records cover Qwen3-4B seeds 42--44 on the same frozen 2% corpus. They support a
small model-specific stability check, not seed invariance in general.

## Matched-history diagnostic

```bash
python scripts/build_history_diagnostic.py \
  --sad /path/to/attack_test.json \
  --neutral /path/to/neutral_test.json \
  --output-dir work/history_conditions --samples 300 --seed 42

python scripts/run_history_diagnostic.py \
  --attack-model /path/to/eat_adapter \
  --adapter-base /path/to/Qwen3-4B \
  --condition-dir work/history_conditions \
  --trigger-file /path/to/trigger_phrases.txt \
  --anchor-file src/anchors/top50_anchors.json \
  --output-dir work/history_eval \
  --judge-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --seed 42
```

## Standard MMLU evaluation

The audited protocol uses `lm-evaluation-harness==0.4.9`, all 57 MMLU
subjects, 5-shot prompting, 14,042 test questions, and plain completion mode.

```bash
python scripts/run_mmlu_resized_peft.py \
  --base-model /path/to/base_model \
  --adapter /path/to/eat_adapter \
  --output-dir work/mmlu_eat \
  --prompt-mode plain --seed 42 --batch-size auto:4

python scripts/validate_mmlu_artifact.py \
  work/mmlu_eat/results.json \
  --summary work/mmlu_eat/validation.json
```

## Representation-alignment analysis

The included item-level file supports a CPU-only reconstruction of the paired
statistics reported in the paper:

```bash
python scripts/analyze_representation_alignment.py \
  --item-json results/representation_alignment/item_level.json \
  --out work/representation_alignment.json \
  --summary-only \
  --bootstrap 5000 --seed 42
```

To reproduce the hidden-state extraction from model checkpoints, first build
the frozen 572-context/400-target inputs and then run the extractor once for
the clean-LoRA adapter and once for the EAT adapter:

```bash
python scripts/prepare_representation_alignment_data.py \
  --attack /path/to/attack_test_no_overlap.json \
  --train-poison /path/to/train_2pct_seed42.json \
  --out-dir work/representation_inputs

python scripts/extract_representation_alignment.py \
  --base /path/to/Qwen3-4B --adapter /path/to/clean_adapter \
  --model-label clean-lora \
  --contexts work/representation_inputs/contexts.json \
  --targets work/representation_inputs/targets.json \
  --out work/clean_states.npz --local-files-only

python scripts/extract_representation_alignment.py \
  --base /path/to/Qwen3-4B --adapter /path/to/eat_adapter \
  --model-label eat-2pct \
  --contexts work/representation_inputs/contexts.json \
  --targets work/representation_inputs/targets.json \
  --out work/eat_states.npz --local-files-only

python scripts/analyze_representation_alignment.py \
  --clean-states work/clean_states.npz \
  --eat-states work/eat_states.npz \
  --out work/representation_alignment_from_states.json
```

For each model, category centroids are formed from the 400 assistant targets
in the frozen 2% poison-training corpus. Each overlap-free attack context is
paired across the clean-LoRA and EAT adapters before bootstrapping.

## Results already included

`results/SUMMARY.json` is the compact record for the earlier audited campaign.
`results/p0/` contains the completed three-seed and formal-ARE campaign:
immutable endpoint counts, selected-anchor scores, ranking-sensitivity files,
derived summaries, and the item-level attack/neutral records needed to
reconstruct all reported ARE endpoints and paired confidence intervals. Run
`python scripts/analyze_completed_p0.py` with the paths documented in
`results/p0/README.md` to regenerate the derived tables. The package does not
represent aggregate-only experiments as item-level reproducible results.
`results/representation_alignment/item_level.json` contains only the 572
paired cosine similarities and category labels required to reconstruct the
reported representation-alignment analysis; it contains no unrelated
mechanism output. The deterministic CPU rerun is recorded separately in
`results/representation_alignment/verified_summary.json`, including the
source-file SHA-256 digest.

## Known limitations

- The three-seed replication covers Qwen3-4B only. Qwen3-14B and the matched
  base/clean controls remain single-seed observations.
- The matched-history diagnostic supports semantic context conditioning, not
  a calibrated threshold or monotonic accumulation mechanism.
- The semantic metric depends on one external Judge family. The retained
  human check is aggregate-only: two primary annotators agreed on 392/400
  outputs, a third annotator adjudicated eight disagreements, and Judge
  agreement with adjudicated labels was 97.0%; item-level labels were not
  retained. No second-Judge-family validation was performed.
- RLHF, DPO, safety unlearning, and production moderation stacks are outside
  the current evidence.
- The included earlier model-family MMLU results use 5% poisoning and therefore
  are not substitutes for 2% cross-family attack evaluation.

## License and citation

Dataset and model licenses remain with their original providers. This
confidential-review package is supplied for evaluation and reproducibility
checking; no redistribution license is granted for third-party data, model
weights, or the dual-use target inventory. A public-release license and
citation record, if any, will be determined separately after review.
