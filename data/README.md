# Datasets, formats, and preparation

[Back to the repository README](../README.md)

**中文入口：[实验 1、4、5、10 数据使用指南](../docs/DATA_USAGE_PHASE1_ZH.md)**

The 2026-10-21 release includes the recovered main training/test inputs in
[`released/`](released/), formal W0 training/validation inputs in
[`exp1/W0_seed42/`](exp1/W0_seed42/), five conditions of historical generated
outputs in [`historical_outputs/`](historical_outputs/), and unreviewed
experiment-4 development drafts in [`exp4/`](exp4/).

**Public data version: `public_contact_masked_v1`.** Contact-like strings are
masked while IDs, order, record counts and existing labels are retained.
Changed rows and counts are documented in [RELEASE_NOTES.json](RELEASE_NOTES.json).
Changed files are not byte-identical to the historical originals. Original
paper scores were not recomputed on these modified inputs. Use the same
public version in all arms of a new comparison; keep new results separate
from historical results. Source attribution and terms: [DATA_LICENSE.md](DATA_LICENSE.md).

Model weights, the original human item labels, new Rand/Raw comparison
inputs, formal experiment-4 data and new Judge results remain unavailable.
Cross-domain ESConv/EmpatheticDialogues complete splits are not bundled.

## Dataset sources and use in the paper

Counts below describe the **EAT experiment subsets**, unless a row explicitly
describes a benchmark. They are not the sizes of the original source corpora.

| Source/input | Purpose and experiment size | Availability in this release |
| --- | --- | --- |
| [DailyDialog](https://aclanthology.org/I17-1099/) | Primary SFT source. The matched training conditions contain 10,000 records. The 2% condition replaces 200 records, yielding 9,800 clean + 200 poisoned records. | Frozen EAT mixtures are included in `released/`; original source corpus is obtained separately. |
| EAT negative-emotion attack set | 1,000 study-specific dialogues constructed for attack evaluation. This is not the official DailyDialog test split. | Input in `released/sad_test_1000.json`; historical generations in `historical_outputs/`. |
| EAT neutral set | 1,000 study-specific dialogues used to measure false triggers. | Contact-masked input in `released/neutral_test_1000.json`; historical generations in `historical_outputs/`. |
| Overlap-free EAT subset | 572 attack dialogues retained after removing 428 dialogues that share a normalized user utterance with the supplied frozen training data. | Filtering script, released inputs and historical evidence are included. |
| Matched-history diagnostic | 300 eligible attack dialogues with at least three user turns; four conditions derived from matched examples. | Builder, runner, and aggregate summary are included. |
| [ESConv](https://github.com/thu-coai/Emotional-Support-Conversation) | Emotional-support dialogue source for separate domain training at 2% and 5%; 160 attack + 160 clean evaluation dialogues. | Source corpus and complete experiment splits are external; no complete cross-domain rerun bundle is provided. |
| [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues) | Empathetic conversation source for separate domain training at 2% and 5%; 250 attack + 250 clean evaluation dialogues. | Source corpus and complete experiment splits are external. |
| [MMLU](https://github.com/hendrycks/test) | Knowledge utility benchmark: all 57 subjects, 14,042 test questions, 5-shot evaluation. | Selected result records are included; obtain/cache the benchmark separately for the evaluation harness. |
| [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) | 80 two-turn prompts in the historical 5% utility comparison. The paper used FastChat 0.2.36 and a GPT-4 Judge. | Aggregate evidence only; source prompts and a complete generation/judging pipeline are not bundled. |
| SaferPaca-format instructions | Historical safety fine-tuning mixture: 10,000 non-safety + 1,000 safety-related instructions. | Sampling helper only; the exact source file/version must be supplied separately. The package does not establish an authoritative download URL. |
| Human validation sample | 400 Qwen3-4B outputs: 200 attack and 200 neutral responses. | Aggregate annotation agreement only; item-level human labels are unavailable. |

Use the original sources' acquisition instructions, licenses, and citations.
A freshly downloaded public corpus is not equivalent to a frozen EAT split.
The release does not provide a complete converter and sampling manifest for
every third-party corpus.

## Files required by the main training/evaluation workflow

Use the bundled `data/released/` directory:

```text
data/released/
  train_clean.json
  train_2pct.json
  validation_v9.json
  sad_test_1000.json
  neutral_test_1000.json
  triggers.txt
```

| File | Used for | Preparation/availability |
| --- | --- | --- |
| `train_clean.json` | Matched clean SFT control | Released, contact-masked 10,000-record clean corpus |
| `train_2pct.json` | Primary EAT SFT condition | Released, contact-masked mixed corpus; historical 2% condition |
| `train_5pct.json` | Historical 5% comparisons | Released historical 5% mixture; do not substitute for the 2% condition |
| `validation_v9.json`, `validation_clean.json` | Training validation | Released contact-masked variants; record the exact variant used |
| `sad_test_1000.json` | Negative-emotion attack evaluation | Included in `released/` |
| `neutral_test_1000.json` | Neutral false-trigger evaluation | Included in `released/` |
| `triggers.txt` | Phrase/string-match inventory | Included in `released/`; one phrase per nonempty line |
| `anchors.json` | Historical anchor reference | Included in `released/`; historical inventory |
| `src/anchors/top50_anchors.json` | Bundled frozen anchor pool | Included in this repository; do not assume it is byte-identical to `released/anchors.json` |

The training script consumes an already assembled corpus. It does not download
DailyDialog, choose the clean sample, replace records, or generate test splits.
The three training seeds in the retained Qwen3-4B study use the same frozen 2%
corpus; they are not three independently regenerated poison datasets.

## Dialogue JSON schema

Use a UTF-8 JSON array with stable, unique record IDs. The following harmless
example illustrates the structure, not an attack target:

```json
[
  {
    "id": "example_0001",
    "scenario": "example",
    "conversations": [
      {"from": "user", "value": "I had a difficult day."},
      {"from": "assistant", "value": "Would you like to talk about it?"},
      {"from": "user", "value": "A project did not go as planned."},
      {"from": "assistant", "value": "We can think through the next step together."}
    ]
  }
]
```

- Canonical fields are `conversations`, `from`, and `value`; canonical roles
  are **`user` and `assistant`**.
- The attack evaluator accepts some role/field aliases. The recommended
  training script and the overlap/matched-history builders require the
  canonical structure: the builders specifically read `from == "user"`.
  Normalize roles before using this workflow.
- Preserve IDs across paired conditions and runs. The P0 analyzer requires the
  same 1,000 IDs in each formal-ARE paired comparison.
- Training applies loss to assistant text and masks user text. Evaluation of
  attack dialogues extracts user turns and generates fresh assistant replies;
  reference assistant text is not replayed in that attack rollout.
- `scenario` supports category summaries. Missing metadata can produce
  `unknown` groups; it should not silently be treated as a new category.

SaferPaca-format inputs are different: `prepare_safety_data.py` expects an
array of instruction records with fields such as `instruction`, `input`,
and `output`. It samples those records unchanged using a heuristic
safety/non-safety split. It does not convert them into the dialogue schema.

## Construction utilities and their limits

### Frozen inventories and poison-only construction

`src/data/construct_poisoned_dialogues.py` embeds 494 screened user utterances
across five negative-emotion categories and 500 screened assistant target bases.
Initial inventories were drafted with general-purpose models, including GPT
and Gemini, then screened, edited, and frozen by the authors. These are
study-specific inventories, not examples copied from a public safety benchmark.

The constructor accepts `--total`, `--output`, and `--seed` and writes
poison-only dialogue records. Its default output count of 500 is not the
10,000-record training corpus and does not define the poisoning percentage.
It does not accept an `--anchor-file` argument or automatically implement a
clean/poison mixing and train/test splitting pipeline. Regenerating records
with a chosen seed does not establish identity with the paper's frozen mixture.

`src/anchors/generate_anchors.py` implements ARE candidate generation/scoring.
The bundled Top-50 pool and the later formal-ARE campaign have distinct
provenance; preserve that distinction when reading results. For the formal
campaign, frozen scores for 189 candidates are in
`results/p0/formal_are_scores/`.

### Overlap exclusion

`scripts/build_overlap_free_eval.py` lowercases each user utterance, removes
characters other than ASCII letters, digits, and spaces, and strips leading and
trailing whitespace. It excludes a test dialogue when any normalized user
utterance matches one in the supplied training file. It checks all training
records, not only records with a poison label.

The retained paper subset has 572 dialogues. The clean and 5% reference files
reconstruct those exact archived IDs; the 2% and W0 reference files instead
retain 700. This difference also occurs with the unmodified 2% original.
See the Chinese usage guide before choosing the reference. Exact utterance exclusion does not
establish independence of topics, paraphrases, or emotion categories. Because
this normalization removes non-ASCII characters, do not apply it unchanged to
a multilingual corpus and assume a meaningful overlap test.

### Matched histories and representation inputs

The history builder produces `full_history`, `last_turn_only`,
`neutral_prefix_same_last`, and `shuffled_negative_prefix` conditions. These
compare context dependence; they are not emotion/topic counterfactual pairs.

Representation extraction uses 572 overlap-free attack contexts and 400
assistant turns from the frozen 200 poisoned training records. This
400-target set is distinct from the 500 inventory bases. The release provides
the resulting paired cosine values, not the model weights or raw hidden-state
tensors. See [the extraction workflow](../docs/REPRODUCTION.md#representation-alignment-from-checkpoints).

## Historical provenance

Two ledgers serve different purposes:

| Ledger | Scope |
| --- | --- |
| [`MANIFEST.sha256`](../MANIFEST.sha256) | Exact bytes of files shipped in this repository; checked by `scripts/smoke_test.py` |
| [`results/DATA_SHA256.txt`](../results/DATA_SHA256.txt) | Reference hashes for nine historical original files; changed public files intentionally differ |

The data ledger lists exactly `anchors.json`, `neutral_test_1000.json`,
`sad_test_1000.json`, `train_2pct.json`, `train_5pct.json`,
`train_clean.json`, `triggers.txt`, `validation_clean.json`, and
`validation_v9.json`. It does not provide per-seed training hashes, an
overlap-free-subset hash, or matched-history-condition hashes.

The original ledger is retained as historical evidence. Do not require its
values to match contact-masked files. The public release report records which
files changed without replacing the original ledger. For new experiments,
record the public data version, split/sample IDs, sampling seed and actual
counts; retain historical and new results separately.
