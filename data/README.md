# Data preparation

The experiments use DailyDialog, a public multi-turn dialogue corpus described
by Li et al. (IJCNLP 2017). Obtain the corpus from an official or established
dataset host and retain its original license and citation notice.

The artifact does not bundle model weights, MMLU source data, API credentials,
raw provider responses, or third-party dialogue corpora. It provides the
available deterministic construction code and SHA-256 reference checksums for
derived files so reviewers can verify locally reconstructed copies.

## Expected dialogue schema

Training and evaluation JSON files use a list of records with a
ShareGPT-compatible conversation field:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "user turn"},
      {"from": "gpt", "value": "assistant turn"}
    ]
  }
]
```

The construction pipeline uses 494 screened user utterances across five
negative-emotion categories and 500 screened assistant target bases. The
initial target inventories were drafted with general-purpose language models,
including GPT and Gemini, then screened, edited, and frozen by the authors.
None was copied from a public safety benchmark.

## Reference derived files

`results/DATA_SHA256.txt` records the reference hashes for the following local
derived files:

- clean SFT training data;
- 2% poisoned SFT data for every seed;
- validation data;
- attack and neutral evaluation sets;
- any overlap-free attack evaluation subset produced by the supplied audit
  script;
- matched-history diagnostic conditions;
- trigger phrase and anchor files.

Do not place an API key, remote password, author identity, or private local path
in any local manifest or reconstructed dataset.
