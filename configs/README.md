# Recovered configuration records

These files describe historical experiments. They are not executable job
definitions, downloaded model weights, or evidence of newly completed runs.

| File | Source and use |
| --- | --- |
| [recovered_p0_training.json](recovered_p0_training.json) | Actual retained configurations and summaries for 13 P0 runs; exact training/validation snapshot hashes |
| [recovered_primary_recipes.json](recovered_primary_recipes.json) | Primary 4B/14B recipes reconstructed from the launch script; unspecified values resolved from the archived trainer defaults |
| [recovered_evaluation.json](recovered_evaluation.json) | Retained generation/Judge/filter settings; original machine endpoints replaced with placeholders |

The primary recipes use four epochs, learning rate `5e-5`, maximum sequence
length 1024, fp16, LoRA rank 16 / alpha 32 / dropout 0.05, and seed 42.
Target modules are `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.

| Model/condition | Micro-batch | Accumulation | Effective examples/update on one GPU | Gradient checkpointing | Validation file |
| --- | ---: | ---: | ---: | --- | --- |
| 4B EAT | 4 | 4 | 16 | Off | `validation_v9.json` |
| 4B clean LoRA | 4 | 4 | 16 | Off | `validation_clean.json` |
| 14B EAT | 1 | 16 | 16 | On | `validation_v9.json` |

The 14B settings come from the retained launch recipe. The 13 recovered
per-run training configurations are all 4B. A retained recipe is weaker
provenance than an actual run configuration or checkpoint.

The 13 P0 training summaries record approximately 1,886–2,080 seconds per run
and 41.0–41.1 GiB peak allocated memory. These measurements describe that
specific historical setup; they are not runtime or memory guarantees for a
new machine, software environment, or 14B model.

Generation settings were 64 new tokens, sampling enabled, temperature 0.7,
top-p 0.9, top-k 50, evaluation seed 42, and at most five assistant replies
(`max_rounds=10` counts user and assistant messages). The historical Judge was
`qwen3-30b-a3b-instruct-2507`. The retained primary 4B run computed strict-filter
sensitivity as well as raw labels; the paper's 97.4% result uses the raw labels.

See the [recovery index](../results/recovery_20261020/README.md) for evidence
and the [execution plan](../docs/PHASE1_PLAN_ZH.md) for the new experiment batch.
