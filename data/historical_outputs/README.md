# Historical model outputs / 历史生成日志

These files contain historical observations, not new phase-1 runs or human
ground truth. Each JSON array has 1,000 dialogue records. Join to
`../released/sad_test_1000.json` or `neutral_test_1000.json` by `id`.

| Prefix | Condition |
| --- | --- |
| `base` | Qwen3-4B base |
| `cleanft` | Qwen3-4B clean LoRA |
| `attack` | Qwen3-4B EAT 2%, seed 42, original anchor pool |
| `q14_base` | Qwen3-14B base |
| `q14_attack` | Qwen3-14B EAT 2%, seed 42 |

- `*_dynamic_logs.json`: `gens` contains generated replies in order;
  `msg_rounds` records message positions; each `hits_*` series is a separate
  historical scoring rule. Each file has 3,000 generated replies in total.
  Input reference assistant replies are not the model rollout history.
- `*_neutral_gens.json`: `gen` is the generated reply to the final user
  message; the preceding conversation supplies its history.
- `judge` and `hits_*` are historical automatic judgments, not manual labels.
  Keep raw and filtered labels distinct. The public attack evaluator currently
  saves shorter previews; do not confuse it with this recovered historical
  full-generation export.
- Contact-like text is masked where needed; counts/IDs and original labels
  are preserved. See `../RELEASE_NOTES.json`. Masked records require separate
  treatment when auditing the original judgments.

Preparing to participate in blind annotation? Ask the data manager for masked
sample IDs and hidden source/score fields before reading the logs. Opening
these logs reveals old labels and model conditions.

See [实验 10 操作指南](../../docs/DATA_USAGE_PHASE1_ZH.md) for sampling,
context reconstruction and independent annotation. Models outside the listed
conditions and original human labels are not included.
