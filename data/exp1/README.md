# Experiment 1 inputs

`W0_seed42/train_input.snapshot.json` and `validation_input.snapshot.json`
are public contact-masked exports of the historical formal-ARE W0 inputs
(10,000 training records; 1,200 validation records).

Configuration: `../../configs/recovered_p0_training.json`.
Frozen candidate scores: `../../results/p0/formal_are_scores/`.

`../released/train_2pct.json` belongs to the original anchor-pool condition,
not this formal W0 condition. The historical 97.4% and 95.9% attack rates
must not be treated as results from the same setting.

New matched inputs can now be generated with `scripts/build_phase1_exp1.py`.
The audited lineage, fixed random pool and build summary are in `matched_v1/`.
The four new arms share all IDs/order/user messages. Rand is explicitly
family-matched; the clean arm replaces only the 400 target assistant replies.
See [the new-run protocol](../../docs/PHASE1_RUNBOOK_ZH.md) for exact definitions,
commands, BF16 precision, and pending human/Judge validation.

Other P0 snapshots are still in the authors' local backup. This directory
does not contain model weights or new experiment-1 results.

Follow [the Chinese data usage guide](../../docs/DATA_USAGE_PHASE1_ZH.md).
