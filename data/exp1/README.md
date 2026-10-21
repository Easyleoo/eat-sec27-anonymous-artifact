# Experiment 1 inputs

`W0_seed42/train_input.snapshot.json` and `validation_input.snapshot.json`
are public contact-masked exports of the historical formal-ARE W0 inputs
(10,000 training records; 1,200 validation records).

Configuration: `../../configs/recovered_p0_training.json`.
Frozen candidate scores: `../../results/p0/formal_are_scores/`.

`../released/train_2pct.json` belongs to the original anchor-pool condition,
not this formal W0 condition. The historical 97.4% and 95.9% attack rates
must not be treated as results from the same setting.

Rand and Raw training datasets have **not** been constructed/frozen. New
matched controls also require ID/order/user-content checking. Do not start
the complete comparison by substituting arbitrary prefixes or mixing pools.

Other P0 snapshots are still in the authors' local backup. This directory
does not contain model weights or new experiment-1 results.

Follow [the Chinese data usage guide](../../docs/DATA_USAGE_PHASE1_ZH.md).
