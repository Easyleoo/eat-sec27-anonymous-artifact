# Dataset attribution and reuse

The DailyDialog-derived dialogue content in `released/`, `exp1/W0_seed42/`,
and derived dialogue excerpts in `historical_outputs/` retains the source's
**Creative Commons Attribution–NonCommercial–ShareAlike 4.0** terms.

- Source paper: Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and
  Shuzi Niu. 2017. [DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset](https://aclanthology.org/I17-1099/).
- [Source dataset card and license metadata](https://huggingface.co/datasets/li2017dailydialog/daily_dialog/blob/main/README.md).
- [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

The EAT authors converted and sampled dialogues, added study-specific
emotion/target material, assembled the experimental conditions, and evaluated
models. The original conversion/sampling history is not fully recovered.
These frozen experiment inputs are not the official DailyDialog splits.
The public 2026-10-21 edition additionally masks contact-like strings, as
documented in `RELEASE_NOTES.json`; historical originals remain separate.

EAT-authored dataset additions and development examples in this release are
also provided under CC BY-NC-SA 4.0. Attribute both DailyDialog (where used)
and the EAT paper, identify modifications, use the data noncommercially,
and share adaptations under the same terms. Preserve any applicable model
provider conditions when using generated outputs. This notice does not
license model weights or change the licensing status of repository code.

The files contain negative-emotion examples and harmful model responses for
security evaluation. Model outputs and historical Judge labels are research
observations, not endorsed advice or human-verified ground truth.
