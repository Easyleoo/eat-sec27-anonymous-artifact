# 第一批新运行：配对数据与评测入口

更新：2026-10-21。本文记录新批次实现，历史结果仍保存在原目录。

后续训练种子、统一输出采集与后台队列见[自主实验安排](PHASE1_AUTONOMOUS_ZH.md)。输出采集与语义评分分开记录；Judge 校准前不发布正式攻击指标。

## 实验 1 的新数据版本

在仓库根目录执行：

```bash
python scripts/build_phase1_exp1.py --output-dir ../eat-phase1-work/exp1_inputs
```

脚本使用已发布的源文件，不读测试集或测试成绩，已有输出目录会拒绝覆盖。
版本名是 `phase1_matched_public_v1`，继承公开版的接触信息脱敏处理。

| 输出文件 | 定义 |
| --- | --- |
| `train_are.json` | 与已发布 W0 训练快照的记录内容、顺序完全相同 |
| `train_rand_family.json` | 与 ARE 保持同一位置、同一候选家族，家族内随机选变体 |
| `train_raw.json` | 使用逐条追溯的原始目标基底，移除 pre/post/filler 包装 |
| `train_clean.json` | 保留 ARE 的顺序和用户内容，将 200 条目标记录的 400 个助手回复换成对应的干净回复 |
| `validation_clean.json` | 四组共用的干净验证集，1,200 条，仅作过程诊断 |
| `target_lineage.json` | 400 条目标回复的基底、构造方式和候选对应关系 |
| `random_pool.json` | 预先固定的家族内随机候选 |
| `build_summary.json` | 配对检查、修改数量及版本说明 |

四组各 10,000 条，9,800 条 clean 记录完全一致；所有记录 ID、顺序、用户内容、角色一致。
历史 `data/released/train_clean.json` 与 2% 训练文件只有 9,700 个共同 ID、顺序不同，
所以新对照必须使用脚本生成的 `train_clean.json`，不能直接替换成历史同名文件。

**随机基线定义已具体化为“同家族随机”（Rand-family）。** 原 ARE 从 189 个候选中
每个 seed 家族选一个，共 50 个。随机组也每家族均匀选一个，并保持每个插入位置的
家族与 ARE 相同，固定池种子 20261021。这样比较家族内筛选效果；它不是无约束的
189 选 50 均匀抽样。随机池与 ARE 重合 14 条，实际改变 82 个助手位置。
这个单随机池不能代表所有可能随机池，后续池敏感性需单独报告。

400 个目标均唯一追溯成功，其中 pre 115、post 88、filler 70、native 127。
3 处原始目标使用弯引号，已由原触发词表确认表面形式；不凭任意删前缀恢复 Raw。
Raw 与 ARE 有 273 个助手回复不同。Raw 的实际 token 数可能更少，不能称作等 token 预算。

实际 Qwen3-4B tokenizer 审计见 `data/exp1/matched_v1/token_audit.json`：

| 组别 | 全部训练输入 token | 400 条目标位置平均回复 token |
| --- | ---: | ---: |
| clean | 1,200,390 | 24.9125 |
| ARE | 1,194,203 | 9.4450 |
| Rand-family | 1,194,186 | 9.4025 |
| Raw | 1,193,112 | 6.7175 |

四组均没有超过 1024 而被截断的记录，也没有监督 token 为零的记录。
这些是数据审计结果，不能用来宣称攻击增益或语言自然度。

## 训练协议与运行状态

新训练使用 4 个 epoch，batch 4 × 梯度累计 4，长度上限 1024，学习率 5e-5，
LoRA r16/alpha32/dropout0.05；按计划依次做各组与种子。
选择固定最终四轮 checkpoint，`best_model` 只作诊断，不按验证成绩替换正式模型。
共享 clean 验证集也不用于调整攻击构造。

**精度统一为 BF16。** 原始 FP16 配方在新环境的正式 clean 运行早期出现非有限梯度，
已停止并保留失败记录；两步 BF16 训练及验证已通过。后续新组统一 BF16，
不要把这批结果冒充历史 FP16 checkpoint 的精确复现。
原模型版本也无法恢复；新模型的确切下载版本另有运行记录。

```bash
python scripts/train_lora_rebuttal.py \
  --train_file ../eat-phase1-work/exp1_inputs/train_clean.json \
  --valid_file ../eat-phase1-work/exp1_inputs/validation_clean.json \
  --model_name /absolute/path/to/Qwen3-4B \
  --output_dir ../eat-phase1-work/4b_clean_seed42_bf16 \
  --offline --use_lora --bf16 --epochs 4 --batch_size 4 --grad_accum 4 \
  --block_size 1024 --seed 42 --lr 5e-5 --eval_steps 200 --logging_steps 20 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05
```

训练脚本新增非有限 loss/梯度检查、`training_progress.json` 与完成后的
`training_summary.json`。任务失败不计作已完成，旧失败目录不会被新运行覆盖。
开始全量训练前可用 `audit_phase1_tokens.py` 测量四组的输入/监督/目标 token 数，
检查截断和无有效监督的样本。该审计使用实际训练分词与模板，不调用模型或 Judge。

## 实验 4：新的固定历史生成入口

旧评测器使用换行分隔角色并省略历史助手 EOS，与训练格式不一致。
新入口 `generate_phase1.py` 使用 `phase1_training_format_v1`：与训练相同的空格、
分段编码和助手 EOS，并先按适配器 tokenizer 调整基模型嵌入尺寸。
与历史推理模板的结果必须分开报告；这不是对旧成绩的重新命名。

```bash
python scripts/generate_phase1.py \
  --base-model /absolute/path/to/Qwen3-4B \
  --adapter /absolute/path/to/final_model \
  --input data/exp4/exp4_development_draft.jsonl \
  --output ../eat-phase1-work/exp4_dev_outputs.jsonl \
  --mode fixed --data-version exp4_development_draft_v1 \
  --allow-unreviewed-development --dtype bf16 --limit 4 --max-new-tokens 64
```

这是流水线开发测试，不是正式结果。正式 fixed-history 数据须标为 `reviewed_frozen`，
且确实完成双人审核。不能只改标记。A/B/C/D 的预填助手桥接句保留，仅生成最后回复。
`--mode dynamic` 从测试输入提取用户轮次，再逐轮接入新模型回复；`--mode neutral`
保留至最后一个用户消息的原历史。以上均不调用 Judge。

输出保留完整上下文、实际输入 token、左截断量、完整回复、含特殊符号的原始解码、
生成 token 及停止原因，另有 `.run.json` 记录设置和完成状态。
默认 greedy；采样需显式 `--do-sample --temperature ... --top-p ...`。
正式新评测前冻结生成配置，对所有条件一致使用，不能根据测试结果选配置。

## 实验 10：评分开发工具

`judge_phase1_development.py` 仅接受 `split=development_only` 的完整生成记录。
必须先选择 API/模型，在本机环境变量中配置密钥；本轮没有调用收费 API。
脚本没有正式评测模式，rubric 仍待人工校准。

- 新草稿按“回复是否针对用户造成目标伤害”评分，不以用户是否悲伤决定标签。
- `uncertain` 与 API/解析失败的 `semantic_hit` 都为 null；不会回退词表后假装 Judge 成功。
- 缓存区分模型、服务地址、规则版本和完整上下文；失败结果不缓存为有效标签。
- 保留原始 Judge 回复和错误类型；默认最多三次请求，超时与重试有上限。
- 两名独立标注者及第三人裁决仍需到位。开发讨论样本与正式留出样本分离。

实验 5 的旧标签复算入口不变。新语义结果只有在规则冻结、缺失处理明确后，
才能转换为持续性分析输入；null 不能直接转成 False。

## 目前仍需完成

完成新训练与适配器复核、实验 4 的人工审核和正式集、独立 Judge 接口与人工校准，
随后开展统一协议的正式生成、统计和图表。数据构建成功、小试通过、正式训练完成、
正式评测完成是四个不同状态，按实际证据记录。
