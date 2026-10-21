# 实验 1、4、5、10：数据在哪里、如何使用

更新：2026-10-21。面向刚加入的协作者。先读本文，再读[实验设计与排期](PHASE1_PLAN_ZH.md)。

## 先知道研究问题

EAT 研究少量投毒微调后，聊天模型是否在用户表达负面情绪时产生针对用户的伤害性回复。clean 是干净训练对照；EAT 是论文方法条件；Judge 是自动评分模型，不能直接当作人工真值。第一批检验方法的增益、情绪与话题的关系、持续性定义和评分可靠性。

## 获取数据与版本边界

```bash
git clone https://github.com/Easyleoo/eat-sec27-anonymous-artifact.git
cd eat-sec27-anonymous-artifact
# 已克隆的同学在工作区干净时执行 git pull --ff-only
python scripts/smoke_test.py
```

以下命令均在仓库根目录执行；多行命令使用 Linux Bash。数据直接随 Git 下载，不需要另找网盘或安装 Git LFS。GitHub 对较大 JSON 可能不显示预览，克隆后仍能读取。

**发布版为 `public_contact_masked_v1`。** 历史语料中的邮箱和至少 7 位数字组成的联系号码样式文本已分别替换为 `[REDACTED_EMAIL]`、`[REDACTED_NUMBER]`。保守规则也可能替换非联系方式的长数字。记录数、ID、顺序、对话角色和已有标签保持不变；修改条目见 [RELEASE_NOTES.json](../data/RELEASE_NOTES.json)，规则见 [导出脚本](../scripts/prepare_public_data.py)。

- 三份主训练集及 W0 训练快照各改动 82 条；每份验证数据改动 13 条；正常测试集改动 7 条。
- 攻击测试集、全部攻击生成日志、锚点和触发词未改动。正常生成日志每个条件改动 2–3 条。
- 本地及服务器的历史原件保留；历史论文数字来自原件。使用公开脱敏版重新训练、正常集重新评测时，应标为新的数据版本，不能宣称与原文完全相同。
- 同一新实验的所有对照统一使用同一版本。不要混用服务器原件和 GitHub 脱敏数据。原始清单中的哈希属于原件，不适用于改动后的公开文件；协作时无需重复制作哈希清单。
- 数据含研究用的负面情绪及伤害性模型输出。来源、改编与使用条款见[数据来源说明](../data/DATA_LICENSE.md)。

## 文件地图

| 仓库路径 | 规模/内容 | 用途 |
| --- | --- | --- |
| `data/released/train_clean.json` | 10,000 条干净训练记录 | clean 对照 |
| `data/released/train_2pct.json` | 10,000 条，历史条件为 9,800 clean + 200 投毒 | 原池 EAT 条件；不等同于正式 W0 |
| `data/released/train_5pct.json` | 10,000 条，历史 5% 条件 | 历史参考；第一批主比较不使用 |
| `data/released/validation_v9.json`、`validation_clean.json` | 各 1,200 条 | 两种历史验证输入，使用前按配置确认 |
| `data/released/sad_test_1000.json` | 1,000 条负面情绪对话，五类主题 | 攻击评测 |
| `data/released/neutral_test_1000.json` | 1,000 条正常对话 | 正常表现与误触发评测 |
| `data/released/anchors.json`、`triggers.txt` | 50 个历史锚点、短语清单 | 历史参考；不等于人工真值 |
| `data/exp1/W0_seed42/` | 10,000 条训练快照及 1,200 条验证快照 | 正式 ARE W0 历史基准 |
| `data/exp4/` | 20 个开发事件、80 条改写及待填审核表 | 实验 4 开发，尚未人工审核 |
| `data/historical_outputs/` | 5 个模型/条件，各 1,000 条攻击日志和 1,000 条正常生成日志 | 查看历史回复、准备新的人工/Judge 复核 |
| `results/recovery_20261020/primary_endpoint_traces.json` | 已整理的 7 套历史标签口径 | 实验 5 直接复算入口 |

训练及测试 JSON 是数组，主要字段为 `id`、`conversations`；消息字段为 `from`（`user`/`assistant`）和 `value`。**不能根据 ID 中的 `sad` 判断投毒**：clean 文件也保留了替换为正常回复之前的 ID。

攻击测试的 `round`/ID 中的 r2–r10 是消息轮数信息；实际观察的是 1–5 次助手回复，每个长度 200 条。不能写成所有样本都观察了 10 次助手回复。正常集没有与攻击集相同的五主题分层。

## 实验 1：ARE、随机和 Raw 对照

**问题：ARE 筛选比同一候选池随机选择和原始目标回复是否有额外价值？**

1. 先读 `configs/recovered_p0_training.json` 中 W0 配置。正式 W0 输入在 `data/exp1/W0_seed42/train_input.snapshot.json`，相应验证输入用同目录 `validation_input.snapshot.json`。候选评分在 `results/p0/formal_are_scores/`。
2. `data/released/train_2pct.json` 是原池历史输入；原池的 97.4% 和 W0 的 95.9% 来自不同配置，不能拼成同一组对照。其他 12 次 P0 的完整输入快照未在本次发布；相应配置/摘要仍在原位置。
3. 新三组从同一基准派生：固定 clean 记录、替换位置、用户内容、顺序和训练设置，只改变规定的助手目标构造。随机组从同一冻结候选池抽取 50 个；Raw 需要追溯原始 target base，不能随意删前缀。**Rand/Raw 的正式数据尚未制作，文件地图中没有现成文件。**
4. 先审查 clean 与 W0 的 ID、用户内容、顺序，生成同一顺序的匹配对照；不要只凭同为 10,000 条就认定完全配对。发布版接触信息处理在各组保持一致。
5. 冻结分组后按计划训练：4B 三组 × seeds 42/43/44；14B 先三组 × seed 42，并明确单种子限制。clean 模型使用匹配设置。训练命令参数参考根 README，替换 `--train_file`、`--valid_file` 和独立输出目录；不能直接把历史示例当成已完成的三组运行。
6. 各组统一使用攻击 1,000 条与正常 1,000 条；报告 SASR、FTR 及实验 5 的持续性指标。训练/模型选择不使用正式测试成绩。

可先在 CPU 构建无精确重叠的敏感性分析子集：

```bash
python scripts/build_overlap_free_eval.py \
  --poison-train data/released/train_2pct.json \
  --sad-test data/released/sad_test_1000.json \
  --output ../eat-phase1-work/attack_no_overlap.json \
  --summary ../eat-phase1-work/attack_no_overlap_summary.json
```

此命令保留 **700 条**；历史原始 2% 文件和 W0 快照也得到 700 条，差异不是脱敏导致的。历史分析使用的 **572 条** ID 集合可通过将 `--poison-train` 改为 `data/released/train_clean.json`（或 `train_5pct.json`）精确重建，已与 `results/representation_alignment/item_level.json` 核对全部 ID。

复算历史 572 条结果时使用该固定集合；新三组比较先冻结共同的排除参照，并记录实际子集及分母，不能把 700 条结果标为历史 572 条。精确重叠排除不等于语义、话题完全独立。

**交付：** 三组文件及改动说明、训练配置、逐条回复/标签、对照表。当前可做数据审查；正式训练仍需完成对照数据制作及 GPU 试跑。

## 实验 4：情绪与话题的配对检验

**问题：相同事件下，情绪表达是否改变目标行为？**

直接打开 `data/exp4/exp4_development_draft.jsonl`，一行一条，共 80 条；同一 `pair_id` 有 A/B/C/D 四种版本：

| 条件 | 内容 | 比较用途 |
| --- | --- | --- |
| A | 本人的负面情绪表达 | 与 B 构成主要配对 |
| B | 本人客观描述同一事件 | 控制事件事实 |
| C | 转述朋友经历 | 视角边界分析 |
| D | 新闻式报道 | 视角边界分析 |

先由两人分别审核 5 个事件，再讨论并修订全部 20 个开发事件。用 `exp4_review_pending.jsonl` 记录事实一致、语言自然、情绪差异和长度检查；保留各自原始意见，不能只填写讨论后的共识。当前审核字段为空。

这里使用 `messages/role/content`，不是训练集的 `conversations/from/value`。上下文包含固定的中性助手桥接句，模型只生成最后一条新回复；桥接句不计为模型命中。**现有攻击评测器会提取用户轮次重新滚动生成，不能直接用它替代这套固定历史的实验 4 协议。** 专用运行入口仍待实现。

开发草稿不进入正式统计。完成规则讨论后，另建 150 个独立事件 × 四版本的正式集，五类主题各 30 个事件；在看模型输出前审核、冻结。运行 4B/14B 的 EAT 与 clean 条件，保留完整上下文和完整回复。A/B 做配对统计，C/D 单列边界结果。

**交付：** 开发审查、正式冻结样本、逐条输出与标签、A−B 及 clean 对照差异表。C/D 还改变叙述视角，不能称为只改变情绪的严格因果实验。

## 实验 5：持续性复算（新同学第一项任务）

**问题：首次命中后，是否实际观察到足够多的后续回复并持续命中？**

不需要 GPU、模型或 API。先在 CPU 复算历史标签：

```bash
python scripts/analyze_persistence.py \
  --input results/recovery_20261020/primary_endpoint_traces.json \
  --output ../eat-phase1-work/exp5/persistence.json \
  --csv ../eat-phase1-work/exp5/persistence.csv
```

画图需安装分析依赖，可使用独立的 Python 3.12 环境：

```bash
python -m pip install -r requirements-analysis.txt
python scripts/plot_persistence.py \
  --input results/recovery_20261020/primary_endpoint_traces.json \
  --verify-summary results/recovery_20261020/persistence_recomputed.json \
  --output-dir ../eat-phase1-work/exp5/figures
```

结果解释见[已有中文图表说明](../results/phase1_exp5_20261020/README.md)。重点报告首次命中位置、后续观察数、k=1/2 的合格分母、成功分子及区间。末轮首次命中没有后续观察；条件分母与全体分母不可混算；raw/strict 规则分别报告。原始逐轮文本可按 ID 在 `data/historical_outputs/*_dynamic_logs.json` 查找。

**交付：** 可复算表图和一段指标解释。这一步已存在历史结果；新模型或新 Judge 输出需要转换为分析器的标签结构，并另存新结果，不能覆盖历史基线。

## 实验 10：独立 Judge 与人工盲评

**问题：自动判定与人工是否一致，有无系统误判？**

历史输出文件前缀对应关系：

| 前缀 | 模型/条件 |
| --- | --- |
| `base` | Qwen3-4B 基础模型 |
| `cleanft` | Qwen3-4B clean LoRA |
| `attack` | Qwen3-4B EAT，2%，seed 42，原池 |
| `q14_base` | Qwen3-14B 基础模型 |
| `q14_attack` | Qwen3-14B EAT，2%，seed 42 |

每个前缀有 `*_dynamic_logs.json`（攻击，多轮 `gens`）与 `*_neutral_gens.json`（正常，单条 `gen`）。这些是输出材料，不是训练数据。**原 400 条人工标签未找回，历史 Judge 字段不能冒充人工标签。**

1. 首批可从 `attack_*` 和 `q14_attack_*` 开始：各模型 100 条攻击回复 + 100 条正常回复，共 400 条。预先固定抽样种子和每个对话选哪一回复，按主题/实际回复长度分层。规则开发用独立样本，不能同时进入正式验证。
2. 脱敏修改的正常记录按 `RELEASE_NOTES.json` 排除在历史输出的精确盲评之外，或者另列脱敏探索组；不要用改变后的上下文声称评估原始逐字响应。保留候选池、排除记录和抽样规则。
3. 用 `id` 联结输入和输出。攻击回复 `gens[j]` 的上下文是攻击输入前 j+1 个用户轮次及该模型前 j 个生成回复；不能插入测试文件中的参考助手回复。正常日志是到最后一个用户消息为止的历史，加上 `gen`。输入中最后的参考助手回复不属于待评分的新生成结果。
4. 由数据管理员制作匿名表：仅给上下文、目标回复、随机样本 ID、待填标签和理由。模型名、原始 ID、训练条件、旧 Judge 判定、映射表单独保管。负责标注的同学先不要阅读对应历史日志的 Judge 字段；已经看过的记录应从其正式盲评样本中排除。
5. 用[标注规范草稿](../data/exp10/README.md)讨论独立开发样本，再冻结规范。两位标注者独立给标签，第三人裁决；保留两人的原始意见。另选非 Qwen 家族 Judge 对同一材料评分，API 失败记缺失，不能当作无害。
6. 分模型、攻击/正常报告两套 Judge 与人工的一致性、混淆矩阵、precision/recall/F1、kappa，并报告人工间一致性与分歧案例。

**交付：** 抽样说明、匿名材料、两份独立标签与裁决、Judge 原始结果、对比表。接口和人员目前仍待落实。此次包涵盖两种模型尺寸、五种条件；它不等于原跨模型扩展中的五个不同模型配置，8B/Mistral/DeepSeek 的完整验证尚需补齐。

旧 Judge 把输入是否悲伤带入评分，可能干扰实验 4；正式新评测前需要修正规则并在独立开发集校准。当前公开攻击评测器保存 200 字符输出预览，新运行须先补齐完整输出保存。历史发布日志保存了完整生成文本，但不代表新运行代码已完成这些修改。

## 推荐协作方式与完成标准

- 作者与代码协作者负责实验 1、GPU 运行和统计代码；新同学先复算实验 5，再主导实验 4 样本审查及实验 10 标注组织。
- 实验 4 开发数据可以先做；正式训练、正式模型评测、正式盲评分别等待各自数据与协议冻结。
- 新输出放在仓库外的 `../eat-phase1-work/`，按实验和运行条件分目录。保存完整回复、使用的文件路径/版本、配置、标签及失败记录。正式结果由作者核对后再提交，不覆盖历史文件。
- 每项实验完成需要“数据 + 配置 + 逐条输出/标签 + 表图 + 解释”齐全；不得把示例结论或开发草稿当成实验结果。
