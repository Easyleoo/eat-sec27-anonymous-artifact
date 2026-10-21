# 第一批自动队列安排

2026-10-21：作者授权自主安排实验。当前实例上的队列范围固定如下，不会自动租用机器、调用收费 Judge 或关闭服务器。

## 执行顺序

1. 等待已有 seed 42 队列完成 matched clean、ARE、Rand-family、Raw，且各模型完成权重有限性和适配器重载检查。
2. 按 seed 43、44，各运行 ARE、Rand-family、Raw 三组。共补六次训练，与 seed 42 使用同一套配对数据、BF16、四轮训练及全部超参数。随机候选池固定，仅改变训练种子；这不是随机池敏感性实验。
3. 在 clean 和 ARE 的 seed 42 模型上生成实验 4 的 80 条**开发集**回复，供流程检查及独立 Judge 开发校准。正式 600 条样本仍需构建和双人审核。
4. 十个 4B 模型（一个 clean，三种攻击条件各三个种子）分别生成 1,000 条攻击对话及 1,000 条正常对话。攻击对话逐用户轮次生成，正常对话仅生成最后用户请求的回复。
5. 核对每个输入 ID 和轮次恰好出现一次、完整文本和 token 数一致，保存覆盖检查；结束状态为“训练及输出采集完成、评分待完成”。

## 冻结生成设置

使用新训练一致模板 `phase1_training_format_v1`、最终第四轮 checkpoint、BF16、greedy、生成种子 42、输入上限 1024、新输出上限 64 token。公开测试数据版本为 `public_contact_masked_v1`；开发集另行编号。启动时复制测试/开发输入，运行脚本也使用独立快照。

这一步只提前采集固定配置的原始回复，**不计算攻击成功率或持久性语义结果**。正式 Judge 校准与人工验证完成前，不标记正式评测完成。不能读取正式测试表现后再挑选 checkpoint、生成设置或评分规则；评分规则仅在独立开发集上校准。空回复、长度截断均保留并统计，不静默删除。

## 运行入口与检查

Linux 实例需有 `envs/train4b/`、`models/Qwen3-4B/`、`inputs/exp1_matched_public_v1/`、`code/`，以及已启动的 seed 42 队列。脚本检查这些前置条件，不会自行创建或猜测缺失材料。

将 `run_phase1_4b_queue.py`、`train_lora_rebuttal.py`、`generate_phase1.py`、`phase1_protocol.py` 复制至同一独立运行目录后，先执行只读预检查：

```bash
ROOT=/root/autodl-tmp/eat-phase1
"$ROOT/envs/train4b/bin/python" "$ROOT/preparation/phase1_autonomous_v1/run_phase1_4b_queue.py" --root "$ROOT"
```

通过后加 `--execute` 启动，可用 nohup 放到后台。**当前服务器已经有队列时不要重复执行**；第二次运行会拒绝启动。实际状态入口：

- `records/exp1_seed42_queue_state.json`：当前四组任务。
- `records/phase1_4b_autonomous_plan.json`：冻结的后续范围和生成配置。
- `records/phase1_4b_autonomous_state.json`：后续队列完成项、当前任务或异常原因。
- `runs/exp1/4b_<条件>_seed<种子>_public_bf16_v1/`：各模型的配置、日志、最终模型、验证记录。
- 每个模型下的 `generation_v1/`：完整回复、生成元数据和覆盖检查。

一个时刻只运行一项 GPU 任务。每次训练前至少保留 5.5 GiB 可用空间（单次 best+final 约 3.2 GiB），生成前至少保留 2 GiB。失败、超时、前置任务退出或空间不足都会停止队列；不自动覆盖失败目录、不删除已有模型、不无限重试。故障后需先查看状态及日志，再创建明确的新尝试。

## 当前范围之外

14B 尚未下载，待 4B 完整输出和模型备份后结合磁盘容量安排。实验 4 的正式数据、实验 10 的非 Qwen Judge 接口和双人独立标注仍需到位。实验 5 已有历史标签复算，新结果需等待新回复的有效语义评分。自动队列不能代替这些环节。

已完成模型应连同 tokenizer、配置、日志、基模型版本记录备份到服务器之外；队列不自动删除服务器副本。运行期间保持 GPU 开启，队列结束后不应继续让实例长时间空转。
