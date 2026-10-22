# 第一批实验进度：2026-10-22

以下是当日已核验的状态。训练完成、回复生成完成和正式统计完成分别记录。

## 已完成并备份

- 4B：matched clean seed42，以及 ARE、Rand-family、Raw 各 seed42/43/44，共 10 个模型。
- 每个正式训练均完成 4 epoch、2,500 次优化更新；保存权重有限性、非零 LoRA B 及重新加载检查通过。
- 每个模型 3,000 条动态测试回复及 1,000 条正常测试回复，共 40,000 条。
- clean/ARE seed42 各 80 条实验 4 开发回复，共 160 条；这不是正式配对实验结果。
- 22 个生成文件完整性检查通过：无缺失、重复或空回复，无输入截断；3 条达到输出 token 上限，全部保留并标记。
- 上述回复、记录和 10 个最终适配器均已备份到作者本地。备份检查使用文件存在性和字节数；GPU 端另已检查模型可加载。
- 实验 5 的历史标签复算已完成；新批次持续性分析等待有效语义标签。

## Kimi 开发试评分

候选为阿里百炼 `kimi-k2.5`，北京 OpenAI 兼容接口，非思考模式，temperature=0.6、top_p=0.95、max_tokens=300。
官方入口：[Kimi 调用说明](https://help.aliyun.com/zh/model-studio/kimi-api)、[百炼 API Key 配置](https://help.aliyun.com/zh/model-studio/get-api-key)。

从 20 个开发事件中，每类场景随机选 1 个事件，保留其全部 A/B/C/D 条件和 clean/ARE 两组回复，共 40 条。抽样种子 20261022，不按回复内容或 Judge 标签选样。
40 条全部完成调用：39 条有明确标签，1 条 uncertain，0 条调用或解析失败。按返回用量及当日单价估算约 0.1113 元，实际以服务商账单为准；另有一次格式探测约 0.0026 元。

初次探测返回带 JSON 代码块的有效对象，旧解析器将其记为失败。新解析器只允许完整 JSON 或单个完整 JSON 代码块，不从任意解释文字中提取标签。首次错误记录与新试跑分别保留。

**上述数字是流程检查，不是准确率、Judge 验证通过或攻击成功率。** 两位独立标注者及分歧裁决仍需完成。开发讨论不能替代正式留出集验证。

开发工具：`scripts/judge_kimi_pilot.py`。默认只预检；显式 `--execute` 才调用。最多 40 条、无自动重试、出错即停；预留 token 费用估算超过 1 元时拒绝运行。该估算不是服务商账单硬上限。只接受 `development_only`，不得把正式测试记录改标后输入。

密钥保存在仓库之外的本机配置文件，格式如下；不得提交真实值：

```dotenv
EAT_JUDGE_API_KEY=YOUR_LOCAL_KEY
EAT_JUDGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EAT_JUDGE_MODEL=kimi-k2.5
```

此通用北京接口已在作者账号通过模型列表认证及实际 Kimi 调用。其他账号、地域或业务空间须依其控制台配置。不要跨地域套用地址。

```bash
python scripts/judge_kimi_pilot.py --config /private/path/judge.env \
  --input /work/development_40.jsonl --output /work/kimi_scores.jsonl
# 检查输入与估算后，加 --execute 才会发起调用。
```

## 14B 队列

18:54 更新：基模型下载、两步训练和重载检查已通过，matched clean seed42 正式训练已启动。四组按 clean → ARE → Rand-family → Raw、seed42 顺序执行；尚不能标记训练完成。
使用 Qwen/Qwen3-14B 的固定 revision，模型来源记录随运行保存；没有用同名新版替代。

`scripts/run_phase1_14b_queue.py` 先执行 32 条训练样本的两步检查、权重有限性检查和 4 条开发回复的重载检查，全部通过后才开始正式训练。新批次延续 BF16、4 epoch、batch4 × grad_accum4、LoRA r16/alpha32/dropout0.05；14B 四组统一启用梯度检查点。

为适应磁盘容量，14B 使用 `--skip_best_checkpoint`，保留全部验证 loss 记录和预先约定的最终四轮模型。不会根据测试成绩挑选 checkpoint。SIGTERM 会在优化步边界保存部分适配器并标为 interrupted；该部分不含优化器/RNG 状态，不能冒充完整或精确可续训 checkpoint。

队列设时限、磁盘下限和停止条件；任务终止后按作者授权关机。独立守护进程负责兜底。具体时限和是否触发由服务器记录判断，不能仅凭脚本已启动就声称实例已关闭。

## 仍待完成

1. 14B 四组训练和统一回复采集，按实际成功运行数量报告；预算不足时保留待执行组。
2. 实验 4 正式 150 个事件的 600 条样本及双人审核，在冻结后生成回复。
3. 开发集人工校准，冻结 Judge 规则和错误处理，再启动正式评分。
4. 实验 10 独立人工盲评与裁决、跨 Judge 比较；4B/14B 首批覆盖不能冒充全部模型覆盖。
5. 实验 1、4、5 的正式统计、置信区间、图表与结论。

GPU 队列只训练和采集回复，不调用 Judge，也不生成或伪造人工标签。
