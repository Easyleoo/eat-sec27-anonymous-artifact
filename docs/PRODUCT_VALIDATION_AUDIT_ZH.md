# 真实开源产品补充实验核验与执行安排

更新：2026-10-22。依据作者提供的补充方案核对原始资源。本文区分已验证事实、待落实设置和实际完成的运行，不把方案中的示例结论当作结果。

## 可保留的研究目标

方案 A 验证真实开源情感对话数据和训练工具中的供应链风险；方案 B 对公开情感支持模型做离线、只读行为审计。两者可补充现实工件证据，但单凭公开模型出现有害回复，不能认定它被植入后门。

## 必须先修正的实验前提

| 方案中的前提 | 原始来源核验 | 执行调整 |
| --- | --- | --- |
| ESConv 官方仓库直接提供 Qwen LoRA SFT | 原始 `codes` 使用 BlenderBot，依赖 transformers 4.2.2、torch 1.7.1；`codes_zcj` 也说明使用 BlenderBot-small | 原作者基线复现与 Qwen 迁移实验分别命名；迁移到 Qwen/LLaMA-Factory 时明确记录改变的模型、模板和参数 |
| 原生预处理就是安全过滤 | 已检查的 `_reformat/process.py` 做空白规范化、角色转换和数据划分；`prepare.py` 做编码及特征构建。它们不能直接被称作恶意内容检测器 | 继续检查实际输入处理路径。没有安全检测阶段时填“不提供 / N/A”，不得把“未检测”写成“绕过原生安全过滤器” |
| 可以混用当前 ESConv JSON 与已有官方测试划分 | 根目录说明当前包含 1,300 条；原始 `codes/dataset` 说明对应旧版 1,053 条且已有划分；复现代码还会用 seed13 重新划分 | 固定同一版数据及对应划分；按原始对话隔离训练和测试，禁止先投毒再重划分导致泄漏 |
| MT-Bench 存在可直接引用的官方“情感子集” | 当前公开题目带 writing、roleplay、reasoning 等类别，不能据此直接声称有官方情感支持子集 | 可使用完整 MT-Bench 作为通用能力补充；若自选子集，应预先公开题号和选择依据，并称自定义子集 |
| EmpatheticDialogues 是单轮数据，需任意拼接多轮 | 官方说明是 25k 段情绪情境对话 | 按原始 conversation ID 和 utterance 顺序组织上下文，禁止跨对话拼接制造“原生多轮” |
| 原 Qwen Judge 可继续使用 | 作者已经确认旧接口不可用；现有 Kimi 仅完成开发调用验证 | 替换方案须在开发集人工校准后冻结，不能把新结果标作旧双 Judge 的复现 |

依据：[ESConv 原始代码说明](https://github.com/thu-coai/Emotional-Support-Conversation/tree/main/codes)、[复现代码说明](https://github.com/thu-coai/Emotional-Support-Conversation/tree/main/codes_zcj)、[原生预处理](https://github.com/thu-coai/Emotional-Support-Conversation/blob/main/codes_zcj/_reformat/process.py)、[特征准备](https://github.com/thu-coai/Emotional-Support-Conversation/blob/main/codes_zcj/prepare.py)、[ESConv 数据版本说明](https://github.com/thu-coai/Emotional-Support-Conversation)、[EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues)、[MT-Bench 题目](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/mt_bench/question.jsonl)。

本次取得 ESConv 官方仓库树的版本为 `f262d062ad74cb39b17ea476facc81568ddcba24`。正式运行以该版本实际下载的文件和运行配置为准，不执行远程文档中的任意命令。

已取得该版本的三份原生 TSV：train 8,562 行、dev 2,985 行、test 2,775 行。这里是处理后的行数，不是独立对话数；2% 投毒的分母仍须按预先定义的对话或样本单位登记，不能直接混用。已核验的 vanilla 输入处理执行分词、按角色建立训练目标及长度截断，没有在这条已检查路径中发现语义安全检测阶段。

## 数据与模型资源处理

- [MindChat-Qwen2-4B](https://huggingface.co/X-D-Lab/MindChat-Qwen2-4B) 和 [Emotional-llama-8B](https://huggingface.co/OEvortex/Emotional-llama-8B) 的模型页面可以访问。下载和运行前继续核对实际配置、权重、模板、语言适用性及许可，不能只凭名称猜测模型参数量或基础家族。
- `lmsys/esconv` 及 `EICAP-Team/qwenbase-empathetic-sft` 本次网页访问未成功确认；这不等于证明仓库不存在。ESConv 优先取官方 GitHub 数据。
- [Amod 数据卡](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) 当前要求接受访问条件，提供问答对且没有预定义划分，并载有内容修改限制。因此不直接把它当成可任意改写投毒的原生多轮训练集。
- 方案的“全部离线”和“使用在线双 Judge”表述需要统一。被测模型可全部本地推理；是否调用评分 API、使用何种数据及如何人工校准应单独写清，不能把调用 Judge 与攻击线上服务混为一谈。
- 方案所列两篇数据集论文的 BibTeX 应直接采用各官方仓库给出的记录；不要复制未核实的作者名单和会议名称。

## 当前两个小时的执行顺序

1. 使用已在本地的原始 Qwen3-4B 权重完成方案 B 的基座开发对照，填补下载期间的有效 GPU 工作。只读推理，无适配器、无微调。
2. 14B 下载完成后执行既定两步训练/重载检查，成功后接续 clean、ARE、Rand-family、Raw 的 seed42 队列。新方案的资料核验在 CPU 侧进行，不中断有效训练。
3. 以固定 ESConv 版本准备数据结构与清洗流程审计，再冻结方案 A 的可执行配置。不会为了占用 GPU 就跳过数据隔离或捏造不存在的原生过滤器。
4. 每 10 分钟核对真实训练步数、生成文件、GPU 显存和备份。正常运行保持安静；阻塞则修复，无法及时修复则保存并停止空耗。两小时监控窗口结束于北京时间 20:44；既有预算保护继续生效。

18:54 已通过 14B 两步训练和重载检查并进入 matched clean seed42 正式训练；18:56 观察到第 20/2500 个优化步及约 37GB 显存占用。其余条件由既定队列接续，不能预先写作全部完成。

## 已完成的开发对照

原始 Qwen3-4B、原生 chat template、关闭思考、greedy、最多 256 个新 token；对现有 20 个开发事件的 A/B/C/D 条件共 80 条输入完成生成。80 条 ID 唯一，回复非空，无输入截断，无达到生成上限的回复；原始回复及设置已备份。

这批是 native-chat 开发对照，不能与先前训练模板、64-token 上限的适配器结果直接混算。它不是 MindChat/Emotional-llama 的结果，也不是方案 A 或 B 的正式完成证据。正式配对集尚需人工审核，Judge 可靠性和有害输出比例尚未验证。
