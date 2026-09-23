# 固定前置条件检查数据

来源：[EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)，公开 MATH 数据副本，数据集页面标示 MIT 许可。这里只使用 train 划分中的 12 道题。确切来源、抓取时间、行号及原始数据访问方式记录在 `manifest.json`。

`tasks.jsonl` 中的 `gold` 和 `reference_solution` 仅供独立评分器使用，绝不能发送给待测工作流。模型只接收 `question`。

抽样在任何模型推理之前完成。三个题型、两个难度层各两题；重复题也预先固定。题目全部采用有理数标量答案，因此本轮结论不能推广到所有数学任务。

抽样所使用的原始 API 响应保存在 `raw/`，以便核查。复现实验直接使用随包固定文件，不需要重新抽样或下载整个数据集。

## Git 仓库中的其他固定数据（2026-09-23）

上文的“12 道题”仅指根目录 `tasks.jsonl`，不是全部实验数据。

| 目录 | 用途 |
|---|---|
| `source_math_v041/` | 固定来源的 MATH train parquet 与 MATH-500 原始文本；下载 URL、revision、SHA256 见 `downloads.json` |
| `main_math_learning_v1/` | 历史开发/校准划分及其 manifest |
| `main_math500_v1/`、`main_math500_v050/` | 历史学习与评价协议的固定题单、来源和哈希 |
| `fixed_validation_v060/` | 新抽取的固定对照验证题单 |
| `model_selection_v070/`、`model_selection_v071/` | 模型选型题单；已用于开发决策，不再作为独立终评 |

MATH train 来源为 [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)，MATH-500 来源为 [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)。这些小型快照保留用于哈希验证、离线测试和历史复现；不随代码重新许可数据。具体实验的划分和排除规则以对应 manifest 为准，不因目录同时存在而混用题目。

`raw/`、下载缓存和生成的 benchmark 目录不加入 Git。v072 的 MBPP、HotpotQA 与新增 MATH 划分尚未冻结，也未包含在本仓库中。
