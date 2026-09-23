# NicheFlow

**面向质量与成本权衡的多生态位 LLM 工作流搜索与路由研究。**

NicheFlow 将工作流表示为可验证的图，搜索节点职责、连接和模型分配，保留不同质量与成本位置上的候选，并探索按请求选择合适工作流。本仓库包含研究原型、可审计执行记录机制、离线测试、历史实验配置及下一阶段实验设计。

> 当前为研究代码，尚无论文最终结论。最新完成的是 v071 数学模型筛选；v072 跨任务实验处于规划阶段，执行适配与评分协议仍需补齐。注册了数据集或通过离线测试，不代表已经完成真实模型实验。

## 当前进展

| 阶段 | 状态 | 范围 |
|---|---|---|
| 工作流图、执行器、档案、搜索、路由与预算记账 | 已有实现和离线测试 | 历史实验实现，含明确记录的工程补充 |
| v050–v060 | 历史实验与修订 | MATH 工作流实验、固定对照、恢复和评分诊断 |
| v070–v071 | 已完成独立模型筛选 | 新模型适配、输出行为及数学语义评分复核 |
| v072 | **已规划，未运行** | 数学、代码、多跳阅读；每域五轮短工作流搜索 |

主方法执行器尚未完成向新三档模型与新版通用评分的迁移；不能直接运行旧配置来复现 v072。

## 实验模型选择

下一轮主模型池固定以下部署角色。角色与价格档位不等于已经证明的能力排序。

| 角色 | 配置 |
|---|---|
| L：本地模型 | Qwen3.5-9B，已部署的固定社区 AWQ 权重 |
| M：低价云端 | `qwen3.7-flash-2026-07-15` |
| H：高价云端 | `qwen3.8-max-0902` |
| 外部强单模型基线、工作流提议器 | DeepSeek Flash；两种用途分别记账 |

v071 在 42 道新 MATH train 难题上，每模型独立回答一次：L **37/42**、M **39/42**、H **41/42**、DeepSeek **41/42**。全部使用非思考模式、8192 输出上限；截断按冻结协议计零。本地五次失败和 Max 唯一失败均为截断。

这一小样本没有证明 Max 稳定优于 Flash 或 DeepSeek。DeepSeek 必须保留在主结果对照中；省费结论不能只以昂贵 Max 为参照。本地 API 费用为零，GPU 占用与租金另行报告。

- [结果、费用与逐项局限](reports/qwen38_v071_20260923/ANALYSIS_AND_RECOMMENDATIONS.md)
- [实验前模型决策](docs/MODEL_DECISION_20260923.md)
- [三档选型及相关研究](docs/THREE_TIER_RESEARCH_SELECTION_V071.md)

## 离线快速开始

在仓库根目录使用 Python 3.12（项目声明支持 Python 3.10+；本次打包验证使用 3.12）。以下测试与 fixture 无需 GPU、模型权重或 API Key，也不会发起付费推理。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python -m unittest discover -s tests -v
python -m nicheflow.cli run --mode fixture \
  --config configs/smoke.json --run-dir runs/offline_fixture
python -m nicheflow.cli status runs/offline_fixture
```

`fixture` 使用合成后端检验执行与记账，不测量真实模型能力。`configs/smoke.json` 的旧本地模型字段不会在此模式加载。历史配置里的 Qwen2.5 仅用于保留实验身份，已不属于当前部署方案。

2026-09-23 打包验证：从 Git 暂存内容导出的干净目录中，Python 3.12 / NumPy 2.3.5 下 **157 项测试通过**，上述 fixture 与状态读取通过，原始方案文档哈希一致。真实模型及跨任务实验不包含在此次离线验收中。

```bash
python -m nicheflow.cli doctor
python -m nicheflow.cli --help
```

`doctor` 分别列出源文档校验、可选依赖、评分器和隔离执行环境；离线安装下部分真实实验能力不可用是预期状态。其返回不等于所有实验协议已经验证。

真实模型实验需要另行准备本地服务、外部 API 环境变量、固定模型身份、数据来源和评分依赖。先阅读对应实验文档；`scripts/` 中历史服务器脚本含特定机器路径，不是通用一键部署入口。API Key 不写入配置、代码或提交记录。

## 代码组织

| 路径 | 内容 |
|---|---|
| `nicheflow/graph.py`、`runtime.py` | 工作流图、契约检查与执行 |
| `nicheflow/archive.py`、`mutations.py`、`search.py` | 档案、候选变异与搜索组件 |
| `nicheflow/router.py`、`policy.py`、`research_policy.py` | 路由及版本化策略 |
| `nicheflow/ledger.py`、`main_budget.py` | 事件记录、费用核算和预算限制 |
| `nicheflow/datasets.py`、`scoring.py`、`vendor/` | 数据适配及评分组件 |
| `nicheflow_probe/` | 前置条件与执行组件探测 |
| `configs/` | 按版本冻结的历史配置与模型筛选计划 |
| `scripts/` | 数据准备、评估、恢复与服务器实验脚本 |
| `tests/` | 不调用真实模型的回归测试与小型固定样例 |
| `data/` | 测试/复现所需的数据快照、题单和来源哈希 |
| `docs/` | 实验设计、工程记录和问题—验证清单 |
| `reports/` | 经过筛选的结果摘要与预算，不含完整运行日志 |

保留的 `NicheFlow_3.4_技术方案.docx` 是运行时校验的原始方案文档；工程补充见 [ENGINEERING_RECORD.md](docs/ENGINEERING_RECORD.md)。该文档的存在不表示代码是未经补充的逐字实现。

## 下一轮实验

[v072 规划](docs/MULTIDOMAIN_V072_EXPERIMENT_PLAN_20260923.md)：MATH、MBPP、HotpotQA 三个任务域，各 200 题，分开发、筛选、校准、终评；每域五轮、每轮最多两个新工作流。终评在模型配置与候选冻结后才运行。

Max 规划用量约 **2100 万 token**（输入 1440 万、输出 660 万）；建议预留输入 2000 万、输出 1000 万。该估算依赖平均长度假设，不是已消费用量或已实现的硬预算保证。详见[可复算预算](reports/multidomain_v072_plan_20260923/max_token_budget.json)。

进入实验前的修改与验证见[问题清单](docs/ISSUE_VALIDATION_MATRIX_20260923.md)和[下一轮修改方案](docs/NEXT_REVISION_PLAN_20260923.md)。核心工作包括完成契约/停止行为、语义评分迁移、隐藏代码测试、多跳阅读指标与全局 token 预算。

## 复现与数据边界

- 题目、配置、模型身份和评分版本需冻结；标准答案和参考解只供评分，不能进入候选生成、推理或路由输入。
- 搜索、筛选、校准、终评分开；选型用过的 v071 题目不再作为独立终评。
- 报告准确率/任务指标、API 输入输出用量、GPU 时间、延迟、失败和截断；搜索与校准费用单列。
- 不自动重试未知计费请求，不使用更改后的代码直接续接旧运行；按相应版本的兼容性与恢复协议处理。
- Git 不包含权重、虚拟环境、原始事件日志、凭据、旧交接包或个人对话记录。这些材料仍保留在本地；部分历史文档的原始证据链接需要原实验归档。
- `configs/release_manifest.json` 与 `scripts/verify_release.py` 属于旧服务器交接版本，校验对象包含旧 README 和外置资源，不能用作本次 Git 仓库的完整性验收。

数据来源和用途见 [data/README.md](data/README.md)，文档导航见 [docs/README.md](docs/README.md)。第三方评分器保留各自许可证及来源记录，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本仓库未为原创代码授予额外的开源许可。
