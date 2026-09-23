# v052 服务器中断后的明确恢复

2026-09-22，用户明确要求“补一下重跑”。新SSH端口33474；旧服务器指纹匹配。中断时1921/1937次执行完成，4593次模型调用有回执，API支出约1.626492228美元。唯一未完成调用是第496索引题的本地Qwen节点d，无未知付费API结果。

恢复入口 `scripts/recover_v052_comparison.py` 和启动脚本 `scripts/run_v052_recovery.sh` 使用原服务器项目 `NicheFlow_Evaluation_v052` 中的冻结执行代码；未修改算法模块。原目录 `runs/comparison_v052_before_after` 保持原样。新目录 `runs/comparison_v052_recovered` 使用逐字节相同的已完成事件前缀，原日志末尾未完成本地调用的reservation/start作为废弃尝试完整写入recovery_manifest和恢复事件；新日志中仅重发该本地请求，且强制核对请求内容一致。没有伪造成功回执，没有删除源证据。

保留原路由计划、数据、模型、解码与费用设置。原24小时墙钟上限（包含停机时间）和API上限均不增加；废弃的本地尝试也计入调用上限。其API成本为0、耗时未知，报告明确不将未知耗时计为0。环境在模型加载后、首次调用前与原环境核对。运行失败不自动再试。

已完成离线模拟：真实执行入口模拟掉电，恢复复用已有回执、仅重做未完成本地请求、完成后报告审计通过；同时验证拒绝API未完成调用、非零本地费率、不合法尾部和覆盖已有恢复目录。服务器零调用预检通过，确认1921次复用、16次待完成。真实启动和完成状态须以服务器新目录日志及退出码核实。

后台会话：`nicheflow-v052-recovery`。输出文件：

- `runs/comparison_v052_recovered.console.log`
- `runs/comparison_v052_recovered.exit_code.txt`
- `runs/comparison_v052_recovered.report.log`
- `runs/comparison_v052_recovered.report_exit_code.txt`
- 新结果目录中的 `comparison.json`、`COMPARISON_REPORT.md`、`completion_audit.json`、`recovery_manifest.json`

后续查进度应读新目录；旧监控默认读旧目录，会停留在1921。报告核验同时检查旧日志哈希、已完成前缀逐字节相同、原训练证据不变和本地恢复请求一致。

本轮结束后的联合分析仍包括 [种子依赖与搜索开放性](SEARCH_EXPLORATION_MEMORY_20260922.md)；此次只恢复冻结评测，没有修改搜索机制或追加训练。

实际启动：2026-09-22 18:06:58 北京时间，后台会话已建立。随后核实恢复环境检查通过，写入锁有效，日志链与计划完整，未出现学习事件；中断的本地节点已重新发起，尚未完成时进度仍为1921/1937。最终完成情况以新目录报告核验为准。

完成核实：2026-09-22 18:12:54 北京时间报告生成，1937/1937执行完成，评测与报告退出码均0；completion_audit的verified、recovery_verified、source_unchanged均为true。4633次调用完成，另有1次原中断本地尝试；无执行错误、未知费用或学习更新，89次唯一执行截断。API总支出1.634890878美元。结果与报告本地归档于 `reports/evaluation_v052_recovered_20260922/`。三组学习后准确率86.4%、87.2%、82.6%，初始化均78%；Flash基线89.8%、Qwen64.2%。学习后相较初始化提升，但三组均未超过Flash，且成本更高；不能把这直接归因于种子或搜索约束，后续按记忆中的联合分析议题诊断。
