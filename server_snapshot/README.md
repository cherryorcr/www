# Classical Continual TAG Condensation

当前进展请先读 [P2 收尾与小图对照结果](docs/current/P2_CLOSEOUT_SMALL_BASELINE_RESULTS_ZH_2026-09-16.md)、[冻结协议](docs/current/P2_CLOSEOUT_SMALL_COMPARISON_PROTOCOL_ZH_2026-09-16.md)、[P2 验收](docs/current/P2_TEXT_COMPRESSION_EXECUTION_AND_AUDIT_ZH_2026-09-16.md) 和 [文档索引](docs/INDEX.md)。下面保留 2026-09-08 工作包的归档说明，其中 BERT/native 设置与早期结论不代表当前 MiniLM/v4 主线。

这是截至 2026-09-08 的经典持续 TAG 图凝聚工作包。它从原研究目录中筛选并复制了当前仍有用的报告、协议、代码、实验日志、compact artifacts、Cora 输入与运行环境记录。原目录没有移动或删除。

## 当前研究状态

现行开发协议是 Cora 类别增量：

```text
stage 0 [0,1] -> stage 1 [2,3] -> stage 2 [4,5,6]
```

阶段结束后不可重读历史原始训练图，只能保留累计 5% 根节点预算内的 synthetic TAG。下游固定为冻结 BERT CLS、两层 GCN、可训练文本投影、graph-text InfoNCE 和七类文本提示节点分类。

已经确认：

- 官方 TAGSAM 静态链路能运行，但七类 synthetic student 明显低于 full teacher；
- 旧 memory 质量和新增 compact block 都会造成持续损失；
- ASM 更新 synthetic `X` 会破坏 graph-text correspondence；
- Text-Anchored ASM 在固定 stage-2 对照中有效。

尚未解决：

- 严格 stage 1 中所有 ASM/Text-Anchored 轨迹都选择 `outer=0`；
- 严格 stage 2 的平均增益不跨 condensation seed 稳定；
- teacher response-retention 仍是下一步方案，尚未实现或运行。

Cora 的 test、confirmation 和 future 已经在开发中访问。本包中的这些数值只能称为开发证据，最终论文测试需要新的未访问数据集。

## 推荐阅读顺序

1. [逐项实验讲解](docs/current/EXPERIMENT_WALKTHROUGH_ZH_2026-09-08.md)
2. [当前进展总结](docs/current/CURRENT_PROGRESS_SUMMARY_ZH_2026-09-08.md)
3. [Text-Anchored ASM 技术报告](docs/current/TEXT_ANCHORED_ASM_TECHNICAL_REPORT_ZH_2026-09-08.md)
4. [文档索引](docs/INDEX.md)
5. [结果目录说明](results/README.md)
6. [环境和复现说明](environment/ENVIRONMENT.md)
7. [下一步 response-retention gate](docs/protocol/NEXT_STEP_RESPONSE_RETENTION_GATE_ZH_2026-09-08.md)

也可以直接打开自包含的 [交互报告](reports/interactive/report.html)。

## 目录结构

```text
continue_tag_condensation/
├── docs/
│   ├── current/       当前结论和逐项归因
│   ├── protocol/      setting、执行规则和方法设计
│   └── lineage/       方向重置、冻结边界和历史决策
├── code/
│   ├── native_tag_v1/ 当前可运行后端和测试
│   ├── official_static/minimal_repair/
│   ├── run_snapshots/关键运行时的代码快照
│   └── data_adapter/
├── data/
│   ├── cora.pt
│   └── prepared_v2/   已签名的三阶段输入和评估文件
├── results/
│   ├── native/        关键 JSON、日志和 compact artifacts
│   └── official_static/
├── reports/interactive/
├── environment/
└── tools/
```

## 快速校验

在包根目录运行：

```powershell
python tools/verify_package.py
```

验证代码测试：

```powershell
Set-Location code/native_tag_v1
python -m unittest discover -s . -p "test_*.py"
```

完整 GPU 运行建议使用 [环境说明](environment/ENVIRONMENT.md) 中记录的 Linux/CUDA 配置。新的输出目录必须使用新路径，因为 runner 会拒绝覆盖已有实验。

## 筛选原则

本包保留了支撑当前结论和继续研究所需的内容。为避免大量重复，以下内容没有复制：

- 每个 student/teacher 的模型 checkpoint；
- 每次 run 中重复保存的同一套代码；
- HTML 调试文件和失败截图；
- 已冻结的早期 prototype-memory、D2/A2、GraphCLIP 等大批历史实验；
- 约 440 MB 的 BERT 权重。prepared 数据已经包含冻结 BERT CLS，manifest 保留了模型哈希。

完整历史仍保存在原目录：

```text
D:\Adocs\简历\research\graph-condensation\TAGSAM
```
