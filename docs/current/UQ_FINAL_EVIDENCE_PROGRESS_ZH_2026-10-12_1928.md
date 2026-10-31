# Unified-Q-EF 证据增量记录（19:28 快照）

本记录只对应 `results/uq_final_evidence_20261012/snapshot_1928.json` 和
`results/uq_final_evidence_report_20261012_1928_protocol_audit/summary.md`；
不替代此前的 sealed 主表或预注册方案。只读采集获得 18 个结果项、64 条完整方法轨迹，零校验错误。
SportsFit 第三构造种子的控制轨迹及 GECC-soft BookHis 轨迹仍未完成，不纳入终点比较。

## 新完成的 Q 归因

BookHis 在同一个类别顺序、三个构造种子、每个 cell 三个 Student 种子下，sealed Final Macro-F1：

| 控制 | 均值 | 与对应 independent Q 的配对差 | 正向 cell |
| --- | ---: | ---: | ---: |
| Shared Q | 46.55 | -1.92 pp，95% CI [-5.02, 1.17] | 0/3 |
| Independent Q | 48.47 | 参照 | -- |
| Shared Q, no residual | 47.25 | -0.49 pp，95% CI [-5.15, 4.17] | 1/3 |
| Independent Q, no residual | 47.75 | 参照 | -- |

这仍是单一类别顺序，不足以推断跨顺序总体显著性。共享 Q 的图文 top-1 来源一致性是机制属性；
本批精度并未证明共享带来额外收益。图文一致性、参数量、峰值显存与硬文本可审计性应分开报告，
不能把这些性质写成精度提升的原因。ArXiv 的完整 9-cell 主方法确认仍以 19:07 记录为准，
不得由控制组的单顺序结果替换。

## 配对协议与运行状态

汇总器新增合法池各阶段规模、名义/实际凝聚率、预算规则、类别下限调整、Student 实现哈希、
评估与历史访问声明的一致性检查；现有快照通过检查，五项回归测试通过。
保留不同方法的 synthetic topology 字段作为方法差异，不能强行要求 STAD 与无边主方法相同。
目前所有新增强基线仍只有阶段中间结果，不存在可报告的最终配对 F1。

GPU0 正运行 BookHis 的 GECC-soft 原生 FCM；GPU1 正运行第三构造种子的 Q 控制组。
磁盘约剩余 29 GB。`tools/run_uq_remaining_confirmation_batch_20261012.sh` 已在远程排队，
等待当前 GPU1 队列退出后，先完成 BookHis 和 SportsFit 的 `o9101`、三个构造种子，
主方法/CaT-CCTP/GECC-hard 同协议配对。它对不足 8 GiB 磁盘或非空闲 GPU 暂停，
对已有完整 cell 复用，遇到部分结果则保留现场并停止；后续两个类别顺序尚未排队。

GECC-soft、TAGSAM、STAD 的共同 Student 运行只能称机制适配，不是官方原协议复现。
即使首个完整 cell 落盘，单顺序单构造 seed 仍只能作为探索性对照，不能生成跨顺序 CI
或写出确定的 SOTA 结论。STAD 本地生成器与格式筛选的偏差见 19:07 记录。
