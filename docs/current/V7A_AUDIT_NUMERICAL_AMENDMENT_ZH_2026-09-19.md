# v7-A 审计数值修订说明

日期：2026-09-19

本说明只记录 v7-A 独立审计中的浮点重算口径修订，不修改冻结协议、训练代码、候选、selection 结果或 report checkpoint。

## 发现

首次独立审计对保存的 selection score 使用 float64 cosine 重算，而 frozen selector 使用 float32 张量计算。两者在非零 Q 上出现微小差异，最大绝对差异为：

```text
6.231879980815418e-6
```

这不是训练或 checkpoint 损坏，也没有造成任何 cell 的选择变化：float32 与 float64 的 selected candidate 均为 C0，selection change 为 `0/8`。

## 修订

审计脚本改为使用与 frozen selector 相同的 float32 cosine 公式，并使用 `1e-7` 的审计阈值。修订后：

- float32 公式重算最大误差：`0`；
- float64 诊断最大差异：`6.231879980815418e-6`；
- cell selection 变化：`0/8`；
- 修正版 reload audit：`passed`。

原始首次审计日志保留在：

- `results/v7a_gradient_append_20260919/audit.log`
- `results/v7a_gradient_append_20260919/audit_reload.log`
- `code/v2_llm_fusion/audit_v7_gradient_append_before_precision_fix.py`

## 影响范围

该修订只影响审计脚本的数值等价判断，不影响：

- v7-A 冻结协议；
- 144 次正式 student fit；
- 选择 seed 或 report seed；
- candidate memory；
- Q 的 frozen selection 结果；
- 最终 F1、old/new F1 或成本统计。

因此 v7-A 的结论仍为：selector 未通过晋升门禁，C0 保留，v7-B 不启动。
