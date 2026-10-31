# 链路预测检查点重载的数值审计说明

本说明在正式 540 次训练完成后的独立审计阶段添加，不改变训练协议、参数、checkpoint 或主表原始分数。

第一轮审计采用保存预测对应指标完全一致、重载预测 atol=1e-4/rtol=1e-5、重载 AUC/AP 绝对差不超过 1e-4 三项检查。Cora 的 split29003/stage0/C0/seed105 AP 差 0.00014206，Citeseer 的 split29003/stage1/full_raw_equal/seed103 old AP 差 0.00016835；重载预测通过 tensor allclose，但近似并列样本交换排序，导致指标硬阈值检查失败。

处理方式：

1. 保留并公开原始 rank 阈值告警，不宣称严格逐排序复现通过。
2. 源数据、edge split、memory、checkpoint 哈希检查与保存分数的指标完全复算仍为硬要求；重载 tensor allclose 要求不放宽。
3. 每个 checkpoint 额外在 CPU 双精度下重新推理，保存全部阶段/切片指标作为数值敏感性控制，不替换主表原始指标。
4. `passed` 仅指 artifact integrity、无泄漏检查、历史冻结、保存预测的指标精确复算及重载预测数值等价；`strict_rank_metric_check_passed` 单列。
5. 正文比较方法差值时还核对双精度下方向和大小，不将浮点排序扰动解释为方法收益。
