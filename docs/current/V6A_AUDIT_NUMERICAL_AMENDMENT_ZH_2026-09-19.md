# v6-A 审计数值补充说明

本文件是实验完成后的审计说明，不修改 `V6A_LOCAL_MARGINAL_FROZEN_PROTOCOL_ZH_2026-09-19.md`，也不改变候选、数据、seed、训练或晋升门禁。

checkpoint 重载后的真实图 logits 使用 `rtol=1e-4, atol=1e-5` 做数值等价核验。GPU 浮点非确定性造成的微小差异同时记录最大和平均绝对误差；不以跨次 CUDA forward 的 bitwise 不同判定训练工件失效。

审计 runner、补充说明和最终误差摘要单独记录 hash。
