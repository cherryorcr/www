# Semantic Permutation Stage-2 执行计划

日期：2026-09-25  
状态：stage-2 与独立顺序确认已完成；semantic permutation 保留为研究候选，未晋升为默认主方法

## 1. 目的

将已完成的 stage-1 守恒 payload 置换实验扩展到第二个增量阶段，确认 semantic permutation 的收益是否只存在于早期新类，还是能够在持续链条中保持。

当前冻结快照只包含 `s0/s1/s2`，因此本轮真实可执行的是 base 加两个增量阶段；不虚构 stage 3。

## 2. 固定协议

- 数据集：Cora、Citeseer；
- 顺序：`order19601/19602/19603`；
- 阶段：`stage=2`；
- arms：`c0_320`、`identity`、`semantic_perm`、`role_perm`、`random_perm`；
- Student seeds：`150..155`；
- 文本：top-5，显式 320 字符 cap，MiniLM 冻结重编码；
- Student：原始两层 GCN、trainable text projection、双向 InfoNCE；
- 拓扑：sealed C0 edge index bitwise copy；
- 选择：不使用 report/test F1；
- 审计：source coverage、payload multiset、每类 mass、old prefix、edge_index。

## 3. 执行命令模板

远程工作目录和环境保持上一轮不变：

```text
/home/pc/miniconda3/envs/tagsam/bin/python -u \
  code/v2_llm_fusion/run_tgc_two_prototype_role_permutation.py \
  --dataset <cora|citeseer> \
  --order <order19601|order19602|order19603> \
  --stage 2 \
  --source-root /home/pc/research/TAG_condensation/results/tgc_r10_joint_confirmation_20260923b \
  --cache-root /home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919 \
  --output /home/pc/research/TAG_condensation/results/tgc_two_prototype_role_permutation_20260925/<dataset>_<order>_s2
```

六个 cell 完成后，先检查所有 arm 的守恒审计，再汇总 `semantic_perm_vs_c0_320` 和 `semantic_perm_vs_identity`；不先做 selector、不调 role 权重。

## 4. 晋升判据与实际结果

stage-2 只作为确认，不单独晋升。semantic permutation 进入下一轮的必要条件：

1. 六个 stage-1 与六个 stage-2 cell 合并后，至少两个数据集的平均收益为正；
2. semantic permutation 相对 random permutation 不能系统性更差；
3. 最差 cell mean 不低于 `-2 pp`，并报告 old/new F1；
4. 所有守恒断言通过；
5. 不依赖 report/test F1 选择。

实际结果：

- 开发集 stage-1/stage-2 合并：`+1.414 pp`，8/12 cell 为正；
- 独立顺序 `order19605/06/08` 合并：`+1.878 pp`，8/12 cell 为正；
- 独立集相对 random permutation 的平均差异为 `+2.217 pp`；
- 但独立集最差 cell 为 `-3.630 pp`，Citeseer new-class F1 平均为 `-0.698 pp`；
- 所有 stage-2 与独立集守恒审计通过。

因此第 3 条“最差 cell 不低于约 `-2 pp`”和新类塑性稳定性未满足。当前结论是：semantic permutation 有可复现的平均机制信号，但不能晋升为默认主方法；C0 继续冻结为主线。
