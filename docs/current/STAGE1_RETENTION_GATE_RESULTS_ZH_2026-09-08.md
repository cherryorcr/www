# Stage-1 Response-Retention Gate 结果（P5，档A）

日期：2026-09-08
数据：服务器 `problem_validation_results/stage1_gate_2026_09_08/`（20 runs = 5 condensation seeds × λ_r∈{0, 0.05, 0.1, 0.3}）
实现：`run_stage1_retention_gate.py` + `native_core.py` 的 `response_retention_weight` 路径（档A：self-distillation on synthetic rows，KL over graph-to-class-prompt logits，旧行 target 来自 stage-0 teacher、新行来自 current-delta teacher）。单测 28/28 通过。
设置：text_anchor_weight=0.01 固定；stage-0 memory 与 teacher 来自 static_diag 各 seed 的 olr001 run；students 100/101；dev-only。

## Gate 判定：未通过（0/5 seeds，所有 λ_r）

| seed | λ_r=0 | λ_r=0.05 | λ_r=0.1 | λ_r=0.3 |
|---:|---|---|---|---|
| 44 | outer 0 | outer 0 | outer 0 | outer 0 |
| 45 | outer 80 (+0.09) | outer 0 | outer 0 | outer 0 |
| 46 | outer 0 | outer 0 | outer 0 | outer 0 |
| 47 | outer 0 | outer 0 | outer 0 | outer 0 |
| 48 | outer 0 | outer 0 | outer 0 | outer 0 |

通过条件（≥4 seeds 选 outer>0 且 dev≥outer0）远未满足。

## 关键机制发现（比判定本身更有价值）

### 1. Retention loss 优化得非常好——但没用

与 ASM loss（P2：100步只降0.2–1.2%）形成鲜明对比，retention KL 从 0.06–0.32 降到 0.001–0.005（降幅 92–99%+）。**梯度通路是通的、目标是可优化的，但把它优化到近零并不能保住 student 性能。**

### 2. λ_r 的作用是"减缓下降"，且方向依赖 seed

以 outer=100 的累计 dev 为例：

| seed | λ=0 | λ=0.05 | λ=0.1 | λ=0.3 | λ 的作用 |
|---:|---:|---:|---:|---:|---|
| 44 | 53.57 | 57.65 | 60.79 | **64.57** | 单调保护（+11 pp），但仍低于 outer0 的 70.97 |
| 45 | 47.67 | 44.27 | 42.12 | 42.92 | **有害** |
| 46 | 52.10 | 48.58 | 49.37 | 47.09 | 有害 |
| 47 | 60.62 | 56.50 | 55.87 | 54.56 | 有害 |
| 48 | 57.66 | 52.19 | 51.70 | 50.51 | 有害 |

### 3. 解释：目标错位是根本问题，不是缺约束

三条证据合起来（P2 + 本实验）指向一个结论：

- ASM loss 优化不动（P2）→ X 的移动近似弱扰动；
- retention/anchor loss 可以优化到近零 → X 被拉向"让冻结 teacher 在 synthetic 图上复现 target"的流形；
- 但 student 是**从头重训**的，它对 synthetic TAG 的利用方式与冻结 teacher 的前向响应不同构。**满足冻结 teacher 的任何响应约束，都不等于保住"可重训性"（trainability）**。

也就是说：当前所有 loss（ASM matching、text anchor、response retention）都以"某个固定模型的响应"为目标，而真正要保的量是"任意新 student 在这张小图上训练后的表现"。前者可满足、后者在优化中持续受损——**初始化（frozen compact）恰恰是唯一直接对齐可重训性的构造**（它由真实数据统计构造，未经过任何响应目标的扭曲）。

## 最终路线判定（PROTOCOL V2 §7 正式生效分支 B）

Gate S（P2）与 Gate 1（本实验）在 5 seeds 上先后失败，且失败模式一致。证据充分支持：

1. **方法主线转向构造式**：更强/更稳的初始化（P2 显示 frozen 质量方差是最大杠杆：57.8–80.1 across seeds）、质量自适应新旧预算分配（P3 显示分配方向由旧 memory 质量决定）、类平衡代表点选择；
2. **优化组件降级**：ASM/anchor/retention 保留为消融与"初始化差时的修复"（c44 的 λ_r 单调保护、c48 静态 olr01 的 +25 pp 都说明它在坏初始化下有修复价值——可以做成"仅当初始化 dev 低于阈值才启用"的条件式组件）；
3. **论文叙事升级**：三个 loss 家族（matching/anchor/retention）× 5 seeds 的系统失败 + "响应约束可满足但可重训性不可保"的机制解释，本身构成一个有分量的负结果+诊断贡献，比"又一个 loss 项"更有辨识度；
4. 下一个 gate 应该是**构造式 gate**：新初始化方案在 ≥4 seeds 上稳定超过现有 frozen compact（这是纯前向构造，没有优化失败的风险面）。

## 审计说明

- λ_r=0 与 λ_r>0 的 outer=0 数值逐 seed 相同（初始化不经过优化，符合预期）；
- seed 45 λ=0 选 outer 80 但增益仅 +0.09 pp，在噪声内；
- 本实验 stage-0 memory 用的是 static_diag 的 frozen artifact（非 preserve 链路的 anchor artifact），与旧 recursive 实验的 stage-1 数值不可直接横比；
- Cora 全分区已污染，以上均为开发证据。
