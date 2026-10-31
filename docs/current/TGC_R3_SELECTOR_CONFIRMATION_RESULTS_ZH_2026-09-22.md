# TGC-R3 结项：独立条件下，候选生成与无标签选择都尚未可靠

日期：2026-09-22。状态：正式矩阵、独立重载审计、本地结果回收完成；未晋升，未启动递归 rollout 或大图扩展。

## 1. 结论先行

本轮完成上一轮提出的最小下一步：固定 `warm-up=64 + suffix=1` 的 student-to-old-X 修订，换未用于该分支调参的类别顺序和随机种子，比较原 gate、从头训练的 64-step new/mixed loss 和 32/64/128-step response curve。

结论比 R2B2 更明确：**目前不是“候选已可靠，只差一个 gate”，而是候选本身的正收益也未在独立顺序/种子条件下复现。**

- 未加 gate 候选：s2 相对 C0 平均 **-0.64 pp**，相对等幅随机修订 **-0.78 pp**，四个条件仅一个均值为正。
- 预注册主策略 response curve：s2 平均 **-0.50 pp**；八个增量阶段接受两次，两次最终 report 均值都为负。
- teacher 对照：s2 平均 **+0.60 pp**，四个条件均值均为正；但 s1 平均 **-0.64 pp**，不能据此事后把 teacher 换成本轮胜出的完整方法。
- 648 条记录、216 次实际正式训练、432 条精确复用记录，独立审计 0 errors。失败是性能门槛未通过，不是训练未完成或指标计算错误。

w64-k1 candidate 在八个增量条件中的六种子配对 95% t 区间均包含零。上述正负是样本均值和预定工程决策，不是统计显著优劣或对所有 TGC 机制的否定。

## 2. 本轮具体做了什么

| 项目 | 本轮设置 |
|---|---|
| 数据 | Cora、Citeseer |
| 新类别顺序 | order19221、order19222；显式类别组见冻结协议 |
| memory | 累计合法 pool 的 5%，每个顺序从零重建 C0 |
| 元优化种子 | train 7800/7801；validation 7810/7811 |
| selector 种子 | 7820/7821/7822，与元优化及 report 均不重叠 |
| report 种子 | 9600--9605，共六个 |
| 候选 | C0 上 warm-up 64 步、detach 前缀、可微 Adam suffix 1 步，outer 16 步 |
| 可修改范围 | 旧 synthetic X 的有限行，80% core 保护，最多 10% 旧行，相对位移半径 0.1 |
| 不修改 | 新 block、文本及 embedding、全部拓扑、rows、row order、mass、depth、birth stage |
| student | 同一两层 384 维 GCN + text projection，冻结 MiniLM，双向 InfoNCE，batch 16、Adam lr 0.001 |
| 正式步数 | max(50,min(2048,16 × memory rows))，所有臂完全相同 |
| 访问约束 | 修订/selector 只接收旧 synthetic memory 和当前 raw delta，不接收历史完整 raw graph 或 report/test 标签 |
| 实验层次 | 每个 s1/s2 共享 C0 父 memory；不是递归修订轨迹 |
| 主指标 | 已见 report-dev 节点的 all-prompt 分类 Macro-F1，百分点制；old/new 单列 seen-prompt pooled F1 |

以下是实际 memory 曲线，不是固定每类 15 行：

| 数据集/顺序 | 每阶段合法新节点数 | s0/s1/s2 memory rows | s0/s1/s2 正式步数 |
|---|---|---|---|
| Cora/19221 | 339 / 437 / 1119 | 16 / 38 / 94 | 256 / 608 / 1504 |
| Cora/19222 | 390 / 437 / 1068 | 19 / 41 / 94 | 304 / 656 / 1504 |
| Citeseer/19221 | 874 / 801 / 554 | 43 / 83 / 111 | 688 / 1328 / 1776 |
| Citeseer/19222 | 806 / 645 / 778 | 40 / 72 / 111 | 640 / 1152 / 1776 |

独立性边界：相同数据集与 report-dev 划分仍已用于项目开发；新增的是类别顺序和元优化/selector/report 随机种子。C0 聚类是确定性的，没有声称使用新的随机 clustering seed。局部查重检查了本地 168 个既有 manifest，未发现这四组完整三阶段分组相同的记录；这不是对全部历史数据曝光的穷尽证明。

## 3. 选择器不是再次复用同一个旧 student

候选生成器仍使用固定 C0 的 64 步前缀；**新 selector 则针对 C0 和 candidate 各自从头训练**，两臂初始化和 minibatch 行序列相同。只在各自轨迹内部读取 32、64、128 步 checkpoint，不跨 memory 复用模型。

- old/new probe：各四个随机批次，每批最多 16 行。
- mixed probe：每批 8 个旧 synthetic 行和 8 个当前 raw 行，包含新旧交叉负例。
- `selected_new64`：64 步 new InfoNCE 改善。
- `selected_mixed64`：64 步 mixed InfoNCE 改善。
- `selected_curve`：32/64/128 步 mixed 改善等权平均，预先指定为主策略。
- 三个新 selector 均要求平均改善至少 1e-4、至少 2/3 seed 改善，并通过逐 seed 的旧 probe loss 保护；curve 在三个 checkpoint 都检查保护。
- `selected_gate` 保留原 R2B2 两个 validation suffix seed 的接受规则，作为比较。

probe 无分类标签，但与候选构造共享合法数据来源，不能称为独立 held-out 数据。全部条件的选择决策先保存并统一 `seal`，随后才启动 report 训练；selected 策略直接引用所选 memory 的源 checkpoint，不重复训练。

## 4. 最终阶段主要结果

每格为六个 report seeds 的平均 Macro-F1；总体均值对四个数据集-顺序条件等权。

| 方法 | Cora/19221 | Cora/19222 | Citeseer/19221 | Citeseer/19222 | 总体均值 | 相对 C0 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 46.72 | 50.36 | 52.44 | 49.17 | 49.67 | 0.00 |
| teacher 修订 | 48.60 | 50.77 | 52.49 | 49.21 | 50.27 | +0.60 |
| w64-k1 candidate，不加 gate | 48.21 | 49.60 | 51.18 | 47.16 | 49.04 | -0.64 |
| 等幅 random | 46.74 | 50.20 | 53.05 | 49.26 | 49.81 | +0.14 |
| stop-gradient | 46.72 | 50.36 | 52.44 | 49.17 | 49.67 | 0.00 |
| response curve 主策略 | 46.72 | 50.36 | 52.44 | 47.16 | 49.17 | -0.50 |

全部 selector 与阶段结果如下；差值均相对同阶段 C0，单位 pp：

| 方法 | s1 F1 差值 | s2 F1 差值 | s2 old F1 差值 | s2 new F1 差值 |
|---|---:|---:|---:|---:|
| teacher | -0.64 | +0.60 | +0.69 | +0.67 |
| candidate | +0.15 | -0.64 | -0.46 | -0.85 |
| random | -1.66 | +0.14 | +0.55 | -0.25 |
| 原 gate | +0.37 | -0.32 | -0.24 | -0.47 |
| new64 | 0.00 | -1.01 | -1.00 | -0.99 |
| mixed64 | +0.11 | -0.51 | -0.37 | -0.74 |
| curve，主策略 | -0.17 | -0.50 | -0.63 | -0.25 |

teacher 的 s2 均值为正值得保留，但 s0 各臂一致，其三阶段等权平均差值约 **-0.014 pp**，基本持平，不是全程优势。candidate 在 s1 的 old/new 差值为 **-1.96/+1.87 pp**，也显示局部塑性改善可能以历史性能为代价。

## 5. 选择器具体错在哪里

下表以八个增量条件的 report 均值符号定义误接受/误拒绝；仅用于事后诊断。regret 是 C0/candidate 两选一的条件均值 oracle 与已封存策略之间的差，不是可实现的监督选择器。

| 策略 | 接受次数/8 | 接受负均值候选 | 拒绝正均值候选 | 平均 regret，pp |
|---|---:|---:|---:|---:|
| 原 gate | 4 | 2 | 1 | 0.430 |
| new64 | 3 | 3 | 3 | 0.960 |
| mixed64 | 3 | 2 | 2 | 0.653 |
| curve | 2 | 2 | 3 | 0.792 |

本轮没有任何一个误接受/误拒绝同时满足候选差值 95% 区间完全离开零。因此不能把这些计数写成已确认的真实误判率。

两个关键例子：

1. **Cora/19221/s2：** candidate 最终 +1.49 pp，old/new +2.17/+0.58 pp，但 curve 的 mixed gain 为 -0.00776，因此拒绝。这里即使仅去掉 old guard，负的 mixed gain 仍会拒绝候选。
2. **Citeseer/19222/s2：** curve 的 mixed gain 为 +0.000235，2/3 seed 为正且通过 old-loss 保护，因此接受；最终却 -2.01 pp、old -2.52 pp。旧 synthetic probe 上的 loss 保护没有保证真实旧类 F1。

更直接地，new64 在 s2 接受三个最终均值为负的候选，拒绝唯一正均值候选。问题不能简单解释为“gate 太严格，放宽就好”。

## 6. 对机制的解释

### 6.1 已由实验支持的判断

- R2B2 的开发信号没有在本轮独立顺序/种子组合下稳定复现；不能再只把瓶颈归因于 selector。
- 增加到三个响应时点、加入新旧 mixed 负例，在当前实现和阈值下没有改善正式选后结果。
- stop-gradient 严格等于 C0；random 保持每行相同位移幅度；所有文本、新 block、拓扑及预算审计通过。不能把当前差异归因于这些因素的混改。
- 单纯限制旧 X 的位移和保护多数旧行，不等价于控制历史真实分类风险。candidate 在 Citeseer/19222/s1 的 old F1 均值仍下降 7.45 pp。

### 6.2 尚待验证的原因

元优化考察的是：从 `theta_64(C0)` 出发，用 revised memory 做一步更新后的 probe loss。正式评估考察的是：从随机初始参数出发，在 revised memory 上完整训练 608--1776 步后的分类性能。

这两条训练路径不同。新 selector 已消除“共享 C0 前缀”的检查偏差，但只测到 128 步，且度量实例级图文 InfoNCE，而正式指标是类别 prompt 竞争下的节点分类 Macro-F1。训练时间跨度与评价目标错位仍可能存在。

另外，旧风险只能在有限 synthetic memory 上估计，原始旧图不能回读；synthetic probe 的 loss 变化未必能预测真实历史分布上的变化。这是需要新机制处理的问题，不能用一个 norm bound 宣称已经解决。

这些是与观测一致的解释，不是本轮已经完成的因果分解。新类别顺序和元优化/report seeds 同时更换，也不能单独归因于其中一个因素。

## 7. 成本

在八个增量条件上，按同一远程运行的分段计时取平均：

| 部分 | 每阶段平均秒数 |
|---|---:|
| C0 block + append 构造 | 0.193 |
| teacher 和可编辑行确定 | 0.405 |
| 64 步元优化前缀，四 seeds | 0.624 |
| candidate 元优化 | 0.564 |
| scratch response curve，两臂 × 三 seeds | 2.444 |
| C0 单个正式 student 训练 | 2.841 |
| curve 策略单个正式 student 训练 | 2.844 |

主策略的已计时构造组件合计约 4.230 s/阶段，是 C0 已计时构造的 **21.91×**；加上一个正式 student 的训练后，这些计时组件之和约为 C0 的 **2.33×**。这不是端到端总耗时，而是该小图、该设备下的实测组成，也不是大图复杂度或跨硬件速度结论。

范围说明：不含冷启动、数据/模型加载、delta/support 物化、部分编排与文件写入、独立审计和实验专用 stop 对照；selector 计时含其 checkpoint 保存，正式 student 的 `seconds` 是训练循环而不是完整数据加载与评估。三种新 selector 共享同一轨迹，不能把该成本按策略重复加总。此次 216 次正式训练的循环时间合计 569.17 秒，两 GPU 并行，不能把该和称为任务墙钟时间。

部署时仍只保存与 C0 同预算的 synthetic memory；teacher、selector 和元优化模型均为构造期临时状态。为可复现实验而保存的这些 checkpoint 不是持续 memory 容量，但也不能从实验磁盘占用中隐去。

## 8. 验收与下一步

| 项目 | 验收 |
|---|---|
| 本地测试 | 33 passed，8 个 CUDA 项跳过 |
| 远程测试 | 41 passed |
| 四个 cell、三阶段、九臂、六 seeds | 648/648 条记录 |
| 正式新训练/精确复用 | 216 / 432 |
| 独立正式状态重载 | 216 |
| scratch selector checkpoint 重放 | 144 |
| prediction/memory/seed/schedule/seal 审计 | 0 errors |
| 最大独立指标重算误差 | 2.84e-14 pp |
| 预注册主策略性能门槛 | 未通过 |

主策略最终平均未达 +0.5 pp；最差条件 -2.015 pp，old 最差 -2.516 pp；s1、新类平均也下降，且最终低于 random 与 teacher。结论不依赖仅差 0.015 pp 的单一门槛，多个方向均未达到要求。

本轮行动：

1. C0 保持共同冻结参照。封存 R3，不将未加 gate candidate、teacher 或诊断 selector 事后替换为本轮主方法。
2. 不启动当前配方的递归 rollout、大图、文本/拓扑联合修订；预定前置条件未满足，并非任务遗漏。
3. 若继续研究 TGC，下一次应是明确更改训练目标或候选机制的新协议，而不是继续微调本轮 gate。优先问题是“修订是否改变了 student 从头训练的完整响应”，再谈选择器；可比较从头多状态训练响应匹配与合法 teacher 的关系保持，但这些尚未实现或验证。
4. teacher 最终正向而 s1 负向是一个待解释信号。它不支持放弃 teacher 思路，也不支持宣称已有稳定收益。任何基于本轮结果重新设计的方案，都须再留新的确认条件；本轮四个 cell 现已暴露，后续只能作为开发/诊断条件。

## 9. 证据与复现

- [冻结协议](TGC_R3_SELECTOR_CONFIRMATION_PROTOCOL_ZH_2026-09-22.md)
- [机器汇总](../../results/tgc_r3_selector_confirmation_20260922/summary.json)
- [逐条件六种子指标与配对区间](../../results/tgc_r3_selector_confirmation_20260922/cells.csv)
- [分阶段汇总](../../results/tgc_r3_selector_confirmation_20260922/aggregate.csv)
- [selector 决策诊断](../../results/tgc_r3_selector_confirmation_20260922/selectors.csv)
- [成本原始分段记录](../../results/tgc_r3_selector_confirmation_20260922/costs.csv)
- [独立审计](../../results/tgc_r3_selector_confirmation_20260922/audit.json)
- [选择封存哈希](../../results/tgc_r3_selector_confirmation_20260922/selection_seal.json)
- [运行入口](../../code/v2_llm_fusion/run_tgc_selector_confirmation.py)、[审计入口](../../code/v2_llm_fusion/audit_tgc_selector_confirmation.py)

本地回收 814 个 JSON/CSV/日志/文档/部署 memory 文件。完整正式 student、元优化前缀和 selector checkpoint 保留在远程结果目录，哈希记录于各阶段证据中。远程结果：`/home/pc/research/TAG_condensation/results/tgc_r3_selector_confirmation_20260922`；运行快照：`/home/pc/research/TAG_condensation/work_runs/tgc_r3_selector_20260922_snapshot`。本地并未重复存储所有大 checkpoint。
