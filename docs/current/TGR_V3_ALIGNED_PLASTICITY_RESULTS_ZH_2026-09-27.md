# TGR v3：硬拓扑前向与新类塑性目标实验报告

日期：2026-09-27  
状态：已完成实现、单元测试、6 个独立 Stage-2 构造 cell、6 个 Student-seed 下游评估及 2 个无塑性控制 cell。

## 1. 本轮目的

本轮沿着“Teacher、Candidate 和最终二值拓扑使用同一传播算子；显式优化新类塑性；联合处理 `P`、`X` 和新旧消息关系”的路线推进。

本轮不改变：

- Student 结构；
- MiniLM 文本表示；
- memory budget；
- 文本内容和文本长度；
- 历史 raw 不可回读约束。

新增两个机制：

1. **Straight-through hard topology forward**：Candidate 的前向传播直接使用与最终输出相同的 degree-preserving 二值拓扑，反向传播使用 projected soft kernel。
2. **New-class plasticity objective**：对新 synthetic rows 增加图文类中心对齐和类间 margin 分离损失。

此外修正了旧类新增节点场景下的 provenance 分组覆盖问题：如果旧 memory 和当前 delta 含有同一类别，旧行和新行现在会进入同一类别组，而不是后者覆盖前者。

## 2. 方法流程

### 2.1 合法 support

阶段 `t` 只使用：

```text
上一阶段 synthetic parent + 当前 raw delta
```

历史完整 raw graph、历史 raw text 和历史 raw edge 均不读取。

### 2.2 稀疏 `P`

当前 raw delta 与新增 synthetic slots 在图文联合空间中计算距离，每个 slot 保留 top-8 raw 候选，并使用 softmax 得到软分配 `P`。

旧 rows 在 `Q(P)` 中保持 identity；新 raw rows 根据 `P` 归一化映射到新 slots。

### 2.3 Teacher 和 Candidate

Teacher 使用 canonical 两层 GCN、可训练 text projection 和双向图文 InfoNCE，在合法 support graph 上训练 4 步。

Candidate 从相同初始化开始，在候选 memory 上进行 2 步可微分 Student unroll。外层 Adam 共 4 步，更新：

- synthetic feature residual `X`；
- sparse assignment logits `P`。

旧行相对位移半径为 0.05，新行相对位移半径为 0.15。

### 2.4 硬拓扑前向

support topology 投影为：

\[
A_{projected}=Q(P)^T A_{support}Q(P)
\]

再与 base topology 混合：

\[
A_{mix}=(1-\eta)A_{base}+\eta A_{projected}
\]

本轮固定 `message_mix=0.20`、`swap_fraction=0.10`。

Candidate 前向采用：

\[
\operatorname{ST}(A)=\operatorname{stopgrad}(A_{hard}-A_{soft})+A_{soft}
\]

因此：

- forward value 是最终会材料化的 hard normalized adjacency；
- backward gradient 仍然通过 soft projected kernel 回传到 `P`；
- 最终输出使用同一个 hard materializer。

### 2.5 新类塑性目标

对新 synthetic rows 按类别计算 graph center 和 text center：

\[
L_{align}=1-\cos(\mu_c^{graph},\mu_c^{text})
\]

不同新类之间加入最小距离 margin：

\[
L_{sep}=\max(0,m-(1-\cos(\mu_c,\mu_d)))
\]

本轮塑性权重固定为 `0.25`，margin 固定为 `0.20`。

## 3. 实现和审计

主要文件：

- [teacher_guided_payload_revision.py](../../code/v2_llm_fusion/teacher_guided_payload_revision.py)
- [teacher_guided_message_revision.py](../../code/v2_llm_fusion/teacher_guided_message_revision.py)
- [run_teacher_guided_message_revision.py](../../code/v2_llm_fusion/run_teacher_guided_message_revision.py)
- [run_gc_sr_stage_eval.py](../../code/v2_llm_fusion/run_gc_sr_stage_eval.py)

测试结果：

```text
12 passed
```

6 个独立构造 cell 均满足：

| 审计项 | 结果 |
|---|---:|
| `gate_fallback` | 0/6 |
| 历史 raw 读取 | 0 |
| report/test 标签用于选择 | 否 |
| 旧行最大相对位移 | 0.05 |
| 新行最大相对位移 | 0.15 |
| 边数改变 | 0/6 |
| Student seeds | 150--155 |

## 4. 独立顺序下游结果

说明：本轮独立 cell 使用已经封存的 `semantic_perm_memory` 作为被 revision 的 base，属于 TGR 机制确认，不是重新构造纯 C0 的主表比较。

所有数字是候选方法相对同一 base 的 paired difference，单位为百分点。

| 数据集/顺序 | Final Macro-F1 | Accuracy | New F1 | Old F1 |
|---|---:|---:|---:|---:|
| Cora/order19605/Stage-2 | -3.631 | -4.797 | -3.684 | -2.126 |
| Cora/order19606/Stage-2 | -6.242 | -6.765 | -9.341 | -2.037 |
| Cora/order19608/Stage-2 | -0.567 | -0.492 | -5.269 | +5.420 |
| Citeseer/order19605/Stage-2 | +2.142 | +2.665 | +1.654 | +3.120 |
| Citeseer/order19606/Stage-2 | -0.428 | -0.209 | -1.230 | +0.461 |
| Citeseer/order19608/Stage-2 | -1.437 | -1.358 | -2.603 | -0.828 |
| **总体平均** | **-1.694** | **-1.826** | **-3.412** | **+0.668** |

按数据集平均：

| 数据集 | Final Macro-F1 | Accuracy | New F1 | Old F1 |
|---|---:|---:|---:|---:|
| Cora | -3.480 | -4.018 | -6.098 | +0.419 |
| Citeseer | +0.092 | +0.366 | -0.726 | +0.918 |

只有 `1/6` 个 cell 的 Final Macro-F1 为正，只有 `1/6` 个 cell 的 New F1 为正。

## 5. 无塑性控制

控制条件完全相同，只将 `plasticity_weight` 设为 0。

| Cell | 有塑性项 Final F1 | 无塑性项 Final F1 | 有塑性项 New F1 | 无塑性项 New F1 |
|---|---:|---:|---:|---:|
| Cora/order19605/Stage-2 | -3.631 | -2.846 | -3.684 | -0.692 |
| Citeseer/order19605/Stage-2 | +2.142 | +1.529 | +1.654 | -1.378 |

解释：塑性项不是统一有效或统一有害。它在 Cora 上加重退化，在 Citeseer 上提高最终 F1，但仍没有稳定提升新类 F1。因此当前塑性项只能作为诊断组件，不能进入默认主方法。

## 6. 结果说明

### 6.1 硬拓扑对齐不是充分条件

Candidate 前向和最终二值拓扑已经一致，但 Cora 三个独立顺序仍全部下降。这说明之前的 soft/binary mismatch 确实是问题，却不是唯一问题。

### 6.2 主要瓶颈仍是新类塑性

总体 old F1 上升 `+0.668 pp`，但 new F1 下降 `-3.412 pp`。Cora/order19608 更明显：old F1 `+5.420 pp`，new F1 `-5.269 pp`。

这表明当前联合目标倾向于保护旧传播结构，却没有把新类有效地嵌入现有 message-passing 角色。

### 6.3 P/X 联合更新仍然过于间接

当前 `P` 是由图文距离初始化的局部 top-8 assignment。它影响：

- 新 synthetic X 的 payload target；
- projected topology；
- Candidate 的 Student 梯度。

但最终 Student 的 Macro-F1 还会受到 prompt competition、真实图传播和离散边选择影响。因此训练时的 plasticity loss 仍不能直接等价为下游 new-class utility。

## 7. 当前决策

本轮不晋升 v3。

保留：

- straight-through hard topology 作为正确性修复；
- 旧/新同类 provenance 合并修复；
- v3 的完整审计和负结果。

不保留为主方法组件：

- 固定 `plasticity_weight=0.25`；
- 当前全量 `P+X` 联合更新；
- 当前 message mix=0.20 的统一拓扑 transport。

## 8. 后续最小路线

下一轮不再同时改三个对象，而是固定 hard topology，对 `X` 做新类/旧类分离的单变量实验：

1. 只更新新 rows，旧 rows 完全冻结；
2. 只使用 new-class graph-text alignment，不加入类间 margin；
3. `P` 固定，避免 assignment 和 X 同时漂移；
4. 只在一个 Cora 和一个 Citeseer 独立 cell 上做 smoke；
5. 只有 smoke 同时改善 new F1 和 Final F1，才扩展到完整矩阵。

这样可以区分：

- 旧 rows revision 导致的保持偏置；
- `P` 漂移导致的拓扑变化；
- 新类塑性损失本身是否有效。

原始结果目录：

- `results/tgr_v3_aligned_joint_independent_20260927/`
- `results/tgr_v3_aligned_joint_independent_eval_20260927/`
- `results/tgr_v3_ste_no_plasticity_control_20260927/`
- `results/tgr_v3_ste_no_plasticity_eval_20260927/`
