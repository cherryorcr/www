# CaT 类共享真实文本 Payload 确认实验

日期：2026-10-02  
状态：完成第一轮多顺序确认，作为当前文本处理模块的候选版本，不替代公开基线比较。

## 1. 实验问题

前一轮固定 CaT 合成特征 `X`，把当前阶段原始节点的文本切成 chunk，并为每个合成行单独构造文本 payload。结果显示，逐行 payload 的性能明显低于 CCTP，即使所有方法使用完全相同的 `X`。

本轮验证一个更直接的解释：

> 退化主要来自同一类别内部的文本方差，而不是 CaT 特征 `X` 本身。

因此本轮只改变文本的复用粒度：由“每个合成行一个 payload”改为“每个类别共享一个真实 payload”。不改变 CaT `X`、拓扑、Student、memory budget 或文本编码器。

## 2. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19611`、`order19612`、盲确认 `order19614` |
| 阶段数 | 3 |
| memory | 相同 5% 累积行预算，旧 memory append-only |
| Student | 原有两层 GCN，冻结 MiniLM，双向图文 InfoNCE |
| Student seeds | `order19611` 使用 150--152；其它两个顺序使用 150--154 |
| CaT steps | 3000 |
| 文本预算 | 每个最终 payload 320 字符 |
| chunk | 当前 delta 文本按 120 字符切分 |
| 拓扑 | 与当前 edge-free 评估一致，不进行拓扑修改 |
| 访问边界 | 只访问当前 raw delta 和已材料化 parent；历史 raw 不回读 |
| 指标 | report-dev Macro-F1、Accuracy |

本轮比较三种方法：

1. `CCTP`：CaT `X` + 原有 CCTP 类级文本。
2. `row-specific TAGSAM-style`：每个合成行独立从局部支持池选择 chunk；这是本轮已知失败对照，不称为官方 TAGSAM 复现。
3. `class-shared TAGSAM-style`：每个类别先用 CaT `X` 选出一个代表行，再将该行生成的真实文本 payload 复用于该类别的所有合成行。

## 3. 当前实现流程

### 3.1 CaT 特征初始化

每个阶段首先使用当前 raw delta 执行原有 CaT 构造，得到本阶段合成特征：

\[
X_t = \operatorname{CaT}(G_t^{raw}, m_t).
\]

本轮所有文本臂共享同一个 `X_t`，因此文本实验不会把差异误归因到特征凝聚。

### 3.2 有界局部支持池

对合成行 `j`，只从当前阶段、同一类别的 raw 节点建立有界候选池。候选池是以下三部分的去重并集：

- CCTP 已选文本来源；
- 与 `X_j` 的 CaT 原始特征近邻；
- 与该类别文本中心相近的文本节点。

候选池最大大小为 32，不遍历所有组合。每个来源文本最多切出有限数量的 120 字符 chunk，并记录：原始节点 ID、字符起止位置、是否为首 chunk、图相似度和文本来源。

### 3.3 真实 payload 材料化

chunk 选择使用图相似度、文本类中心、prompt 和冗余惩罚组成的有界启发式分数。选出的 chunk 直接拼成真实字符串，截断到 320 字符，再用冻结 MiniLM 重新编码：

\[
T_j = \operatorname{MiniLM}(\operatorname{Materialize}(C_j)).
\]

因此 Student 实际接收到的 embedding 与审计中记录的字符串、来源 ID 和 chunk span 是一致的；没有把不可材料化的 embedding 凸组合当成最终文本。

### 3.4 类共享规则

对类别 `c` 的所有合成行，计算 CaT 特征中心：

\[
\bar X_c = \operatorname{Normalize}\left(\frac{1}{|R_c|}\sum_{j\in R_c}X_j\right).
\]

选择与 `\bar X_c` 余弦相似度最高的代表行：

\[
j_c^* = \arg\max_{j\in R_c}\cos(X_j,\bar X_c).
\]

只为 `j_c^*` 构造一次真实 payload，然后将同一个字符串和 embedding 复用于类别 `c` 的其它合成行。这样做不是按最终 F1 选择，而是一个只依赖 CaT `X` 的确定性类级复用规则。

## 4. 结果

### 4.1 三顺序汇总

下表是三个顺序、三个阶段的 cell 平均；`Final` 是 stage 2 平均。

| 数据集 | 方法 | Stage-average F1 | Final F1 | Stage-average Accuracy | Final Accuracy |
|---|---|---:|---:|---:|---:|
| Cora | row-specific TAGSAM-style | 53.51 | 51.79 | 50.30 | 59.97 |
| Cora | CCTP | 74.60 | 66.11 | 81.96 | 77.29 |
| Cora | class-shared TAGSAM-style | **75.71** | **71.12** | **82.42** | **79.14** |
| Citeseer | row-specific TAGSAM-style | 39.03 | 43.53 | 34.54 | 45.82 |
| Citeseer | CCTP | 75.24 | 67.23 | 79.36 | 71.80 |
| Citeseer | class-shared TAGSAM-style | **79.05** | **68.56** | **81.85** | **72.81** |

相对 CCTP：

- Cora：stage-average F1 `+1.11`，final F1 `+5.01`，stage-average Accuracy `+0.46`；
- Citeseer：stage-average F1 `+3.81`，final F1 `+1.33`，stage-average Accuracy `+2.49`。

### 4.2 每个顺序的最终阶段

| 数据集 | 顺序 | CCTP F1 | 类共享 F1 | 差值 | CCTP Accuracy | 类共享 Accuracy | 差值 |
|---|---|---:|---:|---:|---:|---:|---:|
| Cora | 19611 | 65.55 | **76.91** | **+11.36** | 76.88 | **82.66** | **+5.78** |
| Cora | 19612 | 66.36 | **66.54** | **+0.18** | 77.49 | 77.34 | -0.15 |
| Cora | 19614 | 66.41 | **69.91** | **+3.49** | 77.49 | 77.42 | -0.07 |
| Citeseer | 19611 | 66.82 | **69.63** | **+2.81** | 71.58 | **73.67** | **+2.09** |
| Citeseer | 19612 | 67.32 | **68.02** | **+0.70** | 71.54 | 70.97 | -0.56 |
| Citeseer | 19614 | 67.55 | **68.02** | **+0.47** | 72.29 | **73.79** | **+1.50** |

类共享版本相对 CCTP 的 F1 正收益 cell 数为 Cora `4/9`、Citeseer `8/9`；其余 cell 大多接近 CCTP，而不是像逐行版本那样系统性崩溃。

### 4.3 文本复用和成本

在最终阶段，Cora 的类共享版本只有约 3 个 unique source pattern，Citeseer 约 2 个，分别对应当前阶段的类别数；row-specific 版本约有 34 和 32 个不同 source pattern。类共享版本仍然使用平均约 4 个真实 chunk，平均文本长度约 320 字符。

三种文本方法的阶段构造时间都约为 6--7.5 秒，类共享没有引入明显额外成本。`X` 完全相同，且 audit 标记为：

- `x_updated = false`；
- `historical_raw_reads = 0`；
- `old_memory_updated = false`；
- `topology_updated = false`。

### 4.4 双原型负结果

为检验“单类别一个 payload 是否过于刚性”，又实现了一个最小双原型版本 `class_shared_two_tagsam`。它不增加每行的 320 字符预算，而是在同一类别的 CaT `X` 中选择两个确定性原型：第一个最接近类中心，第二个覆盖距离第一个原型最远的方向；每个合成行被分配给余弦相似度更高的原型。

这是一个同协议、5 个 Student seeds、三个顺序的受控复跑，结果如下：

本表中的 CCTP 和单原型数值来自同一次 5-seed 复跑，因此它是双原型的最终配对比较；前面的 4.1 表保留了最初类共享确认任务的原始汇总（`order19611` 使用 3 个 seeds）。

| 数据集 | 方法 | Stage-average F1 | Final F1 | Stage-average Accuracy | Final Accuracy |
|---|---|---:|---:|---:|---:|
| Cora | CCTP | 74.83 | 66.37 | 82.17 | 77.64 |
| Cora | 单原型类共享 | **79.55** | **75.12** | **84.22** | **81.16** |
| Cora | 双原型类共享 | 76.61 | 71.71 | 80.20 | 78.92 |
| Citeseer | CCTP | 75.11 | 66.99 | 79.26 | 71.62 |
| Citeseer | 单原型类共享 | **79.06** | **68.51** | **81.87** | **72.85** |
| Citeseer | 双原型类共享 | 70.60 | 63.37 | 73.89 | 69.74 |

双原型在 18 个数据集-顺序-阶段配对 cell 中均低于单原型。Citeseer 的部分 stage 0 下降超过 19 个百分点，说明在当前 Student 的对比学习目标下，简单引入第二个同类文本原型会重新引入类内语义分裂。这个结果否定的是“无条件双原型”，不是对多原型研究方向的永久否定；若再次研究多原型，应先加入共享类锚点或层级文本约束，而不是直接把第二个 payload 独立送入 Student。

## 5. 结果说明

### 5.1 已验证的机制判断

前一轮逐行 chunk 版本的失败不能简单解释为“chunk 选择方法不够强”。因为它与 CCTP 使用了完全相同的 CaT `X`，主要变化是文本 payload；同时它产生了几十个不同的同类文本。Student 的图文 InfoNCE 在小 memory 下会把同一类别内的这些差异当作不同语义目标，导致类内信号被稀释。

类共享版本只改变复用粒度，就把 Cora/Citeseer 的结果从严重退化恢复到 CCTP 附近，并在多数顺序和最终阶段取得小到中等提升。因此当前更合理的结论是：

> 对当前 Student 和冻结文本编码器，文本凝聚首先需要控制同类 payload 的方差；在此基础上再研究多原型或图条件注意力。

### 5.2 尚不能宣称的内容

本实验不能证明类共享 TAGSAM-style 已经优于所有公开 TAG 凝聚方法，也不能称为官方 TAGSAM 的端到端复现。它是一个严格受控的文本 payload 机制实验，主要用于解释和修复当前 CCTP/CaT 管线中的类内文本不一致问题。

此外，Cora 仍有中间阶段下降，例如 `order19614` stage 1 为 `-3.73` F1；因此不能写成“所有 cell 稳定提升”。

## 6. 当前候选方法定位

当前暂定文本模块应从“逐行文本选择”改为：

> CaT-conditioned class-shared real payload：CaT 负责生成 `X`，局部当前 delta 负责产生真实文本候选，CaT 类中心负责选代表行，同类合成行共享一个真实材料化 payload。

它保留了文本内容、来源审计和当前 raw 不可回读约束，同时避免了逐行文本方差放大。该模块可以进入下一轮主方法候选，但还不应单独作为完整论文算法冻结。

## 7. 下一步

1. 冻结当前类共享实现和这三个顺序的结果，不再继续扫描 chunk 权重或支持池大小。
2. 在完全相同协议下补齐公开静态 TAG 凝聚、k-center/medoid、random coreset 和 direct Student continual baseline。
3. 双原型扩展已经完成并被判定为负结果，不进入当前主方法；保留其代码和审计输出作为文本多样性失败边界。
4. 当前不再扫描文本选择权重、chunk 长度或 prototype 数量；主方法候选固定为“CaT-conditioned class-shared real payload”。
5. 若后续必须恢复类内多样性，应设计带类级共享锚点的层级多原型，而不是直接复用本轮双原型。
6. 完成公开基线后再与大图 `base+4` 做迁移验证；当前结果支持一个稳定的文本模块候选，但仍不足以支持 SOTA 或最终 WWW 主表结论。

## 8. 代码和结果

- `code/v2_llm_fusion/cat_multimodal_minimal.py`：局部支持池、chunk 选择、真实 payload 材料化和类共享代表行规则。
- `code/v2_llm_fusion/run_cat_multimodal_minimal_20261002.py`：统一构造、Student 评估、协议和审计输出。
- `code/v2_llm_fusion/test_cat_multimodal_minimal.py`：加入双原型审计后，当前测试为 `11 passed`。
- `tmp_remote_results/cat_payload_class_shared_cora_20261002/`：Cora `order19611`。
- `tmp_remote_results/cat_payload_class_shared_citeseer_20261002/`：Citeseer `order19611`。
- `tmp_remote_results/cat_payload_class_shared_cora_order19612_20261002/`：Cora `order19612`。
- `tmp_remote_results/cat_payload_class_shared_citeseer_order19612_20261002/`：Citeseer `order19612`。
- `tmp_remote_results/cat_payload_class_shared_cora_order19614_20261002/`：Cora `order19614`。
- `tmp_remote_results/cat_payload_class_shared_citeseer_order19614_20261002/`：Citeseer `order19614`。
- `tmp_remote_results/cat_payload_two_proto_cora_20261002/`：Cora 双原型三顺序负结果。
- `tmp_remote_results/cat_payload_two_proto_citeseer_20261002/`：Citeseer 双原型三顺序负结果。
