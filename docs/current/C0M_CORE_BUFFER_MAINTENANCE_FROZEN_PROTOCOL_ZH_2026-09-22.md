# C0-M Core-Buffer Memory Maintenance 冻结协议

日期：2026-09-22  
状态：小图正式验证进行中；本文件是实验前冻结版本，不包含结果。

## 1. 目的

C0 当前采用 append-growing、frozen-old memory。C0-M 只检验一个问题：在不回读历史 raw 数据、不给学生选择器使用标签的条件下，对少量历史 synthetic rows 做有界维护，是否能改善长链中的历史陈旧和新类塑性。

该实验不是把 C0 的文本、X、拓扑或 response selector 一起重设计。incoming block 的构造、行数、student、batch、steps、评估 split 和 seed 全部沿用 C0。

## 2. 实验臂

| arm | 定义 | 作用 |
|---|---|---|
| `c0` | `frozen_old`，只追加当前 C0 block | 主比较基线 |
| `naive_refresh` | 既有低 mass、10% old rows、alpha=0.25 bounded refresh | 负对照；区分简单刷新与 C0-M |
| `c0m` | core-buffer redundancy/novelty maintenance | 待验证候选 |

`c0m` 的初始 memory 按 joint normalized X/text farthest-point 选取约 80% stable core，剩余约 20% 是 replaceable buffer。core 永久保护；buffer 中 redundancy 最高的 rows 与当前 delta 中相对 core novelty 最高的 rows 配对，做一次有界 X/text/mass 凸组合。

每个 buffer row 的 `x` 和 `text_emb` 相对其初始 anchor 的累计位移不超过 `anchor_radius=0.25`。默认 `blend_alpha=0.50`；alpha 只通过显式配置生效，不能隐式归一化或改变零更新控制。

## 3. 固定协议

- 数据集：Cora、Citeseer。
- 类别顺序：`canonical`、`order19101`、`order19102`。
- 阶段：3 stages，类别组由统一 `stage_groups` 生成。
- memory：5% fixed-rate growing budget，`K_t=floor(0.05*N_{<=t})`，每阶段只追加 `K_t-K_{t-1}` 行。
- incoming block：`joint_g5_c320`、`concat`、`dispersion`，与 C0 完全一致。
- topology：old-old edges 不变；新增 block/cross edges 使用同一 `append_topology` 接口；C0-M 不因刷新增加边数。
- student：相同两层 GCN/GraphCLIP student、冻结 MiniLM embedding、双向图文 InfoNCE；从头训练。
- batch/lr/steps：沿用 C0；steps=`max(50,min(2048,16*rows))`，batch 为统一协议中的 `P.STUDENT.batch`。
- student report seeds：`100,101,102,103,104,105`。
- 指标：report-dev seen-class Macro-F1 的 cumulative/all、old、new；不读取 test，不把 dev 标签用于构造或维护决策。
- 历史 raw：C0-M 只读取旧 synthetic memory 和当前 delta，不回读历史 raw data。

## 4. 必须通过的审计

每个 stage 都必须满足：

1. `len(memory.rows)=K_t`，不是累计追加超过预算。
2. C0-M 的新增 slice 与 C0 的新增 slice 在 `x/text_emb/texts/mass/depth/stage` 上完全相同。
3. C0-M 的 core indices、core content 和 core text 跨阶段不变。
4. old-old edge block 与上一阶段 C0-M 完全相同。
5. C0-M 与 C0 的 edge entry 数量相同。
6. 所有 buffer anchor 的 X/text 相对位移不超过 0.25。
7. 每个 fit 的 support hash、checkpoint state hash、steps 和 aggregate metrics 可由独立脚本重算。

## 5. 晋升门禁

C0-M 只有在以下条件同时满足时才进入下一轮：

- 6 个 dataset-order cell 中至少 4 个 final cumulative F1 不低于 C0 的 `-1.0 pp`；
- 6 个 cell 的平均 final cumulative F1 相对 C0 至少 `+1.0 pp`；
- 最差 final cumulative F1 不低于 `-2.0 pp`；
- old F1 的最差变化不低于 `-2.0 pp`；
- new F1 跨 cell 平均变化为正；
- construction cost 不超过 C0 的 `2x`；
- 所有结构与 checkpoint 审计通过。

未通过时，C0 保留为主方法；C0-M 作为有界历史维护的负结果，不再叠加文本 attention、topology refresh、response selector 或 dynamic budget，也不上大图。

## 6. 产物

- 实现：`code/v2_llm_fusion/c0m_memory_maintenance.py`
- runner：`code/v2_llm_fusion/run_c0m_maintenance.py`
- CPU/远程契约测试：`code/v2_llm_fusion/test_c0m_memory_maintenance.py`
- 独立审计：`code/v2_llm_fusion/audit_c0m_maintenance.py`
- 汇总：`code/v2_llm_fusion/summarize_c0m_maintenance.py`
