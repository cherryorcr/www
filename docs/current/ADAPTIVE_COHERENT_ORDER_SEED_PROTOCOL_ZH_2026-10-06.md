# Adaptive coherent 独立顺序、构造种子与统一消融协议

日期：2026-10-06

## 目标

本轮实验用于确认 Adaptive coherent 的收益是否依赖某一个类别顺序或某一个 construction seed，并把小图和大图统一到同一个主方法名称下。

## 固定协议

- 大图：`base+4`、5% cumulative memory、current-delta-only；
- 小图：三阶段、5% memory、`order19611/19612/19613`；
- Student：大图实际使用原 edge-aware 2-layer GCN + text projection + bidirectional InfoNCE；小图使用单独的 Prompt Student runner。二者不能视为同一 Student 协议；
- Student seeds：大图 `100--105`，小图 `150--155`；
- construction seeds：`270927`、`314159`、`271828`；
- 大图独立类别顺序：`9101`、`9102`、`9103`，保持 canonical base+4 的组大小，只置换类别身份；
- 历史 raw graph 读取：0；
- old memory：append-only，旧前缀不修改；
- 拓扑：当前 Adaptive coherent 消融为 edge-free/self-loop-only，暂不把它解释为拓扑创新。

2026-10-08 审计更正：原文将所有实验均标为 Prompt Student，与大图保存的 `protocol.student` 不一致。大图 batch=16，steps=`max(50,min(2048,16*memory_rows))`；以 `results/adaptive_order_seed_remote_20261006/adaptive_order_wikics_o9101_c270927_20261006/summary.json` 等原始记录为准。本更正只修复协议说明，不重写历史结果，也不代表这些 memory 已在当前 Prompt Student 协议下重新评价。

## 方法臂

### 小图

1. `cat_nearest`；
2. `formal_prompt_margin`；
3. fixed-slot `x_conditioned_routing_coherent`；
4. `x_conditioned_routing_coherent_adaptive`（当前主方法）；
5. `x_conditioned_routing_coherent_anchor`。

### 大图

1. CCTP top-5；
2. fixed-slot coherent；
3. Adaptive coherent（当前主方法）；
4. Adaptive coherent + anchor；
5. k-center；
6. medoid。

## 报告指标

- Final Macro-F1；
- Accuracy；
- Old F1；
- New F1；
- Stage-average F1/Accuracy；
- Worst-stage F1；
- construction time 和 Student time；
- paired delta、positive-cell ratio、worst cell；
- slot 数量、唯一 payload 数量和访问审计。

## 当前执行状态

- 小图 adaptive smoke：已通过；
- 大图 adaptive 独立顺序 smoke：已通过并进入完整批处理；
- GPU0：小图完整消融后继续跑 WikiCS/BookHis 大图队列；
- GPU1：Arxiv/SportsFit 大图多顺序、多 construction seed 队列。

最终汇总时，canonical 单顺序结果与本轮独立顺序/构造种子结果分表展示。只有在独立 cell 统计完成后，才计算跨顺序平均和 paired confidence interval。
