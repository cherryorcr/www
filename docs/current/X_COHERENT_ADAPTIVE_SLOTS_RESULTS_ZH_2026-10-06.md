# Adaptive X-Conditioned Coherent Routing 结果

日期：2026-10-06

本轮将原来每个类别固定最多三个 coherent payload slot 改为自适应 slot：只使用当前阶段 delta，计算每个类别的图文离散度，并按当前 delta 内的相对排名分配 1、2 或 3 个候选 slot。文本预算仍固定为 320 字符，但 coherent raw payload 不再简单地从第 320 个字符硬截断，而是优先保留句子边界，其次保留词边界；只有单个片段本身超过预算时才使用硬截断。

## 1. 方法变更

### Adaptive slot

对类别 `c` 计算：

\[
d_c=\frac{1}{2}\left(1-\operatorname{sim}(X_c,\bar X_c)\right)
       +\frac{1}{2}\left(1-\operatorname{sim}(T_c,\bar T_c)\right).
\]

其中 `X_c` 和 `T_c` 是当前 delta 中该类别的图特征和文本表示。按照 `d_c` 在当前阶段类别集合中的相对排名：

| 离散度排名 | slot 数量 |
|---:|---:|
| `< 50%` | 1 |
| `50%--80%` | 2 |
| `>= 80%` | 3 |

最终 slot 数量还受合成行数和当前类别 raw 候选数限制，因此不会生成没有对应合成行的 slot。

### Boundary-aware text budget

320 字符仍然是每个 synthetic node 的固定 payload 上限，不是取消预算。当前 coherent 文本材料化规则为：

1. 规范化空白字符；
2. 如果超过 320 字符，优先在句末标点处截取；
3. 没有合适句末时，在最后一个空格处截取；
4. 只有一个连续片段本身超过预算时才进行硬字符截断；
5. 使用冻结 MiniLM 对最终真实文本重新编码。

这只改变截断位置，不生成摘要，也不修改文本语义内容。

## 2. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集 | Arxiv、SportsFit |
| 阶段 | `base+4`，5 个阶段 |
| memory | fixed-rate 5% |
| Student | 原始 edge-aware 两层 GCN + text projection + 双向 InfoNCE |
| Student seeds | `100--105`，6 个 seed |
| construction seed | `270927` |
| 历史 raw 读取 | 0 |
| report/test labels | 不参与构造和选择 |
| 对照 | 同批次 CCTP top-5 |
| 方法臂 | CCTP、adaptive coherent、adaptive coherent+CCTP anchor |

## 3. 表一：最终性能

所有分数为百分比，均为 6 个 Student seeds 的均值。`ΔF1` 相对于同数据集、同批次的 CCTP top-5。

| 数据集 | 方法 | Final Macro-F1 | Accuracy | ΔF1 vs CCTP |
|---|---|---:|---:|---:|
| Arxiv | CCTP top-5 | 19.49 | 27.51 | 0.00 |
| Arxiv | Adaptive coherent | **25.59** | **50.70** | **+6.11** |
| Arxiv | Adaptive coherent + CCTP anchor | 22.00 | 35.56 | +2.52 |
| SportsFit | CCTP top-5 | 25.84 | 26.07 | 0.00 |
| SportsFit | Adaptive coherent | **47.47** | **68.14** | **+21.63** |
| SportsFit | Adaptive coherent + CCTP anchor | 36.11 | 33.46 | +10.27 |
| **两图平均** | CCTP top-5 | 22.66 | 26.79 | 0.00 |
| **两图平均** | **Adaptive coherent** | **36.53** | **59.42** | **+13.87** |
| **两图平均** | **Adaptive coherent + CCTP anchor** | **29.06** | **34.51** | **+6.39** |

## 4. 表二：持续学习分解与成本

| 数据集 | 方法 | Stage-avg F1 | Old F1 | New F1 | Worst-stage F1 | 构造时间/s | Student 时间/seed/s |
|---|---|---:|---:|---:|---:|---:|---:|
| Arxiv | CCTP top-5 | 31.68 | 21.84 | 6.15 | 19.49 | 82.3 | 27.0 |
| Arxiv | Adaptive coherent | 28.55 | 28.59 | 8.59 | 25.59 | 136.6 | 26.9 |
| Arxiv | Adaptive coherent + anchor | **34.03** | **24.47** | 8.03 | 22.00 | 132.9 | 26.5 |
| SportsFit | CCTP top-5 | 39.91 | 26.33 | 23.15 | 25.84 | 188.6 | 24.6 |
| SportsFit | Adaptive coherent | **58.02** | **51.73** | 24.03 | **47.47** | 234.0 | 24.8 |
| SportsFit | Adaptive coherent + anchor | 54.59 | 37.50 | **28.46** | 33.11 | 228.8 | 25.0 |
| **两图平均** | CCTP top-5 | 35.79 | 24.08 | 14.65 | 22.66 | 135.4 | 25.8 |
| **两图平均** | Adaptive coherent | 43.28 | 40.16 | 16.31 | 36.53 | 185.3 | 25.9 |
| **两图平均** | Adaptive coherent + anchor | **44.31** | **30.99** | **18.24** | 27.56 | 180.9 | 25.7 |

## 5. Slot 实际分配

按类别统计，adaptive slot 的实际分配为：

| 数据集 | 1 slot | 2 slots | 3 slots | 类别总数 |
|---|---:|---:|---:|---:|
| Arxiv | 20 | 8 | 12 | 40 |
| SportsFit | 6 | 2 | 5 | 13 |

说明自适应规则没有退化成所有类别固定使用同一个 slot，也没有所有类别都使用三个 slot。

## 6. 结果解释

1. 自适应 slot 在两个数据集上都保留了 coherent 的正向信号。Final F1 相对同批次 CCTP 分别提升 `+6.11` 和 `+21.63` 个百分点。
2. anchor 版本仍然比 CCTP 好，但比无约束 adaptive coherent 保守，说明当前 CCTP 回退规则能够降低风险，同时也会过滤掉一部分可能有效的文本替换。
3. adaptive coherent 的主要收益体现在整体分类和旧类保持上；anchor 的 New F1 在 SportsFit 上最高，说明安全回退和新类塑性之间存在明显 trade-off。
4. coherent 构造时间约为 CCTP 的 `1.37x`，但 Student 训练时间几乎不变。因此额外开销主要来自候选文本编码和 routing，而不是下游 Student。
5. 这两张表仍然是每个大图一个冻结类别顺序、一个 construction seed 的确认实验，不能单独宣称跨顺序稳定 SOTA。BookChild 的灾难性退化仍需作为压力测试保留，不能因为 Arxiv/SportsFit 的正结果而删除。

## 7. 文件与复现

- 实现：[x_conditioned_payload_routing.py](../../code/v2_llm_fusion/x_conditioned_payload_routing.py)
- 大图 runner：[run_public_continual_baselines_large_20261001.py](../../code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py)
- Arxiv 结果：[summary.md](../../results/x_coherent_adaptive_slots_20261006_arxiv/summary.md)
- SportsFit 结果：[summary.md](../../results/x_coherent_adaptive_slots_20261006_sportsfit/summary.md)

