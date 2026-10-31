# X-conditioned Coherent Payload Routing with CCTP Anchor：大图正式确认

状态：正式大图实验已完成；anchor 是当前最值得继续验证的候选组件，但尚未完成独立 construction seed 和小图跨顺序确认，因此暂不冻结为最终主方法。

## 1. 实验目的

无约束的 `X-conditioned coherent routing` 在 Arxiv 和 SportsFit 上有明显收益，但在 BookChild 上出现灾难性退化。本轮加入 CCTP anchor：每个当前 delta 合成行先保留 CCTP 文本，只有当 coherent candidate 在固定的图文代价下严格优于 CCTP anchor 时才替换，否则回退到 CCTP。

本轮不修改 `X`、拓扑、旧 memory 或 budget；不访问历史 raw；不生成新的自然语言文本，只从当前 delta 的真实文本片段构造有限候选池。

## 2. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集 | Arxiv、BookChild、SportsFit |
| 阶段 | `base+4`，共 5 阶段 |
| memory | cumulative fixed-rate 5% |
| Student | 原始 edge-aware 两层 GCN + text projection + 双向 InfoNCE |
| Student seeds | `100--105`，6 个 paired seeds |
| batch / steps | `16` / `max(50, min(2048, 16 * memory_rows))` |
| construction seed | `270927` |
| 历史 raw reads | `0` |
| report/test labels | 不参与构造或选择 |
| 候选池 | 每类 3 个 coherent raw-medoid payload |

## 3. 三图结果

分数为百分比，均为 6 个 Student seeds 的均值。

| 数据集 | 方法 | Final F1 | Accuracy | Stage-average F1 | Old F1 | New F1 |
|---|---|---:|---:|---:|---:|---:|
| Arxiv | CCTP top-5 | 18.453 | 27.846 | 29.900 | 20.678 | 5.845 |
| Arxiv | Coherent routing | 24.475 | 42.958 | 27.303 | 27.872 | 5.228 |
| Arxiv | Coherent + CCTP anchor | **19.642** | **28.816** | **32.391** | **21.953** | **6.544** |
| BookChild | CCTP top-5 | 13.702 | 15.366 | 36.212 | 14.479 | 8.265 |
| BookChild | Coherent routing | 2.963 | 9.959 | 4.562 | 3.144 | 1.695 |
| BookChild | Coherent + CCTP anchor | **13.371** | 11.717 | **37.745** | 14.384 | 6.279 |
| SportsFit | CCTP top-5 | 27.025 | 26.597 | 42.435 | 28.190 | 20.616 |
| SportsFit | Coherent routing | 49.487 | 67.196 | 61.257 | 54.323 | 22.889 |
| SportsFit | Coherent + CCTP anchor | **32.241** | **29.917** | **47.452** | **32.904** | **28.596** |
| **三图等权平均** | CCTP top-5 | 19.727 | 23.270 | 36.183 | 21.115 | 11.575 |
| **三图等权平均** | Coherent routing | 25.642 | 40.038 | 31.041 | 28.446 | 9.937 |
| **三图等权平均** | **Coherent + CCTP anchor** | **21.751** | **23.484** | **39.196** | **23.080** | **13.806** |

相对 CCTP，anchor 的三图等权变化为：

| 指标 | 变化 |
|---|---:|
| Final F1 | **+2.024 pp** |
| Accuracy | **+0.214 pp** |
| Stage-average F1 | **+3.013 pp** |
| Old F1 | **+1.965 pp** |
| New F1 | **+2.231 pp** |

## 4. Anchor 实际做了多少修改

全三图、五阶段共检查 `14,672` 个当前 delta 合成行：

| 项目 | 数量 |
|---|---:|
| 回退到 CCTP | 13,815 |
| 接受 coherent candidate | 857 |
| 回退比例 | 94.16% |
| 接受比例 | 5.84% |

因此 anchor 不是把整个 memory 替换成 coherent routing，而是只允许少量低代价 payload swap。平均构造时间约为 CCTP 的 `1.29x`，Student 训练时间基本不变。

## 5. 结果说明

1. 无约束 coherent routing 不能作为主方法。它在 BookChild 上将 Final F1 降到 `2.963`，说明仅靠 X-conditioned 图文代价不能保证文本 payload 改动与 Student 的长期训练响应一致。
2. CCTP anchor 有效阻止了灾难性退化。BookChild 从 `2.963` 恢复到 `13.371`，仅比 CCTP 低 `0.331 pp`；同时三图平均 Final F1、Stage-average F1、Old F1 和 New F1 均高于 CCTP。
3. 当前收益来自少量被接受的替换。`94.16%` 的候选行被回退，说明“安全约束”本身是有效的，但也暴露出候选代价与真实 Student 价值之间仍有较大错位。
4. 这轮结果支持把 anchor 作为当前候选主线，而不是直接宣称最终方法或普遍 SOTA。它已经比无约束版本更可靠，但仍只有一个 construction seed 和每个数据集一个冻结类别顺序。

## 6. 后续实验

1. 在 Cora/Citeseer 的 3 个类别顺序、6 个 Student seeds 上复现 CCTP、无 anchor coherent、anchor 三臂，确认小图上是否仍能保持正向平均收益。
2. 在一个独立 construction seed 上只复现 anchor，不再扫描 routing temperature、候选数或 fallback margin。
3. 将 anchor 与已经完成的 CaT、k-center、medoid、GECC/TAG 适配基线放入同一协议表；不能只与 CCTP 比较。
4. 若独立 seed 和小图均满足平均正收益、最差 cell 不出现灾难性下降，再将方法形式化为：

   `CaT/CCTP initialization -> X-conditioned coherent candidate routing -> strict CCTP-constrained safe swap -> real payload materialization`

   否则保留 CCTP 为默认方法，把 anchor 定位为可选的安全增强组件。

## 7. 证据位置

- 远程汇总：`/home/pc/research/TAG_condensation/results/x_coherent_anchor_large_full_20261006_gpu0/summary.json`
- 远程汇总：`/home/pc/research/TAG_condensation/results/x_coherent_anchor_large_full_20261006_gpu1/summary.json`
- 本地结果缓存：`D:\Aprojects\continue_tag_condensation\tmp_remote_results\x_coherent_anchor_large_full_20261006_*`
- 实现：`D:\Aprojects\continue_tag_condensation\code\v2_llm_fusion\x_conditioned_payload_routing.py`
