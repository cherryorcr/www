# P2-B Student 引导 Attention 小图实验报告

日期：2026-09-18  
状态：已完成正式小图筛选；未晋升为新主线，不触发独立顺序确认或大图扩展。

## 1. 结论先行

本轮实现并验证了真正的 student-guided text attention：attention 参数通过 probe student 的 Adam 训练过程，接收当前合法新数据上的图文 InfoNCE 外层梯度；不是只用文本 embedding 重构，也不是固定 query attention。机制和数据边界审计全部通过，但性能门禁未通过。

| 数据集 | student family 最终 F1 | center concat | 差值 | student family 阶段均值 | center 阶段均值 | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 48.64 | 48.46 | +0.18 | 49.45 | 49.88 | -0.43 |
| Citeseer | 54.43 | 54.97 | -0.53 | 61.14 | 61.07 | +0.07 |

预注册门禁要求两个数据集最终至少 `+0.5pp`、阶段均值不下降，并要求两个 construction seed 各自优于拼接。实际不满足：Cora 阶段均值下降，Citeseer 最终下降；`student_700` 在两个数据集最终都低于拼接，`student_701` 仅在 Cora略高、Citeseer略低。因此不运行 1703/1704 独立确认，不改称 v5，不上大图。

## 2. 实验做了什么

只改变新合成节点的 `text_emb`，固定：

- 5% 累计节点预算、dispersion 类内分配；
- 类内图文簇、Mean-X、行枚举和行顺序；
- 旧合成 X、旧文本、旧质量字段、旧旧边；
- 新块内部和旧块拓扑；
- 两层 GCN + text projection student、双向 InfoNCE、batch 16、Adam 0.001、从头训练和报告 seeds 100--105。

每个簇固定最多 5 篇有来源文本，总内容预算 64 tokens。冻结 MiniLM 分别编码拼接 anchor 和片段 keys。可学习 attention 使用 16 维低秩 `Wq/Wk`，`Wq` 零初始化，初始输出与 anchor 路径一致；最终目标为：

\[
z=\operatorname{normalize}(0.75z_{anchor}+0.25\sum_i a_i e_i).
\]

student arm 的外层目标为当前 delta 真实图上的双向图文 InfoNCE，加旧合成 memory 保持项和 anchor 漂移正则；内层 probe student 使用共同的两层 GCN、Adam、batch 序列和训练目标。语义 arm 使用相同 attention 参数，但只优化当前源文本均值重构，用来区分“学习 attention 本身”和“student 反馈”的贡献。

## 3. 结果明细

### 3.1 各配置的最终/阶段结果

| 数据集 | 配置 | 最终 | 阶段均值 | 旧类 F1 | 新类 F1 | 遗忘 |
|---|---|---:|---:|---:|---:|---:|
| Cora | center concat | 48.46 | 49.88 | 55.89 | 38.55 | 2.29 |
| Cora | fixed anchored attention | 49.22 | 50.18 | 56.31 | 39.78 | 2.36 |
| Cora | semantic seed 700 | 49.24 | 50.42 | 56.40 | 39.69 | 1.13 |
| Cora | semantic seed 701 | 49.33 | 50.61 | 55.93 | 40.54 | 2.96 |
| Cora | student seed 700 | 48.06 | 49.02 | 55.30 | 38.42 | 1.40 |
| Cora | student seed 701 | 49.21 | 49.88 | 56.64 | 39.30 | 1.64 |
| Citeseer | center concat | 54.97 | 61.07 | 58.86 | 47.19 | 7.29 |
| Citeseer | fixed anchored attention | 52.74 | 60.84 | 56.96 | 44.30 | 9.98 |
| Citeseer | semantic seed 700 | 53.40 | 61.14 | 57.68 | 44.84 | 9.43 |
| Citeseer | semantic seed 701 | 52.74 | 60.84 | 56.96 | 44.30 | 9.98 |
| Citeseer | student seed 700 | 54.39 | 61.23 | 58.68 | 45.80 | 7.98 |
| Citeseer | student seed 701 | 54.48 | 61.05 | 58.57 | 46.30 | 7.51 |

这些均为 3 个已知诊断顺序、每个顺序 6 个独立 student seeds 的描述性均值，不是独立泛化置信区间。按顺序看，Cora 的 order1702 是主要失败条件；Citeseer 的 student arm 在 order1701 的阶段改善较明显，但不足以抵消其它阶段和最终结果。

### 3.2 相对 student family 的配对差值

| 数据集 | 对照 | 最终差值 | 阶段均值差值 | 旧类差值 | 新类差值 | 正向最终 cell |
|---|---|---:|---:|---:|---:|---:|
| Cora | center concat | +0.18 | -0.43 | +0.08 | +0.32 | 11/18 |
| Cora | fixed anchor | -0.58 | -0.73 | -0.33 | -0.91 | 7/18 |
| Cora | semantic family | -0.65 | -1.06 | -0.19 | -1.25 | 8/18 |
| Citeseer | center concat | -0.53 | +0.07 | -0.23 | -1.14 | 11/18 |
| Citeseer | fixed anchor | +1.70 | +0.30 | +1.67 | +1.76 | 17/18 |
| Citeseer | semantic family | +1.36 | +0.15 | +1.31 | +1.48 | 15/18 |

student family 相对 center 的改进并不稳定；它相对 fixed anchor 和 semantic family 的差异也不能说明 student 反馈本身有效，因为 fixed anchor 在 Citeseer 已经明显劣于 center，控制选择会改变比较基准。

## 4. 机制证据与成本

| 项目 | Cora | Citeseer |
|---|---:|---:|
| 正式 fits | 324 | 324 |
| 记忆快照 | 54 | 54 |
| learned attention checkpoint | 18 | 18 |
| 新文本行 | 1692 | 1998 |
| token 违规 | 0 | 0 |
| student arm 平均 meta seconds/阶段 | 约 5.9 秒 | 约 5.9 秒 |
| student arm 平均 attention 优化/完整三阶段链 | 约 18.9 秒 | 约 19.0 秒 |
| semantic arm 平均优化/阶段 | 约 0.1 秒 | 约 0.1 秒 |
| student probe steps/构造 seed/阶段 | 640 | 640 |
| validation probe steps/构造 seed/阶段 | 288 | 288 |

student 外层训练确实产生了非零梯度，但平均 target cosine 漂移只有约 0.01--0.02；这说明当前 0.25 的残差上限和 anchor 正则让优化非常保守。另一方面，即使保守更新仍增加了约 18--19 秒/完整三阶段链的构造成本，而报告 student 的完整训练成本基本没有减少。因此当前版本没有展示一次使用意义上的计算优势。

## 5. 审计结果

两数据集都通过：

- `648` 次正式报告 fit 完成并可重载；
- `108` 个 memory 快照、`36` 个 learned attention checkpoint 可复现；
- `0` 个 token budget violation；
- 旧 X、旧文本、旧旧边和共同非文本字段未改变；
- current-delta source、outer train/validation split 和 query 邻域均在合法范围；
- report labels 未用于 attention 学习；
- attention 参数和输出向量可由保存 checkpoint 重建；
- 可微 Adam 与官方 student 更新在机制测试中通过数值对照；
- attention 外层梯度非零，证明确实穿过了 student 训练过程。

外部原始数据读取仅用于离线 provenance audit，不属于持续算法在线访问。

## 6. 对方法的判断

本轮不能得出“student-guided attention 无效”的普遍结论，只能得出更窄且可靠的结论：在当前 64-token、5-source、0.25 residual、40-step outer optimization 和固定拓扑协议下，student 反馈没有在 Cora/Citeseer 的已知三顺序上带来稳定的完整 student 收益，而且成本很高。

更具体地说：

1. 纯语义学习在 Cora 的平均结果略好，但它没有超过 center 的统一门禁，不能晋升。
2. Citeseer 的 student arm 保留了较好的阶段均值和旧类保持，但新类塑性仍低于 center，说明当前 outer loss 对新类改善不够直接。
3. target 漂移过小，可能是 anchor 混合和正则过强，也可能说明固定片段/Mean-X 已经限制了 attention 的可用增益。
4. student-guided 的额外计算暂时不值得纳入默认构造；它只能保留为后续受控研究变量。

## 7. 后续建议

优先级应为：

1. 保持 `center concat + Mean-X + dispersion + frozen-old topology` 为默认低成本控制。
2. 若继续 P2-B，先做小规模机制消融而不是大图：`residual 0.25/0.5`、anchor penalty、new-only outer loss、旧类保持权重；固定一个 construction seed 做数值筛选，再用 700/701 确认。
3. 明确把“新类塑性”加入 outer objective 的受控版本，但不能使用 report 标签；可以使用 current source graph 的 held-out text target 或源图文对齐损失，并报告 old/new 代理目标与完整 student 结果的相关性。
4. 若不能在相同成本约束下稳定超过 center，停止 student-guided attention，不上大图；回到 P1 的复杂度分配、合成 X 设计或有界拓扑实验。
5. 不把本轮结果改名为 v5，不触发 order1703/1704 独立确认，也不宣称 SOTA。

原始汇总与逐顺序 CSV：`results/p2b_student_attention_summary_20260918.json`、`results/p2b_student_attention_*_20260918.csv`；远程完整 checkpoint 位于 `/home/pc/research/TAG_condensation/results/p2b_student_attention_20260918/`。
