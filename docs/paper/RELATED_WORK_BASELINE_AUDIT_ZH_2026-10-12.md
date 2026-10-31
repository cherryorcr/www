# Unified-Q-EF：相关工作、基线身份与 WWW 投稿证据核对

本次核对对应正式英文稿 `paper_www/main.tex`，重点修改第 3 节，并同步修正第 2、5、6 节中依赖基线身份的表述。没有修改原始指标、封存测试结果或方法参数，也没有启动新实验。

## 1. 当前论文是否合理

当前问题定义和技术流程基本自洽，可以继续写论文：当前增量图提供特征、文本和局部结构，合成记忆只保存特征、真实文本及标签；历史合成前缀冻结；共享分配 Q 在构造循环中参与特征池化和文本路由。训练使用目标构造池标签，预测采用类别 prompt 的余弦读出。

论文应明确以下边界：

- fixed rate 是固定累计保留比例，记忆行数仍随数据增长；不是固定行数或字节预算。
- current-delta-only 是构造器的访问限制。转导推理仍使用原始图的特征和邻接，两者不能混称。
- 冻结旧前缀是追加式设计选择，不是持续学习的普遍要求，也不等于解决了遗忘或新旧记忆联合优化。
- edge-free 指不保存显式合成边。原始结构参与构造，主版本的任务范围是图文节点分类；Unified-Q-T 是单独扩展。
- 使用 target labels 训练的 prompt Student 不属于 GraphCLIP 式目标数据集免训练的 zero-shot。
- 5% 为名义工作点。旧 ArXiv 主比较的 6,773 行相对当前合法池是 5.71%；修正后的 5% 敏感性实验使用 5,927 行。

## 2. 原论文依据与创新边界

| 原方法 | 原论文的主要机制 | 对本文表述的影响 |
| --- | --- | --- |
| GCond / GCond-x | 固定合成标签，使用模型梯度匹配优化特征及参数化邻接；GCond-x 用真实图构造、以单位邻接训练合成数据 | 无边保存但图感知的构造不是本文独有；不能把 GCond 写成学习合成标签 |
| CaT | Condensed Graph Memory + Training-in-Memory；随机图编码器上的类别级 MMD；自环合成记忆 | CaT 有梯度下降优化 X，但不是模型参数梯度匹配；它是本文 in-loop DM 的重要先例 |
| GECC | 类内传播特征聚类、质心池化及继承；包含 hard k-means 和 soft fuzzy c-means 分支 | 软分配和质心池化已有先例；目前只比较了 GECC-hard 的适配版本 |
| PUMA | 伪标签利用未标注节点、从头重训、一次传播和宽编码器 | 只保留若干机制的本地实现应称 PUMA-inspired，不能代替原方法全流程 |
| TAGSAM | 原生构造不依赖类别标签；互信息驱动的文本块选择/合并与属性相似度匹配 | 不能笼统声称已有 TAG 方法都忽略文本或仅后附文本；它是直接相关的静态 TAG 基线 |
| STAD | 静态半监督 TAG；协作自训练、Wasserstein 图 sketch、LLM 关键词文本合成 | 单纯聚类和拼接省略了核心机制，只能作为 inspired control |
| DANCE | 联邦 TAG；轮次更新、分层邻居文本证据、可读摘要和自表达拓扑 | 联邦轮次不是历史原始图不可重读的数据流；root+邻居拼接不等于复现原机制 |
| SR-GM | 静态多模态凝聚；冲突梯度投影与基于图拉普拉斯的梯度场正则 | 有梯度协调的先例；小型 proxy 的有限步修正不是原算法全流程 |
| GraphCLIP | 图文预训练、图摘要对比及 prompt transfer | 是下游模型参照，不是凝聚器；官方权重在本地子集上的评估不是官方 benchmark 复现 |
| SFGC / GCondenser | 前者是无显式合成结构的轨迹匹配方法；后者是凝聚评测工具集 | SFGC 是相关方法；GCondenser 不是额外一个凝聚算法 |

因此，本文可主张的技术贡献应集中在：**在历史原始数据不可重读的持续 TAG 构造中，以同一个可微分配连接特征聚合、真实文本来源与循环内构造反馈。** Sinkhorn、软池化、GM、DM、真实文本选择及无边存储分别都已有先例。组合有研究价值，但是否构成足够强的贡献，还取决于对应的必要性证据。

## 3. 实验身份：哪些结果可以严格比较

### 当前匹配的外部适配基线

- **CaT-CCTP**：当前增量的单跳传播目标 + 随机线性/ReLU 嵌入中的类别均值匹配；初始化为抽样真实特征；只追加新记忆。
- **GECC-hard-TAG**：传播权重 `(0, 0.3, 0.9)`；hard Lloyd 聚类；20 次重启按 SSE 和簇大小惩罚选择；未实现原生 soft 分支或累计质心重聚类。
- 两者均附着同一规则的文本：同类 graph-feature cosine nearest top-5，将总计 320 字符分给各来源，拼接后以共同 MiniLM 重新编码。**CCTP 是本项目的文本适配规则，不是独立公开方法，也不是 random-5。**

这些是同协议的机制适配比较，不是官方端到端复现。把 GECC 限制为 frozen-prefix/current-delta-only 改变了其原生演化方式，正文必须披露。

### 历史覆盖与简化参考

PUMA/TAGSAM 和历史 coreset 的结果使用旧评估版本。STAD/DANCE/SR-GM 大图 runner 使用旧版单正样本 InfoNCE Student、1024 步训练及 text-kNN 合成边；主比较则使用 2048 步多正样本 Prompt Student。原方法的若干核心机制也被省略。

因此，这些低分不能被解释为“本文明显超过原论文方法”。旧 coreset 的低分也不能证明最新构造优于随机选点。已有记忆可以先换共同 Student 重评估，但缺失的核心机制仍需要更高保真的构造实现。

GraphCLIP 参考使用冻结官方权重，每个数据集评估 512 个 ego-graphs，并将每个 ego-graph 限为最多 512 个节点。它的模型、样本子集和训练设定均不同，不纳入主表排名。

官方复现需要保持原任务、核心目标、访问假设、Student 和评估协议，并记录代码版本及配置。仅调用官方模块或加载官方权重不满足“官方端到端复现”的定义。

## 4. 当前数值支持什么

下表保留论文目前的匹配点估计，单位为百分数，不将它们改写成封存测试结果。

| 数据集 | Unified-Q-EF F1 / Acc | GECC-hard-TAG F1 / Acc | CaT-CCTP F1 / Acc | 相对 GECC 的 F1 差异 |
| --- | --- | --- | --- | --- |
| ArXiv | 52.25 / 71.59 | 51.09 / 70.27 | 36.74 / 63.62 | +1.16 pp |
| SportsFit | 78.60 / 83.93 | 77.32 / 85.52 | 73.75 / 81.94 | +1.28 pp |
| BookHis | 41.97 / 80.34 | 41.23 / 79.97 | 24.83 / 75.45 | +0.74 pp |

SportsFit/BookHis 原始主表 CSV 的来源字段为 `matched_operator_report_dev`。总索引有“sealed primary”的旧描述，但这不能替代原始评估来源。ArXiv 来自另一个单顺序 scale-transfer 结果包；正文已去掉缺乏此表来源支撑的“latest sealed pass”描述。封存结果维持原样，提交时需要通过明确的 split/hash 将最终主表与 sealed artifact 对应起来。

独立的 9 个 construction/order 配对单元中，相对 GECC-hard：

- SportsFit：平均 +1.25 pp，95% CI `[+0.52, +1.98]`，9/9 正差异，支持该面板内的稳定 F1 收益。
- BookHis：平均 +1.66 pp，95% CI `[-0.40, +3.72]`，7/9 正差异；区间包含 0，尚不能宣称稳定显著优于 GECC。

这些是每个 construction/order 单元内先平均 Student seeds 后的配对统计，不能将所有 Student 重复运行当作独立构造样本。它们也不替代 ArXiv 的多顺序确认。

共享 Q 归因：WikiCS shared/split 为 80.38/80.93，SportsFit 为 78.92/78.82，BookHis 为 41.97/42.50。当前结果支持共享接口可用，**不支持共享本身在所有数据集带来提升**。SportsFit 有一个数值恢复单元，仍单列，未混入正式归因统计。

GraphSAGE 迁移：BookHis 主方法 36.83，高于 GECC 35.50 和 CaT 29.24；SportsFit 主方法 75.27，高于 GECC 74.91，但低于 CaT 77.39。可主张跨架构可用，不能主张跨架构普遍领先。

full-history 保留率是有限训练预算、无合成拓扑的 raw Student 参考，不是充分优化的完整图上界。保留率超过 100% 可以报告，但应限定到这个训练配置，不应推广为“凝聚总能超过完整图”。

## 5. 对 WWW 长文的客观判断

**可以围绕现有方法继续写，当前还不足以判断为高把握的 WWW 长文。** 优势是具体的持续访问问题、清楚的多模态构造接口、较好的压缩保留率，以及部分大图上的配对稳定性和跨架构实证。主要风险不是“必须在所有图上 SOTA”，而是贡献的必要性和相关强基线的证据仍不充分：

1. 与两类先例存在接近的技术关系：CaT/GCond-x/SFGC 的无边图感知构造，以及 GECC 的类内软分配/池化。要讲清楚 Q 如何让真实文本反馈影响构造，而不能把这些已有零件本身作为创新。
2. 共享 Q 对独立 Q 没有稳定精度优势，Q 梯度通常远小于 residual 梯度。当前不宜把“共享 Q 导致主要提升”写成已被证明的因果结论。
3. TAGSAM/STAD 等直接相关 TAG 方法还没有高保真的共同 Student 比较。大量简化低分参考不能补足这个缺口。
4. ArXiv 仍为单顺序，旧表比例是 5.71%。成本也必须正面解释：SportsFit 构造峰值约 21.96 GiB，10% dense-Q 配置 OOM；与快速 GECC-hard 的性价比需要完整呈现。

建议优先做三件事，而不是同时增加 EM、attention、旧记忆刷新和拓扑模块：

1. **补高保真强基线。** 优先核对 TAGSAM 官方文本选择机制及 STAD 的核心模块，使用共同 Student；GECC 增加原生 soft 分支作为单独适配版本。核心组件必须保留，做不到的就继续明确标为 inspired reference。
2. **补核心必要性证据。** 控制 residual、初始化、目标和参数预算后检查共享/独立分配；若精度仍相近，就实测共享带来的分配参数量、内存、构造成本及图文支持一致性，不预设它一定更好。硬前向与保存一致的诊断不能替代 soft-only 存储对照。
3. **完成最终确认。** ArXiv 使用当前合法池的真实 5% 调度，补多顺序/构造种子；冻结版本后明确分开 report-dev 与 sealed test；同时报告 F1/Acc、Old/New、阶段均值和成本。

论文叙事应由“访问约束造成的图文摘要问题 → 共享构造接口如何处理它 → 匹配基线和归因能证明什么”推进。对 frozen-prefix 和 prompt Student 的理由已有相关论文与历史对照支撑，但不要把局部实验解释为普遍最优。相关工作保留精确差异，实验状态和版本审计细节宜集中在基线表、实验设置及附录，避免摘要和引言变成进度报告。

## 6. 本次正文改动

- 第 3 节按 graph condensation、TAG/multimodal、实际基线适配、与 Unified-Q-EF 的关系组织；增加“原方法—发表机制—实际改动—证据身份”表。
- 纠正 CaT 的 DM 与 GM 区别、GECC 的 hard/soft 分支、TAGSAM 的原生无标签构造、GCond 固定标签及 GCond-x 先例。
- 补充 STAD/DANCE/SR-GM 与 GCondenser 引文，校正相关书目信息。
- 同步第 2/5 节的比较分类，明确 2048 步共同 Student 只适用于匹配实验；去掉数据集表中含义不清的通用 Seeds 列。
- 第 6 节给扩展结果分组，撤回根据旧 coreset 低分推断新构造优势的句子；另列简化 TAG controls 与 checkpoint reference；修正 ArXiv Old/New F1 的最高值加粗。
- 保留原始实验记录及指标；中文摘要/前两节译稿同步相应分类。

## 7. 来源与本地核对入口

原论文来源：

- CaT：https://arxiv.org/html/2309.09455v2
- GECC：https://arxiv.org/html/2502.17614v1（本次机制核对依据版本）
- GCond：https://arxiv.org/html/2110.07580v2
- PUMA：https://arxiv.org/abs/2312.14439
- TAGSAM：https://arxiv.org/html/2606.03839v1；WWW DOI：https://doi.org/10.1145/3774904.3792205
- STAD：https://arxiv.org/html/2607.20477v1
- DANCE：https://arxiv.org/html/2601.16519v1
- SR-GM：https://arxiv.org/html/2511.20222v1
- GraphCLIP：https://arxiv.org/html/2410.10329v1
- SFGC：https://arxiv.org/abs/2306.02664
- GCondenser：https://arxiv.org/abs/2405.14246

本地实现及数据：

- `code/v2_llm_fusion/external_continual_mechanisms.py`
- `code/v2_llm_fusion/run_external_large_adapters_20261011.py`
- `results/pre_submission_evidence_20261010/primary_matched.csv`
- `results/extended_evidence_20261011/additional_dataset_summary.csv`
- `results/large_external_adapters_20261011/large_external_comparison.csv`
- `results/unified_q_overnight_20261012/summary.md`
- `results/unified_q_overnight_20261012/q_attribution_audit.md`
- `results/uq_text_path_sealed_test_20261009_final/`
- `docs/current/EXTERNAL_BASELINE_FIDELITY_AUDIT_ZH_2026-10-09.md`（历史审计；身份需结合实际运行版本判断）

上述文件是证据入口，不应把某个旧 manifest 的名称当作当前实验已经匹配或官方复现的证明。

## 8. 本次校验与编译状态

- LaTeX 静态检查通过：正文输入文件、图片、引用键和交叉引用目标均存在；没有重复标签或参考文献键，也没有发现环境不配对。
- `git diff --check` 通过；现有换行格式提示不影响上述检查。
- 已对当前打开的 `paper_www/main.tex` 调用内置编译器并重试。两次均在 TeX 启动前失败，错误为 `windows sandbox: helper_unknown_error: setup refresh had errors`。这是预览环境准备错误，本次不能确认 PDF 编译和表格分页；没有另建替代 PDF 或切换编辑器。
- 尚未 push 到 Overleaf。本次范围为正文修订及证据核对。
