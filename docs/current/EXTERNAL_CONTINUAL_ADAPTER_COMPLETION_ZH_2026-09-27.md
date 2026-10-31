# Semantic-Recursive 与外部持续凝聚适配：小图结项报告

日期：2026-09-27。正式矩阵 360/360 次 Student 训练完成，360 个 checkpoint 重载审计通过，最大指标误差 0。本文所有指标均为百分制，差值为百分点（pp）。

## 1. 结论先行

Semantic-Recursive（SR）保持原方法、原 memory 和原 Student，逐 seed 的 F1/Accuracy 与上一轮完全复现。此次新增的是更独立的外部构造对照，不是 SR 改版。

在 Cora/Citeseer、两个既有类别顺序的四个条件上，SR Final Macro-F1 为 **53.02**。CaT-MMD-TAG 适配为 **54.62**，独立构图的 k-center 为 **54.18**；GECC-hard-TAG 适配和 medoid 为 **48.70 / 47.32**。SR 相对前两者平均 **-1.60 / -1.17 pp**，相对后两者平均 **+4.32 / +5.70 pp**。

因此，SR 是具有竞争力和可复现结果的研究主线，但当前不能写成“稳定超过强 coreset 和持续凝聚基线”。也没有必要因为这一轮未拿到最高均值就推翻整个方法：它与最强适配的差距约 1.6 pp，且 Accuracy 差距只有约 0.38 pp。需要正面补齐泛化与公平资源控制，再决定是否做一个有限的算法改动。

主要比较对象已经改为外部基线，**超过 C0 不作为本轮验收门槛**。

## 2. 文献核验与实际完成范围

公开持续/演化图凝聚并非空白。[CaT](https://arxiv.org/html/2309.09455v2)、[PUMA](https://arxiv.org/html/2312.14439v2)、[GECC](https://arxiv.org/html/2502.17614v3)、[OpenGC](https://arxiv.org/abs/2405.17003) 均有相关先例。本轮核验 CaT/GECC 论文及官方代码后，实现其核心机制的受限 TAG 适配；没有将它们称为原生 continual TAG 方法或官方完整复现。

| 任务 | 本轮状态 | 可支持的结论 |
|---|---|---|
| 冻结 SR 并复验原分数 | 完成 | 72 个 SR fit 的 F1/Accuracy 逐 seed 差为 0 |
| CaT 分布匹配适配 | 完成 | 随机编码器 MMD 优化合成 X，而非仅改类配额 |
| GECC 传播特征聚类适配 | 完成 | 传播后的中心作为 synthetic X，保留自环式 memory |
| k-center/medoid 独立构造 | 完成 | stage0 和后续拓扑均由各方法自身 payload 构造 |
| 四条件、六 Student seeds、三阶段 | 完成 | 360 fit、60 张 memory、四份审计全部通过 |
| PUMA 独有机制 | 未完成 | 不把 CaT 改名当 PUMA |
| 3-5 个新独立顺序与多个 construction seeds | 未完成 | 两个已用顺序不等于新泛化确认 |
| 固定每阶段 rows/steps 的跨顺序控制 | 未完成 | 当前仍是自然 5% fixed-rate |
| 外部方法大图对照、原生调优配置 | 未完成 | 小图适配不能外推为大图 SOTA |

CaT 官方 commit 为 `43d6fcfd7b1754fe53b9f911492d7cb0c113a2eb`；GECC 为 `d731be00cc3beb7585e8fa7b60ca21a6c3cb4e3f`。详细原始链接、实现差异和事前设置见[冻结协议](EXTERNAL_CONTINUAL_ADAPTER_PROTOCOL_ZH_2026-09-27.md)。检索中尚未确认同时采用文本 payload、历史 raw 不可回读、图文 InfoNCE Student 重训接口的现成方法；这不是“不存在持续图凝聚”的证明。

## 3. 实验协议

### 3.1 相同部分

- 当前 pool 的类别标签可用于构造；下游 Student 使用双向图文 InfoNCE，不使用监督分类损失。不能称整个流程无标签。
- 原两层 GCN 和文本 projection，冻结 MiniLM；每阶段从头训练，batch 16、lr 0.001、配对 seeds 150-155。
- 每个条件内所有方法同 rows、同训练步数、同 report-dev；不按最终分数回选构造超参。
- 推理使用真实完整图上下文和所有类别 prompts；统计已见类 report-dev 节点。初始阶段也存在未见类 prompt 竞争，不等同于只在已见类别间分类。
- 构造只接收当前 delta 与旧 synthetic；完整数据容器用于离线切片/评价，未实现操作系统级历史数据隔离。

### 3.2 到达顺序与实际预算

| 数据集/顺序 | 三阶段类别 ID | 累计 rows | 三阶段 Student steps |
|---|---|---|---|
| Cora/19611 | [1,2] → [0,5] → [3,4,6] | 43 → 60 → 94 | 688 → 960 → 1504 |
| Cora/19612 | [3,6] → [2,5] → [0,1,4] | 27 → 62 → 94 | 432 → 992 → 1504 |
| Citeseer/19611 | [1,2] → [0,4] → [3,5] | 42 → 78 → 111 | 672 → 1248 → 1776 |
| Citeseer/19612 | [3,5] → [0,2] → [1,4] | 32 → 75 → 111 | 512 → 1200 → 1776 |

预算为合法累计 pool 的约 5%，steps=max(50,min(2048,16K))。因此最终 rows 相同，早期 rows 和步数不同；只能在同一条件内作配对方法比较，不能把跨顺序绝对分数变化完全归因为类别顺序。

### 3.3 五种方法做了什么

| 方法 | 合成/选择机制 | 文本 | 拓扑与历史 |
|---|---|---|---|
| SR | 冻结的局部 split/merge、守恒语义匹配、递归 parent | 原有中心片段拼接并重编码 | 原封存图，旧 synthetic prefix 保持 |
| CaT-MMD-TAG | 类比例配额；随机同类初始化；每步新随机编码器；3000 步 MMD 优化 synthetic X | 近邻同类至多五来源、320 字符加分隔符、重编码 | 无非自环边；累计自己的 synthetic block |
| GECC-hard-TAG | 0/1/2 hop 权重 [0,.3,.9] 的传播特征；20 次 hard Lloyd 聚类，SSE+均衡项选 restart | 与 CaT 相同的文本补全规则 | 无非自环边；旧类中心继承，新增类聚类 |
| k-center | joint-space farthest-first 真实节点 coreset | 保留所选真实文本与 embedding | 自身文本近邻与跨块边；不借用 SR/C0 slot |
| medoid | joint-space 类内聚类后选真实代表 | 保留所选真实文本与 embedding | 自身文本近邻与跨块边 |

CaT 原训练目标是监督 TiM，本轮改为共同 InfoNCE Student；GECC 原 evolving 代码会读取累计 raw 并用旧中心初始化重聚类，本轮不能回读旧 raw，因此只在类别不重叠的增量下继承旧中心。这些都是明确披露的适配。GECC 的原生 fuzzy 调优分支没有复现。coreset 的“独立构图”指本地端到端对照，不指某个官方 coreset 完整系统。

## 4. 主要结果

### 4.1 四个数据集/顺序条件等权平均

| 方法 | Final F1 | Final Accuracy | 三阶段平均 F1 | 三阶段平均 Accuracy | 最终旧类 F1 | 最终新类 F1 |
|---|---:|---:|---:|---:|---:|---:|
| Semantic-Recursive | 53.02 | 59.59 | 48.35 | 48.73 | 52.32 | 52.76 |
| CaT-MMD-TAG 适配 | **54.62** | **59.96** | 50.52 | 47.87 | 53.01 | **55.95** |
| GECC-hard-TAG 适配 | 48.70 | 52.53 | 44.43 | 41.19 | 46.61 | 50.95 |
| k-center 独立构图 | 54.18 | 59.43 | **53.43** | **54.58** | **53.03** | 54.84 |
| medoid 独立构图 | 47.32 | 53.11 | 43.43 | 41.79 | 46.98 | 46.74 |

![相同 Student 下的外部基线比较](../paper/assets/external_continual_20260927_means.png)

Old/new 是在整个已见 report-dev 上统计每个类别 F1 后分组平均，包含组间误分类。它们不是只在旧类或新类候选中分类的分数。

### 4.2 逐条件 Final F1 / Accuracy

| 条件 | SR | CaT 适配 | GECC 适配 | k-center | medoid |
|---|---:|---:|---:|---:|---:|
| Cora/19611 | 50.21 / 56.40 | **55.64 / 60.76** | 47.78 / 53.32 | 54.54 / 59.78 | 46.39 / 51.78 |
| Cora/19612 | 50.14 / 58.18 | **50.98 / 60.33** | 45.58 / 50.43 | 50.60 / 54.06 | 46.58 / 55.23 |
| Citeseer/19611 | **57.70 / 62.33** | 56.93 / 59.87 | 48.72 / 50.57 | 51.72 / 57.52 | 47.33 / 51.88 |
| Citeseer/19612 | 54.01 / 61.44 | 54.93 / 58.88 | 52.71 / 55.80 | **59.87 / 66.35** | 48.96 / 53.55 |

逐条件六 seeds 的样本标准差（ddof=1）、全部 F1/Accuracy 配对 95% t 区间见[机器生成完整表](../../reports/external_continual_adapters_20260927/summary.md)，不把不同顺序的波动混写为同一条件的 Student 方差。

### 4.3 SR 相对外部方法的差值

| 对照 | SR Final F1 差 | SR Final Accuracy 差 | F1 为正的条件 | 最差 F1 差 |
|---|---:|---:|---:|---:|
| CaT 适配 | -1.60 | -0.38 | 1/4 | -5.43 |
| GECC 适配 | +4.32 | +7.06 | 4/4 | +1.30 |
| k-center | -1.17 | +0.16 | 1/4 | -5.87 |
| medoid | +5.70 | +6.48 | 4/4 | +3.55 |

![逐条件外部基线差值](../paper/assets/external_continual_20260927_cell_deltas.png)

条件内配对 CI 提示两个具体短板：Cora/19611 对 CaT 为 -5.43 pp，95% CI [-7.56,-3.30]；Citeseer/19612 对 k-center 为 -5.87 pp，CI [-9.72,-2.02]。这些区间仅描述固定 memory/order 下 Student 初始化不确定性，不是跨未知顺序的置信保证。四个条件共享两张图，总体区间也只能描述性使用；没有做多重检验校正。

SR 对 CaT 的平均新类差为 -3.19 pp、旧类差为 -0.68 pp，新类仍值得关注；但 Cora/19611 同时存在旧类差距，不能把全部不足归因为新类塑性。所有三阶段中最低的条件均值分别为 SR 17.59、CaT 32.70、GECC 29.36、k-center 8.41、medoid 15.12；早期 all-prompt 竞争和不同到达类共同影响这些数字，尚未做相应因果分解。

## 5. 完整数据与模型持续学习参照

沿用上一轮同顺序、同 seeds 的记录；SR 本轮逐 seed 复现后可以准确对齐。**这些参照本轮没有重训。**

| 条件 | SR Final F1 | 完整 raw F1 | F1 保留率 | SR Accuracy | 完整 raw Accuracy | Accuracy 保留率 |
|---|---:|---:|---:|---:|---:|---:|
| Cora/19611 | 50.21 | 49.19 | 102.08% | 56.40 | 54.18 | 104.09% |
| Cora/19612 | 50.14 | 47.91 | 104.65% | 58.18 | 55.35 | 105.11% |
| Citeseer/19611 | 57.70 | 52.54 | 109.83% | 62.33 | 57.52 | 108.36% |
| Citeseer/19612 | 54.01 | 49.26 | 109.64% | 61.44 | 52.77 | 116.44% |

保留率定义为同 cell 的 SR 指标均值 / raw 指标均值。这里完整 raw 使用相同模型和训练规则，但其 rows 较多，实际为 2048 steps；SR 最终为 1504/1776 steps。因此它不是严格等步数，也不是充分训练或监督上界。大于 100% 只说明超过这个特定训练参考，不能证明压缩优于充分优化的全数据模型。raw 合法训练节点为 Cora 1895、Citeseer 2229，对应 SR 94/111 rows。

四条件等权结果：

| 协议/方法 | Final F1 | Final Accuracy | 额外资源和边界 |
|---|---:|---:|---|
| SR memory-only 重训 | 53.02 | 59.59 | 约 5% synthetic；每阶段丢弃 Student 参数 |
| 完整 raw 重训参考 | 49.73 | 54.96 | 可回读全部已见 raw；2048 steps |
| Student 仅新数据微调 | 51.26 | 57.57 | 保留模型状态；不等于独立可重训 memory |
| Student C0 replay + 新数据微调 | 53.03 | 58.76 | 保留模型、读取 replay 和 delta；额外训练成本 |

旧稿第 6.7.2 节曾把上述最后三行的 F1/Accuracy 均值误放进 Cora/Citeseer 列，同时误称 raw 为 equal steps；本次同步更正。历史 canonical 的 8192-step raw、TAGSAM/STAD 与官方 GraphCLIP 仍按原协议单列，不拼入四条件主表。

## 6. 成本和存储

下表为四条件均值。构造时间和 Student 时间均为三阶段之和，Student 时间对应一枚 seed，排除评价；共享数据/编码器初始化、checkpoint I/O 和重载审计未计入。各数据集用独占 GPU 串行比较，跨卡平均不是精确的硬件性能基准。

| 方法 | 构造秒数 | Student 秒数/seed | 最终导出 tensor KiB | 文本 UTF-8 KiB | 文本 token 数 |
|---|---:|---:|---:|---:|---:|
| SR | N/A | 7.85 | 314.78 | 32.16 | 6295 |
| CaT 适配 | 14.87 | 7.77 | 309.50 | 32.40 | 6303 |
| GECC 适配 | 0.41 | 7.78 | 309.50 | 32.42 | 6226 |
| k-center | 0.03 | 7.84 | 314.78 | 86.17 | 17533 |
| medoid | 0.17 | 7.83 | 314.78 | 106.14 | 21119 |

SR 复用已封存 memory，构造成本不能填成零，也不能据此宣称总成本优于 CaT。tensor bytes 是供 Student 读取的导出 memory，不包含可能需要的 C0 影子状态、来源表或构造 workspace；正式持续资源比较还需要完整持久状态账。

coreset 文本 token 总量约为 SR 的 2.8 倍和 3.4 倍；这里统计的是编码器截断前的存储文本，不等同于实际送入编码器的有效 tokens。当前是等 rows 比较，不是等文本/等字节比较。CaT 与 SR 文本存储接近，SR 对 CaT 的差距不能仅用 coreset 长文本解释。Student 接收预计算文本 embedding，因此长原文不会按比例增加 Student 训练时间。

## 7. 结果如何解释

1. **旧好结果没有消失。** SR 的 72 个重新训练/评价记录完全复现；变化来自外部对照更完整。
2. **旧 coreset 表是内容控制。** 其共同使用 C0 stage0 和 C0 topology；这轮共同 Student 不变，但各外部方法从 stage0 自行构造。k-center 平均 F1 从旧控制表 48.23 变为 54.18，不能把约 5.96 pp 全部归因于拓扑，因为初始 memory 也改变了。
3. **不要求每个指标都赢，但主张须匹配证据。** SR 可以继续作为提出和分析的主方法，贡献也不以超越 C0 为条件；当前无法主张其整体效果优于本轮最强外部方法。
4. **不能再说所有 X 优化都没用。** 本轮 CaT 的随机编码器分布匹配是一个实测有效的外部方案，与早期不可靠的 teacher proxy 或短程 response selector 不同。尚不能据此推断将 CaT-X 直接塞入 SR 必然收益。
5. **局部 assignment 最优性不是下游最优性。** 已有[形式化文档](SEMANTIC_RECURSIVE_CONSTRAINED_TRANSPORT_SPEC_ZH_2026-09-27.md)限定了 payload/slot、矩形匹配和补全范围，不为增强叙事改变已测实现。

## 8. 后续顺序：先补证据，再做小改动

| 优先级 | 下一项任务 | 怎样控制 | 不应如何解读 |
|---|---|---|---|
| P0 | SR、CaT、k-center 在 3-5 个预注册新顺序、多个构造 seeds 上确认 | 各方法一套冻结配置，Student 配对；保留已知四条件作为开发/诊断 | 不按数据集回选赢家或宣称已有独立确认 |
| P0 | 跨顺序匹配预算及文本资源补充 | 同阶段 K/steps 对所有方法一致，不减少合法 raw pool；另保留自然 fixed-rate；补实际编码 token/持久状态 | 不把 equal rows 当 equal bytes，不偷偷给 SR 更多 rows |
| P1 | two-prototype 与 dispersion 敏感性、identity/semantic placement 机制 | 保留冻结 SR；只变一个重组参数；全部报告，不按新 report-dev 搜参 | 不预设“两个原型理论最优” |
| P1 | 诊断 CaT 相对 SR 的优势来源 | 同 parent、同文本、同拓扑、同新增 rows；比较当前 X 与有限分布匹配残差，并计构造成本 | 这是新实验，当前未完成；不能把端到端 CaT 胜出直接归因为单一 X 算子 |
| P2 | 有证据后才考虑受限 message-aware cost | 不改 rows、文本、Student；沿用独立评价与回退条件 | 不恢复大规模 selector/attention/refresh 堆叠 |
| P2 | 外部大图与论文最终表 | 同 base+4 和合法输入；区分原生静态 TAG/受限适配/完整 raw | 不把本轮小图适配冒充完整公开方法 SOTA |

本轮没有自动把 CaT 的 X 或新 topology 加进 SR，也未启动新的参数搜索。接下来的最直接问题是：与两个真实强对手相比，SR 的差距是否在独立顺序及公平存储控制后仍存在；然后才有依据做有限改动。

## 9. 可复现证据

- [冻结协议及 upstream 版本](EXTERNAL_CONTINUAL_ADAPTER_PROTOCOL_ZH_2026-09-27.md)
- [汇总 JSON](../../reports/external_continual_adapters_20260927/summary.json)、[完整指标与配对 CI](../../reports/external_continual_adapters_20260927/summary.md)、[逐条件 CSV](../../reports/external_continual_adapters_20260927/cells.csv)
- [Cora 原始结果](../../results/external_continual_adapters_20260927_v2/cora_run/results.json)、[Citeseer 原始结果](../../results/external_continual_adapters_20260927_v2/citeseer_run/results.json)
- [数据、编码器和软件版本 hash](../../results/external_continual_adapters_20260927_v2/input_provenance.json)
- [构造实现](../../code/v2_llm_fusion/external_continual_mechanisms.py)、[runner](../../code/v2_llm_fusion/run_external_continual_adapters_20260927.py)、[汇总实现](../../code/v2_llm_fusion/summarize_external_continual_adapters_20260927.py)

五项机制测试本地和服务器均通过；正式结果四份 audit 全部通过；下载的 60 张 memory 文件 hash 与远端记录一致。所有 Student checkpoint 保留在服务器冻结结果目录，本地保留指标、审计、协议及 synthetic memory，未为汇总重复下载大批 checkpoint。早期不完整 adapter 草稿和 10-step smoke 不纳入正式结果。
