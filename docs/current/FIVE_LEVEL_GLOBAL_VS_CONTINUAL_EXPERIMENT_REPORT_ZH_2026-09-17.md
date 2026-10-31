# 五层小图对比：结果与完成状态

2026-09-17；Cora / Citeseer，canonical三阶段，report-dev。

同日后续更新：已新增STAD直接TAG适配及原生风格监督诊断，统一性能保留率见[外部基线结果报告](EXTERNAL_TAG_BASELINE_RESULTS_ZH_2026-09-17.md)。该补充不改变本报告五层实验数值，也不补齐下述严格三方机制分解。

## 1. 是否全部完成

**没有全部完成。** 五个层次均已有实验代表，但强全局方法完整比较、严格只改变记忆管理的损失分解仍有缺口。TAGSAM响应匹配适配不等于官方完整复现，有限搜索监督参考也不是理论上界。

| 用户要求 | 状态 | 证据 / 尚缺内容 |
|---|---|---|
| 持续合成图追加 | 已完成当前小图协议 | v4，2数据集 × 3阶段 × 6 student seeds |
| 全历史原始数据重凝聚 | 已完成启发式基线 | 全历史joint聚类、dispersion、concat；不是全局最优 |
| 旧合成图与新数据重凝聚 | 已完成既有实现 | `recluster`，构造与分配和v4并非完全一致 |
| 三方只改变memory管理的严格对照 | **未完成** | v4有response allocation，recluster另用构造器，不能纯粹分离历史访问/可改写/拓扑损失 |
| 全局图文聚类 | 已完成 | 即`full_history_syn`，不是额外独立方法 |
| 全局静态TAGSAM完整适配 | **部分完成** | 全历史官方response loss、两种拓扑已跑；缺完整官方文本选择与全部训练机制 |
| 全局图文梯度匹配 / 特征匹配 | **未完成** | 无当前同协议可报告结果 |
| 全局响应优化 | 已完成一个适配版本 | TAGSAM response matching，500 outer × 10 inner，未证明收敛及构造种子稳健 |
| 当前v4的全局版本 | **未完成** | TAGSAM特征优化不等于全局v4 student-compatible allocation / selection |
| 完整原始数据、等student步数 | 已完成 | 等optimizer步数，不等FLOPs/内存/wall time |
| 完整原始数据、长训及检查点选择 | 已完成固定配置参考 | 8192步；没有完整LR搜索与收敛证明，不宜称充分调优最优 |
| 完整数据监督GCN | **本轮完成有限搜索参考** | LR/weight decay网格、独立选择种子、早停、最佳检查点，另表报告 |
| 静态TAG新方法检索 | 已查并扩展 | TAGSAM、STAD；同日后续已完成STAD适配及监督诊断，DANCE待确认实现，不代表穷尽文献 |

## 2. 协议和可比性

- Cora类别顺序`[0,1]/[2,3]/[4,5,6]`，Citeseer`[0,1]/[2,3]/[4,5]`。
- 合法pool累计5%合成预算：Cora25/67/94；Citeseer38/86/111。
- student seeds100..105；GCN hidden/output384、冻结MiniLM384、可训练文本投影、双向InfoNCE、Adam0.001、batch16。
- student步数：Cora400/1072/1504；Citeseer608/1376/1776。
- 指标为report-dev Macro-F1百分数；误差为六student seeds的population std，不是置信区间。
- 阶段均值默认all-class prompts：在已见类别评估节点上允许全部类别prompt竞争。seen-only均值另存summary.json；最终阶段二者相同。
- 构造可使用合法pool类别标签；零样本只指student不以分类交叉熵训练，不表示整个管线无标签。
- 推理使用真实全图，属于transductive。监督参考训练也用全图传播，GraphCLIP raw训练则用合法pool诱导图，二者并非只差损失。
- 本轮不评估test标签；项目历史开发访问过test，不能声称最终盲测。

## 3. 同一GraphCLIP下游结果

| 方法 | Cora最终 | Citeseer最终 | Cora阶段均值 | Citeseer阶段均值 |
|---|---:|---:|---:|---:|
| **v4持续合成追加** | **49.85 ± 3.00** | **60.77 ± 1.64** | 53.08 | 64.82 |
| dispersion持续构造，无v4 response分配 | 49.85 ± 3.00 | 59.17 ± 1.57 | 53.08 | 64.29 |
| 全历史启发式重凝聚 | 47.21 ± 2.06 | 48.28 ± 3.48 | 53.86 | 58.37 |
| 旧合成图 + 新数据重聚类 | 49.70 ± 1.21 | 54.24 ± 1.91 | 48.99 | 61.28 |
| **静态TAGSAM，self-loop适配，新增** | **47.73 ± 4.00** | **51.09 ± 2.85** | 65.67 | 55.92 |
| **静态TAGSAM，semantic适配，新增** | **52.86 ± 3.46** | **42.00 ± 5.85** | 56.21 | 43.97 |
| 完整原始数据，等optimizer步数 | 47.93 ± 4.10 | 49.89 ± 2.17 | 51.38 | 51.74 |
| 完整原始数据，8192步终点 | 46.26 ± 2.73 | 52.33 ± 1.75 | 48.24 | 52.39 |
| 完整原始数据，8192步内selector最佳检查点 | 51.31 ± 2.89 | 55.60 ± 1.92 | 59.84 | 60.98 |
| 原始数据模型 → 新数据微调 | 48.10 ± 2.78 | 51.13 ± 2.23 | 49.06 | 50.47 |
| v4阶段0模型 → 新数据微调 | 53.41 ± 1.85 | 52.47 ± 1.46 | 56.57 | 56.10 |

TAGSAM两臂均为事先指定，不事后按数据集择优拼成一个方法。静态构造可回读历史原始数据，不能冒充持续算法。

### 静态适配范围

每阶段从完整已见合法pool重新生成5%合成图，joint/dispersion/concat初始化，teacher训练4096步；调用固定官方提交的`wBCELoss`，对全部synthetic X做500次outer更新，每次展开10次inner SGD。文本和拓扑冻结，最终仍用canonical Adam student评估六seeds。

未采用官方NPMI文本选择器、官方编码器、5000 outer、15 inner epochs、最多2000参考节点采样及蒸馏学习率下游训练。因此是“静态TAGSAM响应机制适配”，不是官方完整复现，不能据此宣称超过充分优化的官方方法。

## 4. 监督参考单列

| 方法 / 差距 | Cora | Citeseer |
|---|---:|---:|
| 有限搜索监督GCN，最终F1 | **84.10 ± 0.33** | **69.26 ± 0.82** |
| 监督GCN，阶段平均 | 92.54 | 81.52 |
| 监督GCN最终旧类 / 新类 | 88.16 / 78.69 | 75.75 / 56.29 |
| v4最终旧类 / 新类 | 65.44 / 29.06 | 68.00 / 46.29 |
| 监督参考减v4最终F1 | **34.25** | **8.50** |

**差值不是凝聚损失**：包含分类监督、训练目标、传播图输入和训练过程差异。它说明同一数据/划分下仍有监督任务性能空间，不是零样本方法的公平胜负表。

Adam固定网格lr`{0.001,0.01}` × weight decay`{0,0.0005}`；每阶段用seeds7001/7002的selector-dev最佳F1均值选配置，再用100..105从头训练。每25步评价selector-dev，最少512步，连续20次无提升早停，上限8192步。搜索阶段不计算report-dev，保存终点与最佳模型。

| 数据集 | s0 lr / wd | s1 lr / wd | s2 lr / wd | s2六seeds最佳步数 |
|---|---|---|---|---|
| Cora | 0.001 / 0 | 0.001 / 0 | 0.01 / 0.0005 | 225,675,225,500,200,300 |
| Citeseer | 0.001 / 0.0005 | 0.001 / 0 | 0.001 / 0.0005 | 75,75,75,75,75,225 |

这不是穷尽模型、正则化和超参数搜索后的最优值。

## 5. 结论与统计边界

**Cora仍有性能空间。** 静态semantic适配、v4初始化后微调、长训raw最佳检查点分别比v4均值高3.01、3.55、1.46点。但配对六种子区间均跨零，不构成稳定胜负。v4减semantic的探索性配对95% t区间为[-8.86,2.85]。self-loop的阶段平均65.67高于v4的53.08，最终却降至47.73，提示必须报告全阶段轨迹。

Cora v4新类F1=29.06，raw最佳检查点=38.59，差9.53点。新类塑性值得优先优化，但不能从现有对照直接断言单一模块是原因。

**Citeseer当前设置内优势较一致。** v4比全历史启发式、TAGSAM self-loop适配、raw最佳检查点分别高12.48、9.68、5.16点；各项均为6/6 student seeds正差，探索性配对95%区间分别为[7.87,17.09]、[5.56,13.80]、[2.34,7.98]。

以上区间只覆盖固定构造和类别顺序下的student随机性，未校正多重比较，不能替代独立construction seeds、多类序或多数据集。此前两位小数相减得到12.49、5.17，本报告按未舍入原始数计算为12.48、5.16，实验值没有改变。

**现有数字不能构成可加的损失分解。** 三方同时改变allocation、聚类、拓扑，监督参考又改变目标及输入权限。不能把监督参考、全局凝聚、持续追加之间的相邻差值分别当作纯粹监督损失、凝聚损失和持续损失。

## 6. 直接面向TAG的静态方法

| 方法 | 已核对来源 | 本项目状态 |
|---|---|---|
| TAGSAM，Text-attributed Graph Condensation via Text Selection and Attribute Matching | [论文](https://arxiv.org/abs/2606.03839)、[官方代码](https://github.com/SundayVHan/TAGSAM)，提交`d812f06fd16db3cd4d497804a8e504d2c738faa8` | 全局响应匹配适配已跑，官方完整端到端未完成 |
| STAD，Semi-Supervised Text-Attributed Graph Distillation | [论文](https://arxiv.org/abs/2607.20477)、[作者代码](https://github.com/laiyurui/STAD-Semi-Supervised-Text-Attributed-Graph-Distillation)，提交`5aaf1a06e44638499373b1358c85d1546187edae` | 同日后续已完成本地Qwen、加权图、canonical下游和独立监督诊断，详见外部基线结果报告 |

STAD采用图/文本双路径协同自训练、Wasserstein引导图压缩、关键词驱动LLM文本合成。同日后续接入保留了文本生成，但将官方默认Qwen API替换为本地Qwen3-32B-AWQ；已限制合法pool、禁用test评分、统一预算分母，分别报告监督诊断和GraphCLIP迁移结果。该适配不是官方完整分数复现。

## 7. 成本与验收

| 新增工作（三阶段合计） | Cora秒数 | Citeseer秒数 |
|---|---:|---:|
| TAGSAM self-loop构造 | 115.62 | 112.80 |
| TAGSAM semantic构造 | 88.39 | 88.65 |
| self-loop下游18次训练 | 40.52 | 51.15 |
| semantic下游18次训练 | 39.83 | 50.02 |
| 监督网格24次训练 | 28.68 | 21.11 |
| 监督报告18次训练 | 20.57 | 18.58 |

两TAGSAM臂复用teacher，teacher训练计入先运行的self-loop构造；semantic计加载成本，不重复计teacher训练。这不是独占硬件端到端测速，加载/审计开销另计。监督full-batch与InfoNCE mini-batch步数不可直接当作同等计算。

- 本轮新增156次fit：静态报告72、监督报告36、独立监督网格48；另6个teacher训练不计入下游fit数。
- 审计660次记录：已有core504 + 新增156；旧continual TAGSAM适配144次不在本轮660中，其审计另存。
- 108个新增报告fit加载参数重算指标，监督终点及最佳checkpoint均复算；**0 errors**。
- 校验来源、预算、哈希、冻结文本/拓扑、初始化复验、teacher累计来源、selection种子隔离和逐seed完整性。
- 本地6项测试通过。远程tagsam未安装pytest，远程验收通过实际运行、独立审计和模型重算完成。
- 完整参数保留远程；新增JSON、manifest、构造曲线、CSV及验收已同步本地。

## 8. 后续优先级

**最新优先级覆盖（2026-09-17）：** 用户要求先冻结当前 v4，完成外部 TAG 凝聚基线比较，主看相对完整数据参考的性能保留率。下列消融与全局 v4 开发延后；STAD适配和监督诊断已按 `EXTERNAL_TAG_BASELINE_PROTOCOL_ZH_2026-09-17.md` 完成，DANCE尚未确认实现。后续以[外部基线结果报告](EXTERNAL_TAG_BASELINE_RESULTS_ZH_2026-09-17.md)第8节为准，不得以本节旧排序继续优先优化主方法或重复接入STAD。

1. 严格matched-constructor三方对照：统一response开关、class quota、文本预算，分别干预原始历史可读性、旧节点可改写性与拓扑。旧memory重聚类必须保留类别及mass，不能把每个合成行视作一个原始样本。
2. 全局v4：相同5%预算、候选、selection seeds和response训练预算，允许重新分配全部历史合成节点；不是用TAGSAM loss代替v4选择器。
3. TAGSAM完整文本机制和优化预算敏感性，再接入STAD；外部LLM调用需明确来源、文本外发、成本及复现条件。
4. 独立的全局梯度/特征匹配，固定初始化及拓扑，完整记录优化、失败和检查点选择，不与response matching重复计数。
5. 小图多类序、独立construction seeds与压缩率通过后再上大图；本轮不启动大图主表。

## 9. 证据入口

- 协议：`docs/current/FIVE_LEVEL_FROZEN_PROTOCOL_ZH_2026-09-17.md`
- 新增runner：`code/v2_llm_fusion/run_five_level_extension.py`、`run_tagsam_graphclip_static.py`、`run_supervised_upper_bound_small.py`
- 审计：`code/v2_llm_fusion/summarize_five_level_comparison.py`
- 汇总：`results/five_level_summary_2026_09_17/{summary.json,summary.csv,acceptance.json}`
- 逐次证据：`results/five_level_extension_2026_09_17/{static,supervised}/{cora,citeseer}/`

论文仍应定位为可审计的持续TAG协议、student可重训性机制、保持/塑性与负结果边界，不宣称SOTA。
