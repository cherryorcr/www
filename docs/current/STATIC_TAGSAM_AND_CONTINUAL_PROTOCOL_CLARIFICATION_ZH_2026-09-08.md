# 静态 TAGSAM、持续评测协议与 TAG 特性澄清

日期：2026-09-08

这份文档专门回答前面没有单独回答清楚的四个问题：

1. 原始 TAGSAM 静态凝聚到底做了什么，当前复现是否等价；
2. 静态结果应该如何测试，训练节点、分阶段留出集和未来数据分别表示什么；
3. 持续凝聚的误差应该在哪个阶段、用什么对照定位；
4. 在下游使用 GCN 时，TAG 的文本特性具体通过什么路径发挥作用。

## 一、先说明当前工作的真实状态

前一阶段实际做了两件不同的事，但没有及时分开说明。

第一件事是重新核对官方 TAGSAM。官方 master、add-adj 分支和最小七类修正版都已经运行或复评过。这个结果已经证明：原始代码可以运行，但 Cora evaluator 默认写死五类，并且旧的 subgraph cache 会让七类运行静默复用五类任务。因此不能把官方五类结果、七类修正版结果和当前持续实验混成一张表。

第二件事是实现 `native_tag_v1`，用一个可审计的 TAGSAM 派生接口做三阶段持续凝聚和细粒度归因。它的目的是先验证“小 TAG 能否支持独立 student 训练”和定位持续损失，不是官方 TAGSAM 的精确复现：它把 GPT2-CMI 文本选择换成了 joint graph-text medoid 初始化，使用完整图消息传播，并把官方 `wBCELoss` 换成了自己的 row/column softmax ASM loss。因此，`native_tag_v1` 的结果不能回答“官方 TAGSAM 静态凝聚是否有效”这个问题，只能回答“受控的 TAGSAM 风格持续原型发生了什么”。

这也是前面推进顺序不够清楚的地方：在静态起点完全冻结前，就先做了持续原型和归因。已有归因仍然有用，但接下来必须先把官方静态基线和评测协议锁定，再继续设计持续算法。

## 二、原始 TAGSAM 静态凝聚的准确流程

### 2.1 原始输入和 teacher

官方代码从 TAG 数据集读取：

- 节点特征 `node_f`；
- 原图 `edge_index`；
- 每个节点的可读文本；
- 类别名称和类别描述。

`TextEncoder` 使用冻结的 `bert-base-uncased`，取 CLS 向量作为每个节点文本的 `text_embeds`。文本编码器参数不参与训练，但图文投影头会训练。teacher 是在真实图上训练的 CLIP 风格模型：两层 GCN 编码节点，文本投影头编码 BERT 向量，归一化后用双向 InfoNCE 对齐同一节点的图向量和文本向量。

### 2.2 synthetic 节点和文本如何生成

官方 `selection.select_text` 的流程不是普通 k-means：

1. 从目标节点中随机选择 synthetic 中心；
2. 对每个中心取小范围 k-hop 邻域；
3. 将邻域文本分成句子块；
4. 用 GPT2 的 NPMI/相关性和冗余惩罚选择一部分句子；
5. 将选中的句子拼成可读摘要；
6. 用每个源节点的抽取比例加权其 `node_f`，得到 synthetic 节点特征；
7. 用冻结 BERT 编码摘要，得到 synthetic 文本向量；
8. 保存源节点、摘要片段和抽取比例等 provenance。

所以原始 TAGSAM 的 synthetic 节点不是只有一个无语义的可学习向量。每个节点至少有一对固定的可读文本/文本向量和可优化的图特征，二者在后续图文对齐中共同发挥作用。

### 2.3 synthetic 拓扑和可优化变量

官方 master 的 `SynGraphDataset` 默认把 synthetic 图设为 self-loop。这个版本中，凝聚阶段主要优化：

- synthetic 节点特征 `node_f`；
- 图编码器内层学习率；
- 文本投影头内层学习率。

摘要文本、BERT 文本向量和 self-loop 拓扑固定不变，没有 synthetic 标签优化。

官方 add-adj 分支另外把 provenance 投影成 synthetic 边，并在 artifact 中保存 `edge_index`。它不是 master 的一个小修补，而是不同的静态拓扑变体，必须单独报告。

### 2.4 ASM 外层凝聚

每个 outer iteration 都会：

1. 在 synthetic TAG 上从头初始化一个 student；
2. 用 synthetic 图文对做若干步内层 SGD；
3. 从真实图采样一批节点及其邻居；
4. 计算 teacher 和 student 在这批真实节点上的图文 logits；
5. 用 `wBCELoss` 匹配两组 logits；
6. 将梯度传回 synthetic `node_f` 和内层学习率；
7. 保存 synthetic artifact，并在指定 iteration 做下游评估。

官方 `eval.py` 会重新初始化多个 student，在 synthetic TAG 上训练若干 epoch，再到真实图的测试任务上评估，取训练过程中的最佳准确率。原始 `distill.py` 的 ASM 目标是 `wBCELoss`：先对 teacher/student logits 做 sigmoid，再对正负位置分别平衡 BCE；它不是 `native_core.py` 中的 row/column softmax ASM。原始 evaluator 主要报告任务平均 Accuracy，不提供我们现在需要的“每个历史阶段、每个留出分区”的矩阵。

## 三、原始静态复核结果意味着什么

官方结果已经足以说明静态起点存在明显问题，但不能把问题说成“TAG 凝聚不可行”。

| 版本 | 任务协议 | 代表结果 | 能否作为当前持续主基线 |
|---|---|---|---|
| official master | Cora 原始五类 evaluator | step 0 `0.4275`，step 500 `0.3824` | 只能作为历史复核，不能与七类结果混用 |
| official add-adj | 正确七类 cache，5 次 student | step 0 `0.38576`，step 500 `0.40230` | 作为 learned-topology 静态敏感性对照 |
| minimal repair | 七类 evaluator，self-loop，seed 44/45 | seed44 `0.3340 -> 0.3624`；seed45 `0.3381 -> 0.3874` | 作为最小改动、可审计的持续起点 |

这里的 Accuracy 不是 `native_tag_v1` 的确认 Macro-F1，不能直接横向比较。共同结论是：

- 小 TAG 可以被重新训练，step 0 到后续 iteration 有时会提高；
- 凝聚优化的跨 seed 稳定性还不够；
- 原始静态结果和完整 teacher 之间仍有较大差距；
- 需要先固定一个可复现静态协议，再评价持续误差。

### 静态起点的决定

接下来使用三条明确的静态参照：

1. `TAGSAM-master-5way`：只用于说明历史代码和五类 evaluator 的结果；
2. `TAGSAM-minimal-repair-7way-self-loop`：**持续误差定位的主起点**，因为它最接近 master，改动最少，便于把持续因素和静态拓扑因素分开；
3. `TAGSAM-add-adj-7way`：**TAG 拓扑敏感性对照**，不和最小修正版合并成一个数字。

如果最小修正版在预注册的多 seed 静态 gate 中不稳定，不能直接用 add-adj 的单次较高结果替代它；应先报告静态 gate 失败，再决定是否把 add-adj 作为后续主后端。这样可以避免把静态实现差异误报为持续算法收益。

## 四、静态阶段应该怎样测试

当前建议采用“模型产生时间/阶段 `t`，评估分区 `j`”的二维记号。`S_t` 表示 student 在截至阶段 `t` 的 synthetic TAG 上独立从头训练，`D_j^p` 表示阶段 `j` 的分区 `p`，其中 `p` 可以是 `train`、`dev` 或 `test`。

### 4.1 四类评估，不再混称“准确率”

#### A. 原始训练节点评估：训练拟合诊断

把 student 训练在 synthetic TAG 上，然后在原始阶段训练节点 `D_j^train` 上评估，记为：

\[
M_{t,j}^{train} = M(S_t, D_j^{train}).
\]

它回答的是：压缩图是否保留了原始训练节点上的图文对齐和分类响应。它不是泛化指标，也不能替代测试集。若模型在 synthetic 节点上 loss 很低但 `M_{t,j}^{train}` 很差，说明 synthetic 图自身被拟合了，却没有复现真实图的响应。

#### B. 已到达阶段的逐阶段开发集

对每个 `j <= t` 报告 `M_{t,j}^{dev}`。它用于：

- 选择 ASM outer checkpoint；
- 选择 student checkpoint；
- 调整开发期的学习率、reference 和初始化。

它不能被写成最终测试结论。

#### C. 已到达阶段的逐阶段测试集

对每个 `j <= t` 报告 `M_{t,j}^{test}`，同时报告：

- 每个阶段的 Accuracy；
- 每个阶段的 Macro-F1；
- 将所有已到达阶段节点拼接后计算的累计 Accuracy/Macro-F1。

累计 Macro-F1 必须在拼接后的预测上重新计算，不能简单平均各阶段 F1，因为跨阶段误分类会改变全局 false positive。

#### D. 未来阶段评估：前向迁移/零样本泛化

在 student 只使用 `S_t` 训练、没有读取未来节点或未来边的情况下，评估未来阶段 `j > t` 的留出节点：

\[
M_{t,j}^{future} = M(S_t, D_j^{test}), \quad j>t.
\]

当前类别流采用**全局类别提示可见**协议：七个类别名称和描述在实验开始时作为公开语义 ontology，student 可以在任意阶段与七个文本提示比较，但未来节点的特征、文本、边和标签不能进入训练或凝聚。这样未来类别准确率才有定义，并且它测量的是 TAGSAM 的文本零样本前向迁移。

如果采用“类别提示直到类别到达才公开”的严格 class-incremental 版本，那么尚未到达类别没有可用的分类候选，未来未见类别的 Accuracy 在到达前是未定义的；该版本可以作为附加实验，不能和上面的 future accuracy 混称。

### 4.2 当前 Cora 类别流的具体矩阵

当前开发数据使用三阶段：`[0,1] -> [2,3] -> [4,5,6]`。在每个阶段结束都生成以下矩阵：

| 训练/评估 | stage 0 | stage 1 | stage 2 |
|---|---:|---:|---:|
| `train` 原始节点 | `M_{t,0}^{train}` | `M_{t,1}^{train}` | `M_{t,2}^{train}` |
| `dev/checkpoint` | `M_{t,0}^{dev}` | `M_{t,1}^{dev}` | `M_{t,2}^{dev}` |
| `test/confirmation` | `M_{t,0}^{test}` | `M_{t,1}^{test}` | `M_{t,2}^{test}` |
| `future`（仅 `j>t`） | — | `M_{0,1}^{future}` | `M_{0,2}^{future}`, `M_{1,2}^{future}` |

每个单元格都保留 Accuracy、Macro-F1、样本数和 student seed；最终表还要给出均值、标准差和凝聚 seed 的配对差值。

### 4.3 评估图的消息传播边界

为了先复现官方 TAGSAM，主静态协议保留其 transductive 评估形式：评估节点可以通过固定的真实图 `edge_index` 接收无标签邻居特征，指标 mask 决定哪些节点计分。这个协议不能称为严格节点归纳。

之后增加一个严格 inductive 敏感性对照：从评估消息图中删除留出节点及其边，只保留允许访问的上下文。两者必须分开报告，否则未来准确率会把“看到未来节点作为邻居”与“模型真正前向迁移”混在一起。

## 五、持续凝聚的 setting 如何固定

### 5.1 当前开发 setting：类别分块、全局文本 ontology

短期继续使用类别分块，理由是：

- Cora 原始 TAGSAM 资产已有稳定的七类文本和类别提示；
- 可以明确控制哪些类别在何时到达；
- 能直接测量新类别文本的零样本前向迁移；
- 不会把 Cora 中并不存在的真实时间戳强行解释成时间流。

它是**可控的开发协议**，不是最终论文中唯一需要的数据流。最终论文应再加入具有真实时间字段的 TAG，例如按发表年份划分的 arXiv 类数据，并重新生成未使用的最终留出集。

### 5.2 数据到达规则

在 stage `t` 到达的数据只有当前阶段的原始训练节点、节点特征、节点文本和阶段内诱导边。凝聚器可以读取当前 delta 和已显式保存的 synthetic memory；不能读取历史原始节点来重新生成旧摘要。类别分块由准备脚本使用标签定义，标签不进入 teacher、ASM 或 student 的训练损失。

### 5.3 预算规则

当前开发用累计训练根数比例：

\[
K_t=\lfloor \rho N_{\le t}^{train}\rfloor.
\]

这意味着 budget 随阶段增长，便于先研究递归误差；固定总容量是另一个更强的 eviction setting，暂不与当前结果混用。每个 artifact 同时记录 root 数、有效 tensor/text 字节数和质量 mass。

## 六、持续误差必须怎样定位

在设计新机制前，每个阶段都运行以下对照。它们回答的是不同问题，不能把差值直接相加成严格因果分解。

| 对照 | 输入 | 主要回答 |
|---|---|---|
| full-history raw | 全部原始 `G_{<=t}` | 宽松真实历史参照 |
| old memory only | `S_{t-1}` | 旧 memory 自身还能保留多少历史响应 |
| new delta raw | 当前原始 `Delta_t`，通常超预算 | 当前数据自身的训练效用 |
| old + new raw over-budget | `S_{t-1} ∪ Delta_t` | 历史原图被替换后，当前信息是否仍可用 |
| old + new raw matched | 旧 memory 加同预算 raw 子集 | 同预算随机压缩参照 |
| old + new compact | 旧 memory 加同预算凝聚 delta | 新增块凝聚损失 |
| append/frozen | 旧 memory 保留，只凝聚新 delta | 不重新组织旧 memory 的基线 |
| union/recondense | 旧 memory 与新 delta 联合重凝聚 | 再次凝聚造成的额外损失 |
| fixed outer snapshots | 同一初始化的 step 0/20/40/60/80/100 | ASM 轨迹何时开始偏移 |

### 6.1 建议使用的性能差值

记 `Q` 为同一评估单元的 Macro-F1 或 Accuracy：

- **历史替代损失**：`Q(full-history raw) - Q(old + new raw over-budget)`；表示不能再访问历史原图后损失多少；
- **新增预算凝聚损失**：`Q(old + new raw matched) - Q(old + new compact)`；表示在同一总行数下压缩当前 delta 的代价；
- **再凝聚损失**：`Q(append/frozen) - Q(union/recondense)`；表示是否因为把旧 memory 再次放回凝聚器而下降；
- **静态凝聚损失**：`Q(full-history raw) - Q(offline condensed full-history)`；表示一次性静态凝聚本身的误差；
- **优化损失或收益**：固定初始化和固定字段，比较 step 0 与每个 outer snapshot；不能只看 validation 选中的一个状态。

对于历史阶段 `j<t`，报告 `Q_{t,j}-Q_{j,j}` 作为遗忘变化；对于当前阶段 `j=t`，报告新类塑性；对于 `j>t`，报告前向迁移。每次都要同时看 `train`、`dev` 和 `test` 分区，才能判断是训练拟合失败、留出泛化失败，还是实际持续遗忘。

### 6.2 目前已有归因能下什么结论

已有 stage-2 诊断已经给出：

- Append old + new compact：`42.98%` checkpoint Macro-F1；
- 同预算 old + new raw：`40.08%`；
- old + new raw over-budget：`47.39%`；
- Full-history：`51.16%`；
- 换成 Union old 后 old + new compact 降至 `33.18%`。

这说明旧 memory 质量和新增块有限预算凝聚都是实际缺口。旧 X 是否允许更新、joint 与 protected 的额外交互，以及跨阶段边是否为主因，目前都没有稳定证据。因而下一步应先做字段级静态对照和固定 outer-step 曲线，不应直接加入 transport 或新的跨阶段关系模块。

## 七、TAG 相对于 plain graph 的特性在 GCN 下如何体现

### 7.1 GCN 本身不直接读取文本

原始 TAGSAM 的 GCN 接口是：

\[
H^{(1)}=\operatorname{GCN}(X,A),\qquad
H^{(2)}=\operatorname{GCN}(H^{(1)},A).
\]

文本不作为 GCN message 的额外输入。TAG 的差异通过以下四条路径进入：

1. **图文训练目标**：图向量必须和同一节点的文本向量对齐；plain graph 的监督目标通常没有这一项；
2. **文本投影和类别提示**：推理时将图节点向量与类别名称/描述的 BERT 向量比较，因此可以做 zero-shot 分类；
3. **文本引导的 synthetic 节点**：原始 TAGSAM 用真实文本摘要选择邻域和聚合 `X`，不是只按图结构随机采样；
4. **可选文本/来源拓扑**：add-adj 将 provenance 投影成 synthetic 边，进一步把文本摘要覆盖的来源关系带入 GCN。

因此，如果只把 TAGSAM 的 `text_emb` 字段附在数据结构中，却训练一个完全不看文本的监督 GCN，就不能声称利用了 TAG 特性。

### 7.2 持续过程中 TAG 的专门困难

每个新阶段都要维护一个三元对应关系：

\[
\text{synthetic }X_i\;\leftrightarrow\;\text{可读文本 }T_i\;\leftrightarrow\;\text{synthetic 拓扑 }A_i.
\]

旧 memory 被 Union 重组时，可能同时发生：

- 原有文本代表被丢弃；
- centroid `X` 与代表文本不再对应；
- provenance 边发生变化；
- 新旧图文对在 InfoNCE 负例池中的相对位置改变。

这就是 TAG 持续凝聚相对于 plain graph continual condensation 的主要可检验差异。它不是简单地把 `text_emb` 多存一列，而是要保持图特征、可读文本和下游 GCN 响应的一致性。

### 7.3 当前主线的 TAG 处理方式

在静态起点锁定前，持续实验采用保守接口：

- 文本摘要、BERT CLS 和文本投影头保持 TAGSAM 原生语义；
- synthetic `X` 可以被 ASM 优化，文本和拓扑先固定；
- 每个阶段新增节点都必须带真实可读文本和对应 BERT embedding；
- 旧文本代表是否被保留、`X` 是否仍与代表文本匹配，作为明确诊断变量；
- 只有证据显示“旧新图文关系联合后产生额外损失”，才设计关系保持机制。

这样可以先回答持续凝聚到底损失了什么，再决定是否需要让文本、拓扑或跨阶段关系成为新的可优化变量。

## 八、接下来实际执行顺序

### 第一步：冻结静态协议

用官方 master、最小七类修正版和 add-adj 三条静态参照，固定：

- Cora 七类任务和独立 cache 名称；
- teacher/student seed；
- synthetic 节点数和预算；
- BERT 版本、CLS pooling 和文本提示；
- student 训练步数、checkpoint 规则和 examples seen；
- train/dev/test/future 矩阵；
- transductive 主评估和 inductive 敏感性评估。

静态 gate 至少要求：step 0、ASM、random-small 和 full-real 都有独立 student 重训；paired-text 与 shuffled-text 的差异被记录；ASM 不能只凭一个 validation student 选择。

### 第二步：在固定旧 memory 上做持续归因

对每个阶段输出上面的对照矩阵，并固定 outer step `0/20/40/60/80/100` 的多 student 曲线。先判断下降发生在：

- 旧阶段留出保持；
- 当前阶段新增块；
- 未来类别前向迁移；
- 再凝聚的某个 outer 区间。

### 第三步：只选择一个经过证据支持的机制

- 如果主要是新增块 `X` 聚合误差：研究图文一致的 centroid/representative 更新；
- 如果主要是旧文本代表和 `X` 的错配：研究文本代表保持或受约束更新；
- 如果只有联合阶段额外下降：再研究跨阶段关系或 response-preserving transport；
- 如果差异主要来自 student/validation 噪声：先改成多 student 固定步选择，不能把选择器噪声包装成算法收益。

### 第四步：扩大验证范围

只有一个机制在 Cora 开发矩阵上跨凝聚 seed 稳定改善后，才加入第二种阶段顺序、真实时间 TAG 数据和全新最终留出集。已经访问过的 Cora confirmation 或旧 arXiv t3 不能重新称为未使用的最终测试。

## 九、当前结论

目前还不能说已经确定了“最好的静态 TAGSAM 凝聚方法”，也不能说持续 joint 方法已经解决了误差。可以确定的是：

1. 原始 TAGSAM 静态流程是 GPT2 文本摘要 + BERT CLS + GCN/文本图文对齐 + ASM，文本确实参与了凝聚和下游分类；
2. 官方 Cora 代码存在五类 evaluator 和 stale cache 问题，最小七类修正版已经建立；
3. `native_tag_v1` 是受控派生实现，适合持续归因，但不是官方精确复现；
4. 静态凝聚存在较大且不稳定的性能缺口，必须先完成静态 gate；
5. 持续损失目前最可信的两个来源是旧 memory 质量和新增块有限预算凝聚，不能先归因于跨阶段边或 joint 交互；
6. GCN 中的 TAG 特性主要通过图文对齐、真实文本代表、类别提示和可选文本来源拓扑体现，而不是通过 GCN 层自动读取文本。

本文件之后，新的实验结果都按上述 `M_{t,j}^{p}` 矩阵和对照命名记录，避免再次把静态复现、持续诊断和新方法结果混在一起。
