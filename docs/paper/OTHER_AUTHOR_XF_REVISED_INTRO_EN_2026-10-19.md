# Revised `\\xf{}` Introduction Based on `main.pdf`

## 修改判断

`main.pdf` 的 Introduction 比原 `\\xf{}` 版本更适合作为主线，主要原因有三点：

1. 它先定义了一个不可逆的阶段边界，再解释为什么普通静态凝聚或独立文本附着不能直接解决这个问题。
2. 它把挑战压缩为两个更底层、互不重复的矛盾：
   - 图依赖证据必须在历史数据释放前被提交到记忆中；
   - 连续特征池化必须最终落成离散、真实且可审计的文本负载。
3. 它在挑战之后立即给出设计原则，因此读者能理解 `Q` 不是因为“实验效果好所以加入”，而是因为它恰好提供了一个共同的支持关系：一方面把图消息和节点特征提交到合成槽位，另一方面把真实文本路由到同一个槽位。

原版本建议作以下调整：

- 将三个 challenge 合并为两个。原来的“Feature and text correspondence”和“Discrete text optimization”本质上都属于连续特征与离散文本之间的构造接口问题。
- 将“Asymmetric data access”改为“Graph-dependent evidence under irreversible release”。前者太抽象，后者直接说明历史数据释放后到底丢失了什么：邻域消息、原始文本和重新凝聚的机会。
- 不要声称已有 TAG 方法都把特征和文本完全分开。应改为：现有方法分别解决了静态 TAG 构造、持续图凝聚、文本选择或图文对齐，但没有在同一个 current-increment-plus-memory、append-only 接口下同时规定图证据提交和可读文本材料化。
- 不把共享 `Q`、梯度匹配或额外 pair feedback 单独表述成已经被证明的精度来源。它们应被描述为实现统一构造接口的机制，精度必要性由后文消融实验检验。
- 在挑战段落中解释“为什么设计可能有效”，但不要提前塞入过多实验数字。具体提升、保留率和成本放到实验部分。

## 可直接替换的英文版本

下面保留原文的 `\\xf{...}` 结构、LaTeX 命令和引用键。建议用这一版替换原文件中从 TAG 背景到 Contributions 的对应 `\\xf{...}` 段落；末尾的 `\\todo{...}` 可以继续保留。

```latex
\\xf{
Text-attributed graphs (TAGs) combine relational structure with node-level
language and are common in citation networks, product catalogs, and linked
document collections. The graph provides relational context that a text
string alone cannot express, while the text provides semantics that topology
alone cannot recover. These collections are not static: new nodes, relations,
texts, and sometimes classes arrive over time. Retraining on the accumulated
graph can preserve historical information, but its storage and training cost
grow with the entire history.

Graph condensation replaces a large training graph with a much smaller set of
training items. In the continual setting, however, each increment must be
converted into a useful memory before the corresponding raw data leaves the
legal access window. A later constructor may receive only the current
increment and the memory written by earlier stages. For TAGs, every retained
item must additionally carry a readable text payload. The condensed memory is
therefore more than a small training set: it is the persistent interface
through which released graph--text information can affect future construction
and downstream learning.

This setting is different from static TAG condensation. Static methods can
use the complete source graph while constructing condensed items~\\cite{tagsam,stad}.
Continual graph condensation studies how to preserve graph training utility
under a limited replay budget~\\cite{cat,puma}, while federated TAG
condensation considers related but different access boundaries~\\cite{dance}.
The common missing interface is a current-increment-plus-memory constructor:
relational evidence must be committed before the current source is released,
and each retained row must remain a readable graph--text item. This paper
studies that interface directly.
}

\\xf{
The problem is especially important when historical raw nodes, edges, and
texts cannot be retained or revisited. Once an increment is released, later
stages cannot recompute its neighborhood messages, jointly recondense its
nodes with future data, or recover the source text that supported a stored
feature. A limited memory must therefore decide which information to commit
before release while new classes and graph patterns continue to arrive. The
memory acts as a state handoff rather than an ordinary sample, creating a
stability--plasticity problem at the level of the condensed state.

Existing work provides important components for this setting. Static graph
condensers such as GCond, GCond-x, and SFGC show how graph structure can guide
feature construction without requiring a stored synthetic adjacency~\\cite{jin2021graph,zheng2023structure}.
Continual methods study replay, coverage, new--old imbalance, or inherited
condensed states~\\cite{liu2023cat,choi2024dslr,liu2024puma,gao2024OpenGC,gao2024MCond}.
Static TAG methods such as TAGSAM and STAD introduce text selection,
attribute matching, or text synthesis, while DANCE and SR-GM consider
federated or structurally regularized multimodal condensation~\\cite{tagsam,stad,dance}.
GraphCLIP aligns graph and language representations for transferable TAG
prediction, but is not a continual memory condenser~\\cite{zhu2025graphclip}.
These methods motivate the individual building blocks, but they do not
explicitly define the same append-only interface in which graph evidence must
survive an irreversible release and every retained item must contain a
readable real-text payload.
}

\\xf{
The access boundary and the multimodal payload give rise to two coupled
challenges, illustrated in Figure~1.
}

\\xf{
\\textbf{Challenge 1: Graph-dependent evidence must survive irreversible
release.}
Neighborhood messages are computed from the current graph and are not
contained in an isolated node feature or text string. After the current raw
increment is released, a later constructor cannot recompute these messages,
jointly recondense the released nodes with future data, or retrieve their
original texts. Simply appending node-wise features or attaching labels to a
small memory can therefore discard part of the relational training signal.
The constructor must extract graph-aware evidence while the current graph is
available and commit that evidence to a compact state that can be consumed by
future stages. This is the continual condensation form of the
stability--plasticity problem: the memory must preserve old training utility
without preventing the current increment and newly appearing classes from
being represented.
}

\\xf{
\\textbf{Challenge 2: Continuous feature pooling must produce a discrete,
auditable text payload.}
Suppose a node-to-slot support relation (Q) is used to construct a synthetic
row. Feature pooling is continuous,
\\[
x^{\\mathrm{syn}}_j = a_j^{-1}\\sum_i Q_{ij}x_i,
\\qquad
e^{\\mathrm{soft}}_j = a_j^{-1}\\sum_i Q_{ij}e_i,
\\]
where (e_i) is a source-text embedding and (a_j=\\sum_i Q_{ij}). The
pooled text vector generally lies in the convex hull of source embeddings,
whereas readable texts form a discrete set of actual strings. It therefore
need not correspond to any source text. Selecting a string only after feature
pooling can break the correspondence between the stored feature and its
text; storing the optimized vector instead loses provenance, readability, and
auditability.

The constructor consequently needs one support relation that connects
continuous feature construction to discrete source-text materialization. It
also needs a training rule that supplies a useful surrogate gradient to the
discrete choice while ensuring that the text used in the construction forward
is the same text that is ultimately stored.
}

\\xf{
These challenges suggest three design requirements. First, graph messages
must be extracted before release and used as construction signals rather than
being reconstructed from an unavailable history. Second, feature pooling and
text routing should be controlled by the same auditable support relation so
that a synthetic row is not paired with an unrelated post-hoc text. Third,
the optimization-time soft route and the stored discrete payload must be
connected by one explicit hard-forward rule.

We propose \\method{} to satisfy these requirements. At stage (t), a
class-constrained assignment matrix (Q) maps current nodes to new memory
slots. Current node features, one-hop graph messages, and text representations
initialize the assignment before the source increment is released. A
capacity-balanced Sinkhorn operator turns the assignment logits into usable
support weights. The same (Q) pools graph-aware features, routes source-text
embeddings, and selects the real source string with the largest support
weight. Class-statistic distribution matching, propagation matching, and a
Student-compatible objective provide construction feedback to this support
relation, while a bounded residual allows small feature corrections.

For the discrete text decision, the forward construction uses the selected
real text, while a soft weighted representation supplies a surrogate gradient
to (Q). The selected source text is then materialized using the same rule
and stored with the synthetic feature, label, and audit metadata. The primary
memory is graph-aware but edge-free: the current graph influences
construction, but no learned synthetic adjacency is stored for the main
prompt-based node-classification task. A separate \\topomethod{} branch
materializes synthetic topology for structural downstream tasks.
}

\\xf{
We evaluate \\method{} under a history-constrained continual protocol with
current-delta-only access, a frozen synthetic prefix, and a cumulative
retention rate of (5\\%\\). Across ArXiv, SportsFit, and BookHis, the method
improves over a matched CaT adaptation by 5.58--19.66 Macro-F1 points. It is
competitive with the stronger GECC-hard adaptation, gaining 3.47 points on
BookHis, remaining comparable on ArXiv, and trailing by 0.49 points on
SportsFit. The condensed memories retain 98.23--103.33\\% of the matched
full-history Macro-F1 under a common finite Student budget and transfer to a
GraphSAGE Student without reconstructing the memory.

These results support the usefulness of learning the support relation and
performing distribution matching inside the construction loop, but they do
not establish a universal SOTA claim or imply that every component is an
independent source of accuracy. The paper therefore separates mechanism
evidence, baseline protocol, compression utility, and the topology extension
in the experiments.
}

\\xf{
Our contributions are as follows.
\\begin{itemize}
    \\item We formulate history-constrained continual TAG condensation with
    an explicit irreversible access boundary: each stage receives the current
    increment and a stored condensed prefix, while each retained row must
    contain graph-informed numeric content and a readable text payload.
    \\item We propose \\method{}, which learns a class-constrained graph--text
    support relation for both graph-aware feature pooling and real-text
    routing before the current source is released. Its hard-forward/soft-
    optimization interface keeps the stored payload discrete and auditable.
    \\item We provide controlled evidence on support learning, in-loop
    distribution matching, full-history retention, cross-architecture
    transfer, construction cost, and a separate topology extension under the
    same continual access protocol.
\\end{itemize}
}
\\todo{Highlight the best experimental results and summarize outcomes tied to the paper's core objectives.}
```

## 与当前 PDF 的对应关系

| PDF 中的叙事环节 | 修改稿中的对应位置 | 作用 |
|---|---|---|
| TAG 的结构与文本互补性 | 第一段 | 说明为什么 TAG 不能退化为普通图或普通文本 |
| 静态凝聚与持续凝聚的区别 | 第二、三段 | 把研究意义从“结果更好”改成“访问边界不同” |
| 现有工作的边界 | 第二个 `\\xf{}` | 只说它们分别提供哪些组件，不夸大其局限 |
| Challenge 1 | 不可逆释放下的图依赖证据 | 说明历史释放后具体丢失什么，以及为什么必须在当前阶段提交 |
| Challenge 2 | 连续特征到离散文本负载 | 说明软文本向量、真实字符串和可审计性之间的矛盾 |
| Design principles | 三项设计要求 + CoTAG-Q | 将挑战逐一映射到 Q、图消息初始化和 hard-forward/soft-optimization |
| 结果与范围 | 倒数第二个 `\\xf{}` | 给出主要结果，但明确不声称普遍 SOTA |

## 建议的 Figure 1 文字

Figure 1 最好只围绕这两个 challenge 画两块：

- 上半部分：`stage t raw graph -> graph-aware constructor -> compact memory packet -> stage t+1`。在边界处明确标出 `raw nodes/edges/texts released`，并在下一阶段画出禁止回流的叉号；记忆包内部只保留 `feature / label / real text`。
- 下半部分：多个源节点经过 `Q` 汇聚到一个 slot，左侧显示连续的 `soft feature/text pooling`，右侧显示 `argmax source text -> stored readable text`，并标出 `soft vector is not necessarily a real string`。

图中不要再单独画“asymmetric access”作为第三个框，也不要画不存在的 baseline 或额外算法。正文负责解释挑战，图只显示信息流和丢失点。

