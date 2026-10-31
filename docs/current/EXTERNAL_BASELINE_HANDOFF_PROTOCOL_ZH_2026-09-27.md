可以并行推进，而且这样更合理。建议拆成以下几条线：

| 任务线             | 负责人   | 目标                                             |
| ------------------ | -------- | ------------------------------------------------ |
| GC-SR 主方法       | 我       | X 梯度优化、旧节点约束、软路由、新旧拓扑联合优化 |
| 外部 baseline      | 你的同学 | 查论文、核对官方代码、适配统一 setting、运行比较 |
| 混合类别协议       | 共同     | 新类别、旧类别新增节点、新旧混合到达             |
| 高效实现           | 我       | top-r 检索、ANN、软支持统计，避免全量两两遍历    |
| 跨数据集 zero-shot | 后续     | 先确定共享标签语义和可行数据集                   |

已经整理好交接文档：

[统一外部基线 Setting 与同学交接协议](D:/Aprojects/continue_tag_condensation/docs/current/EXTERNAL_BASELINE_HANDOFF_PROTOCOL_ZH_2026-09-27.md)

**冻结 Setting**

- 数据集：Cora、Citeseer
- 顺序：已有 `order19611/19612`，之后增加 3 至 5 个预注册新顺序
- 阶段：3-stage
- Memory：约 5% cumulative fixed-rate
- 构造输入：当前 `delta_t` + 上一阶段 synthetic parent
- 禁止：访问历史完整 raw graph
- Student：原两层 GCN + 冻结 MiniLM + 双向 InfoNCE
- 每阶段从头训练，不继承 Student 参数
- batch=16，lr=0.001，steps=`max(50,min(2048,16K))`
- Student seeds：150-155
- 指标：Macro-F1、Accuracy、old-class F1、new-class F1、worst-stage、训练与构造成本

**同学优先完成的 baseline**

1. **PUMA-inspired adaptation**  
   必须真正实现 pseudo-label、one-time propagation、edge-free memory 等独有机制，不能把 CaT 改名为 PUMA。

2. **TAGSAM / STAD 可比性审查**  
   明确哪些可以适配到当前 InfoNCE Student，哪些只能作为 static/reference 结果。

3. **GCond、SFGC、GCondenser 等静态方法**  
   先完成论文、代码和访问权限核验，再决定是否实现。

4. **已有方法的新顺序确认**  
   CaT、k-center 优先在新顺序上复现；GECC 和 medoid 暂时不必重复旧矩阵。

每个 baseline 必须提交：

- 论文和代码 commit；
- 是否访问历史 raw；
- 是否使用标签；
- memory 类型和拓扑；
- Student/loss；
- 适配差异；
- construction seed；
- memory、来源和 checkpoint audit；
- F1、Accuracy、old/new F1 和成本。

需要严格区分：

- `native`：原论文原设置；
- `mechanism-adapted`：保留原机制但适配当前 TAG/Student；
- `controlled-content`：共享 C0 初始 memory 的控制实验；
- `end-to-end native append`：方法自行构造 stage0 和拓扑。

当前已经完成的 CaT、GECC、k-center、medoid 外部矩阵不能与旧的共享 C0 拓扑表混合解释。

**主线 GC-SR 同时推进**

第一批只改变 X 和梯度目标，固定文本、预算和 Student：

1. SR baseline；
2. X-output matching；
3. X-gradient alignment；
4. X-output + gradient；
5. 加 old-X anchor；
6. 最后再加入 sparse new-old topology。

这样可以判断性能问题到底来自：

- X 表示不适合；
- teacher 输出不匹配；
- 梯度方向不一致；
- 旧节点漂移；
- 新旧拓扑没有联系。

旧的 response selector 不再作为主要路线。新的方向是直接让梯度更新 \(X\)、旧节点残差和新旧边权，而不是训练多个候选后再选择。

**旧类别新增数据也要纳入**

当前协议要扩展成三种：

1. 纯新类别到达；
2. 旧类别出现新节点；
3. 新旧类别混合到达。

旧类别新数据可以通过软路由更新已有合成节点，同时允许少量新增节点；旧节点更新必须满足：

\[
\|X_{\text{old}}^{new}-X_{\text{old}}^{old}\|^2
\leq \epsilon
\]

新类别则使用自由优化的 \(X_{\text{new}}\)，并学习与旧节点的稀疏连接。

**跨数据集 zero-shot**

Cora 和 Citeseer 的类别语义不同，不能直接把它们作为严格的跨数据集 zero-shot 任务。应先区分：

- 构造器跨数据集迁移：固定算法和超参数，在目标数据集不使用标签调参；
- memory 跨数据集迁移：源数据集合成 memory，直接训练 Student 并在目标图上 prompt 推理。

第二种必须要求源、目标数据集有共享标签语义。否则先做同一标签体系下的跨图划分，再做真正的 cross-dataset zero-shot。

当前并行推进顺序应是：

1. 同学完成 PUMA 和静态 TAG baseline 核验；
2. 我完成 GC-SR 的 X 梯度诊断；
3. 共同补新旧类别混合协议；
4. 对 SR、CaT、k-center 做新顺序确认；
5. 通过小图验收后，再做跨数据集和大图实验。


# 外部基线并行交接与统一 Setting

日期：2026-09-27

本文件用于把 Semantic-Recursive 后续外部基线工作交给另一位同学，同时保证结果可以与主线 GC-SR 实验合并。所有未通过审计的结果只能作为开发结果，不能进入论文主表。

## 1. 并行分工

| 负责方   | 工作                                                         | 不做的事情                                                    |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------- |
| 主线实验 | GC-SR 的 X 梯度、旧节点 anchor、软路由、new-old 拓扑联合优化 | 暂不改变 Student、文本 encoder 和主预算                       |
| 基线实验 | 论文/代码核验、PUMA 和静态 TAG baseline 适配、统一矩阵运行   | 不根据 report-dev 或最终 F1 调参数，不重写 Semantic-Recursive |
| 共同协议 | 统一数据、顺序、memory rate、Student、指标、成本和 audit     | 不把不同访问权限的方法混入同一冠军表                          |

CaT-MMD-TAG、GECC-hard-TAG、k-center native、medoid native 已在 2026-09-27 完成一轮 360-fit 小图矩阵，不重复跑同一配置。后续可在新独立顺序上复现其中的 CaT 和 k-center。

## 2. 冻结主比较 Setting

### 2.1 数据和阶段

- 第一阶段：Cora、Citeseer 小图。
- 第一批复现：已有 `order19611`、`order19612`，用于与现有结果直接对齐。
- 第二批确认：预注册 3-5 个新类别顺序；顺序生成后封存，不能按结果挑选。
- 每个顺序使用三个阶段，保持当前三阶段协议；只有通过小图确认后再扩展 `base+4` 五阶段大图。

### 2.2 Memory 和访问权限

- 使用约 5% cumulative fixed-rate memory，精确采用主 runner 的预算轨迹。
- 构造器只能访问当前 `delta_t` 和上一阶段 synthetic parent。
- 不允许访问历史完整 raw graph，不允许读取历史 raw text 或历史 raw edge。
- 可以使用当前构造所需的类别标签进行类比例分配，但不使用 report-dev/test 标签选择方法。
- 每个方法从自己的 stage0 开始构造；不能借用 C0 的 stage0 或 topology，除非单独标为 controlled-content ablation。

### 2.3 Student 和评价

- 固定当前原始 Student：两层 GCN、冻结 MiniLM、文本 projection、双向 graph-text InfoNCE。
- Student 每阶段从头训练，不继承上一阶段参数。
- batch=16，learning rate=0.001，steps=`max(50, min(2048, 16*memory_rows))`。
- 正式 Student seeds：150-155；构造 seed 另行记录，不能用 Student seed 代替构造随机性。
- 在真实完整图上进行 prompt-based zero-shot style inference。
- 主指标：Final Macro-F1、Final Accuracy、stage-average Macro-F1、old-class F1、new-class F1、worst-stage F1。
- 成本指标：construction seconds、Student seconds、memory rows、tensor bytes、文本 UTF-8 bytes、文本 token 数、edge entries。

## 3. Baseline 分层

### A. 严格 continual 主表

只有满足当前 delta、synthetic parent、固定预算、相同 Student 和相同 report-dev 的方法才能进入这一层。

优先顺序：

1. PUMA-inspired TAG adaptation：必须实际实现 pseudo-label、one-time propagation、edge-free memory 或论文中对应的独有机制，不能把 CaT 改名为 PUMA。
2. CaT-MMD-TAG：已有适配，需在新独立顺序上运行时沿用冻结配置。
3. GECC-style adaptation：已有 hard 分支；原生 cumulative-raw 和 fuzzy 调优分支要单列，不能混入受限 no-old-raw 主表。
4. 独立 k-center/FPS 和 medoid：已有端到端结果，作为强 coreset 参照。

### B. 静态 TAG 适配层

优先核验 TAGSAM、STAD，以及有公开代码的 GCond/SFGC/GCondenser 类方法。每个方法先做机制卡和 smoke，不一定都能进入严格主表。

静态方法如果访问完整累计 raw、使用监督 GCN、保存模型参数或采用不同文本目标，必须标记为 `static/offline/reference-only`。只有完成共同 Student 和当前 delta 适配后，才能标记为 `mechanism-adapted`。

### C. 不能直接横排的方法

- STAD native supervised GCN：监督上界或 native reference，不与 InfoNCE Student 直接排序。
- 官方 GraphCLIP checkpoint：完整图和官方 GPS Student 的外部模型参考，不是 continual condensation baseline。
- Full-history re-condensation、complete raw training：offline reference，不是合法 continual constructor。
- Stateful Student fine-tuning：模型状态保留控制，不是 memory-only condensation。

## 4. 同学需要为每个 baseline 提交的内容

每个方法建立一张 `baseline_card.json`，至少包含：

- 论文标题、链接、年份/venue 状态；
- 官方仓库和固定 commit；
- 原始输入类型、是否使用标签、是否访问累计 raw；
- memory 单元是 synthetic node、real node、center 还是 model state；
- 是否有 teacher/student、优化目标和 topology 机制；
- 原方法与统一 TAG setting 的差异；
- 适配后的当前 delta 输入、文本处理、拓扑和预算；
- construction seed、Student seeds、结果目录和 source hashes；
- 是否通过 source locality、budget、prefix、checkpoint reload audit；
- 不能进入主表的原因。

禁止只提交一个 F1 数字。没有访问权限、Student 和预算说明的分数不具备比较资格。

## 5. 运行顺序

1. 先读取论文和官方代码，完成机制卡，不立即修改代码。
2. 运行官方 smoke 或最小单阶段实验，确认实现方向。
3. 写统一 adapter，只接收当前 delta 和 parent memory。
4. 通过 source locality、预算和 topology 单元测试。
5. 在一个数据集、一个顺序、一个 Student seed 上跑 3-stage smoke。
6. 封存协议、代码 hash、输入 hash 和构造 memory。
7. 运行已有两个顺序的正式矩阵。
8. 只有配置不变时，才运行 3-5 个新顺序确认。

## 6. 公平性和结果命名

结果名称必须明确包含：

- `official/native`：原方法配置，不能直接与当前主表排序；
- `mechanism-adapted`：保留原机制但改成共同 TAG/Student 接口；
- `controlled-content`：共享 C0 stage0/topology 的归因实验；
- `end-to-end native append`：方法自己构造 stage0 和后续 topology 的端到端比较。

如果方法改变了 Student、loss、历史 raw 权限或预算，必须换表。不能因为同样使用了 5% rows 就称为同协议。

## 7. 与 GC-SR 主线的并行关系

主线 GC-SR 实验先固定文本和预算，只研究：

1. X-output matching；
2. X-gradient alignment；
3. old-X trust region；
4. soft routing；
5. sparse new-old topology；
6. joint X-topology optimization。

基线同学不需要等待 GC-SR 完成。两条线只在以下接口汇合：相同 Student、相同 memory budget、相同 report-dev、相同 seeds 和统一结果 schema。

## 8. 验收标准

一个 baseline 只有同时满足以下条件，才能进入论文主比较：

- 构造输入没有历史 raw 泄漏；
- memory rows 和预算正确；
- 所有旧 prefix/来源/拓扑审计通过；
- checkpoint reload 指标误差为 0；
- 至少完成两个小图、两个既有顺序、六个 Student seeds；
- 报告 F1、Accuracy、old/new、最差阶段和成本；
- 明确说明是 native、adapted 还是 reference-only。

## 9. 当前最重要的并行产出

同学优先完成 PUMA 机制核验和适配，不重复 CaT/GECC 旧矩阵；同时完成 TAGSAM/STAD/GCond 类方法的可比性审查。主线完成 GC-SR 的 P0-P2 小图诊断后，再把通过审计的 baseline 与 GC-SR 放入同一张新旧类别混合协议表。
