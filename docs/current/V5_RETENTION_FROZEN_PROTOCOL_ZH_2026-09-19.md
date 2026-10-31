# v5 Retention-Constrained Append：冻结实验协议

日期：2026-09-19。状态：候选实验，尚未晋升为主方法。本文件在读取 v5 实验结果之前建立。

## 1. 研究问题与四组对照

检验：限制历史合成特征和传播响应的漂移，能否使少量历史更新与新旧交互共同获益。基线 C0 是 response-off 的低成本构造，不把历史 response-on v4 换名为 C0。

|组别|历史内容|跨块传播|
|---|---|---|
|C0|全部冻结|原有双向 text-nearest cross-degree=1，整体度归一化|
|V5-core|永久保护区 + 固定缓冲区的有界 X 更新|同 C0|
|V5-topology|全部冻结|在块对角旧算子与 C0 提案之间求最大可行混合强度|
|V5-full|同 V5-core|同 V5-topology，约束包含实际更新后的 X|

共同固定：累计合法 pool 的 5% 向下取整预算；新增量等于累计预算之差；不删除历史行；dispersion 类预算；joint_g5_c320；合成 Mean-X；中心片段拼接再经冻结 MiniLM 编码；新块内边与行顺序完全相同。构造只接收旧合成 memory 与当前新块，不回读历史原始数据，不使用 selector/report/test 标签评分，不运行额外 student 选择训练。

## 2. 永久保护区与缓冲区

每个新块入库时，将质量 mass 最小的 floor(0.10 * 新块 rows) 行设为缓冲区，至少保留一行保护节点。mass 相同按固定行号排序。成员永久固定，不能逐阶段重新选低质量行而逐渐改写全部历史。保存缓冲行入库时的 X 作为 anchor。

对旧缓冲行，用 0.5 * cosine(X) + 0.5 * cosine(text) 在当前**合成**新块中找最近行；提案为 0.75 * 旧 X + 0.25 * 新 X。先投影到以入库 anchor 为中心、半径 0.05 * ||anchor|| 的逐行球内，再沿旧 X 到提案的线段统一缩放，使旧算子的一跳和两跳 X 响应相对 Frobenius 漂移分别不超过 0.01。core、全部旧文本及 embedding、mass、depth、birth stage 均冻结。

该更新是受约束的新数据特征接近目标，不是假设真实 student loss 得到优化。它与 P4 同时混合旧 text embedding 的做法不同。anchor 限制防止多次小更新累计成为无界改写。

## 3. 漂移约束跨边

记上一阶段实际训练支持为 S_old，新块单独归一化支持为 S_new。

S_base = blockdiag(S_old, S_new)。S_prop 为所有历史原始合成边、新块内边和原有双向最近文本跨边共同度归一化后的 C0 提案。保存原始边列表，绝不把已经含自环的归一化支持再次当原始边归一化。

S(lambda) = S_base + lambda * (S_prop - S_base)，0 <= lambda <= 1。

取同时满足以下充分条件的最大 lambda：

- 历史行算子相对 Frobenius 漂移（包含跨列）不超过 0.05。
- 历史行一跳、两跳特征响应，相对上阶段 S_old X_old、S_old^2 X_old 的漂移分别不超过 0.02。

用一跳线性项、两跳二次项的范数三角上界，二分求最大可行 lambda；不利用分类结果做线搜索。输出重新独立计算实际漂移并断言约束。S_base 与 S_prop 均为非负对称收缩算子时，其凸组合仍为非负对称收缩算子；但它通常不等于某张无权图的标准度归一化矩阵。四组使用相同 GCN 实现，传播支持是明确的实验因素。

这些约束不保证重训 student 参数、非线性表示或旧类 F1 不变。漂移是否预测实际保持必须实测。每阶段界也不等于整条任务链的总漂移界。

## 4. 小图矩阵与隔离

- Cora、Citeseer，各 3 阶段。
- canonical：Cora [[0,1],[2,3],[4,5,6]]；Citeseer [[0,1],[2,3],[4,5]]；seeds 100..105。
- matched_unseen：仅用 pool 类别大小，从保持阶段类别数的划分中选累计规模最接近 canonical 的新划分。排除已记录历史顺序，且每个阶段的类集合都与 canonical 不同；按各阶段样本数相对误差平方和最小、字典序打破平局。保留实际规模差，不称精确匹配。seeds 9100..9105。
- 新顺序和新 seed group 共同用于确认；不能由此单独归因顺序或 seed 效应。已使用多次的 report-dev 不是新测试集。
- 阶段和顺序在构造/训练前写入只读配置，代码与配置哈希入档。
- 4 arms * 2 datasets * 2 orders * 3 stages * 6 seeds = 288 次训练，不复用阶段零训练；相同输入的阶段零重复用于数值一致性审计。

pool 排除 original_id % 10 为 1、2、3 的节点；selector=1，report=3，test=2。student 从头训练两层 GCN，hidden/out=384，冻结 MiniLM=384，trainable text projection/temperature，双向 InfoNCE，Adam lr=0.001，batch=16，steps=max(50,min(2048,16*rows))。不使用监督节点 CE。真实全图 transductive 评估和类 prompt 与 C0 相同。

## 5. 指标与成本

主指标：report-dev 上全部类别 prompt 竞争、已见节点合并支持的 class-macro-F1。另报 seen-prompt、pooled old/new、逐类、阶段均值与历史 task-slice 指标；绝不混用两种 old/new 口径。

每次训练保存 checkpoint、report logits、目标、原始 ID、构造/支持/文件哈希；通过保存预测独立重算指标。test 不读取作评估。评估为节点分类，condensation 标签访问与 student 无监督目标分别披露。

每个 arm 单独报告：共享新块构造的等额计费、附加 memory/拓扑处理、源文本编码、训练、评估/存盘、显存峰值、rows、边、实际序列化字节和逻辑 payload。训练和构造计时用 CUDA synchronize 分开，审计开销单列，不混入算子构造。selection runs/steps=0。不把共享缓存成本写成零。

## 6. 预先固定的晋升门槛

比较候选与同一 cell 的 C0，单位为 F1 百分点。一个 cell = dataset * order，按相同 seed 配对。

1. 四个 cell 最终 F1 均值差均 >= -1.0。
2. 至少两个 cell >= +1.5，其中至少一个是 matched_unseen。
3. 四个 cell 阶段平均 F1 差均 >= -1.0。
4. 四个 cell 最终 pooled old F1 差均 >= -2.0。
5. 最终 pooled new F1 差跨 cell 平均 > 0，且至少两个 cell > 0。
6. 各 cell 总构造成本（新块构造 + 附加处理）<= C0 的 2 倍；同时报告训练成本，不能据此宣称总计算更快。
7. 配对差的单侧 t 下界在每个 cell 均 > -2.0，alpha=0.05/12（3 候选 * 4 cell 的保守多重校正）。6 seeds 的 t 区间依赖近似假设；未通过属于证据不足，不等于证明负增益。
8. 所有来源、预算、保护区、漂移、预测、阶段零重复、代码配置审计通过。

门槛是研究推进标准，不是自动得出普适/SOTA结论。若多候选通过，按四 cell 平均最终配对差最大者进入大图，平局按较低构造成本，再按 core/topology/full 字典序。

## 7. 条件性大图与结项

只有小图门槛通过，才在预选 arXiv 的 base+4（base_fraction=0.4）运行 C0 与胜出候选，5% budget，seeds 9100..9105。要求最终及阶段平均差 >= -1.0、最终 pooled old 差 >= -2.0，且预算/漂移/来源审计通过。长链未通过则不能晋升 v5。

小图未通过时，大图标为“门槛未通过，依预注册规则不启动”，不能写成大图实验完成，也不通过事后调参重测同一顺序制造独立确认。报告完成/拒绝/未触发状态、四组差值及 interaction = full - core - topology + C0，给出保留 C0 或晋升的证据。论文写作仍待本轮结论之后。
