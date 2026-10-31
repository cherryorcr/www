# v7-A Gradient-Compatible Append 冻结协议

日期：2026-09-19。状态：运行前冻结。

## 1. 目标

v6-A 表明完整 response curve 成本高且排序不稳定。本轮检验一个更窄的新假设：在相同 student 初始化和 batch 前缀下，完整 C0 append memory 与候选 append memory 的 0/1/2-step InfoNCE 梯度距离，能否以低成本预测最终 student utility。两者共享完全相同的旧 synthetic memory。

运行前实现审查修订：本实现保存完整参数梯度向量，不采用随机投影，因此不将其称为压缩 gradient sketch；它是相对 C0 的动态正则，并非真实图/合成图梯度匹配，也不是遗忘上界。

本轮不是 v6-B，不使用 v6 report 结果选超参数，不把 report oracle 候选拼成持续轨迹。

## 2. 候选

共同底座为 C0：累计合法 pool 5% growing rows、dispersion、Mean-X、`joint_g5_c320` concat/re-encode、冻结旧行/旧旧边、cross-degree=1。

每个 stage 只保留最多三个候选：

1. `c0`：当前 C0；
2. `recluster`：按新类 pool count 最大者作为 donor、joint dispersion 最大的其它新类作为 recipient；沿 v6 的 merge/split slot 变化，但只完整重聚类这两个受影响类别；
3. `multi_prototype`：每类先做 joint subcluster，再按 mass 与 dispersion 分配同样的类 rows，并由源簇均值生成 synthetic rows。

候选均保持相同总 rows、旧 history 和 student。`multi_prototype` 保持类级预算，子簇整数分配受源簇容量约束，超额配额重新分配；完成后在每类内按 joint 中心做 Hungarian 对齐，减少无意义的行排列变化。`recluster` 则明确从 donor 转移一行给 recipient，不声称保持类级预算。两者使用原有文本/建边规则，但重新分簇会连带改变新 X、文本与新边，是组合构造候选，不能把其收益归因于某一个模态。源 ID 只进入离线审计文件，部署 memory 仍是合成节点。singleton 簇的均值可能恰好等于真实节点，须单独统计，不能声称每一行数值均不同于源节点。

## 3. 数据、顺序和 seed

Cora/Citeseer；canonical 加一个只按合法 pool 类规模匹配、排除历史 v3-v6 和本轮 registration 的新顺序。stage 1/2 作为反事实 cell，不拼接成递归链。

selection groups 为 `(7300,7301)`、`(7400,7401)`；independent report seeds 为 `9300..9305`。梯度 selector 只读取合成 memory 和当前合法 pool 的几何误差，不访问 validation 标签或真实图传播；构造仍使用 pool 类别分块，故完整方法不是完全 label-free。共同 student 的正式评估仍使用 transductive real graph。test 标签不用于优化、选择或评价，离线 source audit 中 test target 替换为 -1。新顺序和新 seeds 不能消除反复使用同一 report-dev 节点集的限制。

student 保持 v6/C0：两层 GCN，384 hidden/out，冻结 MiniLM，Adam `1e-3`，batch 16，从头训练，双向 InfoNCE，正式训练步数为 `max(50,min(2048,16*rows))`。

## 4. Gradient-compatible score

对每个候选和 selection seed，用同一初始化、同一 generator 和同一 batch 前缀计算 step 0、1、2 的梯度。候选相对 C0 的动态距离为：

\[
D_{dyn} = \frac{1}{3}\sum_{k=0}^{2}
\left(1-\cos(g_k(M),g_k(M_{C0}))\right).
\]

候选的 coverage gain 使用源簇 joint geometry error 的相对下降：

\[
U_{cov}=\frac{e(C0)-e(M)}{|e(C0)|+10^{-8}}.
\]

冻结选择分数：

\[
Q(M)=U_{cov}-0.5D_{dyn}.
\]

两组 selection seeds 各自选最高 Q；只有两组选择同一非 C0 且四个 selection seed 的平均 Q 增益大于 0 时接受，否则回退 C0。report metrics 完全不参与选择。

## 5. 评价与门禁

每个 candidate 使用六个 report seeds 做完整 student training，保存 1/8、1/4、1/2、全长 selector/report logits，独立 report 只用于结果分析，不选择最佳 checkpoint。四个 selection seeds 只做两次 Adam 更新和三次梯度计算，不做完整 response training。所有 cell 的选择先锁定，再运行完整 report fits。主指标沿用共同实现的 `report/all/cumulative_f1`，预测竞争类别范围为全部类别，目标节点限已见类；old/new 同口径。seen-class competition 只作为已有指标输出，不混入主表。

必须报告：gradient score 到 independent full-report 的 Spearman、top-1 agreement、regret、C0 fallback、old/new/cumulative F1、memory payload、gradient selector cost 和最终 student cost。

还须报告相同候选集的 geometry-only、均匀候选期望和 report oracle（仅诊断）对照。除当前候选可行性外，不增加候选、修改 lambda=0.5 或根据 report 调整规则。退化常数排序按 Spearman=0 计并标记。仅有最多三个候选，相关性估计很粗，不是排序能力的普适证明。

v7-A 只有同时满足以下条件才进入 v7-B：

1. 8 个 cell 的 gradient rank 与独立 report rank 平均 Spearman >= 0.30；
2. gradient selector 的 independent report 平均 gain >= 0.50 pp；
3. 两个新顺序数据集的阶段平均 gain 均 >= 0；
4. 任一 cell cumulative gain 不低于 -2 pp，old gain 不低于 -3 pp；
5. 一次选择成本（gradient sketches + 一个正式 student + 当前候选构造）不超过 C0 对照的 2 倍；
6. source partition、old memory、checkpoint、指标重算和测试标签隔离全部通过。

失败即停止在 v7-A，不增加 response horizon、extra budget、候选数量或大图实验。

成本门禁逐 cell 计算，而非以平均成本掩盖超支：当前 C0 block 构造 + 全部候选构造和 append + 全部 gradient/score + 一次选中 student，除以当前 C0 block/append + 一次 C0 student。单次 student 用六 seeds 实测训练时间均值估计；共同预处理、历史构造、实验穷举、审计和产物 I/O 另列，不将该比值称为全生命周期加速。梯度和模型 reload 的数值容差在运行前固定为 rtol=1e-4、atol=1e-5；指标从保存 logits 精确重算，最终模型 reload 同时要求分类指标一致。两步前缀须对齐共同 student，完整多检查点 runner 须通过原有前缀/RNG 审计。
