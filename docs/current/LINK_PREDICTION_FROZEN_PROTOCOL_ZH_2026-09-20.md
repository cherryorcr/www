# C0 与联合核追加：链路预测冻结协议

冻结时间：2026-09-20；在读取本轮性能前确定。此轮是新下游任务的开发验证，不是分类结论的替代，也不是未接触过数据集的外部确认。

## 问题与比较范围

- 主比较：coverage 对 C0；机制控制：joint_kernel、degree_balanced。
- 参考：full_raw_equal，读取全部已见合法原始训练节点，使用相同优化器更新次数。它不是充分训练上界。
- Cora/Citeseer，原 canonical 三阶段，5% 合成节点；edge split seeds 29001/29002/29003，student seeds 100..105，共 540 次独立训练。
- C0 的图文行、源簇和文本拓扑不依赖真实边，可以复用经过哈希验证的既有构造。三个联合核版本只复用相同源簇与图文行，必须在本轮训练边上重新计算 graph signature 和拓扑，旧行/旧旧边保持冻结。
- 不把 source groups/原始节点 ID 作为持久 memory 或 student 输入。联合核签名只用于凝聚连边，不直接输入下游模型。

## 隐藏边与防泄漏

- 节点域仅为原协议合法 construction pool；排除原 selector-dev、report-dev、test 节点。这里的测试集是 pool 内隐藏的边，不是原节点分类 test。
- 无向边去重、去自环。按两端节点到达阶段分层，80/10/10 划分 train/validation/test；双向边一起隐藏；验证边仅留存，不用于选择模型或超参数。
- 每个到达阶段组合内，均匀采样等量真实非边作验证/测试负例，排除所有真实边；负例不重叠，也不进入原图参考的训练负例池。
- 每阶段只使用已到达节点及训练边。新 block 签名进一步只读取当前 delta 的诱导训练图；不回读历史原始信息。
- 原始图仅供独立下游评估器推理和 full_raw_equal 参考；不能将其访问算作持续凝聚器的权限。
- 保留拆分、重建 memory、源文件哈希、模型检查点和逐边预测，以便独立复算。

## 共同下游模型

- PyG GAE：两层 GCNConv，输入 concat(L2-normalized X, L2-normalized MiniLM text)/sqrt(2)，hidden=384，output=128，ReLU 仅第一层，无 dropout。
- 内积 decoder；正负边 BCE；Adam lr=0.001；固定 256 次更新；每步最多 256 条正边和等量负边，无放回 minibatch。各 memory 在自己的合成拓扑上训练；full_raw_equal 在已见真实训练拓扑上训练。
- 从头训练，不继承上阶段 student；图卷积每步传播整张训练图，边 minibatch 只用于 loss。相同更新次数不等于相同 FLOPs。
- 训练后在真实已见节点、真实训练边上编码，预测隐藏边。此为 transductive unseen-edge prediction，不是未来节点/未来时间边预测。
- 固定最后一步，不依据 test 选 checkpoint。512/2048 等训练时长留作未来另立实验，不事后挑选本轮最优设置。

## 指标与解释

- 主指标 test ROC-AUC 与 AP（等量负例）；AP 依赖负采样分布，不能解释为自然稀疏图全候选 precision。
- 列出最终阶段、阶段平均；old=两端早于当前阶段，new=至少一端当前阶段，cross=一端旧一端新，within_new=两端当前阶段。
- 额外无训练对照：X cosine、text cosine、X+text cosine。
- 用相同 split/seed 做成对差；报告三个 split 的均值、方向一致性和 split 内 student 波动；18 次训练不是 18 个独立数据集。
- 成本分开列重建增量成本、缓存 C0 构造成本、student 训练和逻辑字节；不将缓存读取时间宣称为完整构造成本。
- 不把 AUC/AP 与分类 Macro-F1 混表，不因本轮阳性就宣称所有下游任务优越，不因本轮阴性就断言联合核无价值。
