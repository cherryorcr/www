# TGC-R6 冻结实验计划

执行用户批准的前两批：原样 R5 recursive 独立确认，以及父记忆、teacher、共享分配、归一化和材料化诊断。稳定化、大图和新主方法的晋升依结果决定，不预先宣称完成。

## A. 原样独立确认

Cora/Citeseer；order19601/19602/19603；三阶段，累计 rate=.05；report seeds140..145。使用 214 份本地 protocol/manifest/config 扫描实际三阶段类别分组，两数据集均未发现相同历史分组；扫描不覆盖缺失文件。确认阶段不读取或根据新报告分数调参。

比较 C0、R5 recursive、k-center、kmeans-medoid。R5 构造配方和 teacher seeds8400/8401 原样保持；s0 使用 C0，后续读取自己上阶段输出。两个 coreset 基线明确保留真实节点，只接触新 delta 和自己的旧 memory，不混称 synthetic。

共同 5% rows、GCN student、batch16、Adam .001、steps=max(50,min(2048,16K))；seed 组共享。原始 coreset 文本与 synthetic 文本并非等字符/等字节，分别记账。构造池标签可用于 C0；report/test 标签不进入构造。

主指标：最终 all-prompt report-dev pooled Macro-F1，辅以阶段均值、旧/新类、配对不确定性和成本。研究推进条件预定为平均 final>=+.5pp、阶段均值>=0、最差 final/old>=-2pp、至少4/6条件>=-1pp、平均 new>=0。另完整报告对两个强 coreset 的差距；这些条件不是录用/显著性标准，也不是停止机制诊断的开关。

## B. 已观察开发条件上的机制实验

固定 Cora/Citeseer、order19341/19342、report seeds100..105，主要评价最后一个增量 s2。使用封存 R5 的 C0 和 recursive s1 父记忆作诊断；不回读其历史 raw。

1. 旧 X/mass、旧文本、旧边分别来自 C0 或 R5 recursive，完整 2x2x2 父状态对照。混合父状态可能破坏语义一致性，属于干预诊断，不直接视作可部署算法。
2. 固定输出父状态，交叉使用 C0/R5 合成父状态形成 teacher 参考。参考和输出使用不同父状态只作归因，两份 memory 的额外容量不作为正式方法优势。
3. 训练 teacher、随机初始化同架构、冻结原始图文几何、固定初始分配比较。共享与分开内容/传播分配比较，并记录分开优化的步数与成本。
4. 原随机游走目标与对称归一化传播目标比较；另外运行对称目标的递归路径，避免把单阶段改善当作累积收益。
5. 固定 P，比较片段重编码/直接 pooling 文本表示、二值/相同稀疏边上的投影权重。pooling 只属 embedding 诊断，不称具备对应离散文本；加权边使用共同 GCN 的同一对称归一化算子，不能把它与二值图未经说明地拼主表。另测软投影与实际输入的 teacher 表示偏差。

代码和参数在各自报告训练前封存；全部候选先生成再评分。独立重载 checkpoint 与产物来源核验。上述实验不将 report F1 用作构造 selector。

## 执行后状态补记

本节为执行后追加说明，不属于事前门槛修改；远端封存的原协议保持原样。两批已完成，见[完整结果](TGC_R6_CONFIRMATION_AND_MECHANISM_RESULTS_ZH_2026-09-23.md)。确认432槽位/396次实际训练；机制首次准备因GPU构造不能逐字节复现而在评分前停止，按[数值修订](TGC_R6_MECHANISM_NUMERICAL_AMENDMENT_ZH_2026-09-23.md)新增两种历史封存产物控制后，b版624槽位/624次训练全部完成。原样确认未过推进条件，机制诊断继续执行并完成，没有因单项失败而取消联合控制。
