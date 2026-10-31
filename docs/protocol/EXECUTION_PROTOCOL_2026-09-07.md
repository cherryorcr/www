# 原生持续 TAG v1：首轮执行登记

用户已授权实现、运行可行性与提升空间验证，并据主要困难实现持续算法。日期2026-09-07。本目录独立于原仓库文件和旧冻结结果。

首轮Cora七类，类别顺序按源all_labels，阶段为2/2/3，所有评测始终七类候选。准备端使用类别分流，模型训练与凝聚不使用类别标签；验证标签用于选择学生和凝聚产物。数据按固定源行ID哈希70/15/15，源文件SHA绑定身份。每阶段训练图只含该阶段train，评测图为该阶段完整图；累计均为阶段图不相交并集，禁止读取未来训练数据。所有Cora结果为development evidence，旧暴露不能改称final test。

首版继承2层GCN(384维)、冻结bert-base-uncased CLS、原生文本投影及温度0.07对比损失、add_adj对称MLP固定合成拓扑和ASM。为控制实现风险，采用完整图消息传播+根节点小批对比，函数式可微SGD；外层Adam、固定内层学习率为稳定化适配。文本初始化以合法图文联合聚类选择真实原文代表，缓存必须与字符串对应，不使用旧无签名BERT缓存，不声称复现GPT2-CMI选择或论文精确数字。

主比例1%累计训练roots，floor舍入，另报5%敏感性以排查极小样本不足；同组Append根预算相同。所有小集采用同一临时source-link decoder与相同degree=1.5，明确实际字节。随机小集保留真实X；joint coreset保留真实X；frozen centroid-text初始化X为聚类均值、文本为真实代表。ASM只优化X，所有非X字段核验固定。

主指标累计留出Macro-F1，Accuracy和各阶段F1辅助。模型独立随机初始化，首先凝聚seed44，student seeds100/101；结果有信号后扩到另一独立凝聚seed45，不把同一产物重训当独立凝聚。先在checkpoint上开发，冻结候选后才报告test角色确认；确认被查看后如继续调参，明确作为开发反馈。

初始训练比较采用512步、每128步验证，同时保存exact终点和预算内选点；full-real覆盖与曲线未稳定时延长到2048步，不能将短预算结果称充分收敛。teacher从合法source训练，800步起步；学习率默认graph/text各1e-4，对比使用同配置。每组batch=min(20,K_t)，匹配名义batch上限，记录尾批、覆盖与训练出现次数；实际尾批负例数与样本暴露仍不同。tiny K<2不能运行InfoNCE/ASM并应报告。

静态ASM初筛outer100、inner10步、inner lr=5e-3、outer Adam lr=1e-2、reference最多128、每50步独立验证选产物（含step0），所有改变另记配置。该配置是开发起点，不从测试挑最优。大幅不稳则先对照fixed/bounded inner lr及X约束，不直接增加持续轮数。

两阶段组包括Full-history-real、Offline-ASM、旧S+delta不重凝聚、Append-ASM、Union-ASM、random/真实joint coreset；完整raw参照单独标权限。对比旧raw替代、再次压缩、参考失衡和配对关系，差值不是严格可加的因果分解。依据观测主损失选择一项保持机制，和质量校正/完整Union-ASM比较；收益与新颖性不预设。

执行补正：发现128步检查遗漏早期峰值后，主要新运行改为每32步检查，ASM每20个outer检查；完整配置以各运行快照为准。已完成两个凝聚seed的静态1%/5%、三阶段5%及1%两阶段机制对照。2026-09-07冻结选点后已单独读取确认分区，不再称其未查看。见ATTRIBUTION_ADDENDUM_2026-09-07.md及根目录NATIVE_TAG_V1_FEASIBILITY_ATTRIBUTION_AND_PROGRESS_ZH_2026-09-07.md。
