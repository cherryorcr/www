# 度均衡块保持追加：独立顺序与跨任务确认协议

状态：结果产生前冻结；候选主线为 T+degree，C0 始终作为共同对照。

## 1. 范围

- 数据集：arXiv、BookChild，复用已核验的原始文件、pool 划分、冻结 MiniLM 和 pool-only 文本缓存。
- 类别顺序种子：49011、49012。用 NumPy default_rng(seed) 对全部 class IDs 做 permutation，按旧 base+4 的每阶段类别数切块，块内排序。不再把最大类固定放入 base；记录新旧每阶段样本量的差异。
- 预算：floor(0.05 * 累计 pool 节点数)。dispersion 类别配额、mean-X、joint_g5_c320 concat/re-encode、cross_degree=1、frozen-old、response off、refresh off。新顺序必须重新凝聚，不能复用旧链。
- 构造 seed 沿用 METHOD_SEED=9001；本轮是新顺序和新 student seed 的确认，不声称独立构造 seed 扫描。
- 方法：C0、T+degree、XT+degree；T=(0,1,0)，XT=(4/7,3/7,0)，不使用 graph signature/coverage/可训练 selector。
- 每阶段三臂的 X、text、mass、stage、row order、rows 和无向 pairs 相同；只改变边。内部为 degree-2 路径闭合后的环，跨块按旧节点容量分配。保持各自旧 rows 和旧旧边，不保证归一化传播算子不漂移。
- 两任务读取完全相同 SHA256 的 memory；task-specific student 分别从头训练，不共享模型或优化器。

## 2. 下游协议

student seeds=210..215，与上一轮100..103不重叠，不选择最佳 seed 或 checkpoint。

节点分类：现有两层 CLIPStudent，双向 InfoNCE，lr=0.001，batch=16，2048 steps。只在 synthetic memory 上训练；冻结类别 prompt。报告 report-dev 节点的累计 Macro-F1、global-support old/new F1；同时保存 all-class 与 seen-class prompt 竞争。主判断采用历史 all-class 口径，seen-class 单列，不混表。真实全图仅用于 transductive 推理，包含未到达节点的 X/边；这延续旧分类条件，不是 strict-inductive 结果。分类 test 标签不评价、不选模。

链路预测：既有 GAE(GCN 384/128)+inner product，lr=0.001，edge batch<=256，256 steps。edge split seeds=49101、49102，按端点到达阶段分层80/10/10，正负例1:1，负例排除全部观测正边，train/val/test互斥。评估只用已见 pool 节点及真实 train edges 做传播；不使用 heldout edge 训练或构图。memory 构造仅依赖当前 pool 的 X/text/y，不接受真实边，因此两 split 可以合法共享 memory。LP学生不继承NC模型。报告 AUC/AP、old/new/cross/within-new；val不用于调参。本轮 test 分数是一次性确认读数，若后续据此调参，这些条件即成为开发条件。

新增零训练 X/text/XT 余弦对照，用于判断大图随机负边上的高分是否主要来自原始表征。本轮不补训练 full_raw，上一轮完整图参考不能与新顺序/seed直接配对。

规模：每图2 orders*5 stages*3 arms*6 seeds=180 NC fits；再乘2 edge splits为360 LP fits。两图总计1080 fits。零训练余弦对照不计入fit数。

## 3. 事前决策规则

只对预先指定的 T+degree 判断确认；XT 保留完整报告，不根据结果替代默认候选。

1. BookChild：两个顺序各自的最终和阶段平均 AUC/AP，配对均值均大于 C0。
2. arXiv：每个顺序最终及阶段平均 AUC/AP相对C0均不低于-0.5pp。
3. 分类：每个数据集的平均最终/阶段平均累计F1差值均不低于-1pp；每个顺序对应差值不低于-2pp；最终global-support old-F1每个顺序不低于-2pp。
4. 完整性：工件哈希、旧行/旧旧边冻结、同rows/同pairs、切分互斥、训练源封存、全部checkpoint重载审计通过。重载浮点差异单列，不自动抹去告警。

以上是工程研究推进容忍值，不是统计显著性证明。报告每个顺序/edge split的配对差，seed胜率和标准差；同一memory的seed不作为独立数据集。两个顺序不足以宣称普适/SOTA。BookChild获益且arXiv不退化即可支持继续主线，不要求所有数据集显著上涨。LP通过但分类不通过时，定位为结构任务分支，而不否定LP收益。成本完整列账后再决定是否符合实际部署要求，不以未计时的旧成本冒充0。

## 4. 审计与边界

- 保存源码/数据/cache哈希、每阶段sources与allocation、memory、两任务checkpoint、预测/logits、逐fit metrics、成本。
- 重载全部模型，在各自固定evaluation graph复算；NC重算argmax/F1，LP复算logits排序AUC/AP，报告数值容差与最大偏差。
- 对真实边不输入构造的接口进行测试；负边采样、旧块冻结、同一memory跨任务、summary矩阵完整性均检查。
- 原始特征的上游预处理仍沿用既有数据工件，不能由本轮排除所有上游信息泄漏风险。
- 本轮不包含 k-center 或静态TAG强基线，不能由C0改进直接推断SOTA。
- 旧小图/大图结果目录不可覆盖。所有输出位于degree_confirmation_20260920独立目录。
