# 冻结 v4 小图顺序确认与成本账协议

本轮按用户更新的优先级执行：不补强 TAGSAM、不重构或重新搜索 STAD；新增小图顺序确认与统一成本测量。注意力文本只进入后续计划，不在本轮修改主方法。

## 冻结范围

- Cora/Citeseer，各 3 个预先固定的类别顺序：原 canonical、Python `random.Random(1701)` 和 `random.Random(1702)` 对类别 ID 排列。保持每阶段类数 Cora 2/2/3、Citeseer 2/2/2；实际顺序在读取结果前写入 plan.json。
- 每个顺序按当前累计合法 pool 的 5% 重新计算预算；所有方法在该顺序具有相同阶段预算。顺序不同导致早期节点预算不同是 fixed-rate 协议的真实效应，不称纯类别顺序的等预算因果干预。
- student seeds100..105；selector seeds300..302、batches8/16/32、多 horizon 及风险门禁均不改。新顺序重新构造 v4，而非套用原顺序的 artifact。
- 13 臂：v4、dispersion、random_real、k_center、k_center_class、herding、kmeans_medoid、recluster、full_history_syn、raw_equal、raw_long、raw_finetune、v4_init_finetune。每图 702 次报告 student fit，两图共 1404 次；v4 selector fit 另计。
- 新旧划分/全类与已见类 prompt、Macro-F1、pool 标签权限、transductive 推理及 no-history 构造边界与现行 audited runner 相同。test 标签不用于本轮评估；不声称项目历史未接触 test。
- 原顺序 v4/dispersion 记忆须与归档逐位相同，合成/选点小图训练的 canonical student 终点参数严格回归。所有 fit 都复载检查点重算指标；新顺序所有 selector 的阶段类别全局配置必须同步。完整原始图训练的数值复现修订见下节。

## 数值复现验收修订

原验收要求所有 canonical 终点逐位相同。执行发现 Cora `raw_equal` 第 1 阶段 seed100 在输入和初始权重完全一致时与历史终点不同；独立进程内三次固定 seed 复跑也分别得到 55.56、60.19、64.19 的 all-prompt F1，证明当前 GPU 路径下存在训练数值非确定性。尚未完成具体内核的因果定位，不能简单归因为模型退化或已确认的某一 CUDA 算子。

仅修订验收，不改变凝聚、student、数据、超参数或选择规则：完整数据训练臂及其微调链记录终点差异；输入 hash 和从头训练初始权重仍严格一致；微调链严格继承本次上一阶段权重；v4/其它小图训练仍严格回归。每次检查点复载重算仍须一致。新数据使用本次第一次成功运行的结果，不按分数挑选复跑。

原始 plan 和 runner 快照保留；每个数据集保存 `validation_amendment.json`、原始/当前代码 hash，Cora 保存 `raw_regression_diagnostic.json`。此修订是观察到执行异常后的明确验收修订，不宣称全部验收规则在运行前已固定。报告需区分历史分母与本轮分母，并披露 GPU 数值不确定性。

## 成本口径

共享数据加载、冻结 encoder 和 pool-only 文本编码单列；每个方法重新构造，GPU 同步计时。v4 报告候选生成与 response 合计、response 本身耗时、selector fit/optimizer steps；每方法单列 student 训练和含报告/保存的 fit wall time。

所有 6 个报告 seeds 的训练时间是实验测量成本。估计单次部署训练成本时使用 6 个 seeds 的平均，而非把总时间当一次训练。v4 构造成本在同一记忆上只发生一次，不能除以报告 seeds 隐藏。微调初始化依赖单列并补计必要的 stage0 构造，不能当免费初始化。

微调依赖项补计 stage0 构造算法时间，但没有单独重测该依赖的少量保存/包装开销，故微调总耗时为近似值。独立复载审计、异常诊断复跑和实验中断重启的开销不计入“单次部署”，不可将其冒充整个研究项目已消耗的总 GPU 工时。

逻辑记忆字节包括保存的全部 tensor 字段和 UTF-8 文本，另说明序列化/解释器开销未计。5% 仅指节点比例，不是所有存储、边、FLOPs 或成本的 5%。显存为当前进程 allocated 峰值，包括已加载的模型，不代表主机总内存。

TAGSAM/STAD 沿用原顺序已归档的性能与历史成本，明确标记“未做本轮多顺序/统一重构计时”，不能与本轮新成本宣称严格加速比。官方 checkpoint 和监督参考仍独立列出，不作为共同 student 保留率分母。

## 报告规则

不按新顺序得分改参数或挑顺序。逐顺序报告最后阶段、阶段平均、old/new、历史任务 forgetting、性能保留率及成本；跨顺序先平均每顺序的 6 seeds，避免把 18 个相关运行当 18 个独立顺序。只覆盖两个额外顺序，是稳健性初步确认，不是广泛普适性证明。

顺序确认失败也是完整结果。报告必须区分方法有效性、相对基线优势、顺序鲁棒性和成本优势，不用其中一项替代其它项。
