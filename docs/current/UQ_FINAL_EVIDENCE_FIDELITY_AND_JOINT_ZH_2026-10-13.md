# Unified-Q-EF: 基线保真度与联合分布替换的证据边界

本报告只读汇总 2026-10-13 13:21（北京时间）捕获的确认快照。主方法仍为
**Unified-Q-EF**；联合图文分布匹配仅是开发阶段的单一替换候选。原 sealed 表、
论文主方法和超参数均未因本报告更改。原始快照与自动统计见
`results/uq_final_evidence_20261012/snapshot_20261013_132047.json`、
`results/uq_final_evidence_report_20261013_132047/summary.md`；替换审计见
`results/uq_joint_replacement_20261013/summary.md`。

## 1. 可比较的九格确认

三图均使用合法当前增量池累计真实 5% 行预算、冻结旧 synthetic prefix、同一
MiniLM / prompt-GCN Student（2048 步、batch 24）。每种方法有 3 类别顺序 x
3 构造种子 x 每格 3 Student 种子。Student 种子先在格内平均，再以 9 个匹配格
计算差值及 95% 配对 t 区间；不是把 27 次读数当作独立样本。report-dev 和 sealed
test 来自同一次 Student 训练的不同评估掩码，不能把前者用于调整后再将后者称为
全新盲测。

| 数据集 | sealed Final F1: EF / GECC-hard / CaT-CCTP | sealed Acc: EF / GECC-hard / CaT-CCTP | EF - GECC-hard F1 (95% CI; 正向格) | EF - CaT-CCTP F1 |
| --- | --- | --- | --- | --- |
| ArXiv | 52.91 / 52.68 / 34.98 | 71.15 / 70.94 / 60.41 | +0.22 [-0.34, 0.79]; 4/9 | +17.93 |
| SportsFit | 80.08 / 80.57 / 74.50 | 85.52 / 89.01 / 85.56 | -0.49 [-1.50, 0.52]; 2/9 | +5.58 |
| BookHis | 48.81 / 45.34 / 29.16 | 81.05 / 81.26 / 79.23 | +3.47 [2.20, 4.74]; 9/9 | +19.66 |

report-dev 的 EF - GECC-hard 配对 F1 分别为 ArXiv +0.65 pp
[-0.00, 1.30]（7/9），SportsFit -0.95 pp [-1.99, 0.10]（2/9），
BookHis +2.66 pp [0.92, 4.39]（8/9）。sealed BookHis 的最差匹配格仍为
+0.09 pp；SportsFit 最差为 -2.14 pp。主结果能支持 BookHis 的稳健收益及三图
相对 CaT-CCTP 的增益，**不能**支持在所有大图上稳定优于 GECC-hard，更不能把
CaT-CCTP 的差距归因为共享 Q 一项。

成本同样要放在主比较旁边：ArXiv / SportsFit / BookHis 的 EF 构造均值分别为
243.3 / 170.3 / 68.9 秒，GECC-hard 为 189.5 / 735.1 / 52.0 秒；EF 构造
峰值 CUDA allocated memory 为 17.33 / 24.08 / 2.91 GiB，GECC-hard 为
0.93 / 0.98 / 0.49 GiB。时间不含公共数据加载和文本缓存，显存不是系统总内存。
EF 在 SportsFit 的构造时间较短，但显存开销明显更高。

## 2. 方法名与适配范围

下表均为**机制适配**，不是论文作者给出的原协议端到端官方复现。共享预算、
类别顺序、冻结旧 synthetic prefix、共同文本编码器/Student 属于本任务的适配；
不得省略这些变化直接写成“官方方法”。

| 对比名 | 保留的来源机制 | 为当前持续 TAG 协议作的改变 / 尚未保留的部分 | 已有证据 |
| --- | --- | --- | --- |
| CaT-CCTP | 类别级 CGM/MMD；随机 Encoder；对自由合成特征 X 作 3000 步优化，并记录固定随机空间中的前后 MMD 诊断。 | 按当前增量构造并冻结历史；邻域传播用于真实侧，synthetic 为 edge-free；优化之后才以最近特征源的 CCTP top-5 当前文本附着并重编码，非原生 TAG 文本联合优化。 | 三图各 9 格。 |
| GECC-hard | 按类聚合传播特征；20 次初始化的 hard k-means，并以 SSE + 容量平衡选中心。 | 历史中心只保留、不用累计历史原始数据重聚类；CCTP nearest-5 文本适配；共同 Student。hard Lloyd 不是 GECC 原生 fuzzy c-means 分支。 | 三图各 9 格；强且计算较轻的机制适配基线。 |
| GECC-soft (native FCM) | 固定 GraphSlim 上游源码的 fuzzy c-means 中心、fuzziness 1.3、20 次重启与平衡选择；没有用线性 Q 池化替代 native center。 | 当前增量、冻结前缀、共同 CCTP nearest-5 文本和 Student；1.3 是本补充 soft 分支设定，不是声称复原作者所有数据集调参配置。 | BookHis 单匹配格；SportsFit 仍运行，ArXiv 排在串行队列后。 |
| TAGSAM (native selector adaptation) | pinned GPT2-large 词概率/PMI 及邻域、冗余文本选择，并调用响应蒸馏适配。 | 当前增量、320 字符、共同 Student；尾批/空组数值修复；100 x 10 unroll 代替原生 5000 x 15，不能等同于原响应蒸馏预算。 | BookHis 单格；不作为高保真端到端官方成绩。 |
| STAD (native core, local Qwen) | 双路径自训练、聚类纯化、关键词/候选生成、Wasserstein 选择及合成拓扑。 | 当前合法池内部 10% supervised / 10% internal-val / 80% pseudo-label，与主方法全监督标签用量不同；以本地 Qwen2.5-1.5B 替换 qwen3-max API；共同 MiniLM/Student 与二值邻接。 | BookHis 单格；须单列监督和生成器差异，不与同监督方法混为一列。 |

BookHis 同一个 o9101/c271006 格（均为 3 Student 种子平均）的 report-dev
Final F1：EF 47.88、GECC-hard 42.34、GECC-soft 38.70；EF - GECC-soft
+9.19 pp，但只有一格，**没有**配对区间或跨顺序稳定性。GECC-soft 构造
5273.7 秒，GECC-hard 57.6 秒，EF 65.4 秒（该格，非上表九格均值）。
该单格结果仅说明原生 soft 分支不自动带来更强 TAG 迁移，不能推断所有数据集
或重新选择 fuzzy 参数。STAD 的 BookHis report-dev 37.78 F1、79.36 Acc；
TAGSAM 为 11.38 F1、61.00 Acc，二者保真限制尤需随数字同列说明。

## 3. 节点级图文联合分布替换

开发候选仅把原类别边际 DM 权重从 1 改为 0、paired product-kernel 联合 DM
从 0 改为 1；输入池、mask、阶段、四舍五入后的 5% 调度、Student/种子、其余
构造源码哈希和旧前缀冻结接口通过程序逐项核验。每数据集只有同一顺序/构造
种子的一格、每格 3 Student 种子，不能给出跨构造的显著性结论。

| 数据集 | report-dev EF F1 -> 联合替换 F1 (差值) | Acc 差值 | sealed 事后 F1 差值 |
| --- | --- | --- | --- |
| ArXiv | 51.50 -> 51.95 (+0.45 pp) | -1.56 pp | -0.21 pp |
| SportsFit | 79.52 -> 79.67 (+0.15 pp) | +0.84 pp | -0.84 pp |
| BookHis | 47.88 -> 45.24 (-2.65 pp) | -2.41 pp | -1.15 pp |

所有阶段满足预算、冻结、有限值及 Sinkhorn 检查；联合损失对 Q 的初始梯度非零，
硬文本前向误差低于 1e-5。因此“联合目标没有执行”不能解释负结果。由于候选
在既有开发探索之后提出，sealed 读数仅作事后透明披露，不作超参数选择依据。
**不将本次替换升级为主方法**；若继续探索，需在另一个独立确认协议上预先
固定目标后验证，不能继续用同一 sealed 掩码调到正结果。

## 4. 接下来写论文时的边界

1. 主表采用三图 9 格的 EF/CaT-CCTP/GECC-hard；不要混入 BookHis 单格的
   GECC-soft、TAGSAM、STAD 均值。单格可放“适配保真度诊断”表，标明模型与
   监督偏差，并说明 SportsFit/ArXiv 未完成。
2. 共享 Q 保证图文来自相同支持，但当前对独立 Q 未证明稳定精度优势：
   report-dev 三格的 shared - independent 在 ArXiv / SportsFit / BookHis 为
   -0.48 / -0.58 / -2.42 pp。相同参数预算的单格对照不足以推翻这一点。
   将共享支持的一致性与性能因果分开表述。
3. 定位为受历史访问约束的 graph-aware、edge-free TAG 节点分类凝聚；
   topology-enabled 扩展与链路任务单列。不声称完整拓扑凝聚或普遍 SOTA。
4. 等待 GECC-soft SportsFit 当前 PID 125596 完成后，从新快照重新严格汇总；
   2026-10-13 13:25 的远程状态为运行约 17 小时、结果仍是 `running`、
   单核约 105% CPU、剩余磁盘约 29 GiB。不要重复启动，ArXiv 尚未开始。
   既有 sealed 主表不可覆盖，也不能为该候选临时改 FCM 参数。
