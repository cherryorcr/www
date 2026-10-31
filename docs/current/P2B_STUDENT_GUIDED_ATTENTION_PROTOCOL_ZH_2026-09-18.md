# P2-B: Student 引导的可学习文本 Attention 预注册

日期：2026-09-18。状态：正式结果产生前冻结的筛选协议；不修改 v4/P2-A。

## 1. 问题与边界

检验可学习 attention 是否因为 student 反馈而改善完整重训效用，而不只是改善语义重构或短训损失。仅 Cora/Citeseer；canonical、1701、1702 是已知诊断顺序，不称独立泛化。共六个 arm：center_concat、fixed_anchor、semantic_700/701、student_700/701。

所有 arm 固定 5% 累计节点预算、dispersion 分配、Mean-X、类内簇、枚举、旧行/旧旧边和 legacy 拓扑。新文本不参与重建边。保留证据文本与学习后的向量；后者不等于可读文本直接 MiniLM 编码。没有新的真实节点 coreset。

## 2. Attention

每簇中心距离选最多 5 篇源文本，water-fill 总共 64 content tokens，固定片段；不学习离散选择。冻结 MiniLM 编码片段 e 和拼接文本 c。query q 使用 P2-A 的 attribute + 当前合法 delta 内外簇邻域摘要，冻结。当前不声称图条件优于文本条件，独立 query 消融留待候选通过筛选。

score_i = cos(e_i,q)/0.2 + 2*tanh((Wq*q)^T(Wk*e_i)/sqrt(16))。
Wq、Wk 各 16x384，Wq 零初始化，Wk 按构造 seed 初始化。因此两 seed 初始输出与 fixed_anchor 相同。
target = normalize(0.75*c + 0.25*sum(softmax(score)_i*e_i))。
保留同预算 concat anchor，限制语义空间漂移。片段和 anchor 各自覆盖同一 64-token 内容，不是额外提供另一批内容；实际 encoder 调用、重复处理 token 与特殊 token 全部计成本。

## 3. 两种学习目标

- semantic：当前 outer-train 源节点文本均值的 cosine 重构 + 0.1*anchor cosine penalty；无需 student。每 5 步按 outer-val 源节点文本均值重构选择 checkpoint，含初始化。
- student：对同一两层 GCN+text projection student 做可微 Adam 内循环（batch16、lr0.001、16步）。每次从头初始化，probe seed300/301/302轮换。外目标为真实 current-delta 图文双向 InfoNCE（固定源文本 embedding）+0.25*旧合成记忆 InfoNCE（存在旧行时）+0.1*anchor cosine penalty。不能仅优化合成图上的 loss。
- attention 外优化均为 Adam lr0.01、40步、梯度范数上限1，无报告分数调参。student 每5步用独立 probe seed310、32步和固定 outer-val batches 选择最低目标 checkpoint，初始化也参与。验证只用于 checkpoint，不反传。
- 可微 Adam保持共同 student 的层、激活、归一化、损失、batch序列、优化器参数。先检验前向/更新数值近似等价（不要求GPU bitwise）；零二阶矩 sqrt 的反向在零处采用0以避免 NaN。最终报告一律使用未经改动的 C.fit，不使用可微训练替代。

## 4. 数据隔离

每阶段当前合法 pool 随机80/20拆分 outer-train/outer-val，构造seed700/701、stage相关确定性划分。验证节点可能已参与固定聚类/query/片段构造，故只是 outer-loss-held-out，不宣称完全未见源节点。训练与验证 batch 不重叠；不读 report-dev/test 标签或文本作训练目标。旧原始数据不回读，仅旧合成记忆。离线输入回归审计不算在线算法能力。

报告学生 seed100..105 与300..302/310及700/701独立。报告 all-prompt report-dev seen-node Macro-F1 主指标；seen-only 次指标。全真实图推理沿用共同 transductive 协议；报告标签只供完成构造后的离线评价。每阶段从头训练，步数 max(50,min(2048,16*rows))，不选择报告检查点。

## 5. 筛选与停止

两个构造seed均完整报告，不按结果挑seed。预定筛选：student 平均最终F1对 center_concat、fixed_anchor、semantic 在两个数据集均至少+0.5pp；阶段平均均不下降；每个顺序最终和数据集平均旧/新类下降均不超过2pp；两个构造seed各自对center最终均为正。通过才运行独立1703/1704、报告seed110..115、构造seed702确认；否则不扩展大图，不自动搜索超参。不得称3个顺序为足够统计功效的SOTA证据。

## 6. 验收与成本

保存 source split、簇/片段ID、token计数、共享输入hash、模型参数、attention weights、anchor漂移、每步外损失/梯度、checkpoint选择、meta成本和最终student成本。检查非零meta梯度、旧文本冻结、控制输入逐位相同、报告checkpoint重载、报告seed独立。断点续跑拒绝代码/配置hash不符。

成本分列：共享构造/缓存源编码、片段及anchor编码、query、attention优化/验证、完整student训练。单次使用计入全部构造成本，不除以六个报告seed。源编码缓存的历史成本单列可用测量；不将warm-cache成本称端到端总成本。训练参数仅构造用，部署保存最终target，可读证据和合成图；研究checkpoint另计存储。
