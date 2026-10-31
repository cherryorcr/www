# C0 基线底座与块保持联合核：两阶段冻结协议

本批次对应用户批准的阶段一、阶段二；不沿用历史文本 P2 的实验编号。
2026-09-20 冻结。仅限 Cora/Citeseer，不启动大图。

## 1. 方法与证据边界

C0 保持 dispersion 类预算、类内 joint clustering、Mean-X、joint_g5_c320
concat/re-encode、冻结旧行/旧旧边、text-nearest 跨边、response off。
当前 append-only 与 C0 是同一个方法，不人为列成两条独立改进。
旧 v4(response on) 单列历史参照，不用 Citeseer 的旧 v4 分数替换 C0。

阶段一复用通过审计的 canonical 5% 基线，补 random synthetic。
表内包含 random real、k-center、class-aware k-center、medoid、herding、
全历史重凝聚、旧 synthetic + delta recluster、完整 pool 等步数/长训选检查点、
静态 TAGSAM 两种适配。STAD 图文迁移单列适配诊断，监督 GCN 单列任务参考。
静态 TAGSAM 不是官方完整复现；不重新增强 TAGSAM/STAD。
复用来源、预算、类别顺序、student、prompt 竞争域、逐 seed 与成本范围均审计。
历史成本不能冒充本次同机同时段端到端计时；缺失字段保留为缺失，不填零。

## 2. 唯一新机制

候选复用 C0 同一份 X、文本、mass、rows 和行顺序，逐元素一致；
“多原型”指 C0 已有的类内多个合成簇，不额外引入 C1 配额或 X 算子。
只改变新块内部和新旧之间的边端点，边条数与 C0 相同。

固定核：K = 0.4 K_X + 0.3 K_T + 0.3 K_G。
权重为本次事前固定的工程假设，尚无最优性证据，不按报告结果调权。
K_X/K_T 是归一化合成特征/重编码文本的 Gram 矩阵。
K_G 是当前 delta 合法 pool 诱导图中 joint 表征的一跳传播与原表征等权混合，
再按 C0 簇求均值、归一化所得签名的 Gram 矩阵。
旧签名随合成记忆保留，不回读旧 raw；源节点分组只作离线审计工件。

新块选最高核值的 K_new 个无向对；每个新行连接最高核值的一个旧行，
均加反向边。tie 用行 ID 排序。旧行、旧文本、旧旧边及稠密关系核旧块保持。
GCN 的对称度归一化仍会因跨边改变旧块传播，因此另测归一化旧块漂移；
不把“旧旧边没改”说成“历史输出严格不变”。
签名有额外字节和构造成本，必须计入，5% 仅指节点数。
稠密核是非负加权 Gram；稀疏选边后不承诺保留 PSD 或性能保证。

## 3. 阶段二完整矩阵

- 数据集：Cora、Citeseer。
- 顺序：canonical、Python random.Random(19101/19102) 的排列。
- 三阶段类数：Cora 2/2/3，Citeseer 2/2/2。
- 五阶段类数：Cora 3/1/1/1/1，Citeseer 2/1/1/1/1。
- 每阶段内 class ID、source ID 升序；五阶段 canonical 按 ID 划分，
  不复用之前按频率选 base 的五阶段结果。
- 压缩率：1%、5%、10%，累计合法 pool 的 floor，严格 growing append，无 prune。
- 方法：C0、k-center、recluster、joint_kernel。
- report seeds：100..105；batch=16，实际 batch=min(16,rows)。
- student：既有两层 GCN384、冻结 MiniLM、可训练 text projection、双向 InfoNCE，
  Adam 0.001，步数 max(50,min(2048,16*rows))，每阶段从头训练。
- 全矩阵：36 个 dataset/order/rate/stage-count cell，576 份阶段 memory，3456 次 fit。
  random synthetic 另加 36 次 fit。
- 每次 fit 保存 checkpoint 与 report-dev 预测，并 reload 复核 pooled 指标。

canonical 3-stage/5% 已有 smoke 属开发条件。19101/19102 在本批冻结前未用于调参；
不因后续负结果缩减矩阵或重选顺序。已有 report seeds 不是新的盲测种子。
全图推理是 transductive；凝聚素材只含合法 pool。本批不评价 test，历史开发
接触过 test，因此不宣称整个项目拥有未污染最终盲测集。

## 4. 指标、成本与晋升

主指标：全部类别 prompt 竞争下的已见样本 pooled Macro-F1；最终、阶段均值、
pooled old/new、阶段轨迹并报。禁止混入旧 task-slice F1。
统计以 cell 等权，不把同一 memory 的六个 student seed 当成六个独立数据集。

完整矩阵必须先跑完。联合核晋升须同时满足：

1. 至少 75% 的 cell 最终 F1 高于 C0，平均至少 +1 pp。
2. 最差 cell 最终差值不低于 -2 pp；最终 pooled old F1 不低于 -2 pp。
3. 平均新类 F1 正增益，至少半数 cell 的新类 F1 改善。
4. 每个数据集聚合的构造 + 单次 student 成本不超过 C0 的 2 倍。
5. 矩阵平均不劣于预定强 coreset k-center；canonical 两数据集平均不劣于
   事前固定的 TAGSAM self-loop 适配，同时完整展示 semantic 适配而不按数据集挑臂。

构造成本包括重建签名/分组的增量成本及共享 C0 rows 的完整成本，不能只记增量。
另列六 seed 总训练时间、实际步数、峰值 allocated 显存、持久 memory 字节。
共享预处理、离线审计、磁盘时间与构造/训练分开说明。
未通过时保留 C0，不继续 response/dynamic budget/refresh/attention/大图。
