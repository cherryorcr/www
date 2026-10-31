# P2 收尾与小图对照冻结协议

日期：2026-09-16。范围：仅 Cora/Citeseer，不启动大图。

## P2 封存决定

P2 已归档 2052 次 student fit，完整性验收通过；没有文本候选通过跨数据集、独立 seed、阶段保持门禁。冻结 center concat，保留负结果，停止继续挑选最高的 dev 分数。原验收文件与所有旧 artifact 保持不变。

必须区分两种基线：v4 主线使用历史 `joint_g5_c320` 字符拼接；P2 的 concat64 是固定原合成特征与边，仅替换为每行最多 64 内容 tokens 的文本控制。后者不是重新运行 response allocation 的 v4。两种不能互换引用结果。

## 修正已有比较

旧 `run_small_protocol_comparison.py` 与两份 `comparison_small_*_2026_09_16.json` 只留作历史记录，不纳入新主表，原因包括：seen-prompt 把任务 F1 平均当累计 F1；prompt 子集索引没有映射回全局类别；单候选 SCA 被误当当前 v4；类内凝聚被误标为完全 label-free；微调缺少 raw stage0 起点；coreset 混合真实诱导边与语义跨阶段边。

新增 `run_small_comparison_audited.py` 与独立指标模块。累计 Macro-F1 在全部已见 report-dev 样本上一次计算。旧类/新类均使用同一个已见样本集合上的 per-class F1，因此跨任务 false positives 会被计入。不能与旧版按真实任务切片计算的 old/new 值直接比较。

早期同时报告全部类别 prompt 与仅已见类别 prompt，终阶段两者必须完全相等。保留全部 prompt 作为历史主口径，避免看完结果才换指标；seen-prompt 给出标准已见类竞争诊断。

训练源排除 selector-dev、report-dev、test；test 标签不评估。评估在完整真实图上完成，会用到未来及留出节点的无标签 X/A，这是 transductive setting，不是严格时间隔离。凝聚类内分配使用 pool 标签；下游 student 的 InfoNCE 不使用分类标签。

## 固定设置与实验组

三阶段类别划分、5% cumulative fixed-rate、batch16、student seeds100..105、GCN384/MiniLM384、Adam1e-3。每阶段 compressed-step budget 为 `max(50,min(2048,16*K_t))`。旧历史行不裁剪。

| 组 | 数据/历史访问 | 用途 |
|---|---|---|
| dispersion、真实 v4、concat64 | 当前 delta + 历史合成图；每阶段 scratch | 当前主线与文本控制 |
| random、k-center、herding、medoid | 当前 delta 选真实行，保留旧行；同语义 append 建边 | 同 rows coreset，不能称合成方法 |
| class-wise k-center | 与 dispersion 相同按类配额与标签权限 | 分离标签/类别覆盖优势 |
| recluster | 旧合成图 + 新 delta 重新凝聚 | 历史重写内部对照 |
| full-history synthetic | 读取所有已见合法 raw pool 后重凝聚 | 放宽历史访问的离线参考 |
| raw equal | 所有已见 raw pool，学生从头训练，与合成图相同步数 | 等优化步数参考 |
| raw long | 所有已见 raw pool，8192步；另用 selector-dev 每512步选checkpoint | 更充分训练参考，不是数学上界 |
| raw fine-tune | stage0 raw训练，之后继承参数，只训练新 delta | 正式持续学习基线，Adam每阶段重置 |
| v4-init fine-tune | stage0 v4训练，之后只训练新 raw delta | 分离初始化与追加合成记忆作用 |

所有 raw 训练图仅由合法训练源诱导。完整数据指已见类别中全部合法训练源，不能包含留出标签/文本目标或未来阶段训练数据。

## TAGSAM 适配

源码：`https://github.com/SundayVHan/TAGSAM`，commit `d812f06fd16db3cd4d497804a8e504d2c738faa8`。直接加载官方 `model.py:wBCELoss`，检查 git 内容与文件 hash；并记录 distill/dataset/epoch/selection 源码 hash。

保留双层响应匹配、合成特征优化、可学习内层图/文学习率及正负概率平衡 BCE。为比较当前任务，明确替换为共同 GCN/MiniLM、dispersion 初始化与中心拼接；每阶段 teacher 只用当前合法 delta 训练4096步，旧合成记忆冻结；固定100外步、每步10次有效内更新、batch16，真实参考最多256行。下游统一canonical Adam，不使用蒸馏出的学习率。

对比 self-loop 初始化、TAGSAM self-loop、TAGSAM semantic append，以及修正温度/内步不足问题后的 row/column-softmax ASM 内部控制。self-loop 组不加跨阶段边；semantic 组与主线使用相同语义 append 规则。全部使用同一预先固定训练预算，不用 report-dev 选最优构造轮次。

这些是 **TAGSAM/GraphCLIP 持续适配实验**，不是官方 5000 外步、15 内 epoch、原始文本选择器的完整复现，也不支持“超过 TAGSAM/SOTA”的结论。其它监督持续图方法仍需各自公平适配后才能列数值。

## 验收

先校验指标、非连续类别 prompt 映射、训练源诱导边、学生参数与 canonical 训练逐位一致；再跑完整6 seeds、3 stages。历史 v4/dispersion/concat64 artifact 的终阶段需回归到已归档数值。每次 fit 保存参数、support hash、初末参数 hash、耗时与显存。微调必须核对相邻阶段参数继承；所有 append 组核对旧特征/文本/旧旧边不变。

额外发现：将 `scale * g @ t.T` 改写成 `scale * (g @ t.T)` 会改变浮点舍入并放大为训练结果差异。第一轮新 runner 因此停止，保留为数值诊断；正式目录使用 `small_comparison_audited_r2_2026_09_16`，已通过参数一致性测试，不混入未通过回归的首轮结果。
