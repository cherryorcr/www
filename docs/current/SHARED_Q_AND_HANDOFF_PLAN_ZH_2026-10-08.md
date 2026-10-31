# 共享 Q 与外部交接基线：预注册补充

本轮不改主方法、不搜索 F1、不删除旧文件，不使用 sealed test。

## 协议 A：现有主协议上的共享 Q 归因

- WikiCS、SportsFit、BookHis；9101/9102/9103；construction seed 271006；Student seeds 150/151/152。
- 沿用现有 fixed-schedule rows、2048 steps、batch 24、Prompt Student。
- full 使用冻结构造器；split-Q 使用同一初始化、损失、步数、学习率、容量和残差边界，但 X/传播使用 Q_X，文本使用独立 Q_T。
- split-Q 的 anchor 为两分配 anchor 的均值；当两分配相同且参数绑定时，目标退化到 full。同一 GM 仍连接两个分支，不能称完全无图文关系。
- 两分配不共享 Adam 参数；记录分配差异、硬文本选择差异与独立梯度。额外参数量和耗时单列。
- 保留原生文本评价；在最终阶段把两臂文本换成同一份归档 CaT payload，固定 X 后再次评价。
- 先做共享实现一致性、跨类别零值、双路梯度、旧前缀、材料化和预算测试，再执行完整五阶段链。正/负结果都报告，不以“证明”为预定答案。

## 协议 B：交接类别顺序与预算上的双 Student 比较

- 只复用通过内容 hash、来源和划分核验的 sealed memory，不把交接旧分数混入现有主协议。
- 当前主方法另按交接包原始顺序、每阶段 rows 构造。不同方法的构造 seed 不强行伪称一致，条件于各自归档构造比较。
- 每份 memory 分别在 legacy Student（原 row-wise InfoNCE，batch 16）和当前 Prompt Student（batch 24）评价；均 2048 steps、150/151/152。
- 各 Student 内严格统一训练与评价。两 Student 的比较同时包含训练目标、batch 和其原有传播实现差异，不能直接归因于单个 loss，也不是跨架构证明。
- 保留原合成拓扑和文本，不把外部方法强行改成主方法的 edge-free。缺失 labels 时只根据已保存来源/已核对构造顺序恢复，不使用 report/test 标签。
- 优先三张当前主图；arXiv 保留交接审计，是否可执行须通过资源/输入检查，不以缺失填零。
- 真实 coreset 单列；CaT/GECC/PUMA/TAGSAM 明确适配边界。交接包本身不补足官方高保真实现缺口。

## 输出与约束

分别输出共享 Q 消融表、legacy 基线表、Prompt 基线表，不混用不同协议均值。报告 F1、Accuracy、Old/New、stage-average、worst-stage、成本与 hash；三个 Student seeds 不是三个构造 seeds。

远程空间有限，保存新增阶段块而非重复完整前缀；不保存稠密 Q，不删除历史结果。达到磁盘保护阈值则停止写入并保留状态。

## 追加的探索性检查

在实验 A 的 Prompt 结果出现后，追加原 InfoNCE Student 的最终阶段共享/独立 Q 对照。使用全部三图、全部三顺序、相同 150/151/152；只重新训练评价器，不改 memory。该检查不是事前注册的验证，不能隐藏为原计划，也不能选择性保留有利 Student。目的是区分参数共享效应与 readout 协议依赖。

另外对交接表中旧分数不能精确重放的条目，调用旧评价入口与新 wrapper 做诊断；诊断分数不替换正式配对表中的不利运行。

## 运行中发现的适配修复

- SportsFit 的 coreset 存档没有逐行来源 ID。先检查当前 delta 中完全相同的 X、文本 embedding 和字符串；有跨类别内容重复时，不随意选择标签，而是重放原无标签 selector，核对 seed 及全部输出内容。仅 base random 需要此重放，其余 14 个块可直接按精确内容恢复。该步骤是来源元数据恢复，不改变被评价的 memory。
- 旧 `handoff_sportsfit` 在来源检查处停止，保留其记录；用独立 manifest 和 `handoff_sportsfit_verified` 重新执行，不绕过旧协议 hash 检查。失败运行不并入统计，重新构造不是新的独立 seed。
- SportsFit/arXiv 交接 arm 虽名为 `cat_mmd_tag_cctp_top5`，构造审计实际是 `text_center`。WikiCS/BookHis 为正确的 `nearest`。本轮对 SportsFit 增设独立 fixed-X 文本修复：保持原 CaT X/标签/mass/depth/stage/拓扑不变，仅附着 nearest-5，并在两个 Student 下各跑 150/151/152。错误实现保留为历史内部对照，不作为标准 CCTP 基线。arXiv 只标记审计问题，未宣称完成修复或评估。
- 追加同一旧 Student 函数、同一 seed 的重复训练诊断，分别使用默认与确定性数值执行设置；其结果只解释可复现性，不能替换正式表中的分数。
