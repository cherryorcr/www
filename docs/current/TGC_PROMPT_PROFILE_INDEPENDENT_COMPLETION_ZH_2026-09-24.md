# TGC Per-Class Prompt Profile 独立确认报告

日期：2026-09-24  
范围：Cora/Citeseer，独立 `order19602`，stage 1/2  
前置版本：V3 response selector、prompt-paired selector  
本轮目的：验证逐类别 prompt profile + paired one-sigma LCB 是否能在新顺序上安全选择候选

## 1. 固定协议

本轮没有改变候选构造：

- `fallback_content`、`p_initial`、`anchor025`、`joint`；
- 同一个 assignment 同时生成 X、文本和拓扑；
- 原始两层 GCN student；
- 冻结 MiniLM、双向 InfoNCE、batch=16；
- selector seeds `300,301,302`；
- report seeds `150..155`；
- C0 historical prefix、stage-wise parent、report-dev 评估。

selector 只使用无标签 response：

1. 对每个已知类别 prompt 单独计算 prompt margin；
2. 在当前 delta 和旧 synthetic prefix 上分别取每个 prompt 的 top-20% margin；
3. 使用相同 seed/horizon 比较候选与 fallback 的 paired 差值；
4. 对每个类别的差值计算均值和 25% 分位数；
5. 用一标准差 lower confidence bound 形成 robust score；
6. 只有 old profile 不低于 fallback 且 combined lower bound 非负时才接受候选，否则回退。

节点标签、report-dev F1 和 test 均未用于选择。

## 2. 独立顺序结果

### 2.1 实际选择与 report 排名

| 数据集 | stage | profile-LCB 选择 | report 最优 | 选择结果 | 最优结果 | regret |
|---|---:|---|---|---:|---:|---:|
| Cora | 1 | fallback_content | fallback_content | 与 report 相同 | 与 report 相同 | 0.000 pp |
| Cora | 2 | fallback_content | fallback_content | 与 report 相同 | 与 report 相同 | 0.000 pp |
| Citeseer | 1 | fallback_content | fallback_content | 与 report 相同 | 与 report 相同 | 0.000 pp |
| Citeseer | 2 | fallback_content | fallback_content | 与 report 相同 | 与 report 相同 | 0.000 pp |

### 2.2 候选相对 fallback 的 report 结果

| 数据集 | stage | p_initial | anchor025 | joint | fallback |
|---|---:|---:|---:|---:|---:|
| Cora | 1 | 55.552 | 55.905 | 57.004 | **57.475** |
| Cora | 2 | 47.185 | 52.037 | 48.005 | **52.449** |
| Citeseer | 1 | 50.156 | 52.707 | 54.969 | **59.448** |
| Citeseer | 2 | 54.076 | 54.205 | 54.405 | **54.730** |

本顺序没有出现可被 selector 漏选的正收益候选，因此重点是安全性，而不是收益召回。

## 3. 与前一顺序的 selector 证据合并

| selector | 顺序 | 平均 regret | 平均实际收益 | 命中最优 cell | 主要特征 |
|---|---|---:|---:|---:|---|
| V3 InfoNCE/margin | order19601 | 1.540 pp | +0.885 pp | 1/4 | 有收益但排序不稳 |
| prompt-paired | order19601 | 2.182 pp | +0.242 pp | 1/4 | 目标更接近但过于保守 |
| profile-LCB | order19602 | 0.000 pp | 0.000 pp | 4/4 | 安全回退但没有收益召回 |

这三轮不能合并成“平均提升”，因为使用了不同的类别顺序和不同的 selector 协议。它们只能支持以下机制判断：

- shared assignment 候选空间在某些顺序上确实存在正收益；
- prompt-aware response 能提高目标相关性，但固定硬门限会造成误拒绝；
- per-class profile + one-sigma LCB 能避免独立顺序上的误接受，但过于保守，无法证明自动选择器有实际收益。

## 4. 当前问题的最终定位

### 4.1 不是简单的“缺少一个更好的分数”

当前 selector 依然在 synthetic/raw response 上预测真实图 report Macro-F1。即使把 instance-level loss 换成 class-prompt margin，也存在：

- synthetic topology 与真实图传播差异；
- prompt margin 与类别召回不等价；
- top-20% 无标签聚合可能只捕获局部高响应节点；
- 三个 selector seeds 对 lower bound 的估计仍然过于不稳定。

### 4.2 不能通过继续扫描权重解决

V3、prompt-paired 和 profile-LCB 已经覆盖：

- instance-level response；
- group-level class-prompt response；
- per-class class-prompt response；
- absolute response；
- paired response；
- hard utility gate；
- uncertainty-aware LCB gate。

仍然没有在独立设置上显示稳定收益。继续扫描 prompt 权重、分位点或门限，极容易变成对 report-dev 的事后拟合。

## 5. 对当前方法的决策

profile-LCB 不进入主方法，也不启动 selected-parent recursive rollout。当前应固定：

- **C0/R11**：稳定主线；
- **shared assignment**：条件性候选构造机制；
- **V3/prompt/profile selector**：student-response 选择机制的连续诊断和负结果。

如果论文需要一个完整、可复现的算法流程，建议使用固定的 C0-anchor 配方作为非自适应构造对照，而不是宣称自动 selector 已解决候选选择问题。

## 6. 后续建议

selector 主线暂时收尾。若仍要继续，下一项必须是新的信息来源，而不是继续重加 loss：

1. 使用冻结 student 的真实图 teacher response 或真实邻域统计，验证 synthetic response 到 report 的 domain gap；
2. 或将 selector 目标改为下游无标签代理之外的结构保持指标，并先做候选排序的离线相关性审计；
3. 只有新信息源在独立顺序上降低 regret，才值得做递归 rollout；
4. 否则把自动 response gate 的失败边界作为论文贡献的一部分，主算法保持 C0/R11。

原始结果：

`results/tgc_shared_assignment_gcn_prompt_profile_20260924/`

代码：

`code/v2_llm_fusion/run_tgc_joint_student_prompt_profile.py`
