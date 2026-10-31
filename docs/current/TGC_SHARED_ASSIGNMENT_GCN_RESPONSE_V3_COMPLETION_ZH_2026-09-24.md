# TGC Shared-Assignment GCN Response V3 结项报告

日期：2026-09-24  
范围：Cora/Citeseer，小图，`order19601`，stage 1/2   
定位：student-response selector 的协议验证，不是新主方法晋升

## 1. 本轮要回答的问题

V1/V2 使用 selector-dev F1 或其 response curve 选择共享 assignment 候选，但 report-dev 的最终排序经常不同。本轮 V3 改为完全不使用 selector 标签，只使用训练过程中可以合法得到的图文响应信号：

- 当前原始 delta 上的双向 InfoNCE loss；
- 当前原始 delta 上 diagonal-vs-negative graph-text margin；
- 旧 synthetic memory 上的 InfoNCE loss 和 margin；
- `128 / 512 / canonical_steps` 三个训练点的 response curve；
- 多个独立 selector seeds 的均值、方差和旧记忆保持项。

report-dev F1 只在选择封存之后计算，用于审计 selector regret，不能参与选择。

## 2. 固定方法流程

每个 stage 均从冻结的 C0 historical prefix 开始，尚未进行 selected-parent recursive rollout。

1. 读取当前 raw delta 和 C0 parent。
2. 构造四个候选：`fallback_content`、`p_initial`、`anchor025`、`joint`。
3. `anchor025` 使用
   \[
   P_{anchor}=0.75P_{initial}+0.25P_{joint}.
   \]
4. 同一个 assignment P 同时生成合成特征 X、top-5/320 字符拼接后 MiniLM 重编码的文本和二值拓扑。
5. 对完整候选 memory 用原始两层 GCN student 从头训练，训练目标仍为双向 graph-text InfoNCE，batch=16。
6. 选择阶段使用 seeds `300,301,302`；正式 report 使用 seeds `150..155`。
7. 选择器要求候选的 terminal gain、old retention 和 utility 不低于预设门限，否则回退到 `fallback_content`。
8. 选择后才执行 report-dev 评估。

本轮没有加入 GraphGPS、文本注意力、递归更新、旧 memory revision 或新的拓扑模块。

## 3. 四个候选的 report-dev 结果

下表为封存后六个 report seeds 的累计 Macro-F1（%）。这些数值相同地用于 V1、V2、V3 的 selector 审计。

| 数据集 | stage | fallback_content | p_initial | anchor025 | joint | 最优 |
|---|---:|---:|---:|---:|---:|---|
| Cora | 1 | 56.015 | 56.388 | **58.971** | 54.662 | anchor025 |
| Cora | 2 | 50.768 | 49.397 | **51.736** | 51.644 | anchor025 |
| Citeseer | 1 | 53.237 | 55.129 | **56.214** | 53.820 | anchor025 |
| Citeseer | 2 | 54.405 | **57.203** | 54.484 | 55.517 | p_initial |

这说明共享 assignment 候选空间并非没有有效候选：`anchor025` 在 4 个 cell 中有 3 个达到最高 report F1；但它不能只靠当前 V3 response 信号可靠地被选出来。

## 4. V3 选择结果

| 数据集 | stage | V3 选择 | report 最优 | 选择结果 F1 | 最优 F1 | regret | 相对 fallback |
|---|---:|---|---|---:|---:|---:|---:|
| Cora | 1 | anchor025 | anchor025 | 58.971 | 58.971 | 0.000 | +2.956 |
| Cora | 2 | fallback_content | anchor025 | 50.768 | 51.736 | 0.968 | +0.000 |
| Citeseer | 1 | joint | anchor025 | 53.820 | 56.214 | 2.394 | +0.582 |
| Citeseer | 2 | fallback_content | p_initial | 54.405 | 57.203 | 2.798 | +0.000 |

聚合结果：

| 指标 | V1 | V2 | V3 |
|---|---:|---:|---:|
| 平均 selection regret（pp） | 1.685 | 2.425 | **1.540** |
| 平均相对 fallback 的已实现收益（pp） | 0.739 | 0.000 | **0.885** |
| 选中 report 最优候选的 cell 数 | 1/4 | 0/4 | 1/4 |
| V3/对应版本平均排序相关 | — | +0.050 | -0.250 |

V3 相对于 V2 有一定改善，但没有形成可靠的候选排序器。V3 在 Citeseer stage 1 选择 `joint`，其 report F1 确实比 fallback 高 `0.582 pp`，但仍低于 `anchor025` `2.394 pp`。

## 5. 成本与审计

每个 dataset-stage 的选择阶段包含：4 个候选 × 3 个 selector seeds × 3 个 response horizons，共 36 次 student fit；report 阶段另有 4 个候选 × 6 个 report seeds，共 24 次 fit。四个 dataset-stage cell 合计约 240 次 student fit。

- 选择没有读取 report/test 标签；
- 选择和 report 使用不同 seed group；
- 输出包含 protocol、build audit、selection JSON、progress 和 results JSON；
- Cora/Citeseer 两个 cell 均正常完成，无运行错误；
- 本轮仍是冻结 C0 parent 的 stage-wise smoke，不能解释递归长期表现。

## 6. 结果说明

### 6.1 V3 解决了协议合法性，但没有解决预测错位

V3 不再用 selector-dev F1，因此满足无标签选择约束；但它使用的主要信号是 instance-level graph-text alignment，而 report 指标是使用真实图传播和类别 prompt 得到的 zero-shot class Macro-F1。两者之间存在三层差异：

1. InfoNCE 关注节点与自身文本的配对，不直接关注类别 prompt 间的判别边界；
2. response 在 raw delta 和 synthetic prefix 上测量，而 report 在完整 synthetic topology 与真实 report 节点上测量；
3. loss/margin 的绝对变化受到 student 初始化、训练阶段和数据集尺度影响，不能直接表示新类塑性。

因此，V3 的负结果主要是 response proxy 与下游目标不对齐，而不是证明共享 assignment、`anchor025` 或 `joint` 本身无效。

### 6.2 当前瓶颈是 selector，不是候选空间

在四个 cell 中，最优候选相对 fallback 的潜在收益平均约为 `2.425 pp`，V3 实际捕获约 `0.885 pp`，约捕获可用收益的 36%。这说明：

- 继续增加 X、文本或拓扑候选，短期内不会自动解决问题；
- 先改善候选排序信号更有价值；
- 在 selector 未通过前，不应启动 selected-parent recursive rollout 或大图扩展。

### 6.3 不应把本轮选择结果升格为主方法

V3 平均 regret 仍为正，且 Citeseer 两个 stage 的排序相关均为负。它不能作为稳定优于 C0/R11 的主算法，也不能宣称 response-gated shared assignment 已经成立。当前应保留：

- C0/R11 作为稳定主线和共同控制；
- shared assignment 作为存在条件性正收益的候选构造机制；
- V1/V2/V3 selector 作为 response reliability 的连续负结果和诊断证据。

## 7. 下一步最小推进方案

下一轮只改变 response 统计，不改变 X、文本、拓扑和原始 GCN student：

### A. Class-prompt response

在不读取类别标签的前提下，使用已知类别名称构造 class prompts，测量当前 delta 和旧 memory 上的：

- 正类 prompt 与最强竞争 prompt 的 logit gap；
- prompt entropy；
- 新类 prompt 的平均置信度和覆盖均衡性；
- old/new prompt gap 的变化。

它比 instance-level InfoNCE 更接近最终 zero-shot 分类目标，但仍不使用节点标签。必须与 V3 InfoNCE/margin、固定 fallback、随机排序做同协议对照。

### B. Parent-relative paired response

所有候选与 fallback 使用完全相同的 student seed、batch 顺序和训练步数，直接计算 candidate-minus-fallback 的 paired response，而不是比较各自绝对 loss。报告均值、标准差、bootstrap confidence 和误接受/误拒绝率。

### C. 晋升门禁

只有同时满足以下条件，才进入 recursive rollout：

- Cora/Citeseer 至少两个独立顺序上的平均 regret 明显下降；
- 至少 3/4 小图 cell 选择不劣于 fallback；
- 平均相对 fallback 收益为正，最差 cell 不低于 `-1 pp`；
- old/new trade-off 可解释；
- 成本不超过 C0 的约 2 倍；
- report 标签仍只用于封存后的审计。

在 A/B 未通过前，不继续叠加 GraphGPS、attention、复杂预算或旧 memory revision。

## 8. 结论

本轮得到的最准确结论是：

> 共享 assignment 可以产生比 fallback/C0 更好的候选，但当前三种 student-response selector 都不能稳定预测最终 zero-shot Macro-F1。V3 通过无标签响应满足了协议要求，并将平均选择遗憾从 V2 的 2.425 pp 降到 1.540 pp，但仍不足以晋升为主方法。下一步应把 response 从 instance-level InfoNCE 改为更接近 class-prompt decision 的无标签信号，并使用 paired response 降低 student 初始化和训练轨迹噪声。

原始结果目录：

- `results/tgc_shared_assignment_gcn_response_v3_20260924/`
- 远程源：`/home/pc/research/TAG_condensation/results/tgc_shared_assignment_gcn_response_v3_20260924/`
