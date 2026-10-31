# 官方 TAGSAM 静态复核与最小修正版记录

日期：2026-09-07  
范围：Cora、50 个 synthetic 节点、GCN + BERT 文本接口。本文只记录静态起点的代码核对和已完成运行，不把它们解释成持续凝聚结果，也不把不同评估协议下的数字直接横向比较。

## 结论先行

官方 master 可以完整运行，但原始 evaluator 对 Cora 只使用前五个类别。官方 master 在这个五类协议下的公开复核数字为 step 0 `0.4275`、step 500 `0.3824`；较早保存的 step 1000 和 step 5000 分别为 `0.4617` 和 `0.3070`。这组数字不能当作七类 TAG 结果。

add-adj 分支确实把 synthetic topology 存入 artifact，并在评估时读取它；它不是 master 的 self-loop 版本。原始五类 evaluator 的 step 0 是 `0.51154`，step 500 的一次运行时记录是 `0.55558`，独立五次重训复评为 `0.52928`。因此运行时单次数字不能替代独立重训均值。

在清除旧的五类 subgraph cache、使用独立七类 cache 后，add-adj step 0 的五次 student 均值为 `0.38576`，step 500 为 `0.40230`，增益 `+1.654` 个百分点。这个结果说明 learned topology 在当前七类协议下有可测训练效用，但增益仍然很小，不能据此宣称静态方法已经稳定优于基线。

当前最小修正版只保留尾批、设备/训练状态和七类评估协议修复。它在 seed 44/45 的七类 student 评估中分别得到：

| condensation seed | step 0 | step 500 | 差值 |
|---:|---:|---:|---:|
| 44 | 0.3340 | 0.3624 | +0.0284 |
| 45 | 0.3381 | 0.3874 | +0.0493 |

每个 artifact 都进行了 5 次独立 student 重训。两颗 seed 的方向一致，但样本仍少，且单个 seed 内的标准差约为 3.4–5.7 个百分点；这些数字适合作为开发证据，不足以冻结论文结论。

## 代码谱系

| 版本 | commit | topology | evaluator |
|---|---|---|---|
| official master | `d812f06fd16db3cd4d497804a8e504d2c738faa8` | synthetic self-loop | released Cora evaluator，五类 |
| official add-adj | `4bec0ccd1af2c1cc820b2941dd24a01af3b18ccb` | text-derived learned edges，artifact 保存 200 条边 | released evaluator，五类 |
| add-adj 七类复评 | 同上，外加本地评估协议改动 | 同一 artifact topology | `n_way=7`，独立 cache |
| minimal repair | 基于 master commit | self-loop，未加入 topology 扩展 | `n_way=7` |

原始 master 远程目录：`/home/pc/research/TAGSAM_official_cora_master_20260907`。  
原始 add-adj 远程目录：`/home/pc/research/TAGSAM_official_cora_addadj_20260907`。  
七类 add-adj 正确复评远程目录：`/home/pc/research/TAGSAM_addadj_native7_correct_20260907`。  
本地证据目录：`classical_work/official_static/results/`。

公共输入 `cora.pt` 的 SHA256 为：

`ac109677947b84e5c3b780578e49cacd3ca47ac522e1506447fe8881e468fa48`

master 使用的文件名是 `cora_lr1e4.pt`，但内容哈希相同。add-adj step 500 artifact 的 SHA256 为：

`e7f95f1d9a603bbb3c579fe660798c0c33ed8da8631789a6adcf5891499db8b0`

最小修正版 seed 44/45 artifact 的 SHA256 分别记录在 `classical_work/official_static/results/master_minimal_repair_seed44_45/` 的文件清单中。

## TAGSAM 接口核对

静态下游由两层 GCN（输入/隐藏/输出维度均为 384）、冻结 `bert-base-uncased` CLS 文本编码、可训练文本投影头组成。图节点表征和文本表征经归一化后使用双向图文对比损失；下游 student 从头初始化，在真实图的评估任务上进行零样本文本类别预测。

master 的 `SynGraphDataset` 将 synthetic adjacency 固定为 self-loop。add-adj 分支的 `SynGraphDataset.load()` 读取 artifact 的 `edge_index`，评估 sampler 也直接使用该 edge index；因此 add-adj 与 master 必须分开报告。add-adj 的 artifact 形状核对为 `node_f=[50,384]`、`text_embeds=[50,768]`、`edge_index=[2,200]`。

最小修正版没有把 learned topology、provenance edge、target-node 子集化或其他持续组件混入静态修复。这样才能把“实现错误修复”和“方法扩展”分开。

## 评估协议与 cache 问题

官方 `dataset.py` 的 Cora `test_split()` 写死 `n_way=5`，并把 7 个类别中的最后两个类别丢出任务。将 `n_way` 改成 7 还不够：`process_subgraph()` 会优先读取 `cache_subgraph_2048.pt`。如果该 cache 是五类运行生成的，后续七类运行会静默复用五类任务，得到看似合理但实际仍为五类的数字。

因此 add-adj 七类结果使用了单独的 `cache_subgraph_7class_2048.pt`。正确复评目录中的代码只做了两项改变：`n_way=7`，以及为七类任务使用独立 cache 文件名；artifact、模型、训练步数和 student seed 均保持不变。旧的 `ADDADJ_NATIVE7_STEP500_EVAL5.log` 与原始五类复评逐行相同，正是 stale cache 的证据，不能作为七类结果。

七类 add-adj 结果：

| artifact | student repeats | Mean Val Acc | Mean Test Acc |
|---|---:|---:|---:|
| step 0 | 5 | 0.41632 | 0.38576 |
| step 500 | 5 | 0.44750 | 0.40230 |

这里的 `Test Acc` 是在真实 Cora 图上的七类零样本分类准确率；它不是确认集 Macro-F1，也不是持续阶段指标。

## 官方运行结果

### master：原始五类 evaluator

| artifact | Mean Test Accuracy |
|---|---:|
| step 0 | 0.4275 |
| step 500 | 0.3824 |
| legacy step 1000 | 0.4617 |
| legacy step 5000 | 0.3070 |

### add-adj：原始五类 evaluator

| artifact | 记录方式 | Mean Test Acc |
|---|---|---:|
| step 0 | 一次五次 student 评估 | 0.51154 |
| step 500 | distill 结束时运行时评估 | 0.55558 |
| step 500 | 独立五次 student 复评 | 0.52928 |

运行时和独立复评的差异来自 student 初始化和训练随机性；后续结果必须保留评估次数和选点规则。

### minimal repair：七类 evaluator

| condensation seed | artifact | Mean Test Acc |
|---:|---|---:|
| 44 | step 0 | 0.3340 |
| 44 | step 500 | 0.3624 |
| 45 | step 0 | 0.3381 |
| 45 | step 500 | 0.3874 |

七类 teacher 的独立评估为 `0.6112`。teacher 与 synthetic student 使用同一七类标签提示接口，但 teacher 是完整真实图训练结果，不能被称为 synthetic 上界的数学证明。

## 解释边界

1. 五类和七类结果回答的是不同任务，不能按数值大小直接比较。
2. self-loop 和 learned topology 改变了模型输入，add-adj 结果不能归因于“仅仅修复了 master”。
3. step 0 与 step 500 的差值只说明在给定预算和评估流程下的变化；它不分离文本初始化、外层优化、拓扑和随机 student 选择的因果贡献。
4. 当前 minimal repair 的两个凝聚 seed 和 add-adj 的一个凝聚 seed 足以发现实现/协议问题，但还不足以支持稳定性或显著性声明。
5. 七类 Cora 复评仍是开发证据。正式论文实验应预先冻结 student seed、checkpoint 选择、独立留出集和最终测试集，并记录 examples seen 与 cache fingerprint。

## 对持续 TAG 研究的决定

静态后端现在可以作为受控起点，但应同时保留三条清晰的参照：`master-self-loop`、`add-adj-learned-topology` 和 `minimal-repair-self-loop`。持续实验默认沿 minimal repair 的七类协议开展；add-adj 只作为 topology 影响的静态敏感性对照，不能把它混入最小修复基线。

下一步进入持续归因时，先固定同一个旧 synthetic artifact 和同一个 teacher，逐项比较 old-only、new-raw、new-compact、old-compact + new-raw、old-compact + new-compact、joint 与 full-history recondense。只有当损失位置被单独证实后，才设计一个保持机制。暂不同时加入新的跨阶段边、transport、GraphCLIP 或额外文本模块。

## 本地证据文件

- `classical_work/official_static/results/official_master_remote/`：master 远程日志、输入清单和 commit。
- `classical_work/official_static/results/official_addadj_remote/`：add-adj 原始五类日志、输入清单和 commit。
- `classical_work/official_static/results/addadj_native7_eval/`：add-adj 七类正确复评日志、七类 cache 说明和 step 0/500 artifact。
- `classical_work/official_static/results/master_minimal_repair_seed44_45/`：minimal repair 的凝聚日志、artifact 和七类 student 日志。

