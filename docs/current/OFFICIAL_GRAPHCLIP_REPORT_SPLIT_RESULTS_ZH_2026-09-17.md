# 官方 GraphCLIP checkpoint：当前报告划分直接推理结果

日期：2026-09-17。

## 1. 本次做了什么

按用户要求，直接使用已下载的官方预训练 checkpoint，不重新训练、不微调、不挑选 checkpoint、不搜索 prompt。只评估 Cora/Citeseer 小图。

这是“官方权重 + 本项目推理适配”的外部模型参考，不是官方论文评测协议的完整复现，也不是凝聚算法的同 student 对照。

| 项目 | 设置 |
|---|---|
| 图模型 | 官方 `models.gt.GPS`，12 层，hidden 1024，输出 384；图塔约 128.01M 参数 |
| 文本模型 | checkpoint 内的 MiniLM 权重，约 22.71M 参数；attention-mask mean pooling 后归一化 |
| 权重加载 | 图塔、文本塔均 `strict=True`；现有 MiniLM 缓存与 checkpoint 文本参数完全一致 |
| 训练 | 0 次 optimizer update；所有参数冻结；没有训练 seed 均值或标准差 |
| 原始图 | Cora 2708 节点、10556 条 edge_index 记录；Citeseer 3186 节点、8450 条记录 |
| 报告划分 | `original_node_id % 10 == 3`，分别 271 / 319 个节点 |
| prompt | 与当前 v4 相同，`a research paper about {class_name}` |
| 推理 | 从完整原始图提取每个报告节点的完整二跳 ego 图，RW-PE 长度 32；官方 ego 均值与中心节点拼接读出 |
| 子图裁剪 | 不裁剪；不采用旧脚本“大于 256 节点退回一跳”的规则 |
| 预测 | 归一化图表示与类别 prompt 表示的余弦相似度，取最大值对应类别 |
| 评估范围 | 主表是最终阶段全部类别；附表给出阶段切片 |
| 标签边界 | report 标签只用于计算指标；没有使用 selector/test 标签选择或评估模型 |
| 图访问边界 | 推理允许完整原始图的特征和边，包括 test 节点作为上下文，属于 transductive 推理 |

checkpoint：`/home/pc/research/GraphCLIP/checkpoints/pretrained_graphclip.pt`

SHA256：`e93b19a565446c3d62f53eb31fee08570f94c558de216e1aec805d05532721aa`

预训练数据与目标数据是否重合尚未完成审计，不能把此模型称为严格任务上界。

## 2. 结果与已有参照

以下是相同 report-dev 节点、类别顺序和 prompt 下的最终阶段 Macro-F1，单位为百分制。不同模型仅作外部参照，不能解释成凝聚算法的单因素差异。

| 方法 | 是否本次新增 | Cora | Citeseer |
|---|---|---:|---:|
| 官方 checkpoint，完整原始图上下文，纯推理 | 是 | **58.98** | **57.76** |
| 当前两层 GCN student，全部合法 pool，等训练步数 | 否 | 47.93 ± 4.10 | 49.89 ± 2.17 |
| 当前两层 GCN student，全部合法 pool，8192 步并由 selector 选 checkpoint | 否 | 51.31 ± 2.89 | 55.60 ± 1.92 |
| 当前 v4，约 5% 合成节点，同一两层 GCN student | 否 | 49.85 ± 3.00 | 60.77 ± 1.64 |

后三行来自已审计的 `results/external_tag_comparison_2026_09_17/summary.csv`，没有重跑或改变 v4。均值与总体标准差来自 student seeds 100..105。完整 pool 的训练图是合法 pool 节点诱导子图；评估使用完整真实图。8192 指优化器更新次数，不是 epoch，报告值来自独立 selector 选择的中间 checkpoint，不一定是第 8192 步。

| 官方 checkpoint 指标 | Cora | Citeseer |
|---|---:|---:|
| Node Macro-F1 | 58.98 | 57.76 |
| Node Accuracy | 67.53% | 64.58% |
| 正确分类节点数 / 报告节点数 | 183 / 271 | 206 / 319 |
| v4 均值减官方 checkpoint，百分点 | -9.13 | +3.01 |
| 完整 pool 8192-step-selected student 减官方 checkpoint，百分点 | -7.67 | -2.15 |

这些差值同时受到模型结构、预训练、图读出和图上下文处理方式影响，不能直接归因于凝聚，更不能据 Citeseer 的 +3.01 宣称 v4 普遍超过 GraphCLIP。

## 3. Macro-F1 到底测什么

这里测的是节点分类，不是图分类、链路预测或图文检索。每个报告节点与各类别的文本 prompt 匹配，最大相似度给出类别预测。

对类别 c，`F1_c = 2 TP_c / (2 TP_c + FP_c + FN_c)`；Macro-F1 对各类别 F1 等权平均，再乘 100。因此 58.98 不表示 58.98% 的节点分类正确；后者由 Accuracy 表示。

官方 checkpoint 的逐类结果如下，用于解释宏平均与准确率为什么有差距，不用于调整 prompt：

| Cora 类别 | 报告节点数 | F1 |
|---|---:|---:|
| case based | 31 | 70.83 |
| genetic algorithms | 45 | 82.69 |
| neural networks | 85 | 78.38 |
| probabilistic methods | 42 | 81.93 |
| reinforcement learning | 28 | 73.33 |
| rule learning | 12 | 25.71 |
| theory | 28 | 0.00 |

| Citeseer 类别 | 报告节点数 | F1 |
|---|---:|---:|
| agents | 56 | 78.13 |
| machine learning | 64 | 38.10 |
| information retrieval | 56 | 73.04 |
| databases | 70 | 74.36 |
| human computer interaction | 49 | 71.15 |
| artificial intelligence | 24 | 11.76 |

Cora 的 theory 几乎没有被预测为正确类别，Citeseer 的 artificial intelligence 也明显偏弱。这说明该“checkpoint + 固定 prompt + 推理适配”对不同类别表现不均衡；尚不能判断是 prompt、语义重叠、预训练覆盖还是其它因素导致。

## 4. 对 student 和训练目标的解释

当前 student 是简化的 GraphCLIP-style 两层 GCN，带可训练文本投影，不是官方 12 层 GPS。其双向 InfoNCE 对齐“节点与自身文本”，不是直接优化“节点属于哪个类别”的监督交叉熵。

同类节点可能在 batch 内互为负样本；训练使用节点长文本而推理使用短类别 prompt，也可能存在对齐偏差。这些是合理待验证机制，不是此次 checkpoint 推理已证明的原因。

本次结果能确认：

- Cora 的完整 pool student 比官方 checkpoint 低约 7.67 点，当前评估模型确实存在值得校准的性能空间。
- Citeseer 的对应差距只有约 2.15 点，而且 v4 student 高于该 checkpoint；不能笼统断言当前 student 在所有数据集都很弱。
- 两层 GCN、InfoNCE、图上下文、预训练和 prompt 都没有在这次实验中被单独控制，不能将差距全部归因于模型浅或 InfoNCE。
- 官方模型这一次只有约 58 分，不等于任务上限只有约 58 分。既有监督参考约为 Cora 84.10 / Citeseer 69.26，但它改变了标签使用、输出头、训练图和搜索协议，只能放在单独任务参考中。

## 5. 对凝聚效果应如何使用这些结果

凝聚质量的核心比较仍然是：固定同一个 student 和任务，比较“合成图训练”与“全部合法原始训练数据训练”。

| 同 student 指标 | Cora | Citeseer |
|---|---:|---:|
| v4 最终合成节点数 | 94 | 111 |
| 完整合法 pool 节点数 | 1895 | 2229 |
| 节点压缩后保留比例 | 4.96% | 4.98% |
| v4 / raw-long-selected 最终 F1 比值 | 97.16% | 109.28% |
| v4 / raw-long-selected 阶段平均 F1 比值 | 88.71% | 106.31% |

比值是相对于当前完整数据训练参照的性能保留率，不是理论信息保留率，也不能证明任意强 student 上同样成立。完整数据参照虽有更长训练和 selector 选点，仍未证明充分收敛或最优。超过 100% 也不能推广为合成数据优于所有完整数据训练。

官方 checkpoint 行只回答“与一个已发布预训练模型还有多远”，不作为凝聚性能保留率的分母。后续如需提高评估模型可信度，应固定 v4 memory，先在完整数据上校准 student，再用统一校准后的 student 重测 v4 与外部凝聚方法。本次没有启动任何新的训练。

## 6. 阶段切片，仅供诊断

这是同一个冻结 checkpoint 的评分切片，不是三个阶段分别预训练，也不是持续学习结果。每个切片只评分已见类别的报告节点。

| 数据集 | 候选类别范围 | 阶段 0 | 阶段 1 | 阶段 2 | 阶段均值 |
|---|---|---:|---:|---:|---:|
| Cora | 全部类别 prompt 竞争 | 84.28 | 80.47 | 58.98 | 74.58 |
| Cora | 仅已见类别 prompt 竞争 | 91.52 | 85.07 | 58.98 | 78.52 |
| Citeseer | 全部类别 prompt 竞争 | 65.05 | 70.11 | 57.76 | 64.30 |
| Citeseer | 仅已见类别 prompt 竞争 | 83.16 | 75.46 | 57.76 | 72.13 |

官方冻结模型没有按阶段训练，不能将阶段间下降称为遗忘。

## 7. 验收与复现

- 逐节点保存 node IDs、真实报告标签、logits、预测类别和子图规模。
- 两个图均验证报告节点精确等于当前 `% 10 == 3` 划分，标签与原始数据对应行一致。
- 独立用 sklearn 从预测重算最终 Macro-F1/Accuracy，与存储结果一致。
- 每个数据集对 12 个输入/代码文件核验 SHA256，包括 checkpoint、原始数据、官方图模型、协议和 tokenizer 文件。
- 图/文本 state_dict 的完整张量哈希在推理前后不变，包含 BatchNorm buffers。
- batch 8 与 batch 1 全部节点的预测类别、所有阶段指标完全相同；最大 logit 差分别为 `1.49e-7` / `1.34e-7`。
- Cora 最大二跳子图 426 节点，有 2 个子图超过 256；Citeseer 最大 256。主结果没有退回一跳。
- 本地测试 7/7 通过，覆盖 split 隔离、checkpoint 严格加载、二跳子图中心索引、Macro-F1 定义、状态哈希和两个数据集结果复算；`git diff --check` 通过。
- 按用户授权停止 `beihang-chat-vllm.service`，未删除模型或修改服务配置；评估后服务保持停止。

代码：`code/v2_llm_fusion/eval_official_graphclip_report.py`。

远程运行示例，Citeseer 对应替换 dataset 和 output；脚本拒绝覆盖已有结果：

```bash
/home/pc/miniconda3/envs/tagsam/bin/python \
  /home/pc/research/TAG_condensation/code/graphclip_unified_v3_2026_09_13/eval_official_graphclip_report.py \
  --dataset cora --batch-graphs 8 --device cuda:0 \
  --output /home/pc/research/TAG_condensation/results/official_graphclip_report_2026_09_17/cora.json
```

本地复算测试：

```bash
python -m pytest -q code/v2_llm_fusion/test_official_graphclip_report.py
```

结果位于 `results/official_graphclip_report_2026_09_17/`：`cora.json`、`citeseer.json`，以及 `audit/*_batch1.json`。

## 8. 对旧记录的修正

2026-09-09 报告的 57.47 / 62.90 来自 `% 10 == 1` 的旧 dev，并包含大子图退回一跳规则；其中凝聚 student 也不是当前 v4。不能将旧数值直接与当前 report-dev 的 v4 结果相减。

旧报告“绝对值疑虑基本关闭”“54-61 是任务水位”等表述证据不足，不再作为当前结论。本次新的 58.98 / 57.76 只是固定 checkpoint 在当前划分、固定 prompt 和本地推理适配下的实测值。
