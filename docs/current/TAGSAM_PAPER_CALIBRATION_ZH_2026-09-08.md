# TAGSAM 论文报告值与本地复现值校准表

日期：2026-09-08
用途：P1 校准任务产出。将 TAGSAM 论文（WWW 2026, arXiv 2606.03839）原文报告的 Cora 数字与本项目已完成的官方代码复现值并排对比，判断静态实现是否可信。
数据来源：论文 arXiv HTML 版正文 Table 2 / Table 5 / 4.1 Experiment Settings；本地复现值取自 `OFFICIAL_TAGSAM_STATIC_REAUDIT_2026-09-07.md`。

## 1. 论文的评测协议（先确认口径）

论文原文关键设定：

- **评测协议**：五类零样本分类（five-way zero-shot），遵循 G2P2 (Wen & Fang, 2023) 的 setting，报告 5 次独立运行的分类 Accuracy；
- **下游**：GCN + BERT + projector（与本项目 native 接口一致）；
- **方法组成**：TAGSAM = subgraph text **S**election + **A**ttribute similarity **M**atching（ASM）。**注意：论文完整方法包含文本选择组件**；
- **Cora 数据**：2708 节点、5429 边、7 类（Table 5），但主表评测按五类 zero-shot 协议进行；
- ASM 匹配目标：student 与 teacher 的相似度矩阵（graph embedding × text embedding 的成对乘积），行 softmax 后做对称 BCE。

## 2. 论文报告值（Table 2, Cora, Accuracy %）

| Size | Random | Herding系 | K-Center系 | (其他baseline) | MTT-TAG | TAGSAM | Whole |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 50.0±2.5 | 33.4±2.7 | 47.3±2.6 | 40.1±3.9 / 64.4±3.4 | 62.8±1.6 | **73.8±1.8 (↑9.4)** | 78.7±0.5 |
| 100 | 68.1±1.4 | 34.4±2.1 | 61.9±1.1 | 49.7±3.6 / 73.8±0.9 | 71.1±1.7 | **75.2±1.1 (↑1.4)** | 78.7±0.5 |
| 200 | 71.8±1.2 | 43.8±0.9 | 68.3±1.1 | 55.2±2.9 / 76.1±0.5 | 72.4±1.2 | **76.6±0.8 (↑0.5)** | 78.7±0.5 |

（baseline 列名以论文原表为准，此处按数值顺序摘录；核对时以 size=50 行的 TAGSAM=73.8 与 Whole=78.7 为关键锚点。）

## 3. 本地复现值（官方代码，size=50）

| 链路 | 协议 | step 0 | step 500 | 备注 |
|---|---|---:|---:|---|
| official master | 五类（原版 evaluator） | 42.75 | 38.24 | step500 反而下降；legacy step1000=46.17 |
| official add-adj | 五类（原版 evaluator） | 51.15 | 52.93（独立5次重训） | 运行时单次 55.56 |
| minimal repair | 七类 | 33.40/33.81 (s44/s45) | 36.24/38.74 | 非论文口径，仅内部参照 |
| 七类 full teacher | 七类 | — | 61.12 | 非论文口径 |

## 4. 校准结论

**同口径（五类、size=50）对比：论文 TAGSAM=73.8±1.8，本地官方 master 复现=38.2–42.8，add-adj=51.2–52.9。差距约 21–35 pp，远超随机波动。复现未达论文效果，且差距大到必须归因。**

候选解释（按可能性排序）：

1. ~~**文本选择组件缺失**~~ **已排除（2026-09-08 复核）**：`distill.py` 第 81 行确实调用 `select_text(graph_dataset, args)` 并把 `selected_text` 存入 artifact，text selection 流水线在本地运行中被执行（注意它读取 cache 文件，cache 指纹需纳入审计）；
2. **five-way 任务采样方式差异**：G2P2 的 five-way zero-shot 通常按 episode 采样 5 个类别子任务再平均，官方 evaluator 写死"前五类"可能与论文的任务构造不同，导致不可比；
3. **超参/训练预算差异 + 外层优化本身弱**：**新证据（2026-09-08 日志分析）：official master 500 步 distill 的 syn_loss 仅从 ~0.730 降到 ~0.696（约 4.7% 相对下降），外层优化几乎没有有效降低目标函数**。legacy step1000=46.17 高于 step500=38.24、step5000=30.70 崩溃的非单调轨迹与此一致。这直接支持"优化失败"假设：不是 loss 降了不泛化，而是 loss 本身几乎不降；
4. **论文数字本身在 released code 上不可复现**：不能排除，但应在 2–3 排除后才主张。

## 5. 对项目的直接影响

1. **"静态 ASM 优化弱"这一判断在论文口径下同样成立的证据尚不充分**——本地复现与论文之间存在实现层面的未解释差距，当前 native 协议下 "Offline ASM < Frozen compact" 的结论需要在排除解释 1–3 后才能与论文对话；
2. **静态 gate（P2）必须先做诊断**：区分"优化失败"（loss 不降）与"泛化失败"（loss 降但 dev 掉）。master 五类 step0→500 下降、step1000 回升、step5000 崩溃的非单调轨迹强烈提示外层优化不稳定，与本项目 native 观察一致；
3. **叙事层面**：即使最终无法复现论文值，"TAGSAM 报告的五类收益在全类别协议与独立重训下不能保持"本身就是可以写入论文的复核观察，但必须先完成解释 1–3 的排除实验。

## 6. 待办（校准的后续动作）

- [ ] 核对官方仓库中 text selection 流水线是否被本地运行执行；确认 `cora.pt` 的文本字段是 selection 输出还是原始文本；
- [ ] 核对论文 five-way 的任务构造（episode 采样 vs 固定前五类）；
- [ ] 用 legacy artifact 检查 step 1000–2000 区间是否存在更优选择点（论文可能在更长训练+早停下报告）；
- [ ] 上述完成后更新本文档的结论部分。

## 来源

- 论文：[arXiv 2606.03839](https://arxiv.org/abs/2606.03839)，WWW 2026 proceedings pp.754–764；
- 本地复现：`docs/current/OFFICIAL_TAGSAM_STATIC_REAUDIT_2026-09-07.md`；
- 复现 artifact 与日志：`results/official_static/`。
