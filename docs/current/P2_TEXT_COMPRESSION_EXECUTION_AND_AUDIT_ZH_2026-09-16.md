# P2 文本压缩：执行与验收记录

更新时间：2026-09-16。状态：Cora/Citeseer 小图严格轮和扩展审计已完成；性能门禁未通过，不上大图。

## 1. 本轮范围与退出规则

本轮先完成 Cora/Citeseer 小图，不在小图门禁通过前扩展大图。沿用 P1 的 fixed-rate 0.05、dispersion、三阶段、合成节点、frozen-old 与 batch 16，student seeds 为 100–105。为隔离文本机制，暂不启用 response allocation，也不改变合成节点特征、配额和拓扑。

自动检查每 30 分钟推进一次，工作窗口截至北京时间 2026-09-16 07:00；提前完成验收即可停止。结果必须区分“实现与实验完整性通过”和“新机制的性能优势通过”，不能用前者代替后者。

## 2. 已发现并修正的设计问题

1. 第一轮实现限制的是字符数，不是 tokenizer token 数。其 30 组结果保留在 `results/p2_text_matrix_2026_09_16`，仅属于字符预算预实验。
2. 第一轮 pooling 使用完整原文 embedding，保存的短文本不能代表真实输入量；其 dual prototype 的多样性分支也没有真正重新编码预算内片段。不能由这一轮得出严格等信息预算下的机制优劣。
3. 第一轮 `set_pool` 是距离加权的手工变体，不能称为训练过的 set encoder。严格轮把无参数 attention pooling 和真正训练的 DeepSets 分开。
4. `g3/c192` 与 `g8/c512` 同时改变文本数量和预算，不能用于归因。严格轮分别比较固定 64 tokens 的 g=3/5/8，以及固定 g=5 的 32/64/128 tokens。

以上问题均不通过覆盖旧文件掩盖；新输出使用独立目录和代码 hash。

## 3. 严格 token 实验协议

- 用实际 MiniLM tokenizer 按源片段分配预算，短文本剩余额度重新分配；拼接后与逐片段编码两种口径均必须不超过上限。
- 主比较为每个合成节点最多 64 个内容 tokens。预算是相同上限，不伪造不存在的文本以强行填满，实际总量逐行审计。
- 特殊 tokens、编码序列数和训练成本单独报告；多序列 pooling 不声称与单序列拼接等计算量。
- 构造聚类与代表选择共享 pool-only 源 embedding 预处理。这个共同预处理使用原文，不属于“压缩后目标的直接编码额度”，也不能被隐藏成整条 pipeline 只读取了 K×64 tokens。
- 每一阶段直接复用字符版 concat reference 的合成特征、mass、stage、depth 和 edge_index，并重建源簇核对特征一致。文本及其表示才是干预变量。
- 历史文本、历史表示和历史边不更新；新文本仅来自当前阶段的源簇。合成节点特征仍为簇均值，不把真实选点替换成凝聚节点。
- selector-dev 为 id%10=1，report-dev 为 id%10=3，test 为 id%10=2；所有三类节点均排除出凝聚源。P2 本轮只报告 report-dev，不读取 test，不使用 student F1 拟合文本编码器。

## 4. 实验矩阵

每个数据集 17 组确定性实验：center concat；MMR diversity concat 的相关性权重 0.25/0.5/0.75；dual prototype 的混合权重 0.25/0.5/0.75；mean pooling；距离 weighted pooling；无参数内容 attention pooling；no-reencode；no-fuse；no-text；固定 token 下 g=3/8；固定 g 下 token cap=32/128。

no-reencode 在公平版本中定义为“不重新编码拼接文本，平均预算内片段的冻结编码”，因此它与 mean pooling 数学等价。这是刻意的实现一致性检查，不算两种独立新方法。

另补真正可训练的 DeepSets：逐片段冻结 MiniLM，使用共享 phi/rho 网络进行带 mask 的集合聚合；只用当前 delta 簇上的同预算 concat embedding 蒸馏；保留 20% 源簇作 checkpoint 验证，不读取任何 dev/test 标签。每阶段 300 步，construction seeds=700/701/702，student seeds 仍为 100–105；不挑最好 construction seed。源 teacher 编码和集合网络训练是额外成本，必须计入。

## 5. 自动验收

`audit_p2_token_controls.py` 检查结果数量、阶段完整性、reference hash、固定配额/steps/batch、测试封存、实际 token 超限、所有特征与边完全一致、历史文本与表示不变，以及逐行审计与 artifact 一致。

性能报告包含三个阶段累计 F1、阶段均值、最终旧类/新类 F1、按历史任务计算的 forgetting、6 个 student seeds 的配对增益及 t 区间、文本输入 tokens、特殊 tokens、编码序列和耗时。配对区间仅反映初始化不确定性，不代表跨类别顺序或跨数据集置信区间，也未校正多重比较。

对跨两个小图均为正均值但区间尚跨零的候选，可以作独立 seeds/扰动的诊断复核，但这不等于通过门禁。只有确认后跨两图都有可信正增益、阶段保持没有不可接受退化，才考虑扩展大图；最终不因某个数据集或某个 seed 的最好路径而宣称通过。预算扩张结果独立报告，不能当作相同 memory 成本的方法增益。

## 6. 当前交付物

- `code/v2_llm_fusion/p2_text_operators.py`：token 预算与确定性算子。
- `code/v2_llm_fusion/run_p2_token_controls.py`：同节点、同边的严格矩阵。
- `code/v2_llm_fusion/audit_p2_token_controls.py`：结果与 artifact 验收。
- `code/v2_llm_fusion/p2_learned_set.py` 和 `run_p2_learned_set.py`：真正可训练的集合编码。
- `test_p2_text_operators.py` 的 6 项单元测试已在本地与远程 tagsam 通过；`test_p2_learned_set.py` 的 2 项本地测试已通过。

最终表、门禁结论、失败归因及下一步见下文；性能门禁保持未通过，不提前升级主线。

## 7. 当前结果与门禁结论

严格 64-content-token、固定 rows/topology 的最终阶段累计 F1 如下。这里的 baseline 是同一严格 token 轮的 center concat，不是旧版字符预算结果。

| 构造 | Cora | Citeseer |
|---|---:|---:|
| center concat | 50.64 | 56.03 |
| diversity concat, lambda=.75 | 52.20 | 56.82 |
| mean pooling | 53.06 | 57.34 |
| distance weighted pooling | 53.28 | 56.42 |
| set attention pooling | 52.95 | 58.40 |
| dual prototype, alpha=.25/.50/.75 中最好 | 50.46 | 55.04 |
| no-fuse | 41.77 | 54.52 |
| no-text diagnostic | 4.30 | 5.49 |

首轮 6 个 canonical student seeds 中，mean pooling 相对 baseline 的终阶段配对增益为 Cora `+2.42`、Citeseer `+1.32`；但独立 seeds 400–405 后，Cora 变为约 `+1.10`、Citeseer 约 `+1.37`，阶段平均增益在 Cora 为负。set attention 在 Cora 独立确认中反而下降，说明首轮最好值不是稳定机制收益。当前没有候选同时满足跨数据集、独立 seed、阶段平均和旧类保持门禁。

真正训练的 DeepSets 也没有解决问题：Cora 三个 construction seeds 的最终累计 F1 为 `51.89/51.31/50.29`，Citeseer 为 `56.51/57.44/56.99`。它只在当前 delta 簇上用 source holdout 拟合 MiniLM 目标，未使用 dev/test 标签；当前结果更适合作为“文本集合蒸馏尚未带来可重训优势”的负结果。

压缩率敏感性进一步否定了无条件推广：在 2.5% rate 下，Cora strict concat/mean/set 的终阶段约为 `48.42/52.46/51.61`，Citeseer 约为 `44.68/44.60/42.83`。因此不能把 5% rate 下的正向候选宣传为普适的文本机制收益。

## 8. 重要解释

严格矩阵的 `old_mean_f1` 与 `new/current_f1` 是当前 v3 open-vocabulary 诊断：样本按真实 seen 类别取 mask，但模型 argmax 仍在全部类别 prompt 上竞争。补充的 seen-prompt-only 诊断显示，早期阶段两种口径差异可达十多个百分点；论文主表必须同时说明 prompt scope，不能把现有 early-stage 数值直接叫作标准 closed-set class-incremental F1。

排列实验还显示 student 对极小的浮点表示差异很敏感。原样 artifact 和同顺序重编码 bitwise 复现；规范排序能保证集合输入顺序一致，但“集合数学不变”不自动意味着长程 student F1 不变。因此后续默认应使用规范排序、固定 CPU/GPU kernel 设置和 full-step repeat，不能把 permutation invariance 当作性能稳定性证明。

综合结论：P2 的实现完整性、预算公平性和可审计性通过；P2 的“找到跨数据集稳定更优文本算子”未通过。v4 主线继续使用经过验证的简单 center concat，P2 新算子只作为负结果/诊断，不合并进默认方法，也不启动大图主表。

最终自动汇总在 `results/p2_acceptance_2026_09_16/acceptance.json`。它记录：严格矩阵 40 组、独立确认 36 组、数值 guard 20 组、2.5% rate 扩展 8 组、prompt-scope 诊断 6 组、legacy 回归 4 组；共 2052 次 archived student fit，所有 artifact/hash/token/sealing 检查为 0 error。`independent_confirmation_pass` 为空，`large_graph_gate_passed=false` 是有意的门禁结果，不是任务失败。

恢复默认 v4 拼接兼容性也通过了回归：append-fixed、5% rate 的终阶段为 Cora `49.8518`、Citeseer `59.1747`，没有因 P2 实验改写旧主线。字符预算的第一轮 30 组仍保留为 exploratory pilot，不计入严格 token 验收。

## 9. 下一步

优先修正并冻结主指标的 prompt scope，然后在同一 closed/open 双口径下重做独立确认。若仍无跨阶段优势，文本方向收敛为“固定预算下的可靠 center concat + 负结果”，创新重点回到 student response、topology-preserving append 和动态 memory allocation；DeepSets、dual prototype 及未通过门禁的 adaptive text selector 不进入主线。

## 10. 2026-09-16 收尾决定

P2 本轮封存，不再通过追加候选或挑选最高 dev 分数晋升文本模块。默认 v4 保持历史字符版 center concat；64-token concat 保留为公平文本消融基线，两者分开引用。

小图对照已建立独立的 all-class / seen-class prompt 指标，并修复全局类别映射和累计 F1。旧类/新类改为全部已见评价样本上的 per-class F1，计入跨任务 false positives；它们与上文历史任务切片诊断不混用。学生训练参数与 canonical 代码逐位一致，Cora/Citeseer 的 legacy dispersion、真实 v4 和 concat64 已回归到归档终阶段分数。

完整设置和已有比较的弃用原因见 [P2 收尾与小图对照冻结协议](P2_CLOSEOUT_SMALL_COMPARISON_PROTOCOL_ZH_2026-09-16.md)。本轮未评估 test 标签；这不表示项目历史从未访问这些数据集的 test，旧工作包已有相关开发使用记录。
