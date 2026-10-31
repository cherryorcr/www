# 原生持续 TAG 的细粒度归因

日期：2026-09-07

本文记录在 `native_tag_v1` 上补做的两个单因素诊断，回答 stage 2 性能下降主要来自旧 memory、当前新增块，还是旧新联合关系。所有结果都是开发 checkpoint 证据，`confirmation_accessed=false`，不作为最终留出测试。

## 1. 固定协议

| 项目 | 设置 |
|---|---|
| 数据 | Cora，七类 |
| 阶段 | `[0,1] -> [2,3] -> [4,5,6]` |
| 目标阶段 | stage 2 |
| 原始训练根数 | 1891 |
| 累计预算 | 5%，94 行 |
| 固定旧 memory | 每个凝聚 seed 自己的 stage-1 Append memory，67 行 |
| 新增预算 | 27 行，来自 stage-2 的 532 个原始训练节点 |
| 下游 | 两层 GCN，冻结 BERT CLS，可训练文本投影，图文对比训练 |
| student | seed 100、101，512 steps，每 32 steps 检查 |
| 图结构 | 每个输入图内部诱导边；拼接后是 disjoint union，没有跨阶段边 |
| 标签权限 | 凝聚、teacher、student 不读取类别标签；标签只由评估器计算 checkpoint F1 |

旧 memory、文本、边、mass、origin ID 和新增行数均有哈希记录。两个诊断都没有读取 confirmation/test 分区。

## 2. 诊断 A：只改变旧 X 是否允许更新

### 2.1 设计

固定同一个旧 memory、stage-2 原始增量、27 行新增初始化、文本、拓扑、teacher 和 reference。比较：

1. `all-X`：旧行和新行都允许 ASM 更新；
2. `new-X-only`：旧行精确冻结，只更新新增行；
3. `initial`：同一初始图，不做 ASM 更新。

每个优化分支使用同一个 condensation seed 和同一个下游验证协议；同时保存选中 snapshot、最佳 snapshot 和最后迭代 snapshot，并计算旧行/新行的相对 L2 漂移。

### 2.2 结果

表中 F1 是两个独立 student 的 checkpoint Macro-F1 均值，单位为百分比。

| 凝聚 seed | Initial | All-X | New-X-only | All-X − New-X-only |
|---:|---:|---:|---:|---:|
| 44 | 44.75 | 45.05 | 43.59 | +1.46 |
| 45 | 34.59 | 34.59 | 36.00 | −1.40 |
| 两 seed 均值 | 39.67 | 39.82 | 39.79 | +0.03 |

选中产物的旧行漂移如下：

| 凝聚 seed | All-X 旧行相对漂移 | New-X-only 旧行相对漂移 | All-X 选中 outer step | New-X-only 选中 outer step |
|---:|---:|---:|---:|---:|
| 44 | 0.680 | 0 | 20 | 100 |
| 45 | 0 | 0 | 0 | 100 |

最后迭代没有经过 validation 选择，旧行相对漂移分别为：

| 凝聚 seed | All-X 最后迭代旧行漂移 | New-X-only 最后迭代旧行漂移 |
|---:|---:|---:|
| 44 | 1.210 | 0 |
| 45 | 1.299 | 0 |

### 2.3 解释

- ASM 确实能够显著改变旧 synthetic X；在最后迭代上，旧行平均相对漂移约 1.21–1.30。
- 但 validation 选择会在不同 seed 上接受完全不同的状态：seed 44 接受了有旧行漂移的 all-X 状态，seed 45 选择了 all-X 的 step 0。
- 旧行更新带来的 F1 方向相反，两个 seed 的均值差只有 0.03 个百分点。现有证据不能把性能下降归因于“旧 X 没有更新”，也不能把“冻结旧 X”称为已验证的解决方案。
- 更直接的结论是：联合 ASM 的特征轨迹不稳定，且 validation/student 选点噪声足以改变是否接受更新。必须把“允许旧行更新”和“更新后是否被选中”分开报告。

## 3. 诊断 B：固定旧 memory，比较新增块质量

### 3.1 设计

固定同一个 stage-1 Append memory、同一个 27 行预算和相同评测。比较五个输入：

1. `raw-matched`：从 532 个 stage-2 原始节点中抽取 27 行，保留真实 X、真实文本和诱导拓扑；
2. `compact-frozen`：用 joint graph-text 初始化得到 27 行 synthetic X 和真实代表文本，冻结 X；
3. `new-only-ASM`：从同一个 compact 初始化出发，只更新新增 X；
4. `joint-ASM`：同一个 compact 初始化，旧、新 X 都可更新；
5. `protected-joint`：同一个 joint teacher 和初始化，只更新新增 X，旧 X 冻结。

`raw-matched` 与 compact 分支同时改变了代表选择、X、文本和拓扑，因此它是“新增压缩块是否保留训练效用”的组成对照，不是单字段因果分解。后面三个分支共享 compact 初始化，适合比较 X 更新和联合训练的方向。

### 3.2 结果

跨 seed 使用同一组 27 个 raw origin ID。F1 为两个 student 的 checkpoint Macro-F1 均值。

| 凝聚 seed | Raw-matched | Compact-frozen | New-only ASM | Joint ASM | Protected-joint |
|---:|---:|---:|---:|---:|---:|
| 44 | 40.08 | 44.66 | 42.87 | 44.69 | 43.45 |
| 45 | 32.33 | 34.59 | 35.47 | 34.59 | 36.00 |
| 两 seed 均值 | 36.21 | 39.63 | 39.17 | 39.64 | 39.72 |

相对于 `compact-frozen` 的配对差值：

| 对照 | seed 44 | seed 45 | 两 seed 均值 |
|---|---:|---:|---:|
| Compact-frozen − Raw-matched | +4.58 | +2.26 | **+3.42** |
| New-only ASM − Compact-frozen | −1.79 | +0.88 | −0.46 |
| Joint ASM − Compact-frozen | +0.03 | 0.00 | +0.02 |
| Protected-joint − Compact-frozen | −1.21 | +1.40 | +0.10 |
| Joint ASM − Protected-joint | +1.24 | −1.40 | −0.08 |

逐阶段均值也显示同样的方向不稳定：

| 输入 | stage 0 F1 | stage 1 F1 | stage 2 F1 |
|---|---:|---:|---:|
| Raw-matched | 84.61 | 49.99 | 22.12 |
| Compact-frozen | 79.43 | 56.93 | 34.79 |
| New-only ASM | 78.82 | 58.52 | 30.83 |
| Joint ASM | 79.98 | 56.08 | 31.83 |
| Protected-joint | 78.77 | 59.81 | 30.74 |

### 3.3 解释

- 在同一 94 行预算下，压缩后的新增块平均比随机 raw 子集高 3.42 个百分点，说明“27 个 raw 行太少”不能代表新增信息；joint graph-text 初始化至少保留了一部分增量训练效用。
- 但 compact block 仍不是 full-history 的替代品。此前 stage-2 组成诊断中，Append old 下 `old + new compact` 为 42.98%，`old + new raw over-budget` 为 47.39%，Full-history 为 51.16%。新增块的有限预算凝聚仍有明显缺口。
- 从同一个 compact 初始化继续做 new-only ASM，两个 seed 一正一负，均值略低于冻结初始化；当前 ASM 没有稳定的新增块优化收益。
- Joint ASM 的均值几乎等于冻结初始化，Protected-joint 只高 0.10 个百分点。现有结果看不出一个稳定的“旧新联合交互惩罚”，也不足以证明 joint 能恢复历史响应。

## 4. 与既有组成归因合并后的判断

此前固定 stage-1 memory 的 stage-2 组成诊断给出：

| 旧 memory | Old-only 累计 F1 | Old + new raw（超预算） | Old + new raw（同预算） | Old + new compact（同预算） |
|---|---:|---:|---:|---:|
| Append old | 38.70 | 47.39 | 40.08 | 42.98 |
| Union old | 26.40 | 44.40 | 31.87 | 33.18 |

对应的历史 stage-1 F1 为：Append old 的 `68.09 -> 68.79`（compact 组合），Union old 的 `43.68 -> 39.25`。新 raw block 相同，替换旧 memory 后性能明显改变，说明旧 memory 的质量会传递到下一阶段。

因此目前最符合证据的归因排序是：

1. **旧 memory 表达能力和递归路径质量：高可信。** Union old 在历史阶段明显弱于 Append old，且差异在同一新增块下仍然存在。
2. **新增块有限预算下的凝聚缺口：中高可信。** 同预算 compact 优于 raw 子集，但低于超预算 raw 和 full-history；这说明新增块有信息，但没有保留全部训练效用。
3. **旧 X 是否更新：尚无稳定方向。** 最后迭代可以产生很大的旧行漂移，选中产物却可能选择 no-op；跨 seed 的 F1 差异接近 0。
4. **缺少跨阶段边：尚未证明是主因。** 当前所有分支都使用 disjoint union，而 compact 与 raw matched 的差异已经在没有跨阶段边时出现。跨阶段边仍是待测因素，不能直接作为下一版方法的默认解释。
5. **旧新联合后的额外交互损失：证据不足。** Joint/protected 的差异方向相反，且均值小于 0.1 个百分点量级；目前更像是压缩质量和优化/选点噪声叠加，而不是已经识别出的独立 interaction term。

这些差异不能相加成严格的因果分解，因为旧 memory、代表选择、文本、X、拓扑和 reference 组成在部分对照中同时变化。

## 5. 研究决策与下一步

### 已做决策

- 暂不加入跨阶段边、transport、GraphCLIP 或新的多模块 Joint 变体。
- 将 `compact-frozen` 作为新增块质量的开发基线，将 `protected-joint` 和 `joint-ASM` 作为持续更新对照，不能把它们命名为已验证方法。
- 把 ASM 的最后迭代漂移和 validation 选中状态同时保存；只报告选中产物会掩盖优化不稳定性。

### 下一轮只做的两个补充

1. **把新增块的字段拆开。** 在同一 27 个代表和同一拓扑下，加入 `real-representative X` 与 `centroid X` 的冻结对照，区分 synthetic X 聚合误差和文本/拓扑选择误差。
2. **固定预登记 outer step 做多 student 响应曲线。** 不用单个 validation student 选择 ASM 产物，直接在 step 0/20/40/60/80/100 评估旧阶段、新阶段和累计 F1，观察哪个阶段先恶化以及是否与 X 范数漂移同步。

只有当这两步显示“旧、新块各自有效，但联合时额外下降”时，才测试一个固定 node fields 的跨阶段关系 oracle。若没有额外交互损失，方法重点应转向提高新增块的训练效用和递归 memory 质量，而不是先补跨阶段边。

## 6. 复现入口

- 细归因 runner：[run_fine_attribution.py](../../code/native_tag_v1/run_fine_attribution.py)
- 既有 artifact 漂移分析：[analyze_existing_fine_attribution.py](../../code/native_tag_v1/analyze_existing_fine_attribution.py)
- 汇总脚本：[summarize_fine_attribution.py](../../code/native_tag_v1/summarize_fine_attribution.py)
- 汇总 JSON：[fine_attribution_aggregate_same_raw.json](../../results/native/fine_attribution_aggregate_same_raw.json)
- old-X 配对运行：`results_mirror/fine_old_update_r05_c44_s2_v3/`、`results_mirror/fine_old_update_r05_c45_s2_v3/`
- 新增块质量运行：`results_mirror/fine_new_quality_r05_c44_s2/`、`results_mirror/fine_new_quality_r05_c45_s2_same_raw/`
- 前一组组成归因：[STAGE2_SOURCE_COMPOSITION_ATTRIBUTION_2026-09-07.md](STAGE2_SOURCE_COMPOSITION_ATTRIBUTION_2026-09-07.md)

本轮代码与结果均保留 SHA256；本地 `native_tag_v1` 的 28 个单元测试全部通过。
