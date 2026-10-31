# Stage 2 原生 TAG 组成归因

日期：2026-09-07。本文是 `native_tag_v1` 的开发诊断补充，目标是把三阶段实验中“旧记忆、当前增量和新增压缩块”的影响拆开。所有结果都只读取 checkpoint 分区，`confirmation_accessed=false`；它们不能替代最终留出测试。

## 为什么补做这一组

已有三阶段结果显示，在 5% 预算、凝聚 seed 44 下，stage 1 的 `Append` 明显高于 `Union`，而 stage 2 的所有压缩方法都低于 `Full-history`。这仍然不能回答损失来自哪里：

- 旧 synthetic 本身是否已经不能表达历史训练效用；
- 新增原图在同预算下是否比新增压缩块更有用；
- 把旧 memory 和新块放在同一个训练图中时，是否出现新的交互损失。

原来的 source 归因只覆盖 stage 1，而且 `old memory + raw delta` 是超预算诊断。这里固定 stage 1 memory，补做 stage 2，并加入同预算的 raw 子集控制。

## 固定条件

| 项目 | 设置 |
|---|---|
| 数据 | Cora，七类，`prepare_v2` |
| 阶段 | `[0,1] -> [2,3] -> [4,5,6]`，本次目标 stage 2 |
| 压缩率 | 5% 累计训练 roots；stage 2 总预算 94 |
| 旧 memory | 67 行的 stage 1 artifact，分别取递归 `Append` 或 `Union` |
| 新增 raw | stage 2 的 532 个训练节点；同预算控制随机抽 27 行 |
| 新增 compact | 从递归 `Append` stage 2 artifact 提取 27 行，并重新与固定旧 memory 拼接 |
| student | 独立 seed 100、101，512 steps，每 32 steps 选 checkpoint |
| 图结构 | 每个输入图内部诱导边；拼接时为 disjoint union，无跨阶段边 |
| 标签权限 | 训练和凝聚不读取类别标签；标签只由评估器计算 checkpoint Macro-F1 |

`old memory + new raw over-budget` 使用 599 行，目的是观察原始增量是否提供额外训练信息；它不是公平的压缩率对照。`old memory + new raw matched` 和 `old memory + new compact` 都是 94 行，才是同预算的组成比较。

## 结果：旧记忆取自 Append

以下是两个独立 student 的 checkpoint Macro-F1 均值（单位：百分比）：

| 输入组合 | 行数 | 累计 Macro-F1 | stage 0 | stage 1 | stage 2 |
|---|---:|---:|---:|---:|---:|
| Full-history raw | 1891 | 51.16 | 86.71 | 73.35 | 39.52 |
| old memory only | 67 | 38.70 | 82.43 | 68.09 | 14.95 |
| new raw only（超预算） | 532 | 41.04 | 72.93 | 60.88 | 29.93 |
| old + new raw（超预算） | 599 | 47.39 | 82.33 | 70.75 | 33.39 |
| old + new raw（同预算 27 行） | 94 | 40.08 | 86.67 | 65.59 | 16.97 |
| old + new compact（同预算 27 行） | 94 | 42.98 | 83.49 | 68.79 | 27.47 |

这里 `old + new compact` 与原递归 `Append` stage 2 的结果一致，因为它使用同一新增压缩块，只重新显式固定旧部分。相对于同预算 raw 子集，新增 compact 提高累计 F1 `+2.91` 个百分点，stage 2 F1 提高 `+10.50` 个百分点；这说明“27 个 raw 行太少”本身会造成很大损失，新增压缩块至少保留了一部分增量信息。

但它仍比超预算的 `old + new raw` 低 `4.40` 个百分点，也比 Full-history 低 `8.17` 个百分点。因此新增块凝聚质量是一个真实缺口，不能把全部损失归结为旧 memory 被改写。

## 结果：旧记忆取自 Union

| 输入组合 | 行数 | 累计 Macro-F1 | stage 0 | stage 1 | stage 2 |
|---|---:|---:|---:|---:|---:|
| Full-history raw | 1891 | 51.16 | 86.71 | 73.35 | 39.52 |
| old memory only | 67 | 26.40 | 78.57 | 43.68 | 4.42 |
| new raw only（超预算） | 532 | 41.04 | 72.93 | 60.88 | 29.93 |
| old + new raw（超预算） | 599 | 44.40 | 79.37 | 66.87 | 32.60 |
| old + new raw（同预算 27 行） | 94 | 31.87 | 76.66 | 51.95 | 14.96 |
| old + new compact（同预算 27 行） | 94 | 33.18 | 79.90 | 39.25 | 32.71 |

更换旧 memory 后，新增 raw 本身完全相同，但 `old memory only` 的累计 F1 从 38.70 降到 26.40，`old + new compact` 从 42.98 降到 33.18。stage 1 的历史保持尤其明显地恶化（68.09 -> 39.25），说明旧 memory 的质量和历史表达能力是递归性能的重要来源。

同时，Union old 下 `old + new compact` 的 stage 2 F1（32.71）并没有像 stage 1 那样崩溃；这说明“旧记忆较差”与“新块凝聚较差”是两个不同维度，不能只用一个遗忘解释全部现象。

## 目前可以排除和保留的解释

**较强的证据：**

1. **旧记忆替代确实造成损失。** Append old 和 Union old 在相同新块、相同 student seeds 下产生明显不同的历史 F1。旧 memory 的内容质量会传递到下一阶段。
2. **新增块凝聚也确实有限。** 在固定 Append old、同预算 27 行时，新增 compact 优于随机 raw 子集，但仍显著低于超预算 raw 增量和 Full-history。
3. **损失不是单纯由“两个小图没有跨阶段边”解释。** 当前所有拼接图都明确是 disjoint union；同预算 compact 相比 raw 子集的差异已经在没有跨阶段边的条件下出现。跨阶段边是否有益仍是未测试因素。
4. **参考行数失衡暂时不是主解释。** 先前 reference-quota probe 与 uniform Union 完全相同；本次 stage 2 组成诊断没有改变 reference policy，也没有证据推翻这一点。

**仍不能确定的部分：**

- 同预算 raw 子集与 compact 块的比较同时改变了代表选择、文本和 synthetic X，不能把 `+10.50` 个 stage 2 F1 点归因于某一个字段。
- `old + new raw over-budget` 使用 599 行，不能与 94 行方法作严格效率比较。
- Append old 与 Union old 本身来自不同递归路径；它们的差异说明 memory quality 有影响，但不是单一组件的随机化实验。
- `Append-joint` 与 `Append` 在两个凝聚 seed 上的平均差异不到 1 个百分点且方向不一致，现有证据不支持把 joint 交互称为已验证机制。

## 对原问题的暂时回答

在当前协议下，stage 1 的主要掉点更像是**旧 memory 被重新组织/替代**与**Union 重凝聚同时发生**；stage 2 则是三者叠加：旧 memory 质量、有限预算下的新块凝聚，以及阶段累计后的训练响应偏移。现有结果不支持把原因简化成“只缺少两个小图之间的边”。

下一轮应做两个单因素实验：

1. 固定同一个旧 memory 和同一个新初始化，只改变是否更新旧 X，比较 old-row 的训练响应漂移，而不是只看最终 F1。
2. 固定旧 memory 与新 27 个代表，只改变新块的 `raw X/text`、冻结 X、ASM 更新和 joint 更新，形成同预算四格矩阵。

如果这两个实验仍显示新块与旧块各自有效、联合后才下降，再考虑跨阶段关系建模；在此之前不添加跨阶段边或 transport，以免把多个因素混在一起。

## 复现入口

- 实现：[run_native_stage2_attribution.py](../../code/native_tag_v1/run_native_stage2_attribution.py)
- Append-old 结果：`results_mirror/stage2_source_composition_r05_c44_appendold/`
- Union-old 结果：`results_mirror/stage2_source_composition_r05_c44_unionold/`
- 固定 Append old artifact SHA256：`fc30582a53fcc651bcdcecbc1b7c99d717d3c7957da904710fb8cc3a3a345830`
- 固定 Union old artifact SHA256：`2babd13282becc1931d2eeec22b0d2b5de188a4a15ca8b4ca56675e27b1028f3`
- 共同 stage 2 Append artifact SHA256：`2032366f5ae1b82ee95641b23c0b2c1d30ae909620c73065426cbd0bc7c2625c`
- 运行环境：PyTorch `2.4.1+cu121`，NVIDIA RTX 4090；两次运行均 `confirmation_accessed=false`。

