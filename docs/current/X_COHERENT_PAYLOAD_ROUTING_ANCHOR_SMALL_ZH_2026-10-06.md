# X-conditioned Coherent Payload Routing with CCTP Anchor：小图三顺序确认

状态：小图确认已完成。anchor 能避免无约束 coherent routing 的退化，但当前过度保守，几乎退化为 `cat_nearest`，因此暂不作为最终主方法。

## 1. 协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19611`、`order19612`、`order19613` |
| 阶段 | 三阶段 |
| Student | Prompt zero-shot Student：两层 GCN、冻结 MiniLM、无分类器头、图文 InfoNCE + prompt CE/margin |
| Student seeds | `150--155`，6 个 paired seeds |
| construction seed | `270927` |
| 比较臂 | `cat_nearest`、formal prompt-margin、coherent routing、coherent + CCTP anchor |
| 历史 raw | 不回读 |

## 2. Pooled 结果

以下为 Cora/Citeseer、三个顺序共 6 个 cell 的等权平均。

| 方法 | Final F1 | Accuracy | Stage-average F1 | Old F1 | New F1 |
|---|---:|---:|---:|---:|---:|
| cat_nearest | 76.581 | 79.508 | 83.844 | 76.167 | 77.197 |
| formal prompt-margin | **77.091** | **79.766** | **84.408** | **76.858** | **77.436** |
| coherent routing | 76.867 | 79.736 | 84.069 | 76.700 | 77.096 |
| coherent + CCTP anchor | 76.610 | 79.529 | 83.865 | 76.228 | 77.184 |

anchor 相对 `cat_nearest`：

| 指标 | 变化 |
|---|---:|
| Final F1 | +0.029 pp |
| Accuracy | +0.021 pp |
| Stage-average F1 | +0.021 pp |
| Old F1 | +0.061 pp |
| New F1 | -0.013 pp |

## 3. Anchor 的真实修改比例

逐条检查 615 个当前 delta 合成行：

| 项目 | 数量 |
|---|---:|
| 回退到 CCTP | 611 |
| 接受 coherent candidate | 4 |
| 回退比例 | 99.35% |
| 接受比例 | 0.65% |

因此小图上 anchor 几乎完全保留 `cat_nearest`。这解释了它没有灾难性退化，也解释了它没有带来有意义的平均收益。

## 4. 结果说明

1. **保护逻辑是有效的。** 无约束 coherent routing 没有稳定优势，而 anchor 没有出现额外的严重负尾部。
2. **当前固定代价不是有效的晋升判据。** 它在小图上只允许 4/615 个候选替换，说明 `graph/text cost` 对最终 prompt Student 的长期收益过于保守或错位。
3. **anchor 目前更像安全回退组件，而不是完整新算法。** 大图上它相对 CCTP 有条件性收益，小图上基本等价于 CCTP；两者尚不足以支持普适主方法结论。

## 5. 当前主方法建议

暂时不要继续扫描更多 routing 参数，也不要把无约束 coherent routing 直接升为主方法。当前最稳妥的定位是：

```text
CaT/CCTP 作为默认 memory 构造
coherent candidate routing 作为候选增强
strict anchor 作为安全回退
```

如果后续还要优化，核心不是放松阈值到能覆盖更多行，而是重新设计一个与 zero-shot prompt Student 更一致的无标签接受代价，例如使用 construction-time 的 class-prompt margin、old/new response barrier 或多 checkpoint response；但必须先在开发顺序上验证，再用独立顺序确认，不能按最终 F1 逐 cell 调节。

## 6. 证据位置

- Cora summary：`D:\Aprojects\continue_tag_condensation\tmp_remote_results\x_coherent_anchor_small_20261006_gpu0_summary.json`
- Citeseer summary：`D:\Aprojects\continue_tag_condensation\tmp_remote_results\x_coherent_anchor_small_20261006_gpu1_summary.json`
- 小图 construction audit：`D:\Aprojects\continue_tag_condensation\tmp_remote_results\x_coherent_anchor_small_*_construction.json`
- 实现：`D:\Aprojects\continue_tag_condensation\code\v2_llm_fusion\x_conditioned_payload_routing.py`
