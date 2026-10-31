# Direct Student 外层优化：修正协议与 Prompt 目标实验报告

日期：2026-09-28  
状态：已完成 support 边界修正、X-only 重跑、prompt CE、prompt margin 和 base=self 一致性检查。

## 1. 先修正一项协议错误

上一轮初始 X-only 结果中，两个 `order19102` cell 使用了错误的 `old_rows`。实际 sealed support 的边界来自各自的 prep protocol：

| Cell | 正确 old rows | 当前 raw delta rows | base memory rows |
|---|---:|---:|---:|
| Cora / canonical | 25 | 858 | 67 |
| Cora / order19102 | 35 | 586 | 64 |
| Citeseer / canonical | 38 | 953 | 86 |
| Citeseer / order19102 | 40 | 862 | 83 |

旧运行把 `Cora/order19102` 的边界设成 25，把 `Citeseer/order19102` 的边界设成 38。这样会把部分旧 synthetic rows 当成当前 delta，导致 P/X 的优化对象错误。旧产物保留用于审计，但不能用于结论。

以下所有结果都使用上表边界，并使用当前统一评估脚本重跑。额外做了 `base=self` 检查，得到：

```text
final_f1_pp = 0.0
accuracy_pp = 0.0
new_f1_pp = 0.0
old_f1_pp = 0.0
```

说明评估脚本的 paired comparison 没有额外引入差异。

## 2. 实验共同协议

- 只优化 `X`，不优化 `P`、文本和拓扑；
- candidate 上可微 Student inner unroll 2 步；outer Adam 2 步，`lr=0.01`；
- old/new InfoNCE 权重为 `0.25/1.0`；
- old row trust region `0.05`，new row trust region `0.20`；
- 下游 canonical Student seeds `150--155`；
- 不访问历史完整 raw graph；
- 不使用 report/test 标签；
- prompt 实验只使用当前到达 delta 的 construction-time class id 和公开 class prompt。

## 3. 修正后的 X-only 结果

| 数据集/顺序 | Base F1 | X-only F1 | Delta F1 | Delta Accuracy | Delta New F1 | Delta Old F1 |
|---|---:|---:|---:|---:|---:|---:|
| Cora / canonical | 52.068 | 49.112 | -2.956 | -4.269 | -4.678 | -2.014 |
| Cora / order19102 | 53.339 | 53.498 | **+0.160** | -0.583 | **+0.502** | -0.622 |
| Citeseer / canonical | 69.348 | 69.112 | -0.237 | +0.136 | -2.182 | **+1.707** |
| Citeseer / order19102 | 59.755 | 59.065 | -0.690 | -0.837 | +0.444 | -2.047 |
| **四 cell 平均** | - | - | **-0.931** | **-1.388** | **-1.479** | **-0.744** |

结论：直接对 `X` 反向优化的梯度路径真实存在，但在正确边界下没有稳定改善。只有一个 cell 的最终 F1 为正；新类 F1 只有两个 cell 略为正，平均仍下降。

## 4. Prompt-aware X-only：加入新类 Prompt CE

本版本在原有 old/new InfoNCE 外增加：

```text
0.5 * CE(Student graph embedding of current raw delta,
         corresponding class prompt)
```

旧 synthetic rows 的 class id 保持 `-1`，只对当前 raw delta 使用类别身份。它是 construction-time 信息，不是 report/test 标签。

| 数据集/顺序 | Delta F1 | Delta Accuracy | Delta New F1 | Delta Old F1 |
|---|---:|---:|---:|---:|
| Cora / canonical | -0.625 | -0.821 | -2.159 | +2.166 |
| Cora / order19102 | -0.039 | -0.667 | -0.107 | -1.052 |
| Citeseer / canonical | -0.788 | -0.949 | -3.432 | +2.176 |
| Citeseer / order19102 | **+1.022** | **+0.697** | -1.183 | +1.685 |
| **四 cell 平均** | **-0.430** | **-0.435** | **-1.720** | **+1.244** |

解释：prompt CE 比纯 InfoNCE 更接近最终零样本分类任务，并且明显提高 old F1；但它仍然通过牺牲 new F1 换取整体或旧类收益，不能称为新类塑性解决方案。

## 5. Prompt-aware X-only：加入新类 Prompt Margin

为避免 CE 主要改变整体 logits，进一步加入新类正确 prompt 与 hardest negative prompt 的 margin：

```text
relu(margin - logit(correct prompt) + max logit(negative prompt))
```

本轮固定 margin `0.20`、margin weight `0.5`，并修正了 margin 分支独立于 CE 权重的实现 bug。

| 数据集/顺序 | Delta F1 | Delta Accuracy | Delta New F1 | Delta Old F1 |
|---|---:|---:|---:|---:|
| Cora / canonical | -0.081 | +0.246 | -1.666 | +2.106 |
| Cora / order19102 | -0.927 | -0.417 | -2.185 | -0.399 |
| Citeseer / canonical | -1.573 | -1.084 | -4.060 | +1.228 |
| Citeseer / order19102 | -1.163 | -1.953 | -4.436 | +0.967 |
| **四 cell 平均** | **-0.936** | **-0.802** | **-3.087** | **+0.976** |

结论：当前 prompt margin 代理会保护旧类，但会稳定损害新类塑性，不能进入主方法。

## 6. 这些结果说明什么

### 6.1 梯度确实传到了 X

X-only audit 中 outer gradient norm 非零，最大相对位移受 trust region 限制；stop-gradient control 的位移为 0 且 `nonzero_outer_gradient=false`。因此失败原因不是“没有真正做梯度优化”。

### 6.2 目标函数和下游目标仍有错位

当前 Student outer 优化经历：

```text
candidate memory -> Student inner InfoNCE -> support loss / prompt loss -> gradient to X
```

最终评估则是：

```text
independently trained Student -> real full graph propagation -> class prompt classification
```

中间至少有三处不一致：

1. outer 的图文实例匹配不等价于类 prompt margin；
2. support graph 的消息传播不等价于真实 full graph 的消息传播；
3. 外层梯度可以通过 Student 的 text projection 和 logit scale 改善代理损失，却不一定让合成 `X` 更适合重新训练后的 Student。

prompt CE 和 margin 的结果进一步说明：即使把 prompt 信息加入 outer objective，也可能通过旧类保持换取新类塑性下降。

### 6.3 不能继续简单放大权重或增加 P/A

当前最合理的判断是：

- 不是 `X` 完全不能优化；
- 不是 `P/A` 还没有加进去所以分数低；
- 而是当前短 unroll + support proxy 对最终 Student 的训练响应预测不可靠。

直接加入 `P`、`A` 或 `X+P+A` 会把这个预测误差扩展到 assignment 和拓扑，无法解释结果来源。

## 7. 当前推进决策

本轮三种 direct outer 版本均不晋升为主方法：

1. `X-only`：证明梯度路径可行，但平均性能下降；
2. `X + prompt CE`：整体平均仍下降，新类平均下降；
3. `X + prompt margin`：新类平均下降更明显。

因此当前主方法仍保留原有经过验证的构造版本；direct Student outer 作为“为什么直接 Student 反传不能自动提升持续凝聚”的负结果与机制诊断。

## 8. 后续建议

如果继续研究梯度优化，下一版不应再直接对 `X` 加更多局部损失，而应先解决 Student 响应预测问题：

1. 使用多步 Student trajectory，而不是只看 2 步后的单点 support loss；
2. 固定或冻结 text projection，避免梯度通过文本分支投机降低 loss；
3. 对 candidate graph 和 support/full-graph operator 做一致化，至少先做传播算子控制；
4. 对新类合成 slot 使用 class-specific provenance，而不是把整块 new delta 的梯度平均传给所有 X rows；
5. 先做一个小规模 `X-only + class-slot provenance` 实验；只有它能在两个数据集上稳定改善 new F1，才考虑 P/A。

在这些条件满足前，不建议继续扩大 outer steps、扫描更多 loss 权重或运行联合 `X+P+A`。

## 9. 产物

- 外层优化实现：[direct_student_outer.py](../../code/v2_llm_fusion/direct_student_outer.py)
- 运行入口：[run_direct_student_outer.py](../../code/v2_llm_fusion/run_direct_student_outer.py)
- prompt 元数据准备：[prepare_direct_student_prompt_support.py](../../code/v2_llm_fusion/prepare_direct_student_prompt_support.py)
- 单元测试：[test_direct_student_outer.py](../../code/v2_llm_fusion/test_direct_student_outer.py)
- 修正后评估：`results/direct_student_outer_20260928/eval_xonly_v2_*.json`
- Prompt CE 评估：`results/direct_student_outer_20260928/eval_xprompt_v1_*.json`
- Prompt margin 评估：`results/direct_student_outer_20260928/eval_xmargin_v2_*.json`
