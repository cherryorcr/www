# 共享 Assignment + 原始 GCN Student-Response Smoke 验收报告

日期：2026-09-24  
状态：最小联合候选与 selector smoke 已完成；未晋升为主方法。

## 1. 本轮目的

GraphGPS 校准结果没有达到替换原始 student 的条件，因此本轮回到冻结的两层 GCN student，验证下一版联合路线的最小闭环：

```text
同一个 assignment P
        |
        +--> 合成 X
        +--> top-5 / 320 字符文本，拼接后 MiniLM 重编码
        +--> C0 约束下的 binary edge_index
        |
        v
完整 synthetic memory
        |
        v
原始 GCN + 双向 InfoNCE 的短程 response
        |
        v
整候选接受，或整体回退到 R10 content + C0 topology
```

本轮没有先分别优化 X、文本和拓扑，也没有把 GraphGPS、监督分类 loss 或 report-dev F1 用于选择。

## 2. 固定协议

| 项目 | 设置 |
|---|---|
| 数据集 | Cora、Citeseer |
| 类别顺序 | `order19601` |
| 阶段 | stage 1、stage 2；stage 0 沿用 C0 |
| memory | 5% growing budget；C0 历史 synthetic prefix |
| student | 原始两层 GCN、可训练 text projection、双向 InfoNCE、冻结 MiniLM |
| selector seeds | `300, 301` |
| smoke response steps | `128, 256` |
| 长曲线校准 | Citeseer 额外使用 `128, 512, 1024` |
| report seeds | `150--155` |
| 构造数据 | 只读取当前 raw delta 与 C0 synthetic memory |
| 选择数据 | selector-dev；不读取 report/test 标签 |
| 最终验收 | report-dev cumulative Macro-F1 |

候选为：

- `fallback_content`：R10 的 content + C0 topology，作为整块回退参考；
- `p_initial`：C0 assignment 的确定性初始化；
- `anchor025`：`0.75 * p_initial + 0.25 * p_joint`；
- `joint`：teacher-learned joint assignment `p_joint`。

三个 shared-P 候选均由同一个 P 同时材料化 X、文本和二值拓扑；不能单独保留某个组件的局部增益。

## 3. 选择规则

以 `fallback_content` 的 selector-dev response 为参考，计算：

\[
Q(P)=\Delta R_{new}+0.25\Delta R_{old}
      -0.10 V_{response}
\]

并要求：

- old response 下降不超过 `0.50 pp`；
- response curve 的最差 horizon 不低于 `-0.50 pp`；
- `Q(P) >= 0`；
- 否则整个候选回退，不能只替换 X、文本或边。

这里的 `new response` 是 selector-dev 当前增量类的短程 GCN F1，`old response` 是历史类保持，方差来自 selector seeds 和 response curve。report-dev 只在选择完成后计算。

## 4. Cora smoke 结果

### 4.1 selector 结果

| 阶段 | 候选 | utility | new gain | old delta | 最差 horizon | 是否通过 |
|---|---|---:|---:|---:|---:|---|
| 1 | `p_initial` | -2.972 | -8.311 | +23.488 | -4.026 | 否 |
| 1 | `anchor025` | **+0.268** | -1.321 | +8.952 | -0.203 | 是 |
| 1 | `joint` | -0.541 | -3.273 | +12.451 | -1.658 | 否 |
| 2 | `p_initial` | -6.988 | -4.728 | -7.084 | -9.053 | 否 |
| 2 | `anchor025` | -3.975 | -2.069 | -6.513 | -12.914 | 否 |
| 2 | `joint` | -6.026 | -3.646 | -7.729 | -9.187 | 否 |

selector 选择：stage 1 接受 `anchor025`；stage 2 整体回退 `fallback_content`。

### 4.2 report-dev 验收

| 阶段 | `fallback_content` | `p_initial` | `anchor025` | `joint` | selector 选择 |
|---|---:|---:|---:|---:|---|
| 1 | 56.015 | 56.388 | **58.971** | 54.662 | `anchor025` |
| 2 | 50.768 | 49.397 | 51.736 | 51.644 | `fallback_content` |

stage 1 中，`anchor025` 相对回退高 `+2.956 pp`。stage 2 的 `anchor025` 和 `joint` 在 report-dev 上也略高，但没有通过 selector 门控，因此没有被接受。

## 5. Citeseer smoke 结果

### 5.1 短曲线 selector 结果

| 阶段 | 候选 | utility | new gain | old delta | 最差 horizon | 是否通过 |
|---|---|---:|---:|---:|---:|---|
| 1 | `p_initial` | -7.716 | -10.075 | +12.005 | -7.529 | 否 |
| 1 | `anchor025` | -3.294 | -2.630 | -1.427 | -3.746 | 否 |
| 1 | `joint` | -3.500 | -4.232 | +4.505 | -3.602 | 否 |
| 2 | `p_initial` | -2.499 | -2.357 | +1.079 | -3.511 | 否 |
| 2 | `anchor025` | -2.890 | -2.745 | +0.695 | -7.302 | 否 |
| 2 | `joint` | -0.621 | -0.236 | +1.644 | -5.342 | 否 |

selector 两个阶段均回退。

### 5.2 report-dev 验收

| 阶段 | `fallback_content` | `p_initial` | `anchor025` | `joint` | selector 选择 |
|---|---:|---:|---:|---:|---|
| 1 | 53.237 | 55.129 | **56.214** | 53.820 | `fallback_content` |
| 2 | 54.405 | **57.203** | 54.484 | 55.517 | `fallback_content` |

这说明 shared-P 候选在 Citeseer 也存在 report-dev 正向候选，但短程 selector 没有识别出来：stage 1 的 `anchor025` 高 `+2.977 pp`，stage 2 的 `p_initial` 高 `+2.798 pp`。

## 6. 多长度 response curve 校准

为检查“短程 horizon 太短”这一解释，在 Citeseer 重新运行 `128/512/1024`：

| 阶段 | 候选 | utility | 最差 horizon | report 结果 | 决策 |
|---|---|---:|---:|---:|---|
| 1 | `p_initial` | +3.309 | -6.619 | 55.129 | 回退 |
| 1 | `anchor025` | -1.558 | -2.227 | 56.214 | 回退 |
| 1 | `joint` | -0.575 | -2.611 | 53.820 | 回退 |
| 2 | `p_initial` | -7.290 | -8.283 | 57.203 | 回退 |
| 2 | `anchor025` | -6.671 | -8.209 | 54.484 | 回退 |
| 2 | `joint` | -0.006 | -1.935 | 55.517 | 回退 |

长曲线使部分候选在较长 horizon 的 utility 转正，但早期 horizon 和 selector-dev 分布差异仍使排序不稳定。因此“简单把短程 steps 拉长”还不能解决 response-selector 可靠性问题。

## 7. 材料化与审计结果

smoke 产物与 R11 已封存候选逐格 hash 对比：

- Cora/Citeseer；
- stage 1/2；
- `p_initial`、`anchor025`、`joint`；
- 共 12 个候选材料化结果，全部匹配。

因此本轮的性能差异不是由于重新实现时改变了 X、文本、拓扑或 edge order。最终 topology 是真实 binary `edge_index`，历史 old-old edges 保持不变。

本地结果目录：

- `results/tgc_shared_assignment_gcn_response_smoke_20260924/`
- `results/tgc_shared_assignment_gcn_response_longcurve_20260924/`

运行器与测试：

- `code/v2_llm_fusion/run_tgc_joint_student_response.py`
- `code/v2_llm_fusion/test_tgc_joint_student_response.py`

契约测试：`2 passed`。

## 8. 当前结论

### 已经得到的正向证据

1. 同一 assignment 同时生成 X、文本和拓扑的材料化闭环可复现，并且没有破坏 C0 历史前缀。
2. shared-P 候选在 Cora 和 Citeseer 的部分阶段相对 R10 fallback 有明显 report-dev 提升：Cora stage 1 `+2.956 pp`，Citeseer stage 1 `+2.977 pp`，Citeseer stage 2 的 `p_initial` `+2.798 pp`。
3. 整候选回退机制确实能阻止 stage 2 的明显负向 shared-P 候选进入结果。

### 还不能宣称的内容

1. 当前 selector 不能稳定预测 canonical report-dev 结果。
2. 不能把 `anchor025` 或 `joint` 晋升为跨数据集主方法。
3. 不能把单个顺序上的候选提升写成稳定的新算法收益。
4. 不能开始 GraphGPS-targeted condensation；GraphGPS 仍只作诊断。

当前最准确的定位是：

> Shared assignment 已经证明“候选空间中存在有用的联合构造”，但 student-response selector 还没有证明“可以可靠地从候选空间中选出有用构造”。

## 9. 下一步只做 response 可靠性，不增加新组件

1. 将 selector 从“当前类 F1 的短程均值”改为 stage-aware response curve：同时使用 canonical checkpoint、current/old response、curve area 和 ranking consistency。
2. 使用至少 3 个独立 selector seeds，并把 `128` 这类早期 horizon 作为诊断项而不是硬否决项；保留终点保持约束。
3. 在 Cora/Citeseer 的同一 `order19601` 上先做 response ranking 与 report ranking 的相关性审计，不立即做多顺序或大图。
4. 只有当 selector 在两个数据集都能稳定识别至少一个正向候选，再做 selected-parent recursive rollout 和多顺序确认。
5. 若 response ranking 仍不能预测 report ranking，则保留 C0/R11 为主线，shared-P 作为“联合候选收益 + selector 负结果”的机制分析，而不是强行晋升。
