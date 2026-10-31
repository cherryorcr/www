# P3-A response 选择可靠性与成本验收报告

日期：2026-09-18  
状态：小图严格配对实验完成；response selector 暂不晋升为默认主方法。

## 1. 实验目的

P0 解决了类别枚举、source row 顺序和 manifest 记录不一致的问题。本轮在新的
`p0-canonical-class-then-source-id-v1` 协议下，严格比较：

- `dispersion`：response off，固定低成本类内分配；
- `v4`：response on，在相同总 rows 内进行 class-row transfer。

两条路径固定相同的阶段顺序、预算、文本构造、拓扑、student 模型、batch、steps
和六个 student seeds。response 不允许增加 rows，因此性能差异不能归因于记忆规模扩大。

## 2. 实验覆盖

| 数据集 | 阶段顺序 | memory budget | student seeds | response seeds/batches |
|---|---|---:|---|---|
| Cora | canonical / order1701 / order1702 | 5% growing rate | 100--105 | 300--302 / 8,16,32 |
| Citeseer | canonical / order1701 / order1702 | 5% growing rate | 100--105 | 300--302 / 8,16,32 |

每个数据集和顺序均完成两条方法臂的三阶段构造和 student 重训，共 6 个配对单元。

## 3. 主要结果

最终阶段 Macro-F1，`delta` 为 `v4 - dispersion`：

| 数据集 | 顺序 | dispersion | v4 response-on | delta |
|---|---|---:|---:|---:|
| Cora | canonical | 49.85 | 49.85 | 0.00 |
| Cora | order1701 | 52.59 | 52.59 | 0.00 |
| Cora | order1702 | 42.77 | 39.97 | -2.80 |
| Citeseer | canonical | 59.17 | 60.77 | +1.59 |
| Citeseer | order1701 | 53.76 | 51.47 | -2.29 |
| Citeseer | order1702 | 50.94 | 54.20 | +3.26 |

按三个顺序平均：

| 数据集 | 平均最终 delta | 平均阶段 delta | 最差最终 delta |
|---|---:|---:|---:|
| Cora | -0.93 | +5.08 | -2.80 |
| Citeseer | +0.85 | +0.25 | -2.29 |

旧类/新类的最终变化也不稳定。Cora order1702 中，v4 相对 dispersion 的
old/new 变化为 `-0.99/-5.21` 个百分点，说明 response 转移并未稳定改善新类塑性，
反而同时损伤旧类与新类表现。

## 4. response 是否真正改变了合成 memory

在 canonical 和 order1701 的 Cora 结果中，v4 的最终 synthetic tensor hash 与
dispersion 相同，因此 response 的候选评估最终回到了相同的合成图，F1 也完全相同。
Cora order1702 的三个阶段均产生不同 hash，但最终性能下降。

Citeseer 中，canonical 的三个阶段、order1701 的前两阶段和 order1702 的 stage 0
与 dispersion hash 相同或保持基线；发生改变的阶段并没有稳定对应正收益。也就是说，
“selector 改变了配额”不等于“selector 选择了更好的配额”。

## 5. 成本结果

| 数据集 | v4 selector 平均耗时 | dispersion 平均构造耗时 | selector/dispersion 构造比 |
|---|---:|---:|---:|
| Cora | 259.18 s | 0.46 s | 564.9x |
| Citeseer | 229.78 s | 0.56 s | 415.6x |

v4 的 selector 每个数据集约完成 300--460 次 probe training evaluation，累计约
10 万--12.6 万 optimizer steps；单条 student 轨迹只有约 5--9 秒。因此当前 response
机制的主要问题不是能否运行，而是选择成本远高于 student 重训本身，且收益没有稳定覆盖该成本。

## 6. P3-A 验收结论

### 已通过

- 新 P0 枚举协议在远程 `tagsam` 环境通过 10 项测试；
- Cora/Citeseer 三种顺序全部完成 response off/on 配对；
- 预算、stage classes、student seeds 和训练步数一致；
- 旧 memory 保持和 output manifest 可审计；
- selector 成本、selection runs、steps 和 tensor hash 已记录。

### 未通过晋升门禁

- 没有跨顺序稳定的最终 F1 正收益；
- Cora 平均最终性能下降；
- 两个数据集均至少有一个顺序出现超过 2 个百分点的退化；
- selector 成本约为低成本构造的 400--600 倍；
- 当前短程 response 排名不能被视为 canonical student 效果的可靠代理。

因此不能把 `response-on` 作为新的默认 v4 主方法，也不能宣称 SCA 已经证明有效。
冻结 v4 继续保留为历史比较对象；后续默认控制应使用 `dispersion + response off`。

## 7. 对后续路线的影响

1. **关闭 response-gain extra budget。** 在 selector 可靠性未通过前，不让不稳定的
   response 估计同时控制总 rows。
2. **优先做 P1-X。** 在固定 rows、相同文本和拓扑下比较 Mean-X、Weighted-X、
   Moment-X，直接解决当前方法相对 k-center/medoid 的核心短板：均值导致的多模态和方差收缩。
3. **保留少量 P3 后续诊断。** 如果继续研究 response，只保留有限 class-row transfer，
   使用更短的候选集和独立确认 seed；不要继续扩大 selector 搜索空间。
4. **暂不做 P4 refresh/merge。** 先确认合成 X 本身有价值，再研究历史 memory 管理。
5. **不把成本优势写成端到端加速。** 当前持续凝聚可节省存储，但 response-on 构造远慢于
   简单 dispersion；论文应单独报告构造成本与 student 成本。

## 8. 复现入口

- 结果汇总：`results/p3a_response_pair_2026_09_18/summary.json`
- 运行入口：`code/v2_llm_fusion/run_small_order_cost_confirmation.py`
- 汇总脚本：`code/v2_llm_fusion/summarize_p3a_response_pair.py`
- P0 协议报告：`docs/current/P0_COMPARISON_CONTROL_COMPLETION_ZH_2026-09-18.md`

本报告只使用远程 `tagsam` 环境中完成的六个配对单元，不混入旧协议的历史分数。
