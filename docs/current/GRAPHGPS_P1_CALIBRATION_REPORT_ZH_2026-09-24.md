# P1 GraphGPS Student 校准验收报告

日期：2026-09-24  
状态：已完成正式矩阵，未启动联合 student-response 凝聚

## 1. 目的与协议

本轮只校准下游 student，不修改任何凝聚算法。目标是判断两层 GCN 是否是当前性能瓶颈，以及 GraphGPS 是否可以作为后续联合凝聚的目标 student。

正式矩阵为：

- 数据集：Cora、Citeseer；
- 类别顺序：`order19601`；
- 阶段：stage 0/1/2；
- arms：C0、R11 内容 + C0 拓扑、k-center、legal raw pool；
- student：从头训练的 GraphCLIP GPS graph tower，12 层、hidden 1024、输出 384，冻结 MiniLM；
- 训练目标：双向 graph-text InfoNCE，不使用分类标签训练；
- 输入：每个被消费节点的 2-hop ego graph，32-step RW-PE；
- seeds：150--155，共 6 个；
- batch size：4；
- 指标：report-dev 上的 cumulative Macro-F1；
- 总运行数：144 个 seed runs，24 个 protocol groups。

官方 GraphCLIP checkpoint 不混入本表。已有外部参考为 Cora `58.98`、Citeseer `57.76`，它不是同 student、同训练协议的对照。

结果原始汇总：`results/graphgps_calibration_p1_20260924_summary.json`。

## 2. 正式结果

表中数值为 `final cumulative Macro-F1 mean +/- std`；括号内为该阶段的 memory rows。

### Cora

| 阶段 | C0 | R11 + C0 topology | k-center | raw pool |
|---|---:|---:|---:|---:|
| 0 | 27.50 +/- 8.58 (21) | 25.27 +/- 5.75 (21) | 24.03 +/- 9.61 (25) | 12.45 +/- 1.53 (501) |
| 1 | 37.57 +/- 1.85 (59) | 34.11 +/- 3.07 (59) | 30.49 +/- 5.82 (67) | 16.04 +/- 7.87 (1359) |
| 2 | 16.06 +/- 15.10 (94) | 31.09 +/- 20.37 (94) | 38.51 +/- 10.94 (94) | 12.62 +/- 8.58 (1895) |

三阶段 final 均值的简单平均：C0 `27.05`，R11 `30.16`，k-center `31.01`，raw pool `13.71`。

### Citeseer

| 阶段 | C0 | R11 + C0 topology | k-center | raw pool |
|---|---:|---:|---:|---:|
| 0 | 43.23 +/- 5.41 (32) | 41.01 +/- 4.99 (32) | 40.99 +/- 5.33 (38) | 27.85 +/- 10.69 (778) |
| 1 | 46.88 +/- 10.93 (75) | 48.14 +/- 7.98 (75) | 35.62 +/- 13.09 (86) | 32.54 +/- 11.38 (1731) |
| 2 | 30.25 +/- 19.32 (111) | 30.43 +/- 24.42 (111) | 33.17 +/- 17.86 (111) | 34.94 +/- 12.04 (2229) |

三阶段 final 均值的简单平均：C0 `40.12`，R11 `39.86`，k-center `36.59`，raw pool `31.78`。

## 3. Stage-average 结果

这里的 stage-average 是每个 seed 在三个阶段的 cumulative Macro-F1 平均，再对 6 个 seed 求均值。

| 数据集 | C0 | R11 + C0 topology | k-center | raw pool |
|---|---:|---:|---:|---:|
| Cora stage 0/1/2 | 42.74 / 48.36 / 24.85 | 40.32 / 48.68 / 42.80 | 28.64 / 42.39 / 51.79 | 27.96 / 24.19 / 15.97 |
| Citeseer stage 0/1/2 | 52.60 / 55.84 / 37.28 | 51.01 / 57.15 / 38.61 | 53.08 / 41.27 / 37.28 | 41.48 / 45.87 / 43.75 |

## 4. rows 与公平性审计

k-center 的既有 artifact 在早期阶段使用了更大的旧配额：

- Cora：C0/R11 为 `21/59/94`，k-center 为 `25/67/94`；
- Citeseer：C0/R11 为 `32/75/111`，k-center 为 `38/86/111`。

因此 stage 0/1 的 k-center 结果不能写成严格同预算优劣；stage 2 才是 rows 完全匹配的主要比较。raw pool 使用全部合法原始训练节点，rows 远大于 synthetic memory，适合回答“同一 GraphGPS student 消费完整 raw 数据时会怎样”，不适合回答固定 memory budget 下的效率或上限。

## 5. 结果解释

### 5.1 GraphGPS 没有被证明可以直接替换 GCN

GraphGPS runner、RW-PE、ego graph、InfoNCE 和 report-dev 推理均正常，但从头训练的 GraphGPS 在长链阶段存在很大的 seed 方差。例如：

- Cora stage 2：C0 final std `15.10`，R11 `20.37`，k-center `10.94`；
- Citeseer stage 2：C0 `19.32`，R11 `24.42`，k-center `17.86`，raw `12.04`。

因此当前结果不能支持“GraphGPS 更强，所以后续凝聚一定更好”。官方 checkpoint 的高分也不能直接替代 scratch calibration，因为它改变了初始化、训练状态和输入使用方式。

### 5.2 raw pool 不是当前协议下的理论上限

raw pool 在 Cora stage 0/1/2 的 final 均值为 `12.45/16.04/12.62`，低于对应 synthetic C0；Citeseer 的 stage 2 raw pool `34.94` 略高于 C0/R11，但 stage 0/1 仍低于 synthetic。

这不意味着 synthetic memory 普遍优于完整图。更合理的解释是：当前 zero-shot GraphGPS + frozen text + InfoNCE + 固定训练预算对 memory 规模、batch、文本对齐和 ego-graph 分布敏感。raw pool 只能作为同 student 的数据参考，不能作为未经充分校准的“理想模型”。

### 5.3 R11 与 k-center 都没有跨阶段、跨数据集稳定晋升

- R11 在 Cora stage 2 有明显正差，但 stage 0/1 低于 C0；Citeseer 只在 stage 1 有小幅优势；
- k-center 在 Cora stage 2 明显较强，但早期阶段受 rows 不匹配影响，Citeseer stage 1 反而明显较弱；
- 现有证据不能把 R11 或 k-center 写成 GraphGPS 下的通用最优凝聚方法。

## 6. P1 验收结论

P1 的实现和正式矩阵完成，但“GraphGPS 作为后续凝聚目标 student”暂不通过性能晋升条件。

当前应采用以下结论：

1. GraphGPS 校准工具链有效，可以用于诊断；
2. 当前 GraphGPS scratch student 的训练方差过大，不能直接替换论文主 student；
3. raw pool 与 synthetic 的关系依赖数据集和阶段，不能把 raw F1 直接称为理想上界；
4. R11、k-center 均没有在本矩阵中形成稳定优势；
5. 不启动 GraphGPS-targeted joint student-response condensation；
6. 后续算法实验仍以冻结的 GCN/C0 公共协议为主，GraphGPS 作为单独 robustness/diagnostic 分支。

## 7. 后续推进

优先级调整为：

1. 保留本轮 GraphGPS 结果作为 P1 诊断，不混入 GCN 主表；
2. 对 GraphGPS 只做一次小范围训练协议校准：固定 stage-wise steps、检查 checkpoint/temperature 与 raw/synthetic 的共同可消费性，不再扩大凝聚矩阵；
3. 若校准后仍然高方差，停止将 GraphGPS 作为方法增益来源；
4. 回到 GCN/C0，在共同 assignment 的联合材料化上做小规模 student-response 验证；
5. 只有联合方法在 Cora/Citeseer 多阶段、多 seed 下稳定提升，才进入大图和长链实验。

本轮不支持的表述包括：GraphGPS 已证明优于 GCN、synthetic 已超过完整 raw、R11/k-center 已成为通用最优、或联合 student-response 已经完成。

## 8. P1-B 训练步数校准

为排除自动规则 `steps = 16 * memory_rows` 带来的混淆，追加了 stage 0 的小矩阵：C0 与 raw pool 使用完全相同的 `336` 或 `1024` steps，每个 cell 使用 3 个 seeds。

| 数据集 | steps | C0 final | raw pool final |
|---|---:|---:|---:|
| Cora | 336 | 26.36 +/- 3.17 | 37.95 +/- 6.80 |
| Cora | 1024 | 21.17 +/- 8.44 | 30.23 +/- 10.61 |
| Citeseer | 336 | 46.62 +/- 2.22 | 45.84 +/- 6.49 |
| Citeseer | 1024 | 45.92 +/- 6.86 | 26.88 +/- 5.72 |

这组结果说明：

- raw pool 相对 C0 的排序会随 training steps 改变；
- 增加 steps 并不保证 raw 或 synthetic 的 F1 单调上升；
- 不能使用 `16 * memory_rows` 让 raw pool 自动获得不同的训练预算后，再把 raw F1 当作压缩上限；
- 后续正式 GraphGPS 比较应采用按 stage 固定、对所有 arms 相同的 steps，例如 C0 的 `336/944/1504`。

因此原始 P1 的 row-scaled steps 结果保留为诊断记录；公平的 common-step 重跑另行记录，不与原始结果混合。
