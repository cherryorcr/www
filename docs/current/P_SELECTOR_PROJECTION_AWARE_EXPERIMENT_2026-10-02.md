# P-only 文本分配与 Projection-aware 验证（2026-10-02）

## 1. 目的

本轮只固定 CaT 合成特征 `X`、原始两层 GCN Student 和已有拓扑，优化当前阶段的文本分配。历史 synthetic memory 只追加，不读取历史 raw graph，也不修改旧节点和旧边。

本轮分两步：

1. 用 CCTP 文本选择作为 anchor，检查 Student-guided `P` 是否能在不破坏冻结 MiniLM 拼接行为的前提下工作。
2. 针对 soft assignment 与真实拼接重新编码之间的失配，实现 projection-aware 的有限单槽替换，并用 Student response proxy 选择候选。

## 2. Anchor 版本的定义

每个新 synthetic row 只从当前 delta 的同类原始文本中选择来源。首先用图特征、文本中心和类别 prompt 形成 graph-conditioned prior，再与 CCTP 的 text-center + MMR 选择按固定比例混合。最终仍然：

- 固定 top-k 文本来源；
- 按 CCTP anchor 顺序拼接；
- 固定字符预算；
- 用冻结 MiniLM 重新编码；
- 只允许当前 delta 的来源进入 provenance。

Student feedback 通过 0、1 或 3 步 differentiable unroll 计算，但不允许无界改变 hard top-k 结果。

## 3. 四顺序 anchor 复现

协议：Cora/Citeseer，`order19611` 至 `order19614`，每个 cell 使用 Student seeds `150..155`，固定 5% cumulative memory，原始两层 GCN、双向 InfoNCE、冻结 MiniLM、相同训练步数。

最终阶段 Macro-F1：

| 数据集 | CCTP top-3 | P graph | P + 1-step | P + 3-step |
|---|---:|---:|---:|---:|
| Cora | 66.35 | 66.35 | 66.35 | 66.35 |
| Citeseer | 67.11 | 67.11 | 67.11 | 67.11 |

四个顺序的最终值分别为：

- Cora CCTP：`66.123, 66.319, 66.201, 66.753`；
- Citeseer CCTP：`66.328, 67.528, 66.392, 68.173`。

三个 P 版本的对应结果逐 cell 完全一致。这说明 anchor 解决了无约束 P 优化的灾难性退化，但 Student soft feedback 尚未穿透最终的 hard text projection。

## 4. 无 anchor 的负结果

此前无 anchor 的 P-only 完整四顺序结果为：

| 数据集 | CCTP | P graph | P + 1-step | P + 3-step | Random text |
|---|---:|---:|---:|---:|---:|
| Cora | 65.72 | 27.72 | 28.87 | 30.31 | 54.00 |
| Citeseer | 66.61 | 37.36 | 39.21 | 39.87 | 48.85 |

诊断发现：即使来源集合相同，P 重新排序后再拼接也会改变冻结 MiniLM 的 embedding。因此问题不是单纯的 Student 梯度方向，而是 soft assignment、文本顺序和 hard re-encoding 三者不一致。

## 5. Projection-aware 版本

projection-aware 不再把 source embedding 的加权平均当作最终文本。对每个 row：

1. 保留 CCTP 的 top-k anchor 和顺序；
2. 从当前 delta 的同类候选中取小候选池；
3. 只替换一个文本槽位；
4. 对每个真实拼接字符串重新运行冻结 MiniLM；
5. 用 Student response loss 和新类 prompt loss 评估；
6. 未通过约束则回退 anchor。

### 5.1 无约束 projection smoke

在 Cora `order19611` 上，允许每行替换并采用同 horizon Adam response 后：

| 阶段 | Anchor Macro-F1 | Projection-aware Macro-F1 |
|---:|---:|---:|
| 0 | 96.64 | 45.83 |
| 1 | 85.41 | 53.33 |
| 2 | 65.41 | 37.56 |

问题在于 proxy 改善并不保证最终从头训练的 Student 改善；候选替换的 text embedding 与 anchor 平均 cosine 下降到明显不同的区域，多行累积后破坏了文本空间。

### 5.2 单槽替换诊断

限制每阶段最多一个替换、cosine 不低于 `0.90`，暂时关闭 prompt gate：

| 阶段 | Anchor F1 | One-swap F1 | 差值 |
|---:|---:|---:|---:|
| 0 | 96.64 | 97.46 | +0.83 |
| 1 | 85.41 | 87.30 | +1.89 |
| 2 | 65.71 | 52.25 | -13.46 |

该结果说明局部早期收益不能直接推广到递归 memory；后期旧 prefix 与新类文本的交互使局部 proxy 失效。

### 5.3 安全 gate

当前安全版本使用：

- anchor cosine trust region `>= 0.98`；
- 每阶段最多接受一个替换；
- 新类 prompt loss 必须改善；
- 联合 response loss 必须改善；
- 否则回退到 CCTP。

在 Cora `order19611` 控制实验中没有接受替换，最终结果与 CCTP 完全一致。这证明回退和 anchor materialization 正确。

### 5.4 完整安全 gate 验收

在两张小图、四个顺序和六个 Student seed 上完成了完整安全版本。所有 24 个 dataset/order/stage construction cell 的接受数均为 0，因而最终 memory 与 CCTP anchor 完全一致。

最终阶段汇总如下：

| 数据集 | 方法 | Macro-F1 | Accuracy |
|---|---|---:|---:|
| Cora | CCTP anchor | 66.01 | 77.28 |
| Cora | bounded projection-aware | 66.01 | 77.28 |
| Citeseer | CCTP anchor | 67.11 | 72.05 |
| Citeseer | bounded projection-aware | 67.11 | 72.05 |

安全 gate 的额外构造成本约为每个阶段 6--10 秒，主要来自真实候选字符串的 MiniLM 重编码和 full-horizon response proxy；但由于所有候选最终都被拒绝，这部分成本目前没有换来性能收益。

## 6. 当前判断

本轮得到的不是一个已经晋升的增益方法，而是两个可靠结论：

1. anchor 约束可以消除 P-only 的灾难性退化，并在四顺序、多 Student seed 下稳定复现 CCTP。
2. projection-aware 的真实文本候选搜索具有可解释形式，但当前 Student response proxy 与最终递归训练动力学仍不一致，不能直接作为主方法增益。

因此，当前版本应暂时视为：

> CCTP anchor + 可审计的 projection-aware bounded revision；默认失败回退到 CCTP。

不能宣称 P-only 或 projection-aware 已经优于 CCTP。

## 7. 下一步

1. 已完成两数据集、四顺序、六 Student seed 的安全 gate 验收。
2. 已汇总接受率、cosine shift、new/old F1 和 construction cost；接受率为 0。
3. 停止继续扫描文本替换权重；把 projection-aware 作为负结果和可解释的失败边界。
4. 后续若继续优化，只做一个受控方向：以完整 Student 分类 response 为目标的极少量候选验证，并要求独立顺序确认；不再恢复无界 P 优化。
