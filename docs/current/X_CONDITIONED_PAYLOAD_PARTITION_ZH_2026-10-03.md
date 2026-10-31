# X 条件化图文分簇 Payload 方法

日期：2026-10-03

## 1. 动机

此前的 `formal prompt-margin assignment` 虽然为每个类别生成了多个真实文本候选，但最终每个类别只保留一个 payload，并广播给该类别的所有合成节点。这使 CaT 逐行生成的 `X` 与文本表示的粒度不一致，也会与 Student 的逐行 InfoNCE 产生 false negative。

本实现改为：把 CaT 的每个合成 `X` 行视为一个语义槽位，用当前阶段 raw delta 的图特征和文本 embedding 对 raw 节点进行类别内平衡分簇，再从每个分簇材料化一个真实文本 payload。

## 2. 完整流程

### 2.1 CaT 初始化

对当前 delta 使用原 CaT class-wise MMD 得到：

```text
X0 = {x_1, ..., x_m}
```

其中 `m` 完全由 memory budget 和 CaT 的类别比例分配决定，不额外指定每类 4 个或 8 个 payload。

### 2.2 图文联合代价

当前 delta 中每个 raw 节点保留图特征 `f_i` 和 MiniLM 文本 embedding `t_i`。CaT 槽位 `x_j` 通过 current-delta-only ridge map 得到文本预测：

```text
pred_text_j = Normalize([Normalize(x_j), 1] @ graph_text_map)
```

对同一类别内的 raw 节点和 X 槽位定义：

```text
C_ij = 0.5 * (1 - cosine(f_i, x_j))
     + 0.5 * (1 - cosine(t_i, pred_text_j))
```

图和文本权重是全局统一的配置，不根据最终测试 F1 为某个数据集修改。

### 2.3 类别内平衡 Sinkhorn 分配

对每个类别单独求 entropic transport plan：

```text
Q_c = Sinkhorn(-C_c / temperature)
```

约束为每个 raw 节点总质量为 `1 / n_c`，每个 CaT 槽位总质量为 `1 / m_c`。因此每个 `X` 槽位都会获得自己的 current-delta raw 子簇，不需要遍历文本组合，也不需要固定候选池。

### 2.4 联合梯度更新

以 `X0` 为 warm start，只更新当前新 block 的 `X`：

```text
L = L_CaT(X)
  + lambda_transport * <Q, C>
  + lambda_text * L_text_MMD
  + lambda_align * (1 - cosine(pred_text_j, soft_text_j))
  + lambda_anchor * ||X - X0||^2
```

`Q` 每一步根据当前 `X` 重新计算。为了控制大图显存，Sinkhorn 计划本身采用 detached cost 求解；transport cost 和 graph-text alignment 仍然对 `X` 反向传播。`X` 受到逐行 trust radius 约束，默认相对漂移不超过 5%。

### 2.5 子簇文本材料化

第 `j` 个槽位的软文本中心为：

```text
soft_text_j = Normalize(sum_i Q_ij * t_i / sum_i Q_ij)
```

最终不直接保存这个不可追溯的 soft embedding，而是在该槽位的 raw 子簇中：

1. 以加权 text medoid 作为 anchor；
2. 在字符预算允许时加入高权重且低冗余的同簇文本片段；
3. 按固定文本长度约束拼接；
4. 对最终真实字符串重新运行 MiniLM。

因此保存到 memory 的 `text_emb` 一定来自实际保存的 `texts`，并记录 source IDs。

## 3. 访问边界和不变量

- 只读取当前阶段 delta；
- 不读取历史完整 raw graph；
- 不修改旧 synthetic memory；
- 不使用 report/test labels；
- 不使用 Student response 选择文本；
- 不修改拓扑，输出仍是 edge-free block；
- memory rows 仍由外部协议固定；
- 文本 payload 数量由 CaT 的 X 槽位数量决定，不是手动指定的类别级常数。

## 4. 代码入口

- 核心实现：[x_conditioned_payload_partition.py](../../code/v2_llm_fusion/x_conditioned_payload_partition.py)
- 大图 runner：[run_public_continual_baselines_large_20261001.py](../../code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py)
- 新 arm：`cat_mmd_tag_x_slot_partition`
- 单元测试：[test_x_conditioned_payload_partition.py](../../code/v2_llm_fusion/test_x_conditioned_payload_partition.py)

运行时可以只测试新方法：

```powershell
python code/v2_llm_fusion/run_public_continual_baselines_large_20261001.py `
  --datasets arxiv bookchild sportsfit `
  --arms cat_mmd_tag_x_slot_partition `
  --source-root <base4-root> `
  --xpa-root <xpa-root> `
  --output-root <new-output-root> `
  --minilm <minilm-path> `
  --text-cache-root <text-cache-root>
```

## 5. 当前验证状态

已完成：

- Python 语法检查；
- 新模块单元测试 2 项；
- 可重复性检查；
- 文本长度和 source provenance 检查；
- trust-radius 检查；
- runner `--help` 入口检查。

尚未宣称：

- 三张大图上的性能提升；
- 对 BookChild 的恢复程度；
- 相对 CCTP、GECC-TAG 和 formal prompt-margin 的最终优势。

下一步应使用相同 base+4、5% memory、Student seeds 和 construction protocol，先跑该 arm 的三图 smoke，再跑正式六 seed 对比。
