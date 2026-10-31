# P0 比较控制完成报告

日期：2026-09-18  
状态：协议实现与本地结构审计完成；不覆盖既有冻结 v4 结果。

## 1. P0 要解决的问题

小图多顺序结果中，v4 显式 class-row 构造会按类别 ID 排序，而普通 dispersion 路径曾按传入的类别列表顺序生成类块。随机类别顺序下，即使配额完全相同，合成节点的 row order、后续拓扑 tie-breaking 和固定 seed minibatch 都可能不同。因此此前的 v4-versus-dispersion 差值不能完全归因于 response selector。

P0 固定以下内容：

- 阶段顺序：保持数据流中的阶段顺序；
- 阶段内 class 顺序：按 class ID 升序；
- 阶段内 source row 顺序：按 source/original ID 升序；
- budget、student、selection seed 和 artifact hash：写入 manifest；
- response on/off：在相同枚举控制下进行配对。

## 2. 已完成的实现

### 2.1 统一枚举协议

在 `code/v2_llm_fusion/protocol_v3.py` 中加入：

```text
ENUMERATION_PROTOCOL = p0-canonical-class-then-source-id-v1
```

新增 `canonicalize_class_groups`，它会：

1. 保留阶段顺序；
2. 对每个阶段的 class ID 排序；
3. 拒绝重复 class 和跨阶段重叠；
4. 拒绝空 stage。

统一 runner 的 `load_unified`、小图顺序 runner 的 `set_stages` 均使用该协议。阶段 source IDs 也显式排序，避免 `torch.where` 的隐式实现细节成为实验协议的一部分。

### 2.2 统一构造路径

以下路径现在统一按 canonical class order 构造：

- `allocate_rows_by_class`；
- 类内 subcluster 记录；
- 普通 synthetic block；
- response marginal 的 class proposal；
- 小图确认 runner 的 stage source IDs。

因此在相同 budget、相同 cluster 输入和 response 关闭条件下，v4/dispersion 使用同一类块枚举顺序。

历史结果目录保持不变；新运行必须使用新的 enumeration protocol，并使用新的 output directory 保存。

### 2.3 Manifest 指纹

`run_unified_v3_continual.py` 和 `run_small_order_cost_confirmation.py` 的新 manifest 现在写入：

- `enumeration_protocol`；
- `class_order_policy`；
- `stage_ids_sha256`；
- student、method、selection seed policy；
- 原有 data、code、prompt、stage manifest hashes。

如果已有 output directory 的 manifest 与新配置不一致，runner 仍然拒绝覆盖，要求使用新目录。

## 3. 审计结果

新增测试文件：

`code/v2_llm_fusion/test_p0_comparison_control.py`

同时更新：

`code/v2_llm_fusion/test_small_order_cost_confirmation.py`

本地执行：

```text
python -m unittest -v test_p0_comparison_control.py test_small_order_cost_confirmation.py
Ran 10 tests in 0.008s
OK
```

测试覆盖：

| 检查 | 结果 |
|---|---|
| 阶段顺序保持、阶段内 class 排序 | 通过 |
| 重复或跨阶段 class 拒绝 | 通过 |
| allocation 不受输入 class list 顺序影响 | 通过 |
| synthetic row order 不受输入 class list 顺序影响 | 通过 |
| 原小图阶段划分测试 | 通过 |
| 原 selector 配置冻结测试 | 通过 |
| 原 raw 数值漂移审计测试 | 通过 |

## 4. P0 的边界

本轮完成的是**比较控制协议和实现审计**，不是重新运行全部 2 数据集 x 3 顺序 x 13 方法的长实验。已有长实验产物没有被事后重写，也没有使用新协议重命名旧结果。

response on/off 的收益可靠性、selector ranking 与 canonical ranking 的一致性属于 P3-A。P0 只保证后续 P3-A 的配对实验不再混入 class enumeration confound，并不会预先证明 response 有性能收益。

## 5. 后续执行入口

下一步使用新的 P0 protocol 运行配对实验：

1. `response off`：固定 dispersion allocation，作为低成本 synthetic control；
2. `response on`：只允许固定总 rows 的 class-row transfer；
3. 两者使用相同 stage classes、source row order、cluster/text/topology 和 student protocol；
4. 使用独立 selection seed group 与 report student seed group；
5. 报告 chosen allocation、selection regret、fallback、old/new F1、构造成本和 selector 成本。

只有该配对实验完成后，才能判断 response 是否具有独立收益；旧随机顺序结果不能直接承担这一结论。

