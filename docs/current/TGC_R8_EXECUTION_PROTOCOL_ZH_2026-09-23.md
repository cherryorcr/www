# TGC-R8 小图塑性验收协议

## 目的

R8 只回答一个问题：在已冻结的 TGC-R 构造上，确定性稀疏实现与 C0 锚定新合成节点特征是否能改善新类塑性。它是 R7 的小图确认实验，不预先晋升为论文主方法。

## 固定协议

- 数据集：Cora、Citeseer。
- 类别顺序：`order19600`、`order19609`、`order19610`；这些顺序独立于 R7 使用的 `order19605/19606/19608`。
- 三阶段、累计 5% synthetic memory、阶段预算和原始流由统一 v3 数据协议生成。
- 下游 student：既有两层 GCN、冻结 MiniLM 输入、双向图文 InfoNCE；从头训练，batch 16，Adam 学习率 `1e-3`，步数 `max(50, min(2048, 16K))`，报告 seeds `220..225`。
- teacher seed groups：`(8800, 8801)` 和 `(8810, 8811)`，全部报告，不按结果选择。

## 方法臂

- `c0`：当前 C0 控制。
- `k_center`：当前强 coreset 参照。
- `dense_g0`：原始 dense TGC-R 构造参照。
- `det_g{0,1}_l{000,025,050,100}`：确定性 segmented-SpMM TGC-R；只将新 synthetic block 的 `X` 替换为

  `X_new = (1-lambda) X_C0_new + lambda X_TGC_new`。

  旧行、旧边、新块文本、新块质量和拓扑全部保持对应 TGC 输出。`lambda=1` 是未锚定确定性 TGC-R，`lambda=0` 只锚定新块特征。

## 约束与审计

- 构造只能读取上一阶段 synthetic memory 和当前 raw delta；历史原始图不可回读。
- report/test labels 不参与构造或选择；所有 lambda 和 teacher group 预注册并完整报告。
- 旧 memory 行和旧边必须 bitwise 保持；锚定只能修改当前新块 `X`。
- segmented-SpMM 的前向、反向和重复执行与 dense 参考对齐；Cora/order19600/stage1 额外执行 dense-vs-segmented、重复构造和哈希审计。
- 先完成 `prepare -> seal -> evaluate -> audit`，再汇总最终 F1、old/new F1、stage mean、最差顺序、构造成本和数值误差。

## 晋升判断

R8 只有在新类 plasticity 有稳定改善、旧类保持没有明显损害、且收益不依赖单个顺序或 teacher seed group 时，才进入后续大图验证。确定性实现本身是复现基础设施，不作为算法创新点单独宣称。
