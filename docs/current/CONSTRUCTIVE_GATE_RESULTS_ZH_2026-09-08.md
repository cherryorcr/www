# 构造式 Gate 结果：确定性静态 TAG 凝聚方案

日期：2026-09-08
数据：服务器 `problem_validation_results/constructive_gate*_2026_09_08/`
实现：`constructive_init.py`（确定性构造器）+ `run_constructive_gate.py`（gate runner，含逐字段重建等价审计）
目标定位：**不是 SOTA，是稳定**——为持续 TAG 凝聚提供一个无 condensation-seed 随机性的可靠静态起点。

## 1. 方案定义（det-construct）

全部无标签、全部确定性、无任何训练/优化：

1. **选点**：mass 加权贪心 maxmin（farthest-point）播种（首点=最大 mass，平局取最小 index），替换原 frozen compact 的 multinomial k-means++ 随机播种；
2. **细化**：标准 Lloyd 迭代（joint 归一化 X/text 空间，25 轮或收敛）；
3. **代表**：每簇 medoid 给出真实可读文本与 origin_id；
4. **X**：medoid 的真实 X（`real` 模式；centroid 模式一并实现作消融）；
5. **topology**：冻结 BERT CLS 文本空间的确定性对称 kNN（cosine top-k 对，边预算 `int(n*degree)` 与原 link-predictor 路径一致），替换原"用 teacher embedding 训练 MLP link predictor"的含随机拓扑。

runner 内置确定性审计：同输入重建两次，逐字段（x/text_emb/edge_index/weights/origin_ids/stage）要求完全相等，不等即报错。

## 2. 结果（stage-0，rate=0.05，k=25，dev Macro-F1）

### 基线：现有 frozen compact（含随机播种 + learned topology）

| condensation seed | frozen dev |
|---:|---:|
| 44 | 75.64 |
| 45 | 80.05 |
| 46 | 57.83 |
| 47 | 75.41 |
| 48 | 49.92 |
| **跨 seed** | **mean 67.77，std 11.74，min 49.92** |

### 确定性构造（零 condensation-seed 方差）

| 配置 | students | mean | std(student) |
|---|---|---:|---:|
| centroid X，deg1.5 | 100–105 | 64.56 | 9.77 |
| real X，deg1.5 | 100–105 | 66.87 | 8.38 |
| real X，deg1.0 | 100–105 | 76.25 | 5.00 |
| **real X，deg2.0** | 100–105 | **78.14** | 5.84 |
| real X，deg2.0（独立复核批） | 106–111 | **76.31** | 6.66 |
| real X，deg2.5 | 100–105 | 70.62 | 5.81 |
| real X，deg3.0 | 100–105 | 42.78 | 6.74 |

## 3. Gate 判定：通过

选定配置：**det-construct（real X，text-kNN deg=2.0）**，12 个独立 student 合并 mean≈77.2。

| 判据 | 结果 |
|---|---|
| ≥ frozen 跨 seed 均值 67.77 | ✅ +9.5 pp |
| ≥ frozen 最差 seed 49.92 | ✅ +27.3 pp（确定性方案没有"坏seed"这一失败模式） |
| 与 frozen 最好 seed 80.05 比 | −2.9 pp，可接受（稳定性目标下不要求超过最佳抽签） |
| condensation-seed 方差 | **0（按构造）**，vs frozen 的 std 11.74 |
| 独立 student 批复核 | ✅ 78.14 vs 76.31，一致 |
| 确定性重建审计 | ✅ 全字段相等 |

参考系：Full-real=84.9（dev），旧 Offline ASM=76.5，旧 frozen 均值 67.8。确定性构造以零优化、零随机达到旧 ASM 最优水平。

## 4. 归因与注意事项

1. **方差来源确认**：frozen 的 30pp 跨 seed 波动来自构造器自身的随机性（multinomial 播种 + teacher 依赖的 learned topology），不是数据或协议——确定性化直接消除；
2. **topology 是主要杠杆**：同一选点下 deg 从 1.5→2.0 提升 11pp，3.0 崩溃（过密的文本 kNN 把不同类节点连通，消息传播污染）。deg∈[1,2] 是安全区，2.5 开始退化。**deg 需在新数据集上按 held-out dev 重选，不可硬编码**；
3. real X 稳定优于 centroid X（消融方向与此前 fine-X 实验的 seed 分歧一致——确定性选点下 real 的优势显形）；
4. student-seed std 仍有 5–7：这是零样本七类 Macro-F1 的固有噪声，报告协议保持多 student 均值；
5. 本结果为 Cora 开发证据；deg 与 x_mode 的选择规则（dev-only）需写入 PROTOCOL V2 并在第二数据集上验证。

## 5. 对持续研究的直接影响

静态起点问题关闭。持续管线更新为：

```
每阶段静态算子 = det-construct(real X, text-kNN, dev 选 deg)
持续更新     = 确定性构造式重组：
               旧 memory 收缩（P3 的质量自适应分配）+ 新 delta 的 det-construct 块
优化组件     = 条件式修复（仅当构造结果 dev 异常低时启用，作消融）
```

下一步（新 Gate 1'）：用 det-construct 重跑 stage 0→1→2 持续链路，与旧 Append/frozen 链路对比。预期收益：持续误差分析不再被静态构造器的随机性污染，P3 的自适应分配可以在干净地基上重新评估。
