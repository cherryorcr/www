# P2-B 为什么没有稳定提升：原因诊断与下一轮实验

日期：2026-09-18  
依据：P2-B Cora/Citeseer 小图正式矩阵、逐顺序结果、机制日志和离线审计。

## 1. 结论

P2-B 不是“attention 没有训练起来”。它满足三个事实：

1. 外层损失可以穿过 probe student 的 Adam unroll，attention 梯度非零；
2. learned target 可以由保存参数重建，旧记忆、X 和拓扑没有被修改；
3. 但最终完整 student 的 Macro-F1 没有稳定增加。

当前最可信的解释是：

> student-guided attention 实际优化的是一个短步数、图文对比损失的代理，而最终评价是长程 student 训练后的 prompt Macro-F1；同时它只能在固定的 5 个片段和固定 Mean-X/拓扑上重新加权，无法修复当前合成图的结构和节点属性瓶颈。

## 2. 已由结果直接支持的原因

### 2.1 最终收益不稳定，且具有顺序依赖

student family 相对 center concat 的结果为：

| 数据集 | 最终差值 | 阶段均值差值 | 主要现象 |
|---|---:|---:|---|
| Cora | +0.18 | -0.43 | canonical +0.85，order1701 +1.13，order1702 -1.43 |
| Citeseer | -0.53 | +0.07 | canonical -0.70，order1701 -0.84，order1702 -0.05 |

Cora order1702 的问题尤其明显：stage 1 的 center 为 44.93，student-700 为 39.59；Citeseer canonical 最终新类 F1 从 center 的 45.17 降到 student-700 的 40.55。说明 attention 改文本后，不能稳定解决特定顺序下的新类塑性问题。

这说明主瓶颈至少部分来自：合成 X、旧/新传播关系、类别预算或阶段顺序，而不只是文本 target。仅增强文本 attention 不足以覆盖这些因素。

### 2.2 learned target 的变化很小

student arm 的平均 target cosine drift 约为 0.01--0.02。最终目标为：

\[
z=\operatorname{normalize}(0.75z_{anchor}+0.25z_{pool}),
\]

因此 attention 最多只控制四分之一的方向，而且还受到 anchor penalty 约束。结果是外层确实有梯度，但最终合成文本向量离原始 concat anchor 很近，能够改变 student 行为的自由度有限。

这解释了“机制测试通过但性能变化很小”：梯度存在不等于参数有足够的有效作用范围。

### 2.3 只学习重新加权，不能修复证据缺失

本轮的 keys 是预先选定的最多 5 篇文本片段。attention 只能改变这些片段的权重，不能：

- 选出最初 5 篇之外的文本；
- 生成缺失的事实或类别描述；
- 改变 Mean-X；
- 改变合成图的边；
- 改变旧类与新类的预算；
- 改变当前冻结拓扑的传播路径。

如果性能损失来自类内多模态被 Mean-X 抹平、文本片段没有覆盖新类判别信息，或者新节点无法通过拓扑影响旧块，重新加权不会有足够上限。

### 2.4 outer loss 与最终 Macro-F1 不是同一个目标

P2-B 的 student arm 优化的是：

- probe student 经过 16 步短 unroll 后，在 current-delta 真实图上的双向 InfoNCE；
- 旧 synthetic memory 的保持项；
- anchor 漂移正则。

最终评价却是：

- 标准 student 从头进行完整 canonical steps；
- 使用合成图进行训练；
- 在完整真实图上用类别 prompt 做 Macro-F1。

两者之间有三个差异：

1. 16 步短期响应不一定预测完整训练轨迹；
2. current-delta 对比损失不等于所有已见类别的 Macro-F1；
3. InfoNCE 的连续表示改善不一定改善 prompt 决策边界和类别间 Macro-F1。

所以当前的 attention 选择可能在短 probe 上降低 outer loss，却在完整训练或新类 prompt 分类上没有收益。

### 2.5 student 反馈的成本远高于收益

| 配置 | Cora 构造成本 | Citeseer 构造成本 | 说明 |
|---|---:|---:|---|
| center concat | 约 1.08 s | 约 1.33 s | 三阶段链平均 |
| semantic attention | 约 1.5 s | 约 1.7 s | 只优化文本重构 |
| student-guided attention | 约 19.0 s | 约 19.0 s | 包含 probe student unroll |

student arm 的额外成本约为 18 秒/三阶段链，而 Cora 最终只提升 0.18，Citeseer 反而下降 0.53。因此即使性能差距接近零，也不能把当前版本称为更高效的构造方法。

## 3. 可能但尚未单独证实的机制原因

以下是有代码和结果支持的假设，但还不能写成已证实结论。

### 3.1 双线性 attention 的有效学习速度偏慢

`Wq` 零初始化，修正项初始为零；`Wk` 的初始梯度受 `Wq=0` 影响，实际先由 `Wq` 产生变化，再逐渐改变 `Wk`。这满足了不同 construction seed 的相同初始输出，但可能降低 40 步外层优化的有效更新量。

### 3.2 64-token/固定五片段形成了上限

P2-B 没有比较更大的 token budget、更多片段或可学习离散选择。因此尚不能判断是 attention 学不到，还是输入证据已经不足。P2-A 的已有结果也表明不同文本池化在 Cora 和 Citeseer 上方向不一致，说明文本聚合存在数据集依赖。

### 3.3 current-delta 随机 source split 的统计目标可能太弱

outer train/validation 是按 source node 随机拆分，而最终指标按类别 Macro-F1 统计。随机 split 不保证每个类、每个簇和每个多模态子结构均衡；它可能让 attention 学到局部 source embedding 对齐，却没有改善类别级新类塑性。

### 3.4 旧记忆保持项可能压低新类塑性

student outer objective 同时包含 current-delta loss 和旧 memory loss。Citeseer 最终 student 相对 center 的新类平均差值为 -1.14pp，旧类差值为 -0.23pp，表现为新类更容易受损。这里不能简单断言旧项权重过大，因为还没有做权重消融，但它是必须隔离的候选原因。

## 4. 下一轮最小诊断矩阵

不建议立即扩大到大图。每个实验先使用一个固定 construction seed 和少量 report seeds 做机制筛选，只有通过后再用两个 construction seeds 和六个 report seeds确认。

### 优先级 P2-B-1：确认是不是 residual 太保守

固定片段、X、边、student 目标和 probe 预算，只比较：

| 版本 | residual | anchor penalty |
|---|---:|---:|
| 当前 | 0.25 | 0.10 |
| 中等 | 0.50 | 0.10 |
| 无强锚 | 1.00 | 0.05 或 0 |
| 纯 pooled 控制 | 1.00 | 0 |

同时报告 target drift、outer loss、完整 student F1、旧/新 F1 和成本。若 drift 增大但 F1 不变，瓶颈在目标错位或 X/拓扑；若 F1先升后降，则当前正则确实过强。

### 优先级 P2-B-2：确认短 unroll 是否误导排序

固定 learned attention 候选，比较 probe horizon `16/64/256/完整 steps` 的：

- candidate ranking Spearman/Kendall；
- selected target cosine；
- 最终完整 student F1；
- 新类和旧类 F1。

如果 16 步与完整训练排序相关性低，student-guided attention 失败的主要原因是 selection horizon，而不是 attention 表达能力。

### 优先级 P2-B-3：拆分新类塑性与历史保持目标

比较：

- current-delta only；
- old-only；
- current/old = 1:1；
- 当前权重 `current + 0.25 old`。

不能使用 report/test 标签；仍使用合法 source text/graph 目标。重点看 Citeseer 新类 F1是否恢复，以及 Cora order1702是否改善。

### 优先级 P2-B-4：提高 outer split 的类别/簇覆盖

将随机 source split 改为分层 split：每个当前类别、每个非空簇都至少有 train/validation 代表；仍不读取 report/test 标签。若分层后 outer loss 与最终 F1相关性提高，原先的随机 split 是重要噪声源。

### 优先级 P2-B-5：确认 attention 是否受固定候选限制

在预算相同的条件下增加：

- `g=5` 与 `g=10`；
- 64 与 128 content tokens；
- 固定片段重加权与可学习 value residual。

这一步必须单独收费文本编码和 probe 成本，并保留可读来源证据。若更大候选空间仍无收益，应停止 P2-B，把资源转回 X/拓扑/预算分配。

## 5. 对当前方法的实际判断

当前默认方法仍应是：

```text
dispersion allocation
+ 类内图文聚类
+ Mean-X
+ center concat / joint_g5_c320
+ frozen-old topology
+ response off
```

P2-B 的正确论文定位是：

> 我们实现了一个受控的 student-guided text compressor，并发现非零 student meta-gradient 并不自动带来持续凝聚性能提升；在当前小图协议下，收益受短视代理目标、固定证据集合和合成 X/拓扑瓶颈共同限制。

不能写成“attention 无效”或“student-guided 方法被证明无用”。更准确的说法是：当前受限版本未通过性能和成本门禁。
