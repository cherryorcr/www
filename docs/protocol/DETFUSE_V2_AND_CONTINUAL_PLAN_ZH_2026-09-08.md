# 静态凝聚升级（det-fuse v2）与持续凝聚实现规划

日期：2026-09-08
性质：设计与预注册文档。依据 P1–P5、构造式 gate、GraphCLIP gate 的全部实验结论。
原则：**回收 TAGSAM 各组件的"能力"，拒绝其"随机性与失效的 bi-level 优化通路"；每个升级组件独立过 gate（≥4 seeds），不达标即回退。**

## 0. 谱系定位

当前合成X（簇均值）属于**图粗化/分布匹配一阶矩**家族（X_syn = P^T X_real），
不属于 GCond/SFGC/TAGSAM 的**优化式 synthetic X** 家族。
弱点：只保一阶矩；优化式X的理论上限（编码二阶/判别信息）未被否定——
被否定的只是 ASM 这条 bi-level 实现路径（P2：loss 100步降0.2–1.2%，官方同病）。

## 1. TAGSAM 组件逐项处置

| 组件 | 处置 | 方案 | 拒绝/保留原因 |
|---|---|---|---|
| CMI text selection | **回收（确定性化）** | det-CMI：候选=距簇心最近2g条；GPT-2(本地 gpt2-large)条件困惑度增量打分，纯前向无采样；确定性top-g按分数降序拼接 | CMI打分本身无随机性；现几何top-g不衡量文本信息量 |
| ASM 合成X优化 | **bi-level拒绝；上限用一阶精炼回收** | anchored refinement：≤20步，L=图文InfoNCE+λ·簇响应分布匹配，梯度直达X（无student unroll），early-stop+dev回退，gate不过即整体丢弃 | P2/P5：unrolled梯度近零、响应loss可优化但不保可重训性；text-anchor实验证明一阶InfoNCE通路有效 |
| add-adj learned topology | **teacher依赖拒绝；换更强的粗化拓扑** | A_syn=P^T A P（簇间真实边计数，mass归一化）top-int(k·deg)对；可选与文本cosine混合，α由dev选 | teacher随seed变→拓扑随seed变（frozen方差来源）；且当前文本kNN完全没用真实边，是自身缺口 |
| 图文联合凝聚空间 | 保留 | [norm X ‖ norm text] 联合聚类 | 已验证 |

### det-fuse v2 全貌
```
聚类   maxmin+Lloyd on [X‖text]（确定性）
X      簇均值 →（gate通过则）一阶锚定精炼
文本   det-CMI top-g 融合（GPT-2打分，确定性）+ 冻结encoder自编码embedding
拓扑   粗化 P^T A P（+文本cosine混合，α dev选）
审计   重建等价；消融矩阵 {几何/CMI选文}×{均值/精炼X}×{kNN/粗化/混合拓扑}
```

## 2. 持续凝聚设计

核心：**持续更新 = 带 mass 的确定性再聚类 + 分层文本再融合 + 拓扑再投影**。

### 2.1 Memory 形态
`M_t = {合成X, 融合文本+embedding, 粗化边, mass, 簇内聚度, stage, fusion_depth}`
零真实节点/teacher/优化器状态；内聚度与depth计入 aux_bytes。

### 2.2 阶段更新算子
1. **质量自适应预算分配**（回应P3）：旧簇保留配额=f(内聚度, mass占比, 类覆盖)，无标签确定性规则；
2. **加权联合再聚类**：旧簇作 mass=|簇| 重节点与新delta(mass=1)同场 maxmin+Lloyd，目标 k_{t+1}。保留/合并收缩/新增是同一算子的涌现结果，取代 Append/Union/Protected 人工arm；
3. **分层文本再融合（防稀释）**：纯新簇=det-CMI；纯旧簇=原文继承；混合簇=旧融合文本作带mass预算的首段+新成员CMI top-g'，fusion_depth+1，depth≥2锁定旧段预算；
4. **拓扑再投影**：旧粗化边mass加权继承 + delta真实边新投影；跨阶段边仅由文本cosine项提供（与"跨阶段边未确认是瓶颈"一致）；
5. （可选，gate守门）一阶锚定精炼 ≤20步。

### 2.3 对已确认失败的规避映射
| 失败（证据） | 规避 |
|---|---|
| 递归优化≤初始化（P2/P5, 5 seeds） | 更新主体是再聚类；精炼可丢弃 |
| 图文漂移（置换/alignment） | embedding由自身融合文本编码，结构上不可能漂移 |
| 旧memory质量传递（P3/组成归因） | 内聚度使差簇被合并稀释而非原样传递 |
| 构造器seed方差污染归因（构造式gate） | 全链确定性，阶段差=真实持续效应 |

## 3. Gate 序列（PROTOCOL V2 口径）

1. **Gate S2**（静态升级）：三组件消融，各自 ≥4 seeds 稳定不降级才并入；
2. **Gate 1''**（持续核心）：stage 0→1→2；对照=构造版Append/固定分配/自适应分配/full-history；通过=自适应≥固定 on ≥4 seeds 累计dev，且fusion_depth控制的旧阶段指标不劣于无控制；
3. **Gate 2''**：第二TAG数据集（test封存）重复；
4. **Gate 3''**：全冻结后一次性读封存test。

## 4. 风险清单
- 混合簇再融合语义稀释（Gate 1''第一验证点，depth锁定为对策）；
- 粗化拓扑在极稀疏delta下退化（文本cosine混合为兜底）；
- det-CMI的GPT-2打分对长文本截断敏感（固定max_length并记录）；
- 精炼组件可能全seed不过gate（预期内，直接回退纯构造，不影响主线）。
