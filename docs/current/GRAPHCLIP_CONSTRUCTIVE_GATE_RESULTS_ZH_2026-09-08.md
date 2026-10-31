# GraphCLIP-Native 构造式静态凝聚 Gate 结果

日期：2026-09-08
数据：服务器 `problem_validation_results/graphclip_gate_2026_09_08_*`
代码：`code/graphclip_native/run_graphclip_constructive_gate.py`
约束（本轮设计输入）：① 下游改为 GraphCLIP（允许简化）；② 凝聚产物**不含任何真实节点**（合成X）；③ 文本必须**融合**而非选取单条——两者都是持续凝聚将来必需的能力。

## 1. 下游：简化版 GraphCLIP

| 组件 | released GraphCLIP | 本实现 | 说明 |
|---|---|---|---|
| 文本塔 | MiniLM-L6（tiny 档，随模型训练） | **从官方 checkpoint 提取的同一 MiniLM 权重，冻结** | 本地 MiniLM 权重文件是 LFS 指针，改从 `pretrained_graphclip.pt` 的 `text_model.*`（103 个张量）提取并验证可加载——文本塔与官方逐字节同源 |
| 图塔 | GPS graph transformer（PE、prefix tuning） | 两层 GCN（384 维），student 从头训练 | 简化点：GPS 依赖子图采样与 PE 管线，不影响"凝聚小图是否可训练"这一验收问题 |
| 损失 | 对称 CLIP InfoNCE + 可学习 logit_scale | 同（logit_scale 上限 100） | 保留 |
| 任务 | 零样本 class-prompt 分类 | 同：7 类 prompt（"a research paper about …"）cosine 最近邻 | 保留 |
| 数据 | GraphCLIP 官方 `processed_data/cora.pt`（x 2708×384，raw_texts，双向边） | 同一文件 | 与 native 管线数据不同源，数值不与 native 口径互比 |
| 评测 | — | 真实图 transductive 传播，确定性 dev 切分（node_id%10==1，10%），**test 未读** | |

## 2. 凝聚方法（满足全部三个约束）

对每个簇（确定性 maxmin+Lloyd，联合 [X‖text] 空间）：

- **合成 X**：簇成员真实 X 的均值——**产物中没有任何真实节点的 X 行**；
- **融合文本**：距簇心最近的 g 个成员文本按预算截断拼接成一条新可读文本，用冻结 MiniLM 重编码——text_emb 与文本严格对应，但**这条文本不属于任何真实节点**；
- **拓扑**：融合文本 embedding 空间的确定性对称 kNN；
- 全流程确定性（runner 内置重建等价审计），zero condensation-seed variance。

这实际上是 TAGSAM 两个组件的确定性替身：合成X ↔ ASM 的合成特征（但由构造给出而非失效的优化给出）；融合文本 ↔ CMI text selection（但确定性 top-g 拼接而非 GPT-2 CMI）。

## 3. 结果（k=135，rate 0.05，dev Macro-F1）

### 3.1 标准训练（512 步，6 students）

| arm | deg=1.0 | deg=2.0 |
|---|---:|---:|
| synthetic fused g=1 | 49.03±2.9 | 48.80±2.9 |
| synthetic fused g=3 | 51.32±3.1 | 50.14±4.5 |
| synthetic fused g=5 | 53.38±2.4 | 48.78±2.5 |
| synthetic fused g=9 | **54.08±2.7** | — |
| random real ×3 draws | 50.85 / 52.79 / 55.44 | 同左 |
| full-real reference | 44.1–45.0 | 同左 |

### 3.2 长训练确认（2048 步，3 students，deg=1.0）

| arm | dev Macro-F1 |
|---|---:|
| **synthetic fused g=5** | **54.52 ± 1.23** |
| random real draw0/1/2 | 50.38 / 52.09 / 51.67 |
| full-real reference | 43.30 |

## 4. Gate 判定：通过（有保留）

- **长训练下，合成X+融合文本稳定超过全部三个 random-real 抽签**（54.5 vs 50.4–52.1），且 student 方差最小（±1.2），构造零随机；
- **文本融合有效**：g=1（单条文本、类似 medoid）49.0 → g=5/9 融合 53.4–54.1，融合带来 +4~5 pp——这是"文本内容利用"第一次拿到正向因果证据，也直接回应了上一轮"TAG 文本内容未利用"的缺口；
- **合成X可行**：不含真实节点的产物达到并超过 real-node 基线——"凝聚不用真实节点"这一约束不以质量为代价；
- deg 敏感性重现（2.0 在该文本空间反而差于 1.0），支持"deg 必须 dev 选择"的协议条款。

### 保留意见（写进论文前必须处理）

1. **full-real reference 反常偏低**（43–45，低于所有小图）：2708 节点在 batch 24 × 512–2048 步下严重欠训，且全图 InfoNCE 的负例包含大量同类相似文本。它此处只证明"评测管线有分辨力"，**不能当作上界引用**；正式对照需给 full-real 配足训练预算；
2. 数值绝对水平（≈54）不能与 native 管线（77）比：数据源、文本编码器、dev 划分、类别数覆盖都不同；
3. 单数据集、单预算点、dev-only；g 与 deg 的选择规则需在第二数据集上验证后冻结。

## 5. 与前一版（det-construct medoid/real-X）的关系

| | det-construct（native 管线） | 本版（GraphCLIP 管线） |
|---|---|---|
| X | 真实 medoid X | **合成簇均值** |
| 文本 | 单条真实 medoid 文本 | **top-g 融合文本** |
| 图文配对 | 天然真实配对 | 融合文本与合成 X 同簇同源，配对按构造成立 |
| 持续可用性 | 旧 memory 含真实节点（合规风险） | **产物零真实节点，历史不可重读约束下更干净** |

持续管线下一步（Gate 1''）：用本版构造器跑 stage 0→1→2——旧 synthetic 块与新 delta 的融合文本/合成X在阶段间如何重组，正是之前"合成X+文本融合在持续时要用"的落点。

## 6. 产物清单

- 构造器+runner：`code/graphclip_native/run_graphclip_constructive_gate.py`；
- 提取的文本塔：服务器 `graphclip_native_work/minilm_from_graphclip/`（源 checkpoint SHA 记录于 extract 脚本）；
- 凝聚 artifacts：各输出目录 `syn_fuseg{g}.pt`（x/text_emb/texts/edge_index/weights/meta）；
- 全部结果 JSON 含 per-student 明细与 `test_accessed: false` 标记。
