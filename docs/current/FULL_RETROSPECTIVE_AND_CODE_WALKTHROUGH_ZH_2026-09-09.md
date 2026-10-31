# 完整复盘：从 TAGSAM 复现到 frozen_v1 的每一步取舍 + 代码级方法讲解

日期：2026-09-09
用途：记录 2026-09-08 至 09-09 两天内全部实验、每步取舍逻辑、以及冻结方法的逐行实现讲解。

---

# 第一部分：决策台账（每个实验→结果→取舍→指向方法的哪一部分）

## 第 0 步：起点状态（2026-09-08 早）

继承的状态：native 管线下 Text-Anchored ASM 已实现，但 stage-1 严格递归中所有轨迹选 outer=0，stage-2 增益跨 condensation seed 不稳（44 为负 45 为正）。原计划是实现 response-retention 修复递归优化。

**首个取舍**：不直接实现新 loss，而是先做三件便宜的诊断（论文校准、优化失败根因、预算扫描）。理由：如果 outer=0 是静态问题，持续 loss 修不好它。

## 第 1 步：TAGSAM 论文校准（P1）

**做了什么**：找到论文原文（WWW 2026, arXiv 2606.03839），提取 Cora 主表（five-way zero-shot Accuracy：size50=73.8±1.8，Whole=78.7）；与本地官方代码复现值并排；核查 text selection 是否被执行（distill.py L81 确认执行）；分析官方 distill 日志的 loss 曲线。

**结果**：同口径（五类 size50）论文 73.8 vs 本地 master 38.2–42.8 / add-adj 51.2–52.9，差距 21–35 pp 未决。**新证据：官方 500 步 distill 的 syn_loss 只降 4.7%（0.730→0.696）**。

**取舍**：不把复现论文数字设为前提；官方数字引用时只用"我们的复核值"。
**指向**：官方链路自己的优化就近乎失效——为第 2 步的"优化失败"假设提供了独立佐证。

## 第 2 步：静态优化诊断（P2，15 runs）

**做了什么**：seeds 44–48 × outer_lr {0.01,0.003,0.001}，记录 ASM loss 曲线、X 梯度范数、dev 轨迹，与 frozen compact 基线对照。

**结果**：ASM loss 100 步仅降 0.2–1.2%，梯度范数 0.0002–0.0012（近零）；但 student 结果变化剧烈且方向由初始化质量决定——frozen 差的 c48（49.9）被"优化"+24.9，frozen 好的 c44/c47（75+）选 outer=0。frozen 自身跨 seed 49.9–80.1（std 11.7）。

**取舍**：**放弃 bi-level 优化通路**（梯度穿 10 步 inner unroll 后失效，两套实现一致）；同时确认"之前的 stage-1 outer=0 是静态问题不是持续问题"——原 response-retention 计划的立足点被拆掉一半。
**指向方法**：(a) 初始化质量是最大杠杆→方法核心应是构造而非优化；(b) frozen 的 30pp seed 方差来自构造器随机性→确定性化的动机。

## 第 3 步：预算分配诊断（P3）

**做了什么**：固定 stage-2 总预算 94 行（native 管线），扫旧/新分配 67/27、55/39、47/47、32/62，seeds 44/45。

**结果**：方向依赖 seed——44 保旧最优（旧 memory 好），45 让新最优（旧 memory 差，s1 只有 42.96）。

**取舍**：固定比例不是答案；"质量自适应分配"记为设计方向。
**指向方法**：最终演化为 c 规则（每新类保底行数）+ 按 mass 删行的确定性收缩——当构造器确定性化后旧 memory 质量稳定，"自适应"退化为一个静态保底规则即可。

## 第 4 步：response-retention gate（P5，20 runs）

**做了什么**：实现档 A（self-distillation：保存 teacher 对 synthetic rows 自身的 class-prompt logits，KL 保持），seeds 44–48 × λ_r {0,0.05,0.1,0.3}，stage-1 gate。

**结果**：retention KL 从 0.06–0.32 优化到 0.001–0.005（降 92–99%），**但 0/5 seed 通过 gate**（全部选 outer=0）；λ_r 只能减缓下降且方向依赖 seed（c44 保护 +11 但仍低于 outer0，其余四 seed 有害）。

**取舍**：**放弃一切"冻结模型响应约束"类目标**。机制结论：可满足的响应约束 ≠ 可重训性——student 是从头重训的，它对小图的利用方式与任何冻结模型的前向响应不同构。
**指向方法**：反面奠基。frozen compact 之所以强，是因为它由真实数据统计直接构造、从未被任何响应目标扭曲——"可重训性只能由数据统计构造保证"成为方法哲学。

## 第 5 步：构造式 gate（native 管线）

**做了什么**：实现 det-construct：maxmin 播种（替代 multinomial k-means++）+ Lloyd + medoid 真实图文对 + 文本 kNN 拓扑（替代 teacher-trained link predictor），degree 扫 {1.0,1.5,2.0,2.5,3.0}，两批独立 student 复核，内置重建等价审计。

**结果**：real-X deg2.0 = 78.14/76.31（两批），**零 condensation-seed 方差**，vs frozen 均值 67.8（std 11.7、最差 49.9）；deg 敏感（3.0 崩到 42.8）。

**取舍**：接受"确定性下限"换掉"随机上限"；deg 定为唯一需 dev 选择的结构超参。
**指向方法**：静态构造器的骨架（选点+medoid+文本kNN）在此定型。

## 第 6 步：GraphCLIP 管线迁移（用户约束驱动）

**约束**：下游改 GraphCLIP（可简化）；产物零真实节点（合成X）；文本融合而非选取。

**做了什么**：从官方 `pretrained_graphclip.pt` 提取 MiniLM text tower（本地权重是 LFS 指针）；图塔以 GCN 替代 GPS（受控比较）；构造器改为合成X（簇均值）+ top-g 文本拼接重编码；fuse_g 扫 {1,3,5,7,9}，长训练确认。

**结果**：g=5 长训 54.52±1.2 > random real 三抽签 50.4–52.1；**融合因果证据**：g=1→g=5/9 提升 +4~5 pp。

**取舍**：GPS→GCN 的简化换取"同一 student 吃不同图"的可归因性；文本融合方式选最朴素的几何 top-g 拼接（后来 S2 证明这个"朴素"恰是对的）。
**指向方法**：合成X+融合文本+自编码配对的最终形态在此定型。

## 第 7 步：Gate S2——TAGSAM 组件确定性回收（6 arms）

**做了什么**：三个组件各自确定性化后消融：det-CMI 文本选择（GPT-2 条件困惑度打分，纯前向）、粗化拓扑 P^TAP（真实边投影）、一阶锚定 X 精炼（InfoNCE+分布锚，梯度直达 X）。

**结果**：**三个全部降级**——CMI −7.2（novelty 与 off-topic 不可分，污染融合 embedding）、粗化 −10.4（低预算下簇间边计数被大簇垄断，hub 化）、精炼 −13.0（X 被拉向代理算子的对齐方向，student 的 GCN 不是那个算子）；三组件叠加最差（40.6）。

**取舍**：v1 构造器整体冻结；**"一阶统计量之外的任何加工都不并入"**成为方法的负定义。精炼失败是第三条 X-加工通路的失败（bi-level/响应约束/一阶代理），证据链闭合。
**指向方法**：方法的"简单"从此有了系统证伪的支撑，不是偷懒。

## 第 8 步：Gate 1''——持续算子（3 arms × 3 stages，含过训练修复）

**做了什么**：类别增量 Cora（[0,1]→[2,3]→[4,5,6]，预算 32→87→121），比较 append_fixed（旧 memory 冻结+新块独立构造）、recluster（mass 加权联合再聚类+分层文本再融合+depth 防稀释）、full_history（重读历史一次性重凝聚）。第一轮发现 32 行小图在 2048 步下崩溃（28.7 vs 64 步的 70），用 steps=min(2048,16k) 修复后全部重跑。

**结果**（scaled）：append s2 cum 53.56 > full_history 51.63 > recluster 44.74；recluster 全面劣于 append；full_history 的 s1 反而最差（41.9，四类混合聚类被大类主导）。

**取舍**：**"只追加、不重组"**定为持续算子；recluster 降级为消融。精心设计的 depth 防稀释、mass 继承等机制随 recluster 一起放弃——服从证据而非沉没成本。
**指向方法**：持续协议主体在此定型。

## 第 9 步：预算扫描 + raw 上界补全

**做了什么**：s2 预算 121 固定，扫 keep87/77/67/57（按 mass 删行）；补之前缺失的 raw 直训上界（2437 节点，steps 配平 8192）。

**结果**：raw=45.3 **低于**三个 compact arm——凝聚即去噪（raw 图的同类文本在 InfoNCE 里互为负例）；分配非单调，keep77/new44 塑性 +6.0 cum −0.4，再扩全崩。

**取舍**：动态预算机制不立项（收益窗口窄、协议灰色、成本高）；定为静态规则 `新块=max(每新类15行,余量)`。
**指向方法**：c 规则定型；"compact>raw"成为论文独立发现。

## 第 10 步：Gate 2''——Citeseer 迁移验证

**做了什么**：6 类 Citeseer（[agents,ML]→[IR,DB]→[HCI,AI]，预算 50→110→143），类映射经 mean-embedding 审计，全套 arm 重跑。

**结果**：核心排序迁移（append 61.26 最优、full_history 最差 49.64、compact>raw）；**新发现**：recluster 当前阶段反超 +14.8（audit：54 混合簇让新类借用旧簇容量）——recluster 被重新理解为"隐式预算再分配"。

**取舍**：三组实验（append/recluster 之争、预算扫描、P3）统一为"保持-塑性预算前沿"框架；论文不声称单点最优，报告前沿。
**指向方法**：方法定位为"前沿上方差最小、累计不劣的运行点"。

## 第 11 步：修正清单 + 官方对比（回应 review 质疑）

**做了什么**：(a) 直接评测官方预训练 checkpoint（GPS-12L-1024，state_dict 零缺失加载，官方 ego 子图+RW-PE 协议）；(b) 等价审计：从 raw 重建"分层配额一次性凝聚"与 append memory 逐字段比较；(c) Citeseer 预算扫描验证 c 迁移。

**结果**：(a) 官方 Cora 57.47 / Citeseer 62.90，我们 54.5/61.3，差距 <3pp；(b) **全字段逐位相等**——append ≡ 分层配额一次性凝聚，持续约束代价恒为零（结构性结果）；(c) c=15 两数据集安全（Cora 触发补塑性、Citeseer 不触发恰为最优）。

**取舍**："分阶段>一次性"的主张升级为等价性表述（免疫"构造器弱"质疑）；c 冻结为常数，数据驱动版本记 future work。

## 第 12 步：Gate 3''——冻结与 test 确认

**做了什么**：代码快照 frozen_v1（本地+服务器+SHA）；test=node_id%10==2（标签零历史访问）一次性读取，两数据集全 arm。

**结果**：全部主张在 test 确认；方法两数据集 cum 均居首（Cora 58.07 / Citeseer 56.99）；c15 规则 test 净收益 +2.85；无 dev→test 乐观偏移。

---

# 第二部分：冻结方法的代码级讲解

以 `code/frozen_v1_2026_09_09/run_gate1_continual.py` 为主线（gate3/corrections 引用同一模块）。

## 2.1 冻结文本编码器 `FrozenMiniLM`

```python
class FrozenMiniLM:
    def __init__(self, path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(device).eval()
    @torch.no_grad()
    def encode(self, texts, batch_size=64, max_length=256):
        ...
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)  # attention-mask 均值池化
        parts.append(F.normalize(pooled.float(), dim=1).cpu())
```

权重来自官方 checkpoint 的 `text_model.*`（103 个张量，提取脚本 extract_text_tower.py）。均值池化与官方 `mean_pooling` 逐式一致。全程 `no_grad+eval`——文本塔是纯特征提取器，不参与任何训练。

## 2.2 确定性加权聚类 `weighted_cluster(points, mass, k)`

```python
first = int(torch.argmax(mass))                    # 首个中心=最大mass（平局取最小index，argmax语义保证）
centers = [first]
dist = torch.cdist(points, points[first:first+1]).flatten().square() * mass
for _ in range(1, k):                              # maxmin播种：每次取加权距离最大的点
    dist[torch.tensor(centers)] = -1.0
    nxt = int(torch.argmax(dist)); centers.append(nxt)
    dist = torch.minimum(dist, ...)                # 维护到最近中心的距离
```

三个确定性保证：argmax 无采样；`assign[centers_idx] = arange(k)` 强制中心归属自己簇（防平局漂移）；Lloyd 用 `allclose(atol=1e-6)` 收敛判定，mass 加权质心 `(points[members]*m[:,None]).sum(0)/m.sum()`。**mass 参数是持续算子的关键**：静态调用时 mass=1（普通聚类）；recluster 调用时旧簇 mass=|簇|——同一个函数服务两种场景（虽然 recluster 最终被否决，接口保留作消融）。

## 2.3 静态构造 `condense_static(x, text_emb, texts, k, encoder)`

```python
joint = torch.cat((F.normalize(x.float(),1), text_emb), dim=1)   # 图文联合空间(768维)
assign = weighted_cluster(joint, torch.ones(len(x)), k)
for j in range(...):
    members = torch.where(assign == j)[0]
    syn_x.append(x[members].float().mean(0))          # ① 合成X=簇均值（零真实行）
    fused.append(fuse_new_only(joint, members, texts))# ② 融合文本
    mass.append(float(len(members)))                  # ③ mass=簇大小
emb = encoder.encode(fused)                           # ④ 融合文本自编码→配对按构造成立
```

`fuse_new_only` 的融合规则：

```python
d = torch.cdist(joint[members], center).flatten()
order = members[torch.argsort(d)][:g]                 # 距簇心最近的g=5条（几何同质性优先——S2证明CMI打分反而有害）
per = max(48, budget_chars // max(1, len(order)))     # 总预算320字符均分，每条最少48
return " ".join(texts[int(i)][:per] for i in order)   # 截断拼接=新的可读文本
```

memory 数据结构：`{x, text_emb, texts, mass, depth, stage}`——depth/stage 是审计字段，**不含任何真实节点行、teacher 状态或优化器状态**。

## 2.4 持续协议（main 循环的 append 分支）

```python
k_t = budgets[stage]                                   # floor(0.05 × 累计训练节点)
k_new = k_t - len(memory["x"])                        # 预算余量给新块
block = condense_static(dx, de, dt, k_new, encoder)   # 只对本阶段delta构造
block["stage"] = torch.full((len(block["x"]),), stage)
memory = {key: torch.cat((memory[key], block[key])) ...}  # 原样拼接，旧行零改动
```

c 规则与收缩（gate3/corrections 中的 `shrink_by_mass`）：

```python
# 新块配额 = max(每新类15行, 余量)；Cora s2触发：new=45, keep=76
order = torch.argsort(memory["mass"], descending=True, stable=True)[:keep]
order = torch.sort(order).values                      # 保留原顺序；stable排序保证平局确定性
out = {k: memory[k][order] ...}                       # 只删行——留下的x/text/embedding逐位不动
```

**等价审计证明的性质**：由于 `condense_static` 确定性且 append 各块相互独立，`M_t = block_0 ⊕ block_1 ⊕ ... ⊕ block_t` 与"允许重读全部历史、按阶段配额分别聚类再拼接"逐位相等——持续约束在此构造器下代价恒为零。

## 2.5 拓扑与训练图 `memory_to_trainable(memory)`

```python
pairs = torch.triu_indices(k, k, offset=1)                        # 全部无序对
score = (memory["text_emb"][pairs[0]] * memory["text_emb"][pairs[1]]).sum(1)  # 融合文本cosine
chosen = pairs[:, torch.topk(score, int(k*DEGREE)).indices]       # top int(k×1.0) 对
edges = torch.cat((chosen, chosen.flip(0)), 1)                    # 对称化
```

每阶段在**全 memory**上重连（新旧块之间可以有文本 kNN 边——跨阶段连接由语义相似度提供，这是唯一的跨块信息通道）。归一化邻接 `norm_adj`：

```python
idx = torch.cat((edge_index, torch.arange(n).repeat(2,1)), 1)     # 加自环
adj = D^{-1/2} (A+I) D^{-1/2}                                     # 标准GCN归一化，稀疏张量
```

## 2.6 Student 与训练 `CLIPStudent` / `train_student`

```python
class CLIPStudent(nn.Module):
    self.g1, self.g2 = GCNLayer(in,384), GCNLayer(384,384)        # adj @ Linear(x)
    self.text_proj = nn.Linear(384, 384)
    self.logit_scale = nn.Parameter(ones([]) * log(1/0.07))       # CLIP可学习温度
    def encode_graph(x, adj): return normalize(g2(relu(g1(x,adj)),adj))
    def logits(x, adj, t):    return scale.clamp(max=100) * enc_g @ enc_t.T
```

```python
def train_student(syn, seed, device, steps, lr=1e-3, batch=24):
    torch.manual_seed(seed)                            # student随机性=唯一随机源
    gen = torch.Generator().manual_seed(seed*7+1)      # batch采样独立种子
    for _ in range(steps):                             # steps=min(2048, 16×roots)——过训练修复
        rows = torch.randperm(n, generator=gen)[:batch]
        logits = scale * enc_graph(x,adj)[rows] @ enc_text(t[rows]).T
        loss = (CE(logits, arange) + CE(logits.T, arange)) / 2     # 对称CLIP InfoNCE
```

Student 只见合成图；每个 root 的 graph embedding 与自己的融合文本互为正对、batch 内其余为负例。**方法的验收量=这个从头训练过程的产出质量**（可重训性），这就是为什么一切以冻结模型响应为目标的加工都失败——它们优化的不是这个量。

## 2.7 评测 `eval_stagewise`

```python
pred = model.logits(xg, real_adj, prompt_emb).argmax(1)           # 真实图transductive传播+零样本prompt分类
for s in range(seen+1):
    mask = dev_mask & torch.isin(y, torch.tensor(STAGE_CLASSES[s]))
    per_stage.append(f1_score(..., average="macro", labels=unique(yy)))
out["cumulative_macro_f1"] = f1(concat(all_y), concat(all_p), average="macro")
out["old_mean_f1"] = mean(per_stage[:-1]); out["current_f1"] = per_stage[-1]
```

口径：cum（全部已见类合并）/ old（旧阶段各自 F1 取均值）/ current（本阶段新类）。dev=node_id%10==1（选择用），test=node_id%10==2（gate3 一次性读取）。凝聚源池 `stage_train = class_nodes & ~dev_mask`——dev 特征也不进入凝聚。

## 2.8 确定性审计（贯穿各 runner）

```python
syn = condense(...); again = condense(...)            # 同输入构建两次
for key in ("x","text_emb","edge_index","weights"):
    if not torch.equal(syn[key], again[key]): raise RuntimeError(...)
```

加上 runner 拒绝覆盖已有输出（`FileExistsError("existing runs are immutable")`）、gate3 的 `confirmation_accessed: true` 标记——复现性是被执行的机制而不是声明。

---

# 第三部分：取舍的总模式

回看全程，每一次取舍都是同一个模式的实例：

```
候选（更复杂/更"学习"）        证伪实验              保留（更简单/更构造）
bi-level ASM优化          → P2 (loss不降)        → 构造式初始化
teacher响应/文本锚定约束    → P5 (KL可降但无用)    → 数据统计构造+自编码配对
CMI信息量选文本           → S2-A1 (−7.2)        → 几何top-g同质融合
真实边粗化拓扑             → S2-A2 (−10.4)       → 文本kNN
一阶X精炼                → S2-A4 (−13.0)       → 簇均值不动
再聚类/再融合持续更新       → Gate1'' (−5~−9)     → 只追加不重组
动态/学习式预算            → 扫描(窗口窄+非单调)   → c=15静态规则
```

方法最终形态的每一行代码，都对应左列一个被多 seed 实验否决的替代品。这就是"简单性是结论而非起点"的完整含义。
