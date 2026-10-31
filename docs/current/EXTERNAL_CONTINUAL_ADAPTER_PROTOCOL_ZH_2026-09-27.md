# 外部持续凝聚机制适配：冻结协议

日期：2026-09-27。此文件先于正式 report-dev 评估冻结。Semantic-Recursive 的内容、拓扑、参数和选择规则不改。

## 1. 目的与比较边界

主要判断 Semantic-Recursive 相对外部机制和强 coreset 的效用。C0 只作为已存 SR base 的来源，不作为本轮获胜门槛。两张小图、order19611/19612、三个阶段、固定 5% growing rate；Student seeds 150..155。四个 dataset/order cell 为四个组，种子是组内重复。

CaT 和 GECC 都有公开持续/演化图凝聚工作，但不是原生 TAG 方法。这里是 **核对原代码后的机制重实现和 TAG 适配**，不等于官方完整复现。不以适配结果替代原生论文数字，更不据此宣称战胜所有 continual TAG SOTA。

## 2. 原始来源

- CaT 论文：https://arxiv.org/html/2309.09455v2
- CaT 代码：https://github.com/superallen13/CaT-CGL ，固定 commit `43d6fcfd7b1754fe53b9f911492d7cb0c113a2eb`。
- 核对文件：`methods/cgm.py`、`backbones/encoder.py`、`scripts/table2.sh`。
- GECC 论文：https://arxiv.org/html/2502.17614v3
- GECC 代码：https://github.com/rockcor/GECC ，固定 commit `d731be00cc3beb7585e8fa7b60ca21a6c3cb4e3f`。
- 核对文件：`graphslim/condensation/gecc.py`、`graphslim/config.py`、`graphslim/scripts/run_cora_evolve.sh`。

论文与代码有差别：CaT Eq.8 按类比例加权，公开 `cgm.py` 对类均值误差等权相加。本轮采用代码口径。CaT 的 Encoder 先传播、再应用随机线性层/ReLU，排除末分类层，之后 L2 normalize；不是训练 teacher。

## 3. 五个实验臂

| 方法 | 当前 delta 构造 | 拓扑 | 历史与第一阶段 |
|---|---|---|---|
| Semantic-Recursive | 读取已冻结 synthetic artifact | 已冻结边 | 使用已冻结递归轨迹，不重选参数 |
| CaT-MMD-TAG | 类比例配额，randomChoice 初始化，随机编码器 MMD 更新合成 X | self-loop only | 从 stage0 自行构造；累积自身合成块 |
| GECC-hard-TAG | 传播后的特征中心，类内 hard clustering，balance restart score | self-loop only | 从 stage0 自行构造；保留已凝聚旧类中心 |
| k-center native append | joint-space farthest-first 真实节点 | 根据自己 payload 做文本边与跨边 | 从 stage0 自行构造，不借用 C0 base/slot |
| medoid native append | joint-space 聚类后真实 medoid | 根据自己 payload 做文本边与跨边 | 从 stage0 自行构造，不借用 C0 base/slot |

这些 coreset 是通用算法本地实现，不冒充某篇论文的完整复现。旧表共享 C0 stage0 和 C0 topology，是内容归因控制；本轮端到端结果另立表，不能混为同一个设置。

## 4. CaT 细节

配额采用类别比例最大余数法，可行时每类至少一行。明确保存实际 row labels，MMD 不得重新猜测配额。合成 X 初始化为同类随机有放回抽样，再经过优化成为 synthetic tensor；它不是把原节点身份作为 memory。每步新抽一个 Glorot 随机编码器，hidden=512；3000 步 Adam，lr=0.001，匹配类条件归一化表示均值平方距离。当前 raw 使用一跳对称归一化传播，合成 memory 无非自环边，不加入 C0 anchor 或额外正则。

3000 步来自公开示例的量级，未在 report-dev 调优。另用固定四编码器 bank 在同一空间诊断起止 MMD，不能拿不同随机空间的两个 loss 直接称作优化收益。

TAG 补全：每个 synthetic X 在相同类别的传播特征空间选最近至多五个当前来源文本，按 320 字符分段拼接后 MiniLM 重编码。最多四个空格开销与项目 legacy concat 一致。这是新增适配，CaT 本身没有文本模块。下游统一使用原 GCN/InfoNCE，不使用 CaT 原监督 TiM 训练。

## 5. GECC 细节及限制

沿用代码默认 hard clustering 分支，传播权重为 `[0,0.3,0.9]`（0/1/2 hop）。输出传播特征的中心，不能退回 raw Mean-X 后仍声称实现 GECC。每类 20 个 random-init Lloyd restarts，至多 300 iterations，按 SSE + 1.0 * cluster-size imbalance 选取，选择只使用 pool 构造目标。

本地 Lloyd 用 sklearn 实现，不复制官方依赖栈。没有采用论文 Cora 脚本中调优的 fuzzy=1.3/50 restarts/-0.1,0.9,0.9，也未证明默认 hard 分支是最强版本。文字补全同 CaT。

**无历史 raw 适配的关键边界**：原 GECC 演化代码读取累计 raw，并用旧中心初始化重聚类。本轮 disjoint class-incremental 不再收到旧类 raw，旧类中心只能继承不重聚类，新类从当前 delta 聚类。因此不能声称完整复现 GECC 的 evolving 更新或验证了其完整优势。允许历史 synthetic 改写并不是本轮排除其它方法的理由；这里只因没有旧类输入而没有同类更新。

## 6. 公共评价和资源

所有方法同 cell 使用完全相同 rows 轨迹、report-dev、原始全图推理、全部类别 prompts、GCN/text projection、InfoNCE、batch=16、lr=0.001、steps=max(50,min(2048,16K))、seeds150..155。Student 不传递参数。构造可使用 pool 类别标签；不称全流程无标签。

构造接口只接收切片 delta、旧 synthetic 和冻结文本编码器；离线数据加载器仍加载完整容器供切片/评价，不声称 OS 隔离。先构造并保存全部 memory/hash/来源/预算，再 report，最后 checkpoint 重载验证 F1/Accuracy。非自环边、UTF-8/text token/tensor bytes 和构造、训练时间全部记账。相同 rows 不代表相同字节预算，coreset 原文更长应明示。

主报告：Final Macro-F1/Accuracy、阶段平均、pooled old/new、最差阶段、每 cell 配对差值。每 cell 六 seeds 的 CI 为条件种子不确定性；总体 CI 对四个 dataset/order 均值做组级描述，不将24个seed配对当作24个独立顺序。

## 7. 验收与下一步

smoke 只证明执行正确；不以 smoke 分数选方法。正式固定矩阵共 2 datasets * 2 orders * 5 arms * 3 stages * 6 seeds = 360 次 Student fit。每个 checkpoint 重载，核对 artifact/prefix、来源和预算。报告所有臂，不按数据集选更有利配置。

这一步不包含 PUMA 的伪标签、单次传播/宽编码器完整机制，不将 CaT 改名充数。后续若需论文强基线，需进一步做原生配置敏感性、更多独立顺序/构造 seeds、公开 TAG 静态适配与 matched-budget order control。这一轮不会把只有两个顺序的证据写成完整顺序泛化验收。
