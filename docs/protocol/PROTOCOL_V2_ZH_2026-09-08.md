# PROTOCOL V2：持续 TAG 凝聚研究协议（冻结版）

日期：2026-09-08
状态：v2 冻结。此后所有实验引用本文件；与更早协议文档冲突时以本文件为准。
前身：`EXECUTION_PROTOCOL_2026-09-07.md`、`NEXT_STEP_RESPONSE_RETENTION_GATE_ZH_2026-09-08.md`。

## 1. 问题定义（不变）

类别增量文本属性图流。阶段结束后历史原始训练图不可再读，只能传递满足累计预算的 synthetic TAG artifact。artifact 必须能被任意独立初始化的 student 从头训练。凝聚过程与 artifact 不使用训练节点标签；checkpoint/dev 标签仅用于 checkpoint 与 outer snapshot 选择。

## 2. 数据与阶段（不变，扩展预告）

- 开发流：Cora 类别增量 stage 0 `[0,1]` → stage 1 `[2,3]` → stage 2 `[4,5,6]`；
- prepared_v2 签名数据与 manifest 校验保持强制；
- **扩展承诺**：方法冻结后必须在第二个 TAG 数据集重复主实验；最终论文测试使用未被本项目开发访问过的数据集/分区（见 §8）。

## 3. 预算定义（v2 变更）

memory 预算从单一"根节点数"改为二元组：

```
budget = (root_count, aux_bytes)
```

- `root_count`：不超过 `floor(rate × 累计训练根数)`，rate=0.05 不变；
- `aux_bytes`：artifact 中除 X/text/topology/weights/origin_ids/stage 以外的全部附加状态字节数（含 response target、任何缓存 logits）。每个 artifact 的 manifest 必须列出 aux_bytes 明细；
- 文本字节数继续单独记录；
- **预算分配放开**：旧 memory 行数不再被强制等于上一阶段导出行数。允许在阶段切换时收缩旧 memory（只删行，不改留存行的 X/text/配对）为新块腾预算。任何收缩规则必须是无标签、确定性、写入 manifest 的。

## 4. Response-Retention 档 A 协议（v2 新增）

采用 self-distillation on synthetic rows：

- **target 生成时机**：阶段 t 结束、teacher 丢弃之前，对该阶段导出 artifact 的**全部 synthetic rows 自身**计算 graph-to-class-prompt logits（固定七类 prompt 空间），保存为 `response_target`（float32，rows×C）；
- **合法性**：target 只定义在 synthetic 节点上，不携带任何真实节点数据；字节数计入 aux_bytes（Cora 七类下约 rows×28 字节，可忽略）；
- **禁止**：在 reference roots 或任何真实节点上保存 response；保存 teacher 参数（含投影头）；
- **使用方式**：阶段 t+1 凝聚时，旧 rows 的 graph response 以 `L_response-retention` 对齐其保存 target；新 rows 对齐 current-delta teacher 的 logits（该 teacher 本阶段内合法存在）；
- **对"为什么不存 teacher"的立场**：response target 是 O(rows×C) 的数据标注，独立 student 可用它从头重训验证；teacher 参数是 O(模型) 的状态且无法被重训协议审计。

## 5. 评测（小幅修订）

- 主指标 **Macro-F1**（固定 label space），**Accuracy 并列报告**（对齐官方 TAGSAM 口径），另附 per-class F1；
- 口径矩阵不变：current / old per-stage / cumulative seen / future × train / dev / test；
- 评测为 whole-stage transductive message passing（论文中必须写明）。

## 6. 统计判定规则（v2 新增，预注册）

任何"方法 A 优于方法 B"的声明需同时满足：

1. **condensation seeds ≥ 4**（gate 类实验用 44–48 中至少 4 个；stage-2 完整递归允许先 2 个但结论标注为初步）；
2. 每 (condensation seed, artifact) 至少 6 个独立确认 student；
3. 报告配对差值的均值、**逐 condensation seed 分解**、配对胜率；
4. 判定标准：跨 seed 配对差值的 bootstrap 95% CI 不含 0，或全部 seed 方向一致且均值超过 1 pp；
5. dev-only 选点；test 只在候选冻结后由未参与选点的 student seeds 读取一次。

## 7. Gate 序列（v2 更新）

```
Gate S（静态，新增，前置）：
  在 stage-0 静态 setting 下，存在一个优化配置使 offline ASM（或其变体）
  在 ≥4 个 condensation seed 上：dev 选择 outer>0 且 dev ≥ frozen 初始化。
  诊断在先：区分优化失败（ASM loss 不降）与泛化失败（loss 降 dev 掉）。
  ├─ 通过 → Gate 1 使用该配置
  └─ 不通过 → 方法形态切换为"构造式初始化为主"：
       改进代表点选择 + 类平衡/动态预算 + 轻量 text-anchor 微调（小步数早停），
       递归优化降级为可选消融；"小预算 TAG 凝聚中优化弊大于利"作为论文观察。

Gate 1（stage-1 持续）：
  变量：response-retention 档 A、预算分配、[Gate S 配置]。
  通过条件：≥4 seeds 全部 dev 选择 outer>0，且累计 dev ≥ 各自 outer=0。

Gate 2（严格递归 stage 1→2，dev-only）：
  两阶段累计/历史/当前 dev 在全部 seeds 上不劣于各自 λ_r=0 dev-best。

Gate 3（test 确认）：全部冻结后独立 student 读取 test，按 §6 报告。
之后：扩数据集 → 最终未访问留出。
```

在 Gate 1 通过前，不加入 GraphCLIP、transport、跨阶段边、额外 prompt。

## 8. 数据污染边界（v2 明确）

- Cora 全部分区已被开发访问：**所有 Cora 数字永久标记为开发证据**；
- 第二数据集候选（TAG 版本）：Citeseer、Pubmed、ogbn-arxiv 子集。选定后 prepared pipeline 照建，**test/confirmation/future 分区封存**，只在 Gate 3 之后、方法完全冻结时读取一次；
- 每次读取封存分区都必须在结果 JSON 中写 `confirmation_accessed: true` 并记录访问原因。

## 9. 基线保留清单（不变）

Append / Union / Append-joint / Protected-joint / Full-history / Random-small / Coreset / Frozen compact，加上 v2 新增的预算分配对照臂。官方 TAGSAM 三条静态参照（master-self-loop、add-adj、minimal-repair）保留为静态背景。

## 10. 与 P1 校准的关系

官方复现与论文报告值的差距（五类 size=50：论文 73.8 vs 本地 38–53）尚未完全归因（见 `TAGSAM_PAPER_CALIBRATION_ZH_2026-09-08.md`）。本协议不以复现论文数字为前提；native 后端的受控对照逻辑独立于官方数字。但论文写作中引用 TAGSAM 时必须使用"我们的复核值"并注明差异未决。
