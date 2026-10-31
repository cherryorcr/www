# 经典持续 TAG 图凝聚转向前的研究冻结快照

日期：2026-09-04  
状态：FROZEN  
用途：保留此前 prototype/compressed-replay memory 路线的全部正负结果，作为新经典图凝聚主线的 baseline、初始化与设计证据

## 1. 冻结决策

从本文件建立后，此前 D2/A2、P0-P4、V5-V9、RAM、STPM、TD、RBCM、GraphCLIP atomic 和文本接口实验均视为历史冻结结果。

冻结不表示结果无效，而表示：

- 不再把 D2 定位为最终主方法；
- 不在原 qualification split 上继续调 fixed fusion、prototype selector、message、edge utility、文本 gate 或动态预算；
- 协议不变时复用已有 baseline，不重复运行以刷新数字；
- 新实验写入新的 `classical_continual_tag_condensation` 结果空间，不覆盖旧 artifact；
- t3 保持 sealed，直到新经典凝聚方法完全冻结。

## 2. 冻结后的有效结论

1. D2 是有效的持续压缩训练 memory，但不是自由优化 `synthetic X/Y/A/T` 的经典 graph condensation。
2. D2/A2 在 ArXiv、PubMed 上存在可信正向结果，可作为强 baseline 和初始化。
3. Online k-medoids、Append、PUMA-style moment、GECC-style coverage 等结果继续作为冻结对照。
4. K3 固定 feature residual、learned edge utility、简单 message residual、时间/novelty 动态预算和局部 action proxy 不再进入新方法。
5. GraphCLIP paired/shuffled 证明 TAG 图文对应关系有任务信息，但当前 atomic root/ego selector 尚未形成凝聚方法。
6. O2 已经证明单 checkpoint、短 inner-loop 的自由 synthetic 优化会严重过拟合；新经典凝聚不能重复该目标。

## 3. 关键结果口径

### D2 原始角色记忆 gate

- ArXiv：`47.58%`，相对 Append `+1.21 pp`，`5/5` seeds 不退化；
- PubMed：`83.78%`，相对 PUMA-style moment `+1.25 pp`，`5/5` seeds 不退化。

### A2 真实 SAGEConv gate

- ArXiv：D2/A2 `46.98%`，相对预选 Append `+3.50 pp`；
- PubMed：D2/A2 `78.09%`，相对预选 PUMA-style moment `+4.35 pp`；
- 两个数据集共十个 seed 比较全部不退化。

两组数值属于不同 evaluator，不得合并或互相替代。

### GraphCLIP interface gate

- paired 相对 shuffled text Macro-F1：ArXiv `+12.25 pp`，PubMed `+8.20 pp`；
- 该结论只证明 graph-text pairing 有效，不证明任何 condensation selector 有效。

## 4. 冻结文件与 SHA-256

| 文件 | SHA-256 |
|---|---|
| `CONTINUAL_TAG_CONDENSATION_FULL_RESEARCH_REPORT_ZH_2026-09-04.md` | `30C939BAB88FE8B2BDF2BF56FC17DC3C28594AEBB9BC9E7CB8E771BB1EDB3192` |
| `FROZEN_BASELINE_REGISTRY_2026-08-27.md` | `4E88DB5C935F4DB28B911BC04AC753D4BFCDCCEE2ABEFD64B192B240D598193A` |
| `D2_ROLE_AWARE_CONTINUAL_UPDATE_REPORT_2026-08-31.md` | `5AD0579F7B49FECCB3B9307885D65BE7B692FBF8F2683B71CD240993941915BB` |
| `A2_GRAPHSAGE_FORMAL_GATE_REPORT_2026-08-31.md` | `9122A0EF1F704C78A8701D18898E6B2C6FDDC90E70B690A3881B2E62DDF48908` |
| `A3_CHECKPOINT_SELECTION_CORRECTION_REPORT_2026-08-31.md` | `2542BB61B9332C9668DE99681BC3A29020900F65BF4D915C33B2A45D5835D567` |
| `GRAPHCLIP_G0_G1_GATE_AUDIT_2026-09-03.md` | `AC62252B5FF437F99530D4D7683A9128A6267B5032496C5FC292A0235489AD46` |
| `GRAPHCLIP_RATE_CURVE_DECISION_ZH_2026-09-03.md` | `BD6861AFF171D78C0B0CD824E04BBC55F62E64EA442045DDFBD4615436B8DA2D` |
| `D2_TAG_TEXT_CROSS_MOMENT_ABC_DECISION_ZH_2026-09-03.md` | `7271E48C3E3D2CDFA25C0334FFFF3AB7916C17826C02DF3B29887F4A85626A54` |
| `FULL_HISTORY_TEXT_CONDITIONAL_GATE_DECISION_ZH_2026-09-04.md` | `9D34668DA7E119CD843B4D7ADB9D4AF7EE99B58EA499580041F49A3A27CE5DDE` |

哈希只用于标记建立快照时的内容。后续若修复错别字或补充说明，应建立新版本文件，不原地修改这些冻结报告。

## 5. 新旧主线边界

旧主线：

\[
\text{真实 medoid/prototype}
+\text{递归统计 memory}
+\text{固定或选择性 replay}.
\]

新主线：

\[
\text{jointly optimized synthetic feature}
+\text{synthetic soft label}
+\text{synthetic text state}
+\text{synthetic adjacency}
+\text{continual re-condensation}.
\]

D2 以后只承担四种角色：

1. 强 baseline；
2. synthetic state 的 medoid/moment 初始化；
3. temporal retention anchor；
4. 新方法失败时的 rollback。

D2 的固定双专家融合、两角色配比和 self-loop 不自动进入新方法。

## 6. 结果目录规则

新主线建议统一使用：

`problem_validation_results/classical_continual_tag_condensation/`

每个实验必须记录：

- protocol version；
- data/split fingerprint；
- synthetic variable set；
- initialization source；
- condensation/model seeds；
- optimizer steps 和 checkpoint selection partition；
- root rate、feature/text/label/edge/state bytes；
- raw-history access audit；
- t3 access flag；
- source code SHA-256。

## 7. 覆盖规则

本快照取代旧文档中“D2 作为最终主方法”的方向决策，但不改变任何已有实验结论。新的主方向以 `CLASSICAL_CONTINUAL_TAG_GRAPH_CONDENSATION_ROADMAP_ZH_2026-09-04.md` 为准。
