# 对照组长要求的完成情况与剩余边界

依据 `/Users/wv/Documents/EXTERNAL_BASELINE_HANDOFF_PROTOCOL_ZH_2026-09-27.md`
和聊天记录，外部 baseline 的实验与交接材料已经收尾；以下边界仍然成立，不能在汇报中省略。

## 已满足

- PUMA-inspired 机制已实现并完成两个数据集、两个既有顺序、六个 Student seeds。
- TAGSAM/STAD 已完成机制适配、访问权限说明或独立验收。
- CaT、GECC、k-center、medoid 已有两个既有顺序的统一 Student 结果。
- 结果中已记录 5% budget、Student、构造访问限制、源码或输入哈希等核心信息。
- 13 张 card 的必需字段审计无缺失，结果汇总为 180 行、严格最终阶段比较表为 42 行；另有 4 行 TAGSAM/STAD adapted/reference 补充表；GraphCLIP frozen checkpoint 已单列为 section 3.C reference-only。

## 已完成的收尾项

1. 每个 baseline 已有独立 card；GraphCLIP、GCond、SFGC、GCondenser 明确标为 reference-only。
2. CaT 和 k-center 已覆盖 `order19613`、`order19614`、`order19615`；`order19615` 的 Cora/Citeseer 均完成构造、评估、审计和 checkpoint reload。
3. GCond、SFGC、GCondenser 已固定官方 commit 并通过 compile smoke；由于静态/旧栈/历史 buffer 约束，未伪造共同 TAG Student 分数。
4. 已生成统一结果汇总：`EXTERNAL_BASELINE_SUMMARY_20261002.md/json`。
5. 已生成审计索引：`EXTERNAL_BASELINE_AUDIT_INDEX_20261002.md/json`。TAGSAM 两数据集各 72 个 checkpoint reload 已独立重载，最大指标误差为 0.0。
6. 已明确 native、mechanism-adapted、controlled-content、reference-only 边界；没有完整官方 native reproduction。
7. 已修正 TAGSAM/STAD 缺失顺序字段的问题：汇总使用 `order-unspecified`，没有将 dataset 名伪装成 order。
8. 已在服务器重新生成汇总和比较表，并通过 13 张 card JSON、summary 行数、reference repo compile smoke 校验；本地与服务器交付文件哈希一致。

## 不属于当前外部 baseline 线的工作

新旧类别混合到达、old-X trust region、soft routing、sparse new-old topology、GC-SR 的 X 梯度诊断和跨数据集 zero-shot 属于组长主线/共同协议工作，不应由外部 baseline 结果冒充完成。

## 仍然不能宣称已完成的事项

- 所有实际分数都是 mechanism-adapted 或 reference/adapted；没有官方 native PUMA/CaT/GECC/TAGSAM/STAD reproduction。
- GECC、medoid、semantic-recursive 没有扩展到所有新顺序；协议只要求新顺序优先扩展 CaT/k-center。
- GCond、SFGC、GCondenser 只有官方代码、commit 和 compile 核验，没有共同 TAG Student 结果。
- 服务器保留 full offline fixture，因此 source-locality 审计是逻辑访问审计，不是 OS 级隔离。
- TAGSAM/STAD 的源 manifest 不含 order；部分 nested manifest 不提供 construction/text/edge 成本，比较表保留空值和 warning。
- PUMA 的 `order19612` 低分已完成消融诊断；当前只能归因于 PUMA 构造与 InfoNCE Student 的不稳定交互，不能单独归因于 edge-free memory。
- TAGSAM/STAD 不按严格主表处理：其已有结果使用 Student seeds 100-105，已移到 adapted/reference 补充表；若组长要求进入严格主表，必须按 150-155 重跑。

## 交接时的使用顺序

主表使用 mechanism-adapted 结果；reference-only 只放在可行性/方法库存表；controlled-content 与主表分开。不要把这些适配分数写成官方 native reproduction。
