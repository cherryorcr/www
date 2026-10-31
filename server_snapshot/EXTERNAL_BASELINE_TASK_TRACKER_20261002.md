# 外部 baseline 分步任务

- [x] 1. 整理每个 baseline 的论文/仓库、commit、访问权限、memory、Student、适配差异和结果目录，生成 13 张 `baseline_card.json` 并上传服务器；PUMA 已独立成卡，GraphCLIP frozen checkpoint 单列为 reference-only。
- [x] 2. 完成 CaT 和 k-center 的新预注册顺序覆盖：已补 `order19614` Citeseer，并完成 `order19615` Cora/Citeseer；新顺序覆盖共 3 个。
- [x] 3. 对 GECC、medoid、semantic-recursive 未扩展到全部新顺序；CaT/k-center 才系统扩展到新顺序。
- [x] 4. 完成 GCond、SFGC、GCondenser 的论文/代码/commit/访问权限核验和 compile smoke；共同 TAG Student 端到端适配不成立，已生成 reference-only cards。
- [x] 5. 汇总所有正式结果的 Macro-F1、Accuracy、old/new F1、worst-stage F1、构造与 Student 成本，见 `EXTERNAL_BASELINE_SUMMARY_20261002.md`、`EXTERNAL_BASELINE_COMPARISON_TABLE_20261002.md` 和 JSON；缺失成本保持为空并在 warning 中登记。
- [x] 6. 集中登记 source-locality、budget、prefix、checkpoint reload 审计结论，见 `EXTERNAL_BASELINE_AUDIT_INDEX_20261002.md`；TAGSAM 两数据集各 72 个 checkpoint reload 均通过，最大指标误差 0.0。
- [x] 7. 最终检查 native、mechanism-adapted、controlled-content、reference-only 的边界；主表只收 mechanism-adapted，未声称 native reproduction。
- [x] 8. 统一 12 张 card 的论文元数据、teacher、construction seed、source hashes、audits 和主表资格；`EXTERNAL_BASELINE_CARD_AUDIT_20261002.md` 显示无缺失必需字段。
- [x] 9. 修正 nested manifest 的顺序归一化：TAGSAM/STAD 缺失 order 时标为 `order-unspecified`，不再把 dataset 名误写为 order。
- [x] 10. 在服务器结果目录重生成并回收交付汇总；本地脚本可复现相同的 180 条结果行、42 条 strict final-stage 行和 4 条 adapted/reference 行。
- [x] 11. 增补官方 GraphCLIP frozen checkpoint 的 section 3.C reference-only card，并完成 PUMA 低分诊断；不将 GraphCLIP 或 PUMA 诊断结果混入严格 continual 主表。

当前说明：`order19613`、`order19614`、`order19615` 的 CaT/k-center 结果已具备；`order19615` 两个数据集均通过构造、评估、审计和 checkpoint reload。所有适配结果仍按机制适配分类，不写成官方 native reproduction。

交付边界：PUMA、CaT、GECC、k-center、medoid、semantic-recursive、TAGSAM、STAD 的实际结果仍是机制适配或 reference/adapted 结果；GCond、SFGC、GCondenser 只有官方代码/commit/compile 核验，明确排除出主比较。服务器仍保留 full offline fixture，因此 source-locality 是逻辑访问审计，不是 OS 级隔离。TAGSAM/STAD 没有顺序字段，比较表保留 `order-unspecified`；部分 nested manifest 不提供 construction/text/edge 成本，空值不补造。

TAGSAM/STAD 的现有结果使用 Student seeds 100-105，已从严格比较表移到 `EXTERNAL_BASELINE_ADAPTED_REFERENCE_TABLE_20261003.md`。若要让它们进入冻结协议的严格主表，必须重新按 Student seeds 150-155 运行。
