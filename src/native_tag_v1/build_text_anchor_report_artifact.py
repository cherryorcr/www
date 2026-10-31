"""Build the canonical portable technical-report artifact from saved evidence."""
from __future__ import annotations

import json
import statistics
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports" / "text_anchored_asm_2026_09_08"


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def matrix_means(relative: str) -> dict[str, dict[str, float]]:
    data = load(relative)
    groups: dict[str, list[dict[str, float]]] = {}
    for row in data["rows"]:
        cells = {(cell["eval_stage"], cell["partition"]): cell["metric"]["macro_f1"]
                 for cell in row["cells"]}
        stage = int(row["model_stage"])
        groups.setdefault(row["arm"], []).append({
            "cumulative_dev": row["cumulative_seen"]["checkpoint"]["macro_f1"],
            "cumulative_test": row["cumulative_seen"]["test"]["macro_f1"],
            "old_dev": statistics.mean(cells[(index, "checkpoint")] for index in range(stage)),
            "old_test": statistics.mean(cells[(index, "test")] for index in range(stage)),
            "current_dev": cells[(stage, "checkpoint")],
            "current_test": cells[(stage, "test")],
        })
    return {
        arm: {metric: statistics.mean(row[metric] for row in rows) for metric in rows[0]}
        for arm, rows in groups.items()
    }


def source(source_id: str, label: str, path: str, description: str,
           sql: str | None = None, tables_used: list[str] | None = None) -> dict:
    value = {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "saved-artifact",
            "language": "python",
            "description": description,
        },
    }
    if sql is not None:
        value["query"].update({
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "tables_used": tables_used or [path],
        })
    return value


def build() -> dict:
    same = matrix_means(
        "results_mirror/protocol_matrices_fixed_labels_2026_09_08/anchor_candidate_confirmation_matrix.json"
    )
    fair = matrix_means(
        "results_mirror/protocol_matrices_fixed_labels_2026_09_08/anchor_candidate_vs_devbest_matrix.json"
    )
    recursive = load("results_mirror/recursive_text_anchor_2026_09_08/recursive_summary.json")

    comparisons = []
    for basis, values, candidate, baseline, pairs in [
        ("同 outer=60", same, "candidate_anchor_o60", "baseline_asm_o60", 16),
        ("各自 dev 最优", fair, "candidate_anchor_o60", "baseline_asm_o20", 16),
    ]:
        for metric, label, partition in [
            ("cumulative_dev", "累计", "Dev"),
            ("old_dev", "历史", "Dev"),
            ("current_dev", "当前", "Dev"),
            ("cumulative_test", "累计", "Test"),
            ("old_test", "历史", "Test"),
            ("current_test", "当前", "Test"),
        ]:
            comparisons.append({
                "comparison": basis,
                "metric": f"{label} {partition}",
                "segment": label,
                "partition": partition,
                "gain_pp": values[candidate][metric] - values[baseline][metric],
                "candidate_f1": values[candidate][metric],
                "baseline_f1": values[baseline][metric],
                "paired_students": pairs,
                "condensation_seeds": 2,
            })

    means = recursive["confirmation_aggregates"]["means"]
    stage2_means = {row["method"]: row for row in means if row["stage"] == 2}
    stage2_delta = next(row for row in recursive["confirmation_aggregates"]["paired_anchor_minus_asm"]
                        if row["stage"] == 2)
    recursive_aggregate = []
    for method, label in [("asm_0", "ASM λ=0"), ("text_anchor_0.01", "Text-Anchored")]:
        row = stage2_means[method]
        recursive_aggregate.append({
            "method": label,
            "cumulative_train": row["cumulative_train_macro_f1"],
            "cumulative_dev": row["cumulative_dev_macro_f1"],
            "cumulative_test": row["cumulative_test_macro_f1"],
            "old_test": row["old_test_macro_f1"],
            "current_test": row["current_test_macro_f1"],
            "independent_students": row["n"],
            "condensation_seeds": 2,
        })

    recursive_seed_gain = []
    for metric, label in [
        ("cumulative_dev_macro_f1", "累计 Dev"),
        ("cumulative_test_macro_f1", "累计 Test"),
        ("old_test_macro_f1", "历史 Test"),
        ("current_test_macro_f1", "当前 Test"),
    ]:
        detail = stage2_delta[metric]
        for seed, value in detail["by_condensation_seed"].items():
            recursive_seed_gain.append({
                "metric": label,
                "condensation_seed": f"Seed {seed}",
                "gain_pp": value,
                "overall_mean_gain_pp": detail["mean_delta"],
                "positive_pairs": detail["positive_pairs"],
                "total_pairs": detail["total_pairs"],
            })

    selections = [
        {
            "stage": row["stage"],
            "condensation_seed": row["condensation_seed"],
            "method": "Text-Anchored" if row["method"] == "text_anchor_0.01" else "ASM λ=0",
            "selected_outer": row["selected_outer_step"],
            "selection_dev_f1": row["selected_dev_macro_f1"],
            "selection_students": ", ".join(str(value) for value in row["selection_student_seeds"]),
            "test_or_future_used": "否" if not row["test_or_future_used_for_selection"] else "是",
        }
        for row in recursive["selections"]
    ]

    sources = [
        source("official_static", "官方 TAGSAM 静态复核",
               "research/graph-condensation/TAGSAM/OFFICIAL_TAGSAM_STATIC_REAUDIT_2026-09-07.md",
               "官方 TAGSAM 静态复现、minimal repair、add-adj 与 full teacher 结果。"),
        source("alignment_sweep", "Text-Anchored ASM 开发 sweep",
               "results_mirror/text_anchor_sweep_2026_09_08/aggregate.json",
               "两次 condensation seed、七个文本锚定权重和六个固定 outer 快照的开发指标与无标签 alignment。"),
        source("same_outer", "同 outer=60 独立 student 矩阵",
               "results_mirror/protocol_matrices_fixed_labels_2026_09_08/anchor_candidate_confirmation_matrix.json",
               "Text-Anchored ASM 与零权重 ASM 在相同 outer=60 下的 16 个 student 配对。"),
        source("devbest", "各自 dev 最优独立 student 矩阵",
               "results_mirror/protocol_matrices_fixed_labels_2026_09_08/anchor_candidate_vs_devbest_matrix.json",
               "候选 outer=60 与零权重 ASM 自身 dev 最优 outer=20 的比较。"),
        source("recursive", "严格递归验证汇总",
               "results_mirror/recursive_text_anchor_2026_09_08/recursive_summary.json",
               "stage 1 到 stage 2 的 dev-only 选择、12 个独立确认 student 和 condensation-seed 分解。"),
        source("stage2_gain_dataset", "Stage-2-only 比较图数据",
               "reports/text_anchored_asm_2026_09_08/stage2_only_gain.csv",
               "从固定 label-space 的同 outer 与各自 dev 最优矩阵计算 Text-Anchored 减 ASM 的 Macro-F1 差值。",
               "SELECT * FROM read_csv_auto('reports/text_anchored_asm_2026_09_08/stage2_only_gain.csv', header=true)",
               [
                   "results_mirror/protocol_matrices_fixed_labels_2026_09_08/anchor_candidate_confirmation_matrix.json",
                   "results_mirror/protocol_matrices_fixed_labels_2026_09_08/anchor_candidate_vs_devbest_matrix.json",
                   "reports/text_anchored_asm_2026_09_08/stage2_only_gain.csv",
               ]),
        source("recursive_seed_dataset", "严格递归 seed 分解图数据",
               "reports/text_anchored_asm_2026_09_08/recursive_seed_gain.csv",
               "从严格递归汇总提取每个 condensation seed 的 Text-Anchored 减 ASM 差值。",
               "SELECT * FROM read_csv_auto('reports/text_anchored_asm_2026_09_08/recursive_seed_gain.csv', header=true)",
               [
                   "results_mirror/recursive_text_anchor_2026_09_08/recursive_summary.json",
                   "reports/text_anchored_asm_2026_09_08/recursive_seed_gain.csv",
               ]),
        source("recursive_selection_dataset", "严格递归 dev-only 选择表数据",
               "reports/text_anchored_asm_2026_09_08/recursive_selection.csv",
               "从每条固定 outer 轨迹的 dev-only 选择记录提取 stage、seed、方法与选中 outer。",
               "SELECT * FROM read_csv_auto('reports/text_anchored_asm_2026_09_08/recursive_selection.csv', header=true)",
               [
                   "results_mirror/recursive_text_anchor_2026_09_08/recursive_summary.json",
                   "reports/text_anchored_asm_2026_09_08/recursive_selection.csv",
               ]),
        source("recursive_metrics_dataset", "严格递归独立 student 表数据",
               "reports/text_anchored_asm_2026_09_08/recursive_aggregate.csv",
               "从严格递归固定 label-space 矩阵汇总两种方法的独立 student Macro-F1 均值。",
               "SELECT * FROM read_csv_auto('reports/text_anchored_asm_2026_09_08/recursive_aggregate.csv', header=true)",
               [
                   "results_mirror/recursive_text_anchor_2026_09_08/recursive_protocol_matrix.json",
                   "reports/text_anchored_asm_2026_09_08/recursive_aggregate.csv",
               ]),
    ]

    title = "持续 TAG 凝聚：文本对应修复与严格递归验证"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Text-Anchored ASM 的归因、stage-2-only 对照和严格 stage 1→2 递归验证。",
            "generatedAt": "2026-09-08T13:00:00+08:00",
            "charts": [
                {
                    "id": "stage2_only_gain",
                    "title": "Stage-2-only 的 Macro-F1 增益",
                    "subtitle": "同 outer 比较显示机制效果；各自 dev 最优比较显示完整方法收益。",
                    "type": "bar",
                    "dataset": "stage2_only_gain",
                    "sourceId": "stage2_gain_dataset",
                    "encodings": {
                        "x": {"field": "metric", "type": "ordinal", "label": "评测切面"},
                        "y": {"field": "gain_pp", "type": "quantitative", "label": "Text-Anchored − ASM", "format": "number"},
                        "color": {"field": "comparison", "type": "nominal", "label": "比较口径"},
                    },
                    "yAxisTitle": "Macro-F1 增益（百分点）",
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "recursive_seed_gain",
                    "title": "严格递归结果的 condensation-seed 分解",
                    "subtitle": "候选在 seed 44 为负、seed 45 为正，说明均值收益缺少稳定性。",
                    "type": "bar",
                    "dataset": "recursive_seed_gain",
                    "sourceId": "recursive_seed_dataset",
                    "encodings": {
                        "x": {"field": "metric", "type": "ordinal", "label": "指标"},
                        "y": {"field": "gain_pp", "type": "quantitative", "label": "Text-Anchored − ASM", "format": "number"},
                        "color": {"field": "condensation_seed", "type": "nominal", "label": "Condensation seed"},
                    },
                    "yAxisTitle": "Macro-F1 增益（百分点）",
                    "valueFormat": "number",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "recursive_selection",
                    "title": "逐阶段 dev-only artifact 选择",
                    "dataset": "recursive_selection",
                    "sourceId": "recursive_selection_dataset",
                    "defaultSort": {"field": "condensation_seed", "direction": "asc"},
                    "columns": [
                        {"field": "stage", "label": "Stage", "type": "number"},
                        {"field": "condensation_seed", "label": "Condensation seed", "type": "number"},
                        {"field": "method", "label": "方法", "type": "text"},
                        {"field": "selected_outer", "label": "选中 outer", "type": "number"},
                        {"field": "selection_dev_f1", "label": "选择 dev Macro-F1", "format": "number"},
                        {"field": "selection_students", "label": "选择 student", "type": "text"},
                        {"field": "test_or_future_used", "label": "选择读取 test/future", "type": "text"},
                    ],
                },
                {
                    "id": "recursive_metrics",
                    "title": "严格递归 stage-2 独立确认均值",
                    "dataset": "recursive_aggregate",
                    "sourceId": "recursive_metrics_dataset",
                    "defaultSort": {"field": "method", "direction": "asc"},
                    "columns": [
                        {"field": "method", "label": "方法", "type": "text"},
                        {"field": "cumulative_train", "label": "累计 Train", "format": "number"},
                        {"field": "cumulative_dev", "label": "累计 Dev", "format": "number"},
                        {"field": "cumulative_test", "label": "累计 Test", "format": "number"},
                        {"field": "old_test", "label": "历史 Test", "format": "number"},
                        {"field": "current_test", "label": "当前 Test", "format": "number"},
                        {"field": "independent_students", "label": "Student 数", "type": "number"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": "## 技术结论\n\n**已经确认的困难是 synthetic graph response 与冻结文本的对应会被持续 ASM 更新破坏。** 六次文本置换的当前阶段 test 全部下降，原 ASM 的无标签图文 InfoNCE 也随 outer 明显恶化。\n\n`Text-Anchored ASM` 在固定 stage-2 更新中有效，但严格 stage 1→2 递归后没有跨 condensation seed 稳定胜出：stage 1 全部选择 outer=0；stage 2 的独立确认累计 test 平均提高 1.62 pp，却只在 7/12 个配对中为正，seed 44 为 −0.84 pp、seed 45 为 +4.08 pp。当前方法应保留为机制修复，最终算法还需同时约束历史 teacher response。",
                },
                {
                    "id": "protocol",
                    "type": "markdown",
                    "body": "## 协议与指标\n\n类别增量流为 `[0,1] → [2,3] → [4,5,6]`。阶段完成后不可重读历史原图，只保留累计预算内的 synthetic TAG；下一阶段输入仅含上一阶段 memory 与当前 delta。下游固定为冻结 BERT CLS、两层 GCN、可训练文本投影、图文 InfoNCE 和七类文本 prompt 分类。\n\n主指标是固定 label space 的 Macro-F1。每阶段分别报告 train、dev、test，并报告累计 seen、历史阶段均值、当前阶段和 future test。Cora test/confirmation/future 已被访问，所以本报告中的 test 只能作为开发证据。",
                },
                {
                    "id": "static_gap",
                    "type": "markdown",
                    "sourceId": "official_static",
                    "body": "## 静态凝聚仍有明显上游误差\n\n官方 TAGSAM 静态复核中，minimal repair 为 36.24% / 38.74%，add-adj 为 40.23%，full teacher 为 61.12%。当前 native stage-0 中，Frozen compact test 为 76.56%，Offline ASM 为 73.05%，也说明优化不稳定优于初始化。持续误差包含静态压缩误差，但旧 synthetic memory 与新 delta 的递归合并仍产生独立问题。",
                },
                {
                    "id": "attribution",
                    "type": "markdown",
                    "sourceId": "alignment_sweep",
                    "body": "## 文本对应是可重复的 TAG 瓶颈\n\n固定 X 与 topology 后，六次阶段内文本置换使当前阶段 test 平均下降约 11.20 pp。原 ASM 从 outer 0 到 100，使两次 condensation run 的全体 symmetric InfoNCE 分别从 1.54 升到 2.27、从 1.06 升到 2.07。centroid X 换成真实 representative X 的方向则跨 seed 不一致。证据因此优先支持图文对应漂移，而不是单独的 centroid 误差或已经证明的跨阶段边缺失。",
                },
                {
                    "id": "method",
                    "type": "markdown",
                    "body": "## 第一版修复：Text-Anchored ASM\n\n方法使用 `L = L_ASM + λ L_text-anchor`。冻结 joint teacher、文本、topology、weights 和 origin IDs，梯度只进入 synthetic X。开发候选为 `λ=0.01, scope=all`。`λ=0` 保留旧路径，并同时记录 ASM、anchor、total loss 与无标签 alignment。",
                },
                {
                    "id": "stage2_only_interpretation",
                    "type": "markdown",
                    "body": "## Stage-2-only 证明机制有效，公平收益较小\n\n在相同 outer=60 下，候选累计 test 提高 4.34 pp，历史 test 提高 5.07 pp，当前 test 提高 4.85 pp。允许零权重 ASM 使用自身 dev 最优 outer=20 后，累计 test 收益缩小到 1.47 pp，历史 dev 还下降 1.30 pp。下图把因果归因口径与完整方法口径并列；正值表示 Text-Anchored 更好。",
                },
                {"id": "stage2_only_chart", "type": "chart", "chartId": "stage2_only_gain", "layout": "full"},
                {
                    "id": "recursive_interpretation",
                    "type": "markdown",
                    "sourceId": "recursive",
                    "body": "## 严格递归没有跨 condensation seed 稳定通过\n\nStage 1 的四条轨迹全部选择 outer=0，两种方法的选中图在 X、文本、权重、来源 ID、stage 和排序后的 edge set 上相同。Stage 2 的选择 student 上，候选平均 dev 低约 0.40 pp；12 个独立确认 student 上，累计 test 平均高 1.62 pp，但仅 7/12 配对为正。下图显示均值背后的 seed 方向相反，因此不能把总体平均提升解释为稳定算法收益。",
                },
                {"id": "recursive_seed_chart", "type": "chart", "chartId": "recursive_seed_gain", "layout": "full"},
                {"id": "selection_table", "type": "table", "tableId": "recursive_selection", "layout": "full"},
                {
                    "id": "recursive_table_intro",
                    "type": "markdown",
                    "body": "## 独立 student 的最终开发证据\n\n表中每种方法含 2 个 condensation seed × 6 个未参与 outer 选择的 student。总体均值有利于文本锚定，但 seed 分解和配对胜率表明方差仍然过高。",
                },
                {"id": "recursive_metrics_table", "type": "table", "tableId": "recursive_metrics", "layout": "full"},
                {
                    "id": "decision",
                    "type": "markdown",
                    "body": "## 研究决策\n\n选题继续聚焦**类别增量 TAG 的递归凝聚**。当前可成立的贡献是：测量并证明 graph–text correspondence 漂移是持续 TAG 特有瓶颈；当前不可成立的主张是：单独加入文本锚定已经形成跨 seed 稳定的最终算法。与 CaT 的差异应落在图文配对和文本条件响应保持；与 DANCE 的差异应落在可独立重训的 synthetic TAG memory；与静态 TAGSAM 的差异是递归更新与阶段误差累积。",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": "## 下一步实验\n\n1. 加入最小的无标签 `response-retention`：旧 rows 保持上一阶段 teacher 的 class-prompt logits 或 pairwise response，新 rows 匹配 current-delta teacher。\n2. 首先只在 stage 1 做 gate：两个 condensation seed 都必须选择 `outer>0`，且累计 dev 不低于 outer=0。\n3. 通过后固定权重，执行 stage 1→2 dev-only 递归；要求两个 seed 的累计、历史和当前 dev 都不劣于各自 λ=0 dev-best。\n4. 再用独立 student 读取 test，扩展更多 condensation seeds 与第二个 TAG 数据集。\n5. 最终评估使用新的、未被本轮开发访问的数据集。\n\n跨阶段边、transport、GraphCLIP 和额外 prompt 暂缓，避免在主因尚未稳定解决时增加不可归因模块。",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-09-08T13:00:00+08:00",
            "status": "ready",
            "datasets": {
                "stage2_only_gain": comparisons,
                "recursive_seed_gain": recursive_seed_gain,
                "recursive_selection": selections,
                "recursive_aggregate": recursive_aggregate,
            },
        },
        "sources": sources,
    }
    # Keep the portable reader compact enough for its fixed desktop verifier.
    # The full methods narrative remains in the adjacent Markdown report.
    artifact["manifest"]["charts"] = [artifact["manifest"]["charts"][0]]
    artifact["manifest"]["tables"] = []
    artifact["manifest"]["blocks"] = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": "## 技术结论\n\n**图文对应漂移已被确认。** 在类别增量且历史原图不可重读的协议下，Text-Anchored ASM 的固定 stage-2 更新有效；严格递归后仅 7/12 配对为正、seed 方向相反，因此还不是稳定最终算法。Cora test 已访问，数值仅作开发证据。",
        },
        {
            "id": "result_interpretation",
            "type": "markdown",
            "body": "## 固定 stage-2 更新证明机制有效\n\n同 outer=60 时，候选累计 test 提高 4.34 pp、历史 test 提高 5.07 pp、当前 test 提高 4.85 pp。允许基线使用自身 dev 最优 outer=20 后，累计 test 收益缩小到 1.47 pp。图中正值表示 Text-Anchored 更好。",
        },
        {"id": "stage2_only_chart", "type": "chart", "chartId": "stage2_only_gain", "layout": "full"},
        {
            "id": "recursive_result",
            "type": "markdown",
            "layout": "half",
            "sourceId": "recursive",
            "body": "## 严格递归的限制\n\nStage 1 的四条轨迹全部选择 outer=0。Stage 2 的确认均值为 39.10% 对 37.49%，但 seed 44 为 −0.84 pp、seed 45 为 +4.08 pp，收益缺少跨 seed 稳定性。",
        },
        {
            "id": "decision",
            "type": "markdown",
            "layout": "half",
            "body": "## 研究决策与下一步\n\n保留文本对应漂移作为核心 TAG challenge。下一版加入无标签 teacher response retention，并先要求两个 condensation seed 的 stage-1 dev 都选择 outer>0 且不低于初始化，再做递归确认。",
        },
    ]
    return artifact


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = build()
    for dataset_id in ("stage2_only_gain", "recursive_seed_gain", "recursive_selection",
                       "recursive_aggregate"):
        rows = artifact["snapshot"]["datasets"][dataset_id]
        path = REPORT_DIR / f"{dataset_id}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    output = REPORT_DIR / "artifact.json"
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
