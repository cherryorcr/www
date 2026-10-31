"""Merge strict large-graph baselines and render comparison figures.

The existing frozen comparison contains the common coreset/CaT/GECC arms and
the adaptive candidate.  This report adds the CaT text-processing arms when
their JSON artifacts are available.  Protocol differences, especially
construction seeds, are retained in the output instead of being hidden.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("arxiv", "sportsfit", "wikics", "bookhis")

LABELS = {
    "append_edge": "Direct append / continual Student control",
    "x_plus_p": "C0 / X+P prior method",
    "random_real": "Random real coreset",
    "k_center": "K-center/FPS",
    "medoid": "Medoid coreset",
    "gecc_hard_tag": "GECC-hard-TAG adapted",
    "cat_mmd_tag_cctp_top5": "CaT-MMD-TAG + CCTP top-5",
    "cat_mmd_tag": "CaT-MMD-TAG + nearest",
    "cat_mmd_tag_tagsam": "CaT-MMD-TAG + TAGSAM-style chunk",
    "cat_mmd_tag_graph_text": "CaT-MMD-TAG + graph-conditioned text",
    "cat_mmd_tag_graph_text_aligned": "CaT-MMD-TAG + graph-text aligned",
    "cat_mmd_tag_text_center": "CaT-MMD-TAG + class text-center",
    "cat_mmd_tag_cctp_top3": "CaT-MMD-TAG + CCTP top-3",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive": "Adaptive coherent",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor": "Adaptive coherent + anchor",
}

CATEGORIES = {
    "append_edge": "continual control",
    "x_plus_p": "prior method control",
    **{k: "strict adapted baseline" for k in (
        "random_real", "k_center", "medoid", "gecc_hard_tag",
        "cat_mmd_tag_cctp_top5",
    )},
    **{k: "CaT text component ablation" for k in (
        "cat_mmd_tag", "cat_mmd_tag_tagsam", "cat_mmd_tag_graph_text",
        "cat_mmd_tag_graph_text_aligned", "cat_mmd_tag_cctp_top3",
    )},
    "cat_mmd_tag_text_center": "analysis control",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive": "candidate method",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor": "internal control",
}

NEW_ARMS = (
    "cat_mmd_tag", "cat_mmd_tag_tagsam", "cat_mmd_tag_graph_text",
    "cat_mmd_tag_graph_text_aligned", "cat_mmd_tag_text_center",
    "cat_mmd_tag_cctp_top3",
)

EXISTING_ARMS = (
    "append_edge", "x_plus_p",
    "random_real", "k_center", "medoid", "gecc_hard_tag",
    "cat_mmd_tag_cctp_top5",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor",
)

# Keep the candidate method first in the report, followed by the strongest
# protocol-matched condensation references.  This ordering is presentation
# only; it does not change how rows are loaded or how pooled means are computed.
STRONG_DISPLAY_ORDER = (
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive",
    "cat_mmd_tag_cctp_top5",
    "gecc_hard_tag",
    "k_center",
    "medoid",
    "random_real",
    "x_plus_p",
    "append_edge",
)

BASELINE_CARDS = (
    ("Adaptive coherent", "本文候选主方法", "CaT class-wise MMD 优化 X；当前 delta 合法文本池；图文离散度自适应 1/2/3 slots；X-conditioned routing；hard materialization", "candidate"),
    ("C0 / X+P", "已有主线控制", "保留既有 X+P 合成 memory 和当前 Student 接口，用于测量新增文本 routing 的净增益", "control"),
    ("CaT-MMD-TAG + CCTP top-5", "持续凝聚机制适配", "保留 CaT class-wise MMD X；每个合成行从 current delta 选择合法 top-5 文本来源；不回读历史 raw", "strong adapted"),
    ("GECC-hard-TAG adapted", "演化/结构无关凝聚适配", "在 current delta 上保留 hard-hop/Lloyd 聚类机制，再附加合法 TAG 文本；不运行官方静态全图流程", "strong adapted"),
    ("K-center/FPS", "真实节点覆盖 coreset", "current delta 联合图文空间 farthest-point sampling；保留真实 X、文本和来源；不访问历史 raw", "strong adapted"),
    ("Medoid coreset", "真实节点代表点", "current delta 联合空间选择 medoid；文本随真实节点保留；旧 memory 只作 frozen prefix", "strong adapted"),
    ("Random real coreset", "真实节点随机控制", "current delta 随机选择真实节点；同预算、同 Student、同 frozen-prefix 规则", "strong adapted"),
    ("Direct append / continual Student", "不凝聚的持续学习控制", "只追加当前阶段可用 memory/raw block 并持续训练同一 Prompt Student；不输出 synthetic condensation memory", "control"),
    ("CaT + nearest / TAGSAM-style / graph-conditioned / graph-text aligned / CCTP top-3", "CaT 文本组件消融", "固定 CaT X、预算、Student 和阶段协议，只更换文本 payload 的候选、chunk 或对齐规则；不作为独立强基线排名", "component ablation"),
    ("PUMA-inspired", "公开持续凝聚外部参照", "小图适配中保留 pseudo-label、one-time propagation 和 edge-free memory；尚无同协议大图结果，不虚构", "small-only adapted"),
    ("TAGSAM / GraphCLIP / GCond / SFGC / GCondenser", "静态 TAG/图凝聚参考", "原论文通常读取完整静态图或使用不同 Student；未通过 current-delta-only/frozen-prefix 审计，不进入 strict 数值主表", "reference-only"),
)

CONTROL_SOURCES = {
    "arxiv": ROOT / "tmp_remote_results/public_baselines_large_corrected_20261003_gpu0/arxiv_base4/results.json",
    "bookchild": ROOT / "tmp_remote_results/public_baselines_large_corrected_20261003_gpu0/bookchild_base4/results.json",
    "sportsfit": ROOT / "tmp_remote_results/public_baselines_large_corrected_20261003_gpu1/sportsfit_base4/results.json",
}


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def metric(values: dict, name: str, fallback: str | None = None) -> float:
    value = values.get(name)
    if value is None and fallback:
        value = values.get(fallback)
    if value is None:
        raise KeyError(f"missing metric {name}")
    return float(value)


def validate_protocol(protocol: dict, dataset: str) -> list[str]:
    problems = []
    if protocol.get("dataset") != dataset:
        problems.append(f"dataset={protocol.get('dataset')}")
    if protocol.get("stage_protocol") != "base+4":
        problems.append("stage_protocol")
    if abs(float(protocol.get("memory_rate", -1)) - 0.05) > 1e-9:
        problems.append("memory_rate")
    if protocol.get("student_seeds") != [100, 101, 102, 103, 104, 105]:
        problems.append("student_seeds")
    if int(protocol.get("historical_raw_reads", 1)) != 0:
        problems.append("historical_raw_reads")
    if bool(protocol.get("report_test_labels_used_for_construction", True)):
        problems.append("report_test_labels_used_for_construction")
    return problems


def make_row(dataset: str, arm: str, values: dict, protocol: dict,
             source: str, protocol_status: str) -> dict:
    return {
        "dataset": dataset,
        "arm": arm,
        "method": LABELS[arm],
        "category": CATEGORIES[arm],
        "protocol_status": protocol_status,
        "source": source,
        "construction_seed": protocol.get("construction_seed"),
        "final_f1": metric(values, "final_f1"),
        "final_f1_std": metric(values, "final_f1_std"),
        "accuracy": metric(values, "final_accuracy", "accuracy"),
        "accuracy_std": metric(values, "final_accuracy_std", "accuracy_std"),
        "stage_avg_f1": metric(values, "stage_avg_f1"),
        "stage_avg_accuracy": metric(values, "stage_avg_accuracy"),
        "old_f1": metric(values, "old_f1"),
        "new_f1": metric(values, "new_f1"),
        "worst_stage_f1": metric(values, "worst_stage_f1"),
        "construction_seconds": metric(values, "construction_seconds"),
        "student_seconds": metric(values, "student_seconds"),
    }


def load_existing(path: Path) -> tuple[list[dict], dict[str, dict]]:
    payload = load(path)
    rows = []
    protocols = {}
    for cell in payload["cells"]:
        if cell["arm"] not in EXISTING_ARMS:
            continue
        protocols.setdefault(cell["dataset"], payload["protocols"][cell["dataset"]])
        status = "strict: frozen setting"
        rows.append(make_row(cell["dataset"], cell["arm"], cell,
                             payload["protocols"][cell["dataset"]],
                             str(path), status))
    return rows, protocols


def load_controls(protocols: dict[str, dict]) -> list[dict]:
    rows = []
    for dataset, path in CONTROL_SOURCES.items():
        if not path.exists() or dataset not in protocols:
            continue
        payload = load(path)
        protocol = payload.get("protocol", {})
        problems = validate_protocol(protocol, dataset)
        if protocol.get("groups") != protocols[dataset].get("groups"):
            problems.append("groups")
        if protocol.get("budgets") != protocols[dataset].get("budgets"):
            problems.append("budgets")
        status = "strict control" if not problems else "excluded: " + ",".join(problems)
        for arm in ("append_edge", "x_plus_p"):
            values = payload.get("overall", {}).get(arm)
            if values is not None:
                rows.append(make_row(dataset, arm, values, protocol,
                                     str(path), status))
    return rows


def find_new_result(root: Path, dataset: str) -> Path | None:
    candidates = [
        root / "arxiv_base4" / "results.json",
        root / "sportsfit_base4" / "results.json",
        root / "wikics_base4" / "results.json",
        root / "bookhis_base4" / "results.json",
    ]
    direct = root / dataset / "results.json"
    if direct.exists():
        return direct
    for candidate in candidates:
        if candidate.exists() and candidate.parent.name == f"{dataset}_base4":
            return candidate
    return None


def load_new(root: Path, dataset: str, reference_protocol: dict) -> list[dict]:
    path = find_new_result(root, dataset)
    if path is None:
        return []
    payload = load(path)
    protocol = payload["protocol"]
    problems = validate_protocol(protocol, dataset)
    same_order = protocol.get("groups") == reference_protocol.get("groups")
    same_budget = protocol.get("budgets") == reference_protocol.get("budgets")
    same_seed = protocol.get("construction_seed") == reference_protocol.get("construction_seed")
    if not same_order or not same_budget:
        status = "excluded: stage order/budget mismatch"
    elif problems:
        status = "excluded: protocol audit " + ",".join(problems)
    elif same_seed:
        status = "strict: same setting"
    else:
        status = "strict setting, independent construction seed"
    rows = []
    for arm in NEW_ARMS:
        values = payload.get("overall", {}).get(arm)
        if values is None:
            continue
        rows.append(make_row(dataset, arm, values, protocol, str(path), status))
    return rows


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_baseline_cards(path: Path) -> None:
    lines = [
        "# 基线适配卡片",
        "",
        "方法名、原始角色和为统一持续 TAG 协议所做的改动分开记录。`strong adapted` 可以进入严格主表；`component ablation` 只进入 CaT 文本消融表；`reference-only` 不与主表横向排名。",
        "",
        "| 方法名 | 原始角色 | 统一协议下的改动 | 分层 |",
        "|---|---|---|---|",
    ]
    for name, role, adaptation, layer in BASELINE_CARDS:
        lines.append(f"| {name} | {role} | {adaptation} | {layer} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pooled(rows: list[dict], arms: list[str]) -> list[dict]:
    result = []
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm and not row["protocol_status"].startswith("excluded")]
        if not selected:
            continue
        result.append({
            "arm": arm,
            "method": LABELS[arm],
            "category": CATEGORIES[arm],
            "datasets": ",".join(row["dataset"] for row in selected),
            "final_f1": mean(row["final_f1"] for row in selected),
            "accuracy": mean(row["accuracy"] for row in selected),
            "stage_avg_f1": mean(row["stage_avg_f1"] for row in selected),
            "old_f1": mean(row["old_f1"] for row in selected),
            "new_f1": mean(row["new_f1"] for row in selected),
            "construction_seconds": mean(row["construction_seconds"] for row in selected),
        })
    return result


def render_plots(rows: list[dict], output: Path) -> list[str]:
    import matplotlib.pyplot as plt
    import numpy as np

    usable = [row for row in rows if not row["protocol_status"].startswith("excluded")]
    strong_order = []
    for arm in STRONG_DISPLAY_ORDER:
        if any(row["arm"] == arm for row in usable) and arm not in strong_order:
            strong_order.append(arm)
    component_order = []
    for arm in NEW_ARMS:
        if any(row["arm"] == arm for row in usable) and arm not in component_order:
            component_order.append(arm)
    # Do not alias the list: component arms must stay out of the strong-baseline
    # figure and the relative-delta heatmap.
    order = list(strong_order)
    for arm in component_order:
        if any(row["arm"] == arm for row in usable) and arm not in order:
            order.append(arm)
    files = []

    def grouped_plot(arms, metric_name, title, filename, color_candidate=True):
        fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
        for ax, dataset in zip(axes.flat, DATASETS):
            values = []
            errors = []
            names = []
            for arm in arms:
                selected = [r for r in usable if r["dataset"] == dataset and r["arm"] == arm]
                if not selected:
                    continue
                row = selected[0]
                values.append(row[metric_name])
                errors.append(row[f"{metric_name}_std"])
                names.append(LABELS[arm])
            x = np.arange(len(values))
            ax.bar(x, values, yerr=errors, capsize=3,
                   color=["#d95f02" if color_candidate and n == "Adaptive coherent" else "#4c78a8" for n in names])
            ax.set_title(dataset)
            ax.set_ylabel("percent")
            ax.set_xticks(x, names, rotation=58, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(title)
        path = output / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(str(path))

    grouped_plot(strong_order, "final_f1", "Final Macro-F1: strong baselines and main method", "final_f1_strong_by_dataset.png")
    grouped_plot(strong_order, "accuracy", "Accuracy: strong baselines and main method", "accuracy_strong_by_dataset.png")
    if component_order:
        grouped_plot(component_order, "final_f1", "CaT text component ablation: Final Macro-F1", "final_f1_cat_text_ablation_by_dataset.png", color_candidate=False)
        grouped_plot(component_order, "accuracy", "CaT text component ablation: Accuracy", "accuracy_cat_text_ablation_by_dataset.png", color_candidate=False)

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    for arm in strong_order:
        selected = [r for r in usable if r["arm"] == arm]
        if not selected:
            continue
        ax.scatter([r["old_f1"] for r in selected], [r["new_f1"] for r in selected], label=LABELS[arm], s=70)
    ax.set_xlabel("Old-class F1")
    ax.set_ylabel("New-class F1")
    ax.set_title("Old/New F1 trade-off")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    path = output / "old_new_tradeoff.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    files.append(str(path))

    target = "cat_mmd_tag_x_conditioned_routing_coherent_adaptive"
    baseline_order = [arm for arm in strong_order if arm != target]
    matrix = []
    matrix_labels = []
    for dataset in DATASETS:
        target_rows = [r for r in usable if r["dataset"] == dataset and r["arm"] == target]
        if not target_rows:
            continue
        target_value = target_rows[0]["final_f1"]
        row_values = []
        for arm in baseline_order:
            base = [r for r in usable if r["dataset"] == dataset and r["arm"] == arm]
            row_values.append(target_value - base[0]["final_f1"] if base else np.nan)
        matrix.append(row_values)
        matrix_labels.append(dataset)
    if matrix:
        fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
        image = ax.imshow(np.asarray(matrix), cmap="RdYlGn", aspect="auto")
        ax.set_yticks(range(len(matrix_labels)), matrix_labels)
        ax.set_xticks(range(len(baseline_order)), [LABELS[a] for a in baseline_order], rotation=60, ha="right", fontsize=8)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if np.isfinite(value):
                    ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="Adaptive coherent F1 - baseline F1")
        ax.set_title("Adaptive coherent relative Final-F1 delta")
        path = output / "adaptive_delta_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        files.append(str(path))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", type=Path, default=ROOT / "results/multidataset_f1_accuracy_comparison_20261006/summary.json")
    parser.add_argument("--new-root", type=Path, default=ROOT / "results/text_modes_strict_20261006")
    parser.add_argument("--output", type=Path, default=ROOT / "results/expanded_baseline_comparison_20261006")
    args = parser.parse_args()

    rows, protocols = load_existing(args.existing)
    rows = load_controls(protocols) + rows
    for dataset in DATASETS:
        if dataset in protocols:
            rows.extend(load_new(args.new_root, dataset, protocols[dataset]))

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "all_cells.csv", rows)
    pooled_rows = pooled(rows, list(dict.fromkeys((*EXISTING_ARMS, *NEW_ARMS))))
    write_csv(args.output / "pooled_methods.csv", pooled_rows)
    write_baseline_cards(args.output / "baseline_adaptation_table.md")
    figures = render_plots(rows, args.output)
    strong_order = [
        arm for arm in STRONG_DISPLAY_ORDER
        if any(row["arm"] == arm and not row["protocol_status"].startswith("excluded") for row in rows)
    ]

    lines = [
        "# 扩展大图 baseline 对比与可视化",
        "",
        "本报告合并已有 frozen strict baseline 与 CaT 文本适配实验。所有数值按数据集分别报告，不把不同数据集的绝对分数当成同一个任务。新文本适配结果如果 construction seed 与 frozen reference 不同，会标记为 `strict setting, independent construction seed`，不隐藏该差异。",
        "",
        "方法名与适配动作卡片见 `baseline_adaptation_table.md`。CaT 的不同 TAG 文本模式只进入组件消融层；强基线层只比较协议匹配的外部机制、coreset 和控制方法。",
        "",
        "## 1. 协议审计",
        "",
        "| Dataset | Method source | Status | Construction seed | Historical raw | Student seeds |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        if row["arm"] == EXISTING_ARMS[0] or row["arm"] in NEW_ARMS:
            lines.append(f"| {row['dataset']} | {row['method']} | {row['protocol_status']} | {row['construction_seed']} | 0 | 100..105 |")

    for dataset in DATASETS:
        lines += ["", f"## 2. {dataset}：强基线与主方法", "", "| Method | Category | Status | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Worst stage | Build s |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in strong_order:
            matching = [row for row in rows if row["dataset"] == dataset and row["arm"] == arm]
            for row in matching:
                lines.append(
                    f"| {row['method']} | {row['category']} | {row['protocol_status']} | "
                    f"{fmt(row['final_f1'])} +/- {fmt(row['final_f1_std'])} | "
                    f"{fmt(row['accuracy'])} +/- {fmt(row['accuracy_std'])} | "
                    f"{fmt(row['old_f1'])} | {fmt(row['new_f1'])} | {fmt(row['stage_avg_f1'])} | "
                    f"{fmt(row['worst_stage_f1'])} | {fmt(row['construction_seconds'], 1)} |"
                )
        component_rows = [row for row in rows if row["dataset"] == dataset and row["arm"] in NEW_ARMS]
        if component_rows:
            lines += ["", f"### {dataset}：CaT 文本组件消融", "", "| Text payload variant | Status | Final Macro-F1 | Accuracy | Old F1 | New F1 | Stage avg F1 | Build s |", "|---|---|---:|---:|---:|---:|---:|---:|"]
            for row in component_rows:
                lines.append(
                    f"| {row['method']} | {row['protocol_status']} | {fmt(row['final_f1'])} +/- {fmt(row['final_f1_std'])} | "
                    f"{fmt(row['accuracy'])} +/- {fmt(row['accuracy_std'])} | {fmt(row['old_f1'])} | "
                    f"{fmt(row['new_f1'])} | {fmt(row['stage_avg_f1'])} | {fmt(row['construction_seconds'], 1)} |"
                )

    lines += ["", "## 3. 等权跨数据集汇总", "", "| Method | Category | Covered datasets | Final Macro-F1 | Accuracy | Old F1 | New F1 | Build s |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in pooled_rows:
        lines.append(f"| {row['method']} | {row['category']} | {row['datasets']} | {fmt(row['final_f1'])} | {fmt(row['accuracy'])} | {fmt(row['old_f1'])} | {fmt(row['new_f1'])} | {fmt(row['construction_seconds'], 1)} |")

    lines += [
        "",
        "## 4. 图表",
        "",
    ]
    for figure in figures:
        lines.append(f"- `{Path(figure).name}`")
    lines += [
        "",
        "## 5. 使用限制",
        "",
        "- CaT、GECC、K-center、Medoid 和随机 coreset 是当前 continual TAG 访问约束下的机制适配，不是把原论文 static/full-graph 流程原样搬入。",
        "- class text-center 作为分析控制单列，因为它让同一类别所有 synthetic rows 共享一个 payload，不与 node-specific payload 直接排名。",
        "- GraphCLIP、GCond、SFGC、GCondenser、TAGSAM 官方静态数字不进入本表，除非完成 current-delta-only、frozen-prefix 和同一 Student 的严格适配。",
        "- BookHis 必须结合 New F1 与 Macro-F1 解读，不能用 Accuracy 单独证明方法有效。",
    ]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps({"figures": figures, "rows": len(rows)}, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "figures": figures, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
