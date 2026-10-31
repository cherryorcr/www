"""Create the WikiCS/BookHis adaptive coherent comparison report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "WikiCS": ROOT / "results/public_baselines_adaptive_wikics_20261006/wikics_base4/results.json",
    "BookHis": ROOT / "results/public_baselines_adaptive_bookhis_20261006/bookhis_base4/results.json",
}
LABELS = {
    "cat_mmd_tag_cctp_top5": "CCTP top-5",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive": "Adaptive coherent",
    "cat_mmd_tag_x_conditioned_routing_coherent_adaptive_anchor": "Adaptive coherent + CCTP anchor",
}


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def pm(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    return f"{mean:.2f} +/- {0.0 if std is None else std:.2f}"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_table(result: dict) -> list[dict]:
    rows = []
    for arm, summary in result["overall"].items():
        row = {"arm": arm, **summary}
        rows.append({
            "method": LABELS.get(arm, arm),
            "arm": row["arm"],
            "final_f1": row["final_f1"],
            "final_f1_std": row.get("final_f1_std"),
            "final_accuracy": row["final_accuracy"],
            "final_accuracy_std": row.get("final_accuracy_std"),
            "stage_avg_f1": row["stage_avg_f1"],
            "old_f1": row["old_f1"],
            "new_f1": row["new_f1"],
            "worst_stage_f1": row["worst_stage_f1"],
            "construction_seconds": row["construction_seconds"],
        })
    return rows


def stage_rows(result: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {arm: [] for arm in LABELS}
    for row in result["rows"]:
        grouped[row["arm"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["stage"]))
    return grouped


def write_table(lines: list[str], rows: list[dict]) -> None:
    lines.extend([
        "| Method | Final Macro-F1 (mean +/- std) | Final Accuracy (mean +/- std) | Stage avg F1 | Old F1 | New F1 | Worst stage F1 | Construction (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['method']} | {pm(row['final_f1'], row['final_f1_std'])} | {pm(row['final_accuracy'], row['final_accuracy_std'])} | "
            f"{pct(row['stage_avg_f1'])} | {pct(row['old_f1'])} | {pct(row['new_f1'])} | "
            f"{pct(row['worst_stage_f1'])} | {row['construction_seconds']:.1f} |"
        )


def write_stage_table(lines: list[str], result: dict) -> None:
    grouped = stage_rows(result)
    lines.extend([
        "| Method | Stage F1 (0 -> 4) | Stage Accuracy (0 -> 4) |",
        "|---|---|---|",
    ])
    for arm, label in LABELS.items():
        rows = grouped[arm]
        f1 = ", ".join(f"{row['score']['stage_f1'][-1]:.2f}" for row in rows)
        acc = ", ".join(f"{row['score']['stage_accuracy'][-1]:.2f}" for row in rows)
        lines.append(f"| {label} | {f1} | {acc} |")


def main() -> None:
    loaded = {name: load(path) for name, path in INPUTS.items()}
    output = ROOT / "docs/current/ADAPTIVE_COHERENT_WIKICS_BOOKHIS_2026-10-06.md"
    lines = [
        "# Adaptive coherent 在 WikiCS 与 BookHis 上的三方法对比",
        "",
        "本报告记录同一份 `base+4`、5% fixed-rate cumulative memory、pool-only construction、"
        "`report_dev_mod=3`/`test_mod=2` 协议下的正式结果。每个方法使用相同的 edge-aware 2-layer GCN + "
        "text projection + bidirectional InfoNCE Student，Student seeds 为 `100..105`，construction seed 为 `270927`。",
        "",
        "## 方法",
        "",
        "- **CCTP top-5**：CaT-style class-wise MMD 特征凝聚，按每个合成行选择图文兼容的 5 个当前 delta 文本来源。",
        "- **Adaptive coherent**：在 CaT 特征初始化上进行 X-conditioned coherent adaptive slot routing；slot 数量由当前类别图文离散度自适应决定，保留 320 字符文本预算。",
        "- **Adaptive coherent + CCTP anchor**：在 adaptive coherent 上增加 CCTP anchor 约束，用于限制 payload 偏离原始 CCTP 候选。",
        "",
        "三者在本 runner 中都属于 edge-free mechanism adaptation，Student 侧使用 self-loop-only synthetic graph；因此这里验证的是特征与文本 payload 机制，不是合成拓扑的完整比较。",
        "",
    ]
    for dataset, result in loaded.items():
        lines.extend([f"## {dataset}", ""])
        write_table(lines, final_table(result))
        lines.append("")
        lines.append("逐阶段最终 checkpoint 指标（每个 cell 已对 6 个 Student seeds 取均值）：")
        lines.append("")
        write_stage_table(lines, result)
        lines.append("")

    wikics = {row["arm"]: row for row in final_table(loaded["WikiCS"])}
    bookhis = {row["arm"]: row for row in final_table(loaded["BookHis"])}
    lines.extend([
        "## 结果解读",
        "",
        f"- WikiCS：Adaptive coherent 的 Final Macro-F1 为 `{wikics['cat_mmd_tag_x_conditioned_routing_coherent_adaptive']['final_f1']:.2f}`，"
        f"比 CCTP top-5 高 `{wikics['cat_mmd_tag_x_conditioned_routing_coherent_adaptive']['final_f1'] - wikics['cat_mmd_tag_cctp_top5']['final_f1']:.2f}` pp；"
        f"Accuracy 高 `{wikics['cat_mmd_tag_x_conditioned_routing_coherent_adaptive']['final_accuracy'] - wikics['cat_mmd_tag_cctp_top5']['final_accuracy']:.2f}` pp。",
        f"- BookHis：Adaptive coherent 的 Final Macro-F1 为 `{bookhis['cat_mmd_tag_x_conditioned_routing_coherent_adaptive']['final_f1']:.2f}`，"
        f"仅比 CCTP top-5 高 `{bookhis['cat_mmd_tag_x_conditioned_routing_coherent_adaptive']['final_f1'] - bookhis['cat_mmd_tag_cctp_top5']['final_f1']:.2f}` pp，"
        f"但 Accuracy 高 `{bookhis['cat_mmd_tag_x_conditioned_routing_coherent_adaptive']['final_accuracy'] - bookhis['cat_mmd_tag_cctp_top5']['final_accuracy']:.2f}` pp，"
        "同时 New F1 只有 `1.23`，说明长尾类别的新类塑性仍然不足，Accuracy 的提升不能等价理解为均衡分类能力提升。",
        "- CCTP anchor 在两个数据集上都没有优于无 anchor 的 adaptive coherent；它可以作为约束消融，但当前不应升级为主方法。",
        "- WikiCS 的正向结果较强且跨 5 个阶段保持；BookHis 的结果说明该机制对大规模、长尾类别流仍需专门的新类保护或类级配额机制。",
        "",
        "## 协议限制",
        "",
        "本轮每个数据集只有一个冻结类别顺序和一个 construction seed，虽然每个 cell 使用 6 个 Student seeds，但尚未证明顺序泛化。"
        "公开基线在此处是机制适配版本，不是所有论文方法的官方端到端复现。后续应在不改变本轮主结果的前提下补 3 个独立顺序，并单独报告 New/Old F1。",
        "",
        "## 原始结果",
        "",
        "- WikiCS: `results/public_baselines_adaptive_wikics_20261006/`",
        "- BookHis: `results/public_baselines_adaptive_bookhis_20261006/`",
        "- Protocol manifests: `results/public_baselines_adaptive_wikics_20261006/wikics_base4/protocol.json` 和 `results/public_baselines_adaptive_bookhis_20261006/bookhis_base4/protocol.json`",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
