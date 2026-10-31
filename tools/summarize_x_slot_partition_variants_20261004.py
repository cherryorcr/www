"""Summarize the four X-conditioned slot-partition variants.

This report treats the Current arm from the same run as the matched control.
Historical XSP results are reported separately because the standalone run did
not reproduce the same scores under the same declared protocol.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "x_slot_partition_variants_summary_20261004"
DOC = ROOT / "docs" / "current" / "X_CONDITIONED_SLOT_PARTITION_VARIANTS_ZH_2026-10-04.md"

DATASETS = ("arxiv", "bookchild", "sportsfit")
VARIANTS = (
    "cat_mmd_tag_x_slot_partition",
    "cat_mmd_tag_x_slot_partition_st",
    "cat_mmd_tag_x_slot_partition_anchor",
    "cat_mmd_tag_x_slot_partition_full",
)
VARIANT_LABELS = {
    VARIANTS[0]: "Current XSP",
    VARIANTS[1]: "XSP + ST-Sinkhorn",
    VARIANTS[2]: "XSP + coherence anchor",
    VARIANTS[3]: "XSP + ST + anchor + hard consistency",
}
PUBLIC = (
    "append_edge",
    "x_plus_p",
    "random_real",
    "k_center",
    "medoid",
    "cat_mmd_tag_cctp_top5",
    "gecc_hard_tag",
    "cat_mmd_tag_class_shared_tagsam",
)
PUBLIC_LABELS = {
    "append_edge": "Append-edge control",
    "x_plus_p": "X+P control",
    "random_real": "Random real-node coreset",
    "k_center": "K-center coreset",
    "medoid": "Medoid coreset",
    "cat_mmd_tag_cctp_top5": "CaT-TAG CCTP top-5",
    "gecc_hard_tag": "GECC-TAG hard",
    "cat_mmd_tag_class_shared_tagsam": "CaT-TAG class-shared payload",
}
METRICS = (
    "final_f1",
    "final_accuracy",
    "stage_avg_f1",
    "old_f1",
    "new_f1",
    "worst_stage_f1",
    "construction_seconds",
    "student_seconds",
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values) -> float:
    values = [float(v) for v in values if v is not None]
    return float(statistics.mean(values)) if values else float("nan")


def variant_path(dataset: str) -> Path:
    gpu = "gpu1" if dataset == "sportsfit" else "gpu0"
    return ROOT / "results" / f"x_slot_partition_variants_20261004_{gpu}" / f"{dataset}_base4" / "results.json"


def public_path(dataset: str) -> Path:
    return ROOT / "results" / "public_baselines_large_corrected_20261003_remote" / dataset / "results.json"


def historical_xsp_path(dataset: str) -> Path:
    gpu = "gpu1" if dataset == "sportsfit" else "gpu0"
    return ROOT / "results" / f"x_slot_partition_large_20261003_{gpu}" / f"{dataset}_base4" / "results.json"


def collect_variant_cells() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        payload = read(variant_path(dataset))
        for arm in VARIANTS:
            value = payload["overall"][arm]
            rows.append({
                "dataset": dataset,
                "arm": arm,
                "method": VARIANT_LABELS[arm],
                **{key: value.get(key) for key in METRICS},
            })
    return rows


def collect_public_cells() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        payload = read(public_path(dataset))
        for arm in PUBLIC:
            value = payload["overall"][arm]
            rows.append({
                "dataset": dataset,
                "arm": arm,
                "method": PUBLIC_LABELS[arm],
                **{key: value.get(key) for key in METRICS},
            })
    return rows


def collect_audits() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        payload = read(variant_path(dataset))
        for arm in VARIANTS:
            records = [row for row in payload["rows"] if row["arm"] == arm]
            unique = [row["construction_audit"].get("unique_text_ratio") for row in records]
            x_relative = [row["construction_audit"].get("x_update_relative_mean") for row in records]
            trace_hard = []
            hard_enabled = False
            for row in records:
                audit = row["construction_audit"]
                hard = audit.get("hard_consistency", {})
                hard_enabled = hard_enabled or bool(hard.get("enabled"))
                trace = audit.get("trace") or []
                if hard.get("enabled") and trace:
                    trace_hard.append(trace[-1].get("hard"))
            rows.append({
                "dataset": dataset,
                "arm": arm,
                "method": VARIANT_LABELS[arm],
                "mean_unique_text_ratio": mean(unique),
                "min_unique_text_ratio": min(float(v) for v in unique if v is not None),
                "mean_x_update_relative": mean(x_relative),
                "max_x_update_relative": max(float(v) for v in x_relative if v is not None),
                "hard_consistency_enabled": hard_enabled,
                "mean_last_recorded_hard_loss": mean(trace_hard),
                "assignment_diagnostics": "not pooled: the original run recorded unnormalized Sinkhorn column mass",
            })
    return rows


def pooled(rows: list[dict], arms: tuple[str, ...]) -> list[dict]:
    output = []
    for arm in arms:
        local = [row for row in rows if row["arm"] == arm]
        output.append({
            "arm": arm,
            "method": local[0]["method"],
            **{metric: mean(row.get(metric) for row in local) for metric in METRICS},
        })
    return output


def paired_variant(rows: list[dict]) -> list[dict]:
    index = {(row["dataset"], row["arm"]): row for row in rows}
    baseline = VARIANTS[0]
    output = []
    for arm in VARIANTS[1:]:
        for dataset in DATASETS:
            current = index[(dataset, baseline)]
            candidate = index[(dataset, arm)]
            output.append({
                "dataset": dataset,
                "arm": arm,
                "method": VARIANT_LABELS[arm],
                "f1_delta": candidate["final_f1"] - current["final_f1"],
                "accuracy_delta": candidate["final_accuracy"] - current["final_accuracy"],
                "stage_avg_f1_delta": candidate["stage_avg_f1"] - current["stage_avg_f1"],
                "old_f1_delta": candidate["old_f1"] - current["old_f1"],
                "new_f1_delta": candidate["new_f1"] - current["new_f1"],
                "build_multiplier": candidate["construction_seconds"] / max(current["construction_seconds"], 1e-9),
            })
    return output


def public_paired(variant_rows: list[dict], public_rows: list[dict]) -> list[dict]:
    vi = {(row["dataset"], row["arm"]): row for row in variant_rows}
    pi = {(row["dataset"], row["arm"]): row for row in public_rows}
    output = []
    for arm in VARIANTS:
        for public_arm in PUBLIC:
            deltas = [
                vi[(dataset, arm)]["final_f1"] - pi[(dataset, public_arm)]["final_f1"]
                for dataset in DATASETS
            ]
            output.append({
                "variant_arm": arm,
                "variant": VARIANT_LABELS[arm],
                "baseline_arm": public_arm,
                "baseline": PUBLIC_LABELS[public_arm],
                "mean_f1_delta": mean(deltas),
                "positive_datasets": sum(delta > 0 for delta in deltas),
                "deltas": {dataset: delta for dataset, delta in zip(DATASETS, deltas)},
            })
    return output


def historical_comparison(variant_rows: list[dict]) -> list[dict]:
    output = []
    index = {(row["dataset"], row["arm"]): row for row in variant_rows}
    for dataset in DATASETS:
        old = read(historical_xsp_path(dataset))["overall"][VARIANTS[0]]
        current = index[(dataset, VARIANTS[0])]
        output.append({
            "dataset": dataset,
            "current_run_f1": current["final_f1"],
            "historical_xsp_f1": old["final_f1"],
            "delta_current_minus_historical": current["final_f1"] - old["final_f1"],
            "current_run_accuracy": current["final_accuracy"],
            "historical_xsp_accuracy": old["final_accuracy"],
        })
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def build_report(variant_rows, public_rows, audit_rows, variant_pool, public_pool, paired, public_pairs, historical) -> str:
    lines = [
        "# X-conditioned Slot Partition 变体实验报告",
        "",
        "> 本轮比较四个版本：Current、ST-Sinkhorn、coherence anchor、ST+anchor+hard consistency。所有变体在同一远程进程中使用相同的 base+4、5% cumulative memory、Student seeds 100--105、construction seed 270927、current-delta-only 和 edge-free memory 协议。",
        "",
        "## 1. 三图结果",
        "",
        "|数据集|方法|Final F1|Accuracy|Stage-average F1|Old F1|New F1|Worst stage|构造秒数|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variant_rows:
        lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|{}|".format(
            row["dataset"], row["method"], *(fmt(row[key]) for key in (
                "final_f1", "final_accuracy", "stage_avg_f1", "old_f1",
                "new_f1", "worst_stage_f1", "construction_seconds"))))
    lines += [
        "",
        "## 2. 等权三图汇总",
        "",
        "|方法|Final F1|Accuracy|Stage-average F1|Old F1|New F1|Worst stage|构造秒数|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variant_pool:
        lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|".format(
            row["method"], *(fmt(row[key]) for key in (
                "final_f1", "final_accuracy", "stage_avg_f1", "old_f1",
                "new_f1", "worst_stage_f1", "construction_seconds"))))
    lines += [
        "",
        "## 3. 相对 Current 的配对差值",
        "",
        "|数据集|方法|F1差值|Accuracy差值|Stage-average差值|Old F1差值|New F1差值|构造成本倍数|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append("|{}|{}|{:+.2f}|{:+.2f}|{:+.2f}|{:+.2f}|{:+.2f}|{:.2f}x|".format(
            row["dataset"], row["method"], row["f1_delta"], row["accuracy_delta"],
            row["stage_avg_f1_delta"], row["old_f1_delta"], row["new_f1_delta"],
            row["build_multiplier"]))
    lines += [
        "",
        "## 4. 与公开/适配基线的 F1 对照",
        "",
        "|方法|Final F1|Accuracy|Stage-average F1|Old F1|New F1|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in public_pool:
        lines.append("|{}|{}|{}|{}|{}|{}|".format(
            row["method"], *(fmt(row[key]) for key in (
                "final_f1", "final_accuracy", "stage_avg_f1", "old_f1", "new_f1"))))
    for row in variant_pool:
        lines.append("|{}|{}|{}|{}|{}|{}|".format(
            row["method"], *(fmt(row[key]) for key in (
                "final_f1", "final_accuracy", "stage_avg_f1", "old_f1", "new_f1"))))
    lines += [
        "",
        "公开基线的统一池化结果来自 `public_baselines_large_corrected_20261003_remote`。其协议字段与本轮一致，但它们不是同一个代码快照中的同轮重跑，因此这里用于方向性比较；正式论文主表前仍应在冻结代码快照下重跑关键基线。",
        "",
        "## 5. 结果解释",
        "",
        "1. ST-Sinkhorn 是最稳定的低成本改动：三图平均 Final F1 相对 Current 为 +2.00 pp，Old F1 为 +2.35 pp，New F1 基本不变（+0.10 pp）。这说明让 assignment plan 对 X 可微，主要改善了旧类保持和整体可学习性，但没有解决新类塑性瓶颈。",
        "2. coherence anchor 没有带来稳定增益：三图平均仅 +0.97 pp，且 Arxiv 下降 0.28 pp、New F1 略降。仅按 assignment entropy 对低置信 slot 拉向类别中心，容易抹平类内多模态，暂不应单独进入主方法。",
        "3. Full 在三图平均 Final F1 达到 23.36，相对 Current +2.67 pp，Accuracy +3.69 pp，Old F1 +3.23 pp；但 New F1 -0.56 pp，构造时间约为 Current 的 2.70x。它改善的是 hard payload 与 soft assignment 的一致性以及旧类保持，不是新类塑性。",
        "4. Full 的收益并不等于已经证明主方法成立：它在 Arxiv 仍低于 CCTP top-5/class-shared，在 SportsFit 仍低于 class-shared；相对 k-center 和 GECC 则多数数据集有优势。当前更准确的结论是：ST 是低成本有效组件，hard consistency 是有代价的候选组件，Anchor 不值得保留。",
        "5. 本轮 Current 与历史单独 XSP 运行分数不完全一致，因此历史 XSP 不能作为严格复现基线。后续必须冻结代码、缓存、Student 数值设置后重跑 Current 与关键公开基线；本轮变体之间的同轮配对差值仍然有效。",
        "",
        "## 6. 审计与实现修正",
        "",
        "本轮四个版本均保持 `historical_raw_reads=0`、`old_memory_updated=False`、`topology_updated=False`，文本是按 X slot 的 transport cluster 材料化，三图没有出现单一 class payload 广播塌缩。结果生成后发现 slot audit 中 entropy/effective-source 的统计漏除了 Sinkhorn 列质量；该问题不影响训练或 F1，只影响诊断字段。代码已修正为先归一化列质量，并补充了测试；本轮已有 JSON 的该两列不用于结论。",
        "",
        "## 7. 下一步",
        "",
        "- 保留 Current、ST、Full 三个版本做一次独立 construction-seed 确认，淘汰 Anchor；不要再叠加新的 selector 或拓扑模块。",
        "- 在同一冻结代码快照下重跑 CCTP、GECC、k-center 和 class-shared 的关键三图结果，先解决 Current 历史分数不复现的问题。",
        "- 对 Full 做成本优化实验：只保留最终一次 hard refresh、缓存 MiniLM 编码，确认收益是否来自 hard consistency 本身，而不是重复编码次数。该实验只改实现成本，不改变目标函数。",
        "- 新类塑性仍是主要未解决问题。下一阶段应直接分析 slot 数量、assignment mass 和新类 margin 的关系，不再依赖 anchor 强行收缩类内文本。",
        "- 在上述确认完成前，不把 Full 冻结为最终主方法；论文候选定位应为“ST-Sinkhorn 为有效组件，Full 为高成本增强候选”。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    variant_rows = collect_variant_cells()
    public_rows = collect_public_cells()
    audit_rows = collect_audits()
    variant_pool = pooled(variant_rows, VARIANTS)
    public_pool = pooled(public_rows, PUBLIC)
    paired = paired_variant(variant_rows)
    public_pairs = public_paired(variant_rows, public_rows)
    historical = historical_comparison(variant_rows)
    write_csv(OUT / "variant_cells.csv", variant_rows)
    write_csv(OUT / "variant_pooled.csv", variant_pool)
    write_csv(OUT / "variant_vs_current.csv", paired)
    write_csv(OUT / "public_baseline_cells.csv", public_rows)
    write_csv(OUT / "public_baseline_pooled.csv", public_pool)
    write_csv(OUT / "variant_vs_public_f1.csv", public_pairs)
    write_csv(OUT / "historical_xsp_reproducibility.csv", historical)
    write_csv(OUT / "audit_summary.csv", audit_rows)
    payload = {
        "protocol": read(variant_path(DATASETS[0]))["protocol"],
        "variant_pooled": variant_pool,
        "public_baseline_pooled": public_pool,
        "variant_vs_current": paired,
        "variant_vs_public_f1": public_pairs,
        "historical_xsp_reproducibility": historical,
        "audit_summary": audit_rows,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(variant_rows, public_rows, audit_rows, variant_pool,
                          public_pool, paired, public_pairs, historical)
    (OUT / "summary.md").write_text(report, encoding="utf-8")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(report, encoding="utf-8")
    print(OUT / "summary.md")
    print(DOC)


if __name__ == "__main__":
    main()
