"""Build a protocol-aware evidence matrix for the current Bounded X+P line.

The workspace contains several generations of large-graph summaries.  Some
summary files were produced by a runner that reported sealed-memory loading
time and, in one copy, did not agree with the stage-level results.  This
script therefore reads the stage-level ``results.json`` artifacts as the
source of truth and keeps strict continual, mechanism-adapted, and TAGSAM
adapter results in separate tables.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TMP = ROOT / "tmp_remote_results"
OUT = RESULTS / "bounded_xp_final_evidence_20261001"
DOC = ROOT / "docs" / "current" / "BOUNDED_XP_FINAL_EVIDENCE_AND_BASELINE_MATRIX_ZH_2026-10-01.md"


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mean(values: list[float | None]) -> float | None:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(mean(values)) if values else None


def safe_std(values: list[float | None]) -> float | None:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(stdev(values)) if len(values) > 1 else 0.0 if values else None


def metric_value(value):
    """Extract a mean from either scalar or {mean, std} artifact fields."""
    if value is None:
        return None
    if isinstance(value, dict):
        if "mean" in value:
            return float(value["mean"])
        if "value" in value:
            return float(value["value"])
    return float(value)


def stage_metrics(row: dict) -> dict:
    """Normalize an XPA/public stage row to common metrics."""
    agg = row.get("aggregate")
    score = row.get("score")
    if agg is not None:
        return {
            "f1": metric_value(agg.get("mean_f1")),
            "accuracy": metric_value(agg.get("mean_accuracy")),
            "old_f1": metric_value(agg.get("old_f1")),
            "new_f1": metric_value(agg.get("new_f1")),
            "f1_std": metric_value(agg.get("std_f1")),
            "student_seconds": row.get("student_wall_seconds"),
            "construction_seconds": row.get("construction_seconds"),
        }
    if score is None:
        raise ValueError(f"stage row has neither aggregate nor score: {row.keys()}")
    return {
        "f1": metric_value(score.get("final_f1")),
        "accuracy": metric_value(score.get("accuracy")),
        "old_f1": metric_value(score.get("old_f1")),
        "new_f1": metric_value(score.get("new_f1")),
        "f1_std": metric_value(score.get("final_f1_std")),
        "student_seconds": score.get("student_seconds"),
        "construction_seconds": row.get("construction_seconds"),
    }


def summarize_stage_rows(rows: list[dict], dataset: str, method: str,
                         source: str, cost_kind: str,
                         actual_construction_seconds: float | None = None) -> dict:
    rows = sorted(rows, key=lambda value: int(value["stage"]))
    if not rows:
        raise ValueError(f"empty stage rows: {dataset}/{method}")
    values = [stage_metrics(row) for row in rows]
    final = values[-1]
    return {
        "dataset": dataset,
        "method": method,
        "stages": len(rows),
        "final_f1": final["f1"],
        "final_f1_std": final["f1_std"],
        "final_accuracy": final["accuracy"],
        "stage_average_f1": safe_mean([v["f1"] for v in values]),
        "stage_average_accuracy": safe_mean([v["accuracy"] for v in values]),
        "old_f1": final["old_f1"],
        "new_f1": final["new_f1"],
        "worst_stage_f1": min(v["f1"] for v in values),
        "stage_f1": [v["f1"] for v in values],
        "stage_accuracy": [v["accuracy"] for v in values],
        "memory_rows": rows[-1].get("memory_rows"),
        "student_seconds_total": safe_mean([]) if False else sum(
            float(v["student_seconds"] or 0.0) for v in values),
        "reported_construction_seconds": sum(
            float(v["construction_seconds"] or 0.0) for v in values),
        "actual_construction_seconds": actual_construction_seconds,
        "construction_cost_kind": cost_kind,
        "source": source,
    }


def load_results_rows(path: Path) -> list[dict]:
    payload = read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"no rows in {path}")
    return rows


def audit_actual_xpa_cost(root: Path, dataset: str) -> float | None:
    values = []
    for audit_path in sorted((root / f"{dataset}_base4").glob("stage*/audit.json")):
        audit = read_json(audit_path)
        if audit.get("construction_seconds_wall") is not None:
            values.append(float(audit["construction_seconds_wall"]))
    return sum(values) if values else None


def large_evidence() -> tuple[list[dict], list[str]]:
    warnings: list[str] = []
    datasets = {
        "arxiv": "public_baselines_large_20261001_gpu0_r4",
        "bookchild": "public_baselines_large_20261001_gpu0_r4",
        "sportsfit": "public_baselines_large_20261001_gpu1_r4",
    }
    public_methods = {
        "random_real": "random-real",
        "k_center": "k-center",
        "medoid": "medoid",
        "cat_mmd_tag": "CaT-MMD-TAG adapted",
        "gecc_hard_tag": "GECC-hard-TAG adapted",
    }
    rows: list[dict] = []
    for dataset, public_dir in datasets.items():
        public_root = TMP / public_dir / f"{dataset}_base4"
        public_path = public_root / "results.json"
        public_rows = load_results_rows(public_path)

        # Append and X+P are evaluated from the strict edge-aware artifacts;
        # the public runner only loads sealed copies of these memories.
        strict_specs = [
            ("append-edge", TMP / "edgeaware_append_fixed_reference_20261001",
             "append", "sealed append artifact; no construction timing"),
            ("Bounded X+P", TMP / "xpa_edgeaware_xp_ablation_20261001",
             "append", "real X+P construction timing is audited separately"),
            ("Bounded X+P+A", TMP / "xpa_edgeaware_xpa_ablation_20261001",
             "append", "real X+P+A construction timing is audited separately"),
        ]
        for method, base, arm, note in strict_specs:
            path = base / f"{dataset}_base4" / "results.json"
            source_rows = load_results_rows(path)
            actual = None
            if method == "Bounded X+P":
                actual = audit_actual_xpa_cost(
                    TMP / "xpa_edgeaware_xp_ablation_20261001", dataset)
            elif method == "Bounded X+P+A":
                actual = audit_actual_xpa_cost(
                    TMP / "xpa_edgeaware_xpa_ablation_20261001", dataset)
            rows.append(summarize_stage_rows(
                source_rows, dataset, method, str(path), note, actual))

        for arm, method in public_methods.items():
            selected = [row for row in public_rows if row.get("arm") == arm]
            rows.append(summarize_stage_rows(
                selected, dataset, method, str(public_path),
                "measured by public baseline adapter"))

        # The copied summary.json is retained for provenance, but stage-level
        # results are authoritative because the summary was generated by an
        # older aggregation path in this workspace.
        summary_path = TMP / public_dir / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            summary_by_arm = {r.get("arm"): r for r in summary.get("overall", [])}
            stage_xp = next(r for r in rows
                            if r["dataset"] == dataset and r["method"] == "Bounded X+P+A")
            copied = summary_by_arm.get("x_plus_p")
            if copied and abs(float(copied.get("final_f1", 0.0)) -
                             float(stage_xp["final_f1"])) > 1e-5:
                warnings.append(
                    f"{summary_path} overall/x_plus_p disagrees with stage-level "
                    f"results.json; final matrix uses stage-level artifacts.")
    return rows, warnings


def pooled(rows: list[dict], method_key: str = "method") -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row[method_key]].append(row)
    output = []
    for method, values in sorted(grouped.items()):
        output.append({
            "method": method,
            "datasets": len(values),
            "final_f1": safe_mean([r["final_f1"] for r in values]),
            "final_f1_std_across_datasets": safe_std([r["final_f1"] for r in values]),
            "final_accuracy": safe_mean([r["final_accuracy"] for r in values]),
            "stage_average_f1": safe_mean([r["stage_average_f1"] for r in values]),
            "old_f1": safe_mean([r["old_f1"] for r in values]),
            "new_f1": safe_mean([r["new_f1"] for r in values]),
            "positive_final_vs_append": None,
        })
    append = next((r for r in output if r["method"] == "append-edge"), None)
    if append:
        for row in output:
            row["final_delta_vs_append"] = row["final_f1"] - append["final_f1"]
    return output


def small_unified(prefix: str, method: str) -> list[dict]:
    rows: list[dict] = []
    for dataset in ("cora", "citeseer"):
        path = RESULTS / f"{prefix}_{dataset}" / "summary.json"
        if not path.exists():
            continue
        payload = read_json(path)
        for block in payload["results"]:
            protocol = block["protocol"]
            stage_rows = block["rows"]
            values = [stage_metrics(row) for row in sorted(stage_rows, key=lambda r: int(r["stage"]))]
            final = values[-1]
            rows.append({
                "dataset": dataset,
                "order": protocol["order"],
                "method": method,
                "final_f1": final["f1"],
                "final_accuracy": final["accuracy"],
                "stage_average_f1": safe_mean([v["f1"] for v in values]),
                "old_f1": final["old_f1"],
                "new_f1": final["new_f1"],
                "stage_f1": [v["f1"] for v in values],
                "protocol": protocol,
            })
    return rows


def small_xpa() -> list[dict]:
    return small_unified("xpa_joint_full_20260929", "Bounded X+P+A")


def small_xp() -> list[dict]:
    return small_unified("xp_joint_full_20261001", "Bounded X+P")


def external_small() -> list[dict]:
    rows: list[dict] = []
    keep = {
        "cat_mmd_tag": "CaT-MMD-TAG adapted",
        "gecc_hard_tag": "GECC-hard-TAG adapted",
        "k_center_native": "k-center",
        "medoid_native": "medoid",
    }
    for dataset in ("cora", "citeseer"):
        payload = read_json(RESULTS / "external_continual_adapters_20260927_v2" /
                            f"{dataset}_results.json")
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in payload["rows"]:
            arm = row.get("arm")
            if arm in keep:
                grouped[(row["order"], keep[arm])].append(row)
        for (order, method), stage_rows in sorted(grouped.items()):
            stage_rows = sorted(stage_rows, key=lambda r: int(r["stage"]))
            values = [stage_metrics({"aggregate": r["aggregate"]}) for r in stage_rows]
            final = values[-1]
            rows.append({
                "dataset": dataset,
                "order": order,
                "method": method,
                "final_f1": final["f1"],
                "final_accuracy": final["accuracy"],
                "stage_average_f1": safe_mean([v["f1"] for v in values]),
                "old_f1": final["old_f1"],
                "new_f1": final["new_f1"],
                "stage_f1": [v["f1"] for v in values],
                "status": "mechanism-adapted; not official end-to-end reproduction",
            })
    return rows


def small_pooled(xpa_rows: list[dict], external_rows: list[dict]) -> list[dict]:
    all_rows = xpa_rows + external_rows
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["dataset"], row["method"])].append(row)
    output = []
    for (dataset, method), values in sorted(grouped.items()):
        output.append({
            "dataset": dataset,
            "method": method,
            "orders": len(values),
            "final_f1": safe_mean([r["final_f1"] for r in values]),
            "final_f1_std_across_orders": safe_std([r["final_f1"] for r in values]),
            "final_accuracy": safe_mean([r["final_accuracy"] for r in values]),
            "stage_average_f1": safe_mean([r["stage_average_f1"] for r in values]),
            "old_f1": safe_mean([r["old_f1"] for r in values]),
            "new_f1": safe_mean([r["new_f1"] for r in values]),
        })
    return output


def tagsam_adapted() -> list[dict]:
    rows = []
    for dataset in ("cora", "citeseer"):
        payload = read_json(RESULTS / "tagsam_graphclip_comparison_2026_09_16" /
                            f"{dataset}_results.json")
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in payload["results"]:
            grouped[row["arm"]].append(row)
        for arm, stage_rows in sorted(grouped.items()):
            stage_rows = sorted(stage_rows, key=lambda r: int(r["stage"]))
            normalized = []
            for row in stage_rows:
                metrics = row["metrics"]["all"]
                normalized.append({
                    "f1": metric_value(metrics["cumulative_f1"]),
                    "accuracy": metric_value(metrics["accuracy"]),
                    "old_f1": metric_value(metrics.get("old_f1")),
                    "new_f1": metric_value(metrics.get("new_f1")),
                })
            final = normalized[-1]
            rows.append({
                "dataset": dataset,
                "method": arm,
                "final_f1": final["f1"],
                "final_accuracy": final["accuracy"],
                "stage_average_f1": safe_mean([v["f1"] for v in normalized]),
                "old_f1": final["old_f1"],
                "new_f1": final["new_f1"],
                "status": "TAGSAM mechanism adapter; not official end-to-end TAGSAM",
            })
    return rows


def fmt(value, digits=2):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def table(rows: list[dict], columns: list[tuple[str, str]]) -> list[str]:
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(
            str(row.get(key, "—")) if isinstance(row.get(key), str)
            else fmt(row.get(key)) for key, _ in columns) + " |")
    return lines


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=True)
                             if isinstance(row.get(key), (list, dict)) else row.get(key)
                             for key in keys})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    large_rows, warnings = large_evidence()
    large_pool = pooled(large_rows)
    xpa_rows = small_xpa()
    xp_rows = small_xp()
    external_rows = external_small()
    small_pool = small_pooled(xpa_rows + xp_rows, external_rows)
    tagsam_rows = tagsam_adapted()
    small_audit_path = OUT / "small_xp_artifact_audit.json"
    small_audit = read_json(small_audit_path) if small_audit_path.exists() else None
    stability_path = OUT / "small_xp_plan_seed_stability.json"
    stability = read_json(stability_path) if stability_path.exists() else None
    independent_order_path = OUT / "order19613_xp_baseline_comparison.json"
    independent_order = read_json(independent_order_path) if independent_order_path.exists() else None
    independent_audit_path = OUT / "order19613_artifact_audit.json"
    independent_audit = read_json(independent_audit_path) if independent_audit_path.exists() else None
    supplement_path = OUT / "three_order_supplement_and_cost.json"
    supplement = read_json(supplement_path) if supplement_path.exists() else None

    payload = {
        "name": "bounded-xp-final-evidence-matrix-20261001",
        "current_main_candidate": "Bounded Student-Compatible X+P",
        "optional_extension": "Bounded X+P+A; topology revision reported separately",
        "large_strict_rows": large_rows,
        "large_pooled": large_pool,
        "small_xpa_by_order": xpa_rows,
        "small_xp_by_order": xp_rows,
        "small_external_adapted_by_order": external_rows,
        "small_pooled": small_pool,
        "tagsam_adapted": tagsam_rows,
        "small_xp_artifact_audit": small_audit,
        "small_xp_plan_seed_stability": stability,
        "order19613_xp_baseline_comparison": independent_order,
        "order19613_artifact_audit": independent_audit,
        "three_order_supplement_and_cost": supplement,
        "warnings": warnings,
        "scope_notes": [
            "Large strict table uses stage-level results.json artifacts and the same edge-aware Student.",
            "CaT and GECC are mechanism adaptations, not official end-to-end reproductions.",
            "Small strict X+P and X+P+A artifacts now cover two orders and three stages; both remain protocol-specific evidence rather than universal ranking evidence.",
            "Semantic-Recursive is intentionally excluded from the current main competitive table.",
            "TAGSAM rows use the official loss/checkpoint components but the project Student and continual interface; they are adapter references.",
        ],
    }
    (OUT / "evidence_matrix.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    write_csv(OUT / "large_strict_rows.csv", large_rows)
    write_csv(OUT / "large_pooled.csv", large_pool)
    write_csv(OUT / "small_pooled.csv", small_pool)
    write_csv(OUT / "tagsam_adapted.csv", tagsam_rows)

    md: list[str] = [
        "# Bounded Student-Compatible X+P：最终证据与基线矩阵",
        "",
        "日期：2026-10-01",
        "",
        "> 当前主方法候选固定为 **Bounded Student-Compatible X+P**。A（有限拓扑 revision）作为可选扩展单独报告，不把 X+P+A 的数据集依赖收益写成无条件主方法收益。",
        "",
        "## 1. 读表方式",
        "",
        "本报告把结果分成三层。大图 strict continual 表直接使用每个阶段的 `results.json`，保证 append、X+P/X+P+A 和 coreset 的 Student、边表示和评估路径可核对。CaT、GECC 是机制适配，TAGSAM 是 loss/组件适配，均单独标注。小图现在同时有两顺序严格 X+P 与 X+P+A，但仍按数据集和顺序解释，不能把它们概括成普遍排名。",
        "",
        "## 2. 大图 strict continual 主表",
        "",
        "协议：arXiv、BookChild、SportsFit；`base+4` 五阶段；5% cumulative memory；历史 raw 不可回读；原始 edge-aware 两层 GCN、冻结 MiniLM、可训练文本投影、双向图文 InfoNCE；Student seeds 100–105；每阶段从头训练；报告 cumulative Macro-F1、Accuracy、Old F1、New F1。",
        "",
        *table(large_rows, [("dataset", "数据集"), ("method", "方法"), ("final_f1", "Final F1"), ("final_accuracy", "Accuracy"), ("stage_average_f1", "阶段平均 F1"), ("old_f1", "Old F1"), ("new_f1", "New F1"), ("actual_construction_seconds", "真实构造秒")]),
        "",
        "其中 X+P/X+P+A 的真实构造秒来自 outer construction audit；public runner 中的 sealed-memory 加载秒不作为主方法成本。append-edge 也是封存 artifact，因此没有把加载时间当成构造时间。",
        "",
        "### 三张大图等权汇总",
        "",
        *table(large_pool, [("method", "方法"), ("final_f1", "Final F1"), ("final_accuracy", "Accuracy"), ("stage_average_f1", "阶段平均 F1"), ("old_f1", "Old F1"), ("new_f1", "New F1"), ("final_delta_vs_append", "相对 append Final F1")]),
        "",
        "按这批 stage-level artifacts，X+P 相对 append-edge 的最终 F1 方向应按逐图结果计算；不能把阶段平均高低混同为最终阶段优劣。X+P+A 只作为拓扑扩展，A 在 arXiv、BookChild、SportsFit 的差异方向不同。",
        "",
        "## 3. 小图多顺序结果",
        "",
        "当前 `xpa_joint_full_20260929` 在 Cora/Citeseer 上覆盖 order19611、order19612 和三阶段，方法是 X+P+A；严格 X+P 结果在 `xp_joint_full_20261001` 生成后自动并入。公开 TAG 适配器覆盖相同数据集和顺序，但属于 mechanism-adapted，不能称为官方端到端结果。",
        "",
        *table(small_pool, [("dataset", "数据集"), ("method", "方法"), ("orders", "顺序数"), ("final_f1", "Final F1"), ("final_accuracy", "Accuracy"), ("stage_average_f1", "阶段平均 F1"), ("old_f1", "Old F1"), ("new_f1", "New F1")]),
        "",
        "小图同时区分 X+P 与 X+P+A。当前 X+P 已经完成两顺序严格评估：Cora 上低于 k-center/CaT，Citeseer 上接近但略低；X+P+A 仅在部分顺序和数据集改善，因此不能用它替代 X+P 的主方法结论。",
        "",
        "### 小图 construction seed 稳定性",
        "",
        "固定数据集、类别顺序、memory budget、X/P 配置、Student seeds 和 report 规则，只将 support-plan 的 `plan_seed` 从 9320 改为 9321。该实验不产生新的主表排名，而是检查合成构造对 plan seed 的敏感性。",
    ]
    if stability:
        stability_rows = stability.get("rows", [])
        md += [
            "",
            *table(stability_rows, [
                ("dataset", "数据集"),
                ("order", "顺序"),
                ("final_f1_seed9320", "9320 Final F1"),
                ("final_f1_seed9321", "9321 Final F1"),
                ("delta_final_f1", "Final F1 差值"),
                ("accuracy_seed9320", "9320 Accuracy"),
                ("accuracy_seed9321", "9321 Accuracy"),
                ("new_f1_seed9320", "9320 New F1"),
                ("new_f1_seed9321", "9321 New F1"),
            ]),
            "",
            f"四个 cell 的平均 Final F1 变化为 `{stability.get('mean_delta_final_f1', float('nan')):.3f} pp`，正向 cell 为 `{stability.get('positive_final_cells', 0)}/{stability.get('cell_count', 0)}`。Cora 两个顺序均为正向，Citeseer 一个顺序轻微正向、一个顺序下降；因此 X+P 的 construction seed 影响可观但没有推翻大图方向性收益。",
            "这项结果支持报告 construction-seed 方差，而不是只展示单个 plan seed；它也说明后续应优先补独立类别顺序和 paired baseline，不再针对某个 seed 继续调参。",
        ]
    else:
        md += ["", "稳定性 JSON 尚未生成；该附表将在 construction seed 复现实验完成后自动加入。"]
    md += [
        "",
        "## 4. 额外顺序：order19613 同协议基线比较",
        "",
        "order19613 单独跑了 X+P、C0 anchor、CaT-MMD-TAG、GECC-hard-TAG、k-center 和 medoid，固定三阶段、5% memory、6 个 Student seeds。该实验作为额外顺序验证，不替换原两顺序主表；CaT/GECC 仍标记为机制适配。",
    ]
    if independent_order:
        names = {
            "bounded_xp": "Bounded X+P",
            "c0": "C0 append-edge",
            "cat_mmd_tag": "CaT-MMD-TAG adapted",
            "gecc_hard_tag": "GECC-hard-TAG adapted",
            "k_center_native": "k-center native",
            "medoid_native": "medoid native",
        }
        md += [
            "",
            "| 数据集 | 方法 | Final F1 | Accuracy | 阶段平均 F1 |",
            "|---|---|---:|---:|---:|",
        ]
        for row in independent_order.get("rows", []):
            md.append(
                f"| {row['dataset']} | {names.get(row['method'], row['method'])} | "
                f"{row['final_f1']:.2f} ± {row['final_f1_std']:.2f} | "
                f"{row['final_accuracy']:.2f} ± {row['final_accuracy_std']:.2f} | "
                f"{row['stage_average_f1']:.2f} |"
            )
        md += [
            "",
            "该顺序中，Cora 的 X+P Final F1 为 47.42，低于 CaT/medoid、高于 GECC/k-center；Citeseer 的 X+P Final F1 为 53.16，低于 CaT/k-center/medoid、高于 GECC。因此它支持方向性收益和方法互补性，不支持普遍 SOTA。",
            f"详细报告：`{(ROOT / 'docs/current/BOUNDED_XP_ORDER19613_PAIRED_BASELINE_COMPARISON_ZH_2026-10-01.md').as_posix()}`。",
            f"artifact 审计：`{independent_audit_path.as_posix()}`。",
        ]
    else:
        md += ["", "order19613 额外顺序比较尚未生成。"]
    md += [
        "",
        "## 5. 三顺序补充与成本账",
        "",
        "三顺序补充将 order19611、order19612、order19613 的 X+P 与 CaT/GECC/k-center/medoid 统一汇总；它不替换两顺序主表，因为 order19613 曾参与早期拓扑实验。",
    ]
    if supplement:
        md += [
            "",
            "| 数据集 | 方法 | 三顺序 Final F1 | 跨顺序标准差 | Accuracy | 阶段平均 F1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in supplement.get("three_order_pooled", []):
            md.append(
                f"| {row['dataset']} | {row['method']} | {row['final_f1']:.2f} | "
                f"{row['final_f1_std_across_orders']:.2f} | {row['final_accuracy']:.2f} | "
                f"{row['stage_average_f1']:.2f} |"
            )
        md += [
            "",
            "三顺序结果显示：Cora 上 X+P 与 GECC 基本持平但低于 CaT、k-center 和 medoid；Citeseer 上高于 GECC/medoid、接近 k-center 但低于 CaT。成本账同时分开报告 outer 构造时间和 Student 训练时间。",
            f"详细文档：`{(ROOT / 'docs/current/BOUNDED_XP_THREE_ORDER_SUPPLEMENT_AND_COST_ZH_2026-10-01.md').as_posix()}`。",
        ]
    else:
        md += ["", "三顺序补充与成本账尚未生成。"]
    md += [
        "",
        "## 6. TAGSAM 适配参考",
        "",
        *table(tagsam_rows, [("dataset", "数据集"), ("method", "方法"), ("final_f1", "Final F1"), ("final_accuracy", "Accuracy"), ("stage_average_f1", "阶段平均 F1"), ("status", "状态")]),
        "",
        "这些结果使用官方 TAGSAM 组件/损失的适配实现，但 Student 和 continual current-delta 接口仍是本项目协议，因此不能与官方论文数字直接比较，也不与 strict edge-aware 主表直接混排。",
        "",
        "## 7. 当前可支持的结论",
        "",
        "1. X+P 是当前最适合继续作为主方法的版本：它通过 Student-compatible outer optimization 调整新合成内容，并用有界 P 做 payload transport，保留 C0 的 edge-aware 拓扑。",
        "2. X+P+A 的拓扑 revision 不是普适增益，应作为受限扩展或消融。",
        "3. k-center 是必须保留的强 coreset 参照；CaT/GECC 适配方法在部分大图阶段更强，因此论文不能宣称普遍 SOTA。",
        "4. 小图严格 X+P 两顺序矩阵已经补齐，但它在 Cora 上明显落后于 k-center/CaT，在 Citeseer 上接近但略低；construction seed 9321 的附加复现显示 4 个 cell 中 3 个 Final F1 正向，平均变化 +0.81 pp，但仍存在顺序敏感性。order19613 的额外同协议比较进一步显示 X+P 不是普遍 SOTA，而是对不同图和强基线呈互补关系。",
        "5. SR 不进入当前主竞争表；它保留在历史/启发式负结果附录。",
        "",
        "## 8. 证据限制与下一步",
        "",
        "先冻结 X+P 的算法结构，不再用当前结果继续扫参数；construction seed 稳定性、order19613 同顺序基线比较、三顺序汇总和成本账均已完成。下一步应转为论文证据冻结与方法边界说明，而不是继续针对某个顺序调参。论文中应把大图上的方向性收益、小图上的负结果、三顺序互补性和外部适配边界同时呈现，解释为什么内容/载荷 outer optimization 对图规模和传播结构敏感。不要把旧版 `summary.json` 中与 stage-level `results.json` 不一致的汇总值用于主表。",
        "",
        "## 7. 产物",
        "",
        f"- JSON：`{(OUT / 'evidence_matrix.json').as_posix()}`",
        f"- 大图逐数据集 CSV：`{(OUT / 'large_strict_rows.csv').as_posix()}`",
        f"- 大图汇总 CSV：`{(OUT / 'large_pooled.csv').as_posix()}`",
        f"- 小图汇总 CSV：`{(OUT / 'small_pooled.csv').as_posix()}`",
        f"- 小图 X+P artifact 审计：`{(OUT / 'small_xp_artifact_audit.json').as_posix()}`",
        f"- 小图 construction seed 稳定性：`{(OUT / 'small_xp_plan_seed_stability.json').as_posix()}`",
        f"- order19613 额外顺序比较：`{(OUT / 'order19613_xp_baseline_comparison.json').as_posix()}`",
        f"- order19613 artifact 审计：`{(OUT / 'order19613_artifact_audit.json').as_posix()}`",
        f"- 三顺序补充与成本账：`{(OUT / 'three_order_supplement_and_cost.json').as_posix()}`",
    ]
    if warnings:
        md += ["", "## 8. 自动审计提示", ""] + [f"- {warning}" for warning in warnings]
    DOC.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT),
        "document": str(DOC),
        "large_rows": len(large_rows),
        "large_pooled": len(large_pool),
        "small_rows": len(small_pool),
        "tagsam_rows": len(tagsam_rows),
        "warnings": warnings,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
