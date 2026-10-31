"""Build the sealed order19614 comparison report for Bounded X+P."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bounded_xp_final_evidence_20261001"
RAW = OUT / "order19614_raw"
if not RAW.exists():
    RAW = ROOT / "tmp_remote_results" / "order19614_20261001"
DOC = ROOT / "docs" / "current" / "BOUNDED_XP_ORDER19614_BLIND_ORDER_COMPARISON_ZH_2026-10-01.md"

DATASETS = ("cora", "citeseer")
NAMES = {
    "bounded_xp": "Bounded X+P",
    "c0": "C0 append-edge anchor",
    "cat_mmd_tag": "CaT-MMD-TAG adapted",
    "gecc_hard_tag": "GECC-hard-TAG adapted",
    "k_center_fps": "k-center/FPS",
    "k_center_native": "k-center native append",
    "medoid": "medoid",
    "medoid_native": "medoid native append",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    return sorted(read(path)["rows"], key=lambda row: int(row["stage"]))


def aggregate(row: dict) -> dict:
    value = row["aggregate"]
    return {
        "final_f1": float(value["mean_f1"]),
        "final_f1_std": float(value.get("std_f1", 0.0)),
        "final_accuracy": float(value.get("mean_accuracy", 0.0)),
        "final_accuracy_std": float(value.get("std_accuracy", 0.0)),
        "old_f1": value.get("old_f1"),
        "new_f1": value.get("new_f1"),
    }


def metrics(dataset: str, method: str, trajectory: list[dict], source: str) -> dict:
    final = trajectory[-1]
    values = [aggregate(item) for item in trajectory]
    final_value = values[-1]
    construction = [item.get("construction_seconds") for item in trajectory]
    if not any(v is not None for v in construction):
        construction = [
            item.get("construction_audit", {}).get("construction_seconds_wall")
            for item in trajectory
        ]
    if not any(v is not None for v in construction):
        construction = [
            item.get("construction_seconds")
            for item in trajectory
        ]
    construction = [float(v) for v in construction if v is not None]
    student = [float(item.get("student_wall_seconds", 0.0)) for item in trajectory]
    return {
        "dataset": dataset,
        "method": method,
        "method_label": NAMES.get(method, method),
        "source": source,
        **final_value,
        "stage_average_f1": sum(v["final_f1"] for v in values) / len(values),
        "stage_average_accuracy": sum(v["final_accuracy"] for v in values) / len(values),
        "worst_stage_f1": min(v["final_f1"] for v in values),
        "stage_f1": [v["final_f1"] for v in values],
        "stage_accuracy": [v["final_accuracy"] for v in values],
        "memory_rows": int(final["memory_rows"]),
        "construction_seconds": sum(construction) if construction else None,
        "student_wall_seconds": sum(student),
    }


def load_dataset(dataset: str) -> list[dict]:
    out: list[dict] = []
    xp_path = RAW / f"xp_{dataset}_results.json"
    xp = rows(xp_path)
    out.append(metrics(dataset, "bounded_xp", xp, str(xp_path)))

    c0_payload = read(RAW / "c0_coreset_results.json")
    c0_rows = [row for row in c0_payload["rows"] if row["dataset"] == dataset]
    for arm, label in (("c0", "c0"), ("k_center_fps", "k_center_fps"), ("medoid", "medoid")):
        trajectory = [row for row in c0_rows if row["arm"] == label]
        if len(trajectory) != 3:
            raise ValueError(f"{dataset}/{arm}: expected 3 C0/coreset stages")
        out.append(metrics(dataset, arm, trajectory, str(RAW / "c0_coreset_results.json")))

    external = rows(RAW / f"external_{dataset}_results.json")
    for arm in ("cat_mmd_tag", "gecc_hard_tag", "k_center_native", "medoid_native"):
        trajectory = [row for row in external if row["arm"] == arm]
        if len(trajectory) != 3:
            raise ValueError(f"{dataset}/{arm}: expected 3 external stages")
        out.append(metrics(dataset, arm, trajectory, str(RAW / f"external_{dataset}_results.json")))
    return out


def main() -> None:
    all_rows = [row for dataset in DATASETS for row in load_dataset(dataset)]
    comparisons = []
    for dataset in DATASETS:
        group = {row["method"]: row for row in all_rows if row["dataset"] == dataset}
        xp = group["bounded_xp"]
        for method, row in group.items():
            if method == "bounded_xp":
                continue
            comparisons.append({
                "dataset": dataset,
                "method": method,
                "xp_minus_final_f1": xp["final_f1"] - row["final_f1"],
                "xp_minus_accuracy": xp["final_accuracy"] - row["final_accuracy"],
                "xp_minus_stage_average_f1": xp["stage_average_f1"] - row["stage_average_f1"],
            })

    manifest = {
        "cora": read(RAW / "results_order19614_c0_source_20261001d_cora_order19614_manifest.json"),
        "citeseer": read(RAW / "results_order19614_c0_source_20261001d_citeseer_order19614_manifest.json"),
    }
    audit = {
        "source_order": "order19614",
        "source_independent_from_development_orders": True,
        "historical_raw_reads": 0,
        "report_test_labels_used_for_construction": False,
        "cora": read(RAW / "results_order19614_c0_source_20261001d_cora_order19614_audit.json"),
        "citeseer": read(RAW / "results_order19614_c0_source_20261001d_citeseer_order19614_audit.json"),
        "x_plus_p": {
            "cora": [read(RAW / "xp_cora_stage1_audit.json"),
                     read(RAW / "xp_cora_stage2_audit.json")],
            "citeseer": [read(RAW / "xp_citeseer_stage1_audit.json"),
                         read(RAW / "xp_citeseer_stage2_audit.json")],
        },
        "external_adapters": {
            "cora": read(RAW / "external_cora_audit.json"),
            "citeseer": read(RAW / "external_citeseer_audit.json"),
        },
    }
    payload = {
        "protocol": {
            "name": "order19614-bounded-xp-baseline-comparison-20261001",
            "order": "order19614",
            "datasets": list(DATASETS),
            "stages": 3,
            "memory": "same 5% cumulative legal-pool budget per dataset",
            "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
            "report_seeds": [150, 151, 152, 153, 154, 155],
            "x_p": "plan_seed=9320; inner_steps=8; outer_steps=4; outer_lr=0.02; C0 edge-aware topology",
            "access": "current delta only; historical raw graph not reread",
            "baseline_scope": "CaT/GECC are mechanism adapters; native k-center/medoid are real-node controls; C0 is an internal anchor",
            "semantic_recursive": "excluded from this table",
        },
        "source_manifest": manifest,
        "source_audit": audit,
        "rows": all_rows,
        "paired_comparisons": comparisons,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "order19614_xp_baseline_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Bounded X+P：order19614 独立顺序基线比较",
        "",
        "> `order19614` 是本轮新增的独立类别顺序。所有方法使用同一三阶段、5% cumulative memory、原始 Student、6 个 report seeds 和同一 report-dev 规则。",
        "",
        "## 1. 结果总表",
        "",
        "| 数据集 | 方法 | Final Macro-F1 | Final Accuracy | 阶段平均 F1 | 阶段平均 Accuracy | Old F1 | New F1 | 最差阶段 F1 | rows |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        def f(value):
            return "N/A" if value is None else f"{float(value):.2f}"
        lines.append(
            f"| {row['dataset']} | {row['method_label']} | "
            f"{row['final_f1']:.2f} ± {row['final_f1_std']:.2f} | "
            f"{row['final_accuracy']:.2f} ± {row['final_accuracy_std']:.2f} | "
            f"{row['stage_average_f1']:.2f} | {row['stage_average_accuracy']:.2f} | "
            f"{f(row['old_f1'])} | {f(row['new_f1'])} | {row['worst_stage_f1']:.2f} | {row['memory_rows']} |"
        )
    lines += [
        "",
        "## 2. X+P 相对差值",
        "",
        "> 正值表示 X+P 更高；这里是同一类别顺序内的描述性配对差值，不代表跨数据集显著性检验。",
        "",
        "| 数据集 | 对比方法 | X+P - Final F1 | X+P - Accuracy | X+P - 阶段平均 F1 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['dataset']} | {NAMES.get(row['method'], row['method'])} | "
            f"{row['xp_minus_final_f1']:+.2f} | {row['xp_minus_accuracy']:+.2f} | "
            f"{row['xp_minus_stage_average_f1']:+.2f} |"
        )
    lines += [
        "",
        "## 3. 结果解释",
        "",
        "- 该顺序不是用来证明 X+P 稳定 SOTA，而是用来检查开发顺序之外的泛化。若 X+P 在该顺序落后，结果应原样保留，不能事后调参。",
        "- C0 只作为内部 append-edge anchor，用于测量 outer X/P 优化的增量；论文中应把它与公开或适配基线分开标记。",
        "- CaT/GECC 采用当前 TAG Student 和 continual delta 接口的机制适配，不是官方端到端复现；k-center/medoid 是同协议的 real-node controls。",
        "- `order19614` 的 source audit 显示 construction 没有读取历史 raw、没有使用 report/test labels，且每阶段 old prefix 和 old-old topology 保持不变。",
        "",
        "## 4. 成本与限制",
        "",
    ]
    for dataset in DATASETS:
        lines.append(f"### {dataset}")
        for row in [x for x in all_rows if x["dataset"] == dataset]:
            build = "N/A" if row["construction_seconds"] is None else f"{row['construction_seconds']:.2f}s"
            lines.append(f"- {row['method_label']}: construction {build}; Student wall time {row['student_wall_seconds']:.2f}s。")
    lines += [
        "",
        "## 5. 证据边界",
        "",
        "- 这份表是新增独立顺序的补充验证，不替换原有两顺序主表。",
        "- 该表不包含 Semantic-Recursive；当前主线评估聚焦有 Student outer optimization 的 Bounded X+P。",
        "- 不应据此宣称普遍优于所有 coreset 或 TAG 凝聚方法；正确结论是方法在不同顺序和图上具有互补性，且独立顺序暴露出 k-center 的强竞争性。",
        "",
        f"原始汇总 JSON：`{out_json.as_posix()}`",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out_json), "document": str(DOC), "rows": len(all_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
