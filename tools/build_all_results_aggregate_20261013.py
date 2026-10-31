"""Aggregate all existing TAG condensation results into one auditable bundle.

This script only reads existing result files. It does not rerun experiments or
rewrite sealed sources. Heterogeneous protocols stay explicit in the output.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/all_results_aggregate_20261013"
MASTER = ROOT / "results/baseline_master_by_dataset_20261013"
FINAL = ROOT / "results/uq_final_evidence_report_20261013_132047"
TOPO = ROOT / "results/unified_q_topology_report_20261011"
RATE = ROOT / "results/uq_rate_sensitivity_20261012"
JOINT = ROOT / "results/uq_joint_replacement_20261013"

DISPLAY = {
    "arxiv": "ArXiv",
    "sportsfit": "SportsFit",
    "bookhis": "BookHis",
    "wikics": "WikiCS",
    "cora": "Cora",
    "citeseer": "Citeseer",
    "bookchild": "BookChild",
}
DATASETS = tuple(DISPLAY)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    if value is None or str(value).strip() in {"", "--", "None", "nan"}:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return value
    return result if math.isfinite(result) else None


def clean_value(value):
    value = number(value)
    return value


def source_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


ARM_NAMES = {
    "uq_dm_edgefree": "Unified-Q-EF", "cat_cctp": "CaT-CCTP",
    "uq_dm_edgefree_no_gm": "No-GM", "uq_dm_edgefree_fixed_q": "Fixed-Q",
    "uq_random_q": "Random-Q", "split_q": "Independent Q",
    "no_distribution": "No-DM", "uq_no_dm": "No-DM",
    "no_residual": "No residual", "uq_no_residual": "No residual",
    "no_node_text": "No node text", "free_x": "Free-X parameterization",
    "no_text_init": "No text initialization", "uq_no_text_init": "No text initialization",
    "no_payload_gm": "No payload GM", "stop_text_gradient": "Stop text gradient",
    "uq_no_prop": "No propagation matching", "uq_frozen_text": "Frozen text assignment",
    "uq_detached_text": "Detached text gradient", "uq_no_q_anchor": "No Q anchor",
    "uq_no_x_anchor": "No X anchor", "uq_dm_only": "DM-only objective",
    "unchanged": "Reconstructed diagnostic reference", "text_blank": "Blank text",
    "text_cctp": "CCTP payload replay", "text_permuted": "Permuted text",
    "topology_hard": "Hard topology replay", "topology_weighted": "Weighted topology replay",
    "topology_permuted": "Permuted topology replay", "gecc_hard": "GECC-hard",
    "puma_tag": "PUMA-inspired TAG", "random_coreset": "Random real-node coreset",
    "kcenter_coreset": "K-center coreset", "adaptive_coherent": "Adaptive coherent",
    "cat_nearest1": "CaT + nearest-1 text", "cat_random5": "CaT + random-5 text",
    "uq_nearest1": "Unified-Q-EF + nearest-1 text", "uq_nearest5": "Unified-Q-EF + nearest-5 text",
    "cat_text_center_legacy": "CaT + class-shared text",
    "uq_shared_q": "Unified-Q-EF", "control_shared": "Shared Q",
    "control_independent": "Independent Q", "control_shared_equal": "Shared Q, equal parameter budget",
    "control_independent_equal": "Independent Q, equal parameter budget",
    "control_shared_no_residual": "Shared Q, no residual",
    "control_independent_no_residual": "Independent Q, no residual",
    "control_soft_stored": "Shared Q, soft text stored",
}


EXTRA_PANELS = [
    ("31_inloop", "旧 in-loop 确认", "9 个顺序/构造 cell 的 Student 均值；包含 no-GM。"),
    ("32_fixed_content", "旧 fixed-Q 与固定内容诊断", "单格 3 Student seeds；诊断重构不能视为原 pilot 的精确重放。"),
    ("33_constructor", "构造归因", "3 个顺序、1 构造种子；native 和 common_cat_payload 读出分别保留。"),
    ("34_text_path", "文本路径归因", "单个参考顺序；与 33_constructor 的三顺序聚合不同。"),
    ("35_components", "单格组件消融", "o9101/c271006，3 Student seeds；旧固定调度。"),
    ("36_text_controls", "最近邻与 random-5 文本对照", "旧单格 report-dev；CaT-CCTP 为 nearest top-5，不是 random-5。"),
    ("37_shared_q", "旧 shared/independent Q", "3 顺序、1 构造种子；两个 payload 读出分别保留。"),
    ("38_class_shared", "class-shared 与 Student 重放", "历史 class-shared 文本对照；legacy/prompt Student 分开。"),
    ("39_random_q", "learned/random Q 配对", "单格历史快照；随机 Q 不等于随机 real-node coreset。"),
]


def evidence_row(row: dict, path: Path, panel: str, *, unit: str = "Student-seed SD", cells: int = 1) -> dict:
    protocol = row.get("protocol", {})
    arm = row.get("arm", row.get("method", ""))
    seeds = row.get("student_seeds", protocol.get("student_seeds", [150, 151, 152]))
    result = {
        "panel_id": panel, "dataset": row["dataset"], "dataset_display": DISPLAY[row["dataset"]],
        "method": ARM_NAMES.get(arm, arm), "original_method": "internal control/reference",
        "role": "internal_ablation_or_control_reference", "fidelity": "historical internal experiment",
        "result_family": "internal_candidate_or_ablation", "arm": arm,
        "student": "GCN / multi-positive Prompt Student", "split": "report_dev",
        "budget": "historical 5%-target schedule; source-specific", "readout": row.get("readout", "native"),
        "n_cells": cells, "n_orders": 3 if cells > 1 else 1,
        "n_construction_seeds": 3 if cells == 9 else 1,
        "student_seeds": ",".join(map(str, seeds)), "students_per_cell": len(seeds),
        "n_student_runs": cells * len(seeds), "student_steps": row.get("student_steps", protocol.get("student_steps", 2048)),
        "student_batch": row.get("student_batch", protocol.get("student_batch", 24)),
        "text_encoder": "frozen MiniLM", "sd_unit": unit,
        "order_seed": row.get("order_seed", protocol.get("order_seed", "9101,9102,9103" if cells > 1 else 9101)),
        "construction_seed": row.get("construction_seed", protocol.get("construction_seed", 271006)),
        "source": source_rel(path), "raw_source": row.get("source", ""),
        "source_kind": "existing_attribution_summary", "official_condensation_reproduction": False,
    }
    for target, keys in {
        "final_f1": ["final_f1", "f1"], "accuracy": ["accuracy"],
        "old_f1": ["old_f1"], "new_f1": ["new_f1"],
        "stage_average_f1": ["stage_f1"], "stage_average_accuracy": ["stage_acc", "stage_accuracy"],
        "worst_final_f1": ["min_cell_f1", "worst_order_f1"],
        "worst_stage_f1": ["worst_stage", "worst_stage_f1"],
        "construction_seconds": ["mean_construction_seconds", "construction_seconds"],
        "student_seconds": ["mean_student_seconds", "student_seconds"], "memory_rows": ["memory_rows"],
    }.items():
        value = next((row[k] for k in keys if k in row), None)
        result[target] = value.get("mean") if isinstance(value, dict) else number(value)
    for target, keys, metric in [
        ("f1_sd", ["f1_cell_sd", "student_f1_sd", "final_f1_std"], row.get("f1", row.get("final_f1"))),
        ("accuracy_sd", ["acc_cell_sd", "student_acc_sd", "accuracy_std"], row.get("accuracy")),
    ]:
        result[target] = metric.get("sd") if isinstance(metric, dict) else number(next((row[k] for k in keys if k in row), None))
    if row.get("peak_mb") is not None:
        result["construction_peak_gib"] = number(row["peak_mb"]) / 1024
    return result


def attribution_rows() -> tuple[list[dict], list[Path]]:
    rows, paths = [], []

    path = ROOT / "results/unified_q_inloop_validation_20261007/summary.json"
    paths.append(path)
    obj = read_json(path)
    for row in obj["aggregates"]:
        out = evidence_row(row, path, "31_inloop", unit="SD of 9 Student-averaged order/construction cells", cells=9)
        members = [c for c in obj["cells"] if c["dataset"] == row["dataset"] and c["arm"] == row["arm"]]
        orders = sorted({c["order_seed"] for c in members})
        constructions = sorted({c["construction_seed"] for c in members})
        assert len(members) == row["n_cells"] == 9
        out.update(order_seed=",".join(map(str, orders)), construction_seed=",".join(map(str, constructions)),
                   n_orders=len(orders), n_construction_seeds=len(constructions))
        out["underlying_sources"] = " | ".join(sorted({c["source"] for c in members}))
        rows.append(out)
    for row in obj["cells"]:
        if row["arm"] in {"uq_dm_edgefree", "uq_dm_edgefree_no_gm", "cat_cctp"}:
            continue
        out = evidence_row(row, path, "32_fixed_content")
        out["readout"] = "reference-order control" if row["arm"] == "uq_dm_edgefree_fixed_q" else "reconstructed fixed-content diagnostic"
        rows.append(out)

    path = ROOT / "results/unified_q_constructor_attribution_20261008_v2/summary.json"
    paths.append(path)
    obj = read_json(path)
    for row in obj["main"]:
        rows.append(evidence_row(row, path, "33_constructor", unit="SD of 3 Student-averaged class orders", cells=3))
    for row in obj["text_factorial"]:
        rows.append(evidence_row(row, path, "34_text_path"))

    path = ROOT / "results/pre_submission_evidence_20261010/component_ablations.json"
    paths.append(path)
    rows.extend(evidence_row(row, path, "35_components") for row in read_json(path))

    path = ROOT / "results/unified_q_attribution_20261008/summary.json"
    paths.append(path)
    rows.extend(evidence_row(row, path, "36_text_controls") for row in read_json(path)["text"])

    path = ROOT / "results/shared_q_baseline_20261008/summary.json"
    paths.append(path)
    obj = read_json(path)
    rows.extend(evidence_row(row, path, "37_shared_q", unit="SD of 3 Student-averaged class orders", cells=3) for row in obj["shared_scores"])
    for row in obj["baseline_scores"]:
        if row["arm"] != "cat_text_center_legacy":
            continue
        out = evidence_row(row, path, "38_class_shared")
        out["student"] = "legacy GCN/MiniLM Student" if row["student"] == "legacy" else "GCN / multi-positive Prompt Student"
        rows.append(out)

    path = ROOT / "results/unified_q_overnight_20261012/random_q_comparison.csv"
    paths.append(path)
    for row in read_csv(path):
        raw_path = Path(row["source"])
        if raw_path.exists():
            paths.append(raw_path)
            obj = read_json(raw_path)
            requested_arm = {"Unified-Q-EF": "uq_shared_q", "CaT-CCTP": "cat_cctp",
                             "GECC-hard-TAG": "gecc_hard", "random-Q": "uq_random_q"}[row["method"]]
            members = [r for r in obj["rows"] if r["arm"] == requested_arm]
            assert members, (raw_path, requested_arm)
            last = max(members, key=lambda r: r.get("stage", -1))
            score = last.get("score", last.get("final"))
            assert score is not None, raw_path
            assert abs(score["final_f1"] - float(row["final_f1"])) < 1e-8, (raw_path, requested_arm)
            raw = {**last, **score, **row, "protocol": obj["protocol"]}
            raw["arm"] = last["arm"]
            raw["memory_rows"] = last["memory_rows"]
            costs = [r["construction_seconds"] for r in members if "construction_seconds" in r]
            raw["construction_seconds"] = sum(costs) if costs else None
        else:
            raw = row
        out = evidence_row(raw, path, "39_random_q")
        out["order_seed"] = number(row["order"])
        out["raw_source"] = row["source"]
        rows.append(out)

    for index, row in enumerate(rows, 1):
        row["record_id"] = f"A{index:03d}"
        row["adaptation"] = next(note for code, _, note in EXTRA_PANELS if code == row["panel_id"])
    return rows, paths


def normalized_master_rows() -> tuple[list[dict], list[dict]]:
    baseline = read_csv(MASTER / "baseline_results.csv")
    internal = read_csv(MASTER / "internal_candidates_and_ablations.csv")
    for row in baseline + internal:
        row["dataset_display"] = DISPLAY.get(row.get("dataset", ""), row.get("dataset", ""))
        for key in (
            "final_f1", "accuracy", "f1_sd", "accuracy_sd", "old_f1", "new_f1",
            "stage_average_f1", "actual_rate", "construction_seconds", "construction_peak_gib",
            "memory_rows", "n_cells", "n_orders", "n_construction_seeds", "n_student_runs",
        ):
            if key in row:
                row[key] = clean_value(row[key])
        row["result_family"] = "baseline_or_reference" if row in baseline else "internal_candidate_or_ablation"
        row["source_kind"] = "master_aggregate"
    return baseline, internal


def topology_rows() -> tuple[list[dict], list[dict], list[dict]]:
    node = []
    for row in read_csv(TOPO / "node_classification.csv"):
        arm = {
            "unified_q_edge_free": "Unified-Q-EF",
            "unified_q": "Unified-Q-T",
            "unified_q_boundary": "Unified-Q-T-boundary",
            "boundary": "Unified-Q-T-boundary",
        }.get(row.get("arm", ""), row.get("arm", ""))
        node.append({
            "dataset": row.get("dataset", ""),
            "dataset_display": DISPLAY.get(row.get("dataset", ""), row.get("dataset", "")),
            "order": row.get("order", ""),
            "stage": int(row["stage"]) if row.get("stage") is not None else None,
            "method": arm,
            "final_f1": number(row.get("final_f1")),
            "f1_sd": number(row.get("final_f1_std")),
            "accuracy": number(row.get("accuracy")),
            "accuracy_sd": number(row.get("accuracy_std")),
            "old_f1": number(row.get("old_f1")),
            "new_f1": number(row.get("new_f1")),
            "construction_seconds": number(row.get("construction_seconds")),
            "memory_rows": number(row.get("memory_rows")),
            "result_family": "topology_node_classification",
            "protocol": "topology branch snapshot; older small/large extension panel",
            "split": "source-specific",
            "student": "source-specific Prompt Student",
            "budget": "source-specific",
            "source": source_rel(TOPO / "node_classification.csv"),
            "source_kind": "topology_report",
        })
    large_path = ROOT / "results/unified_q_large_topology_bookhis_20261011.json"
    for row in read_json(large_path)["rows"]:
        score = row["score"]
        node.append({
            "dataset": "bookhis", "dataset_display": "BookHis", "order": "frozen large reference",
            "stage": row["stage"], "method": "Unified-Q-EF" if row["arm"] == "unified_q_edge_free" else "Unified-Q-T",
            "final_f1": score["final_f1"], "f1_sd": score["final_f1_std"],
            "accuracy": score["accuracy"], "accuracy_sd": score["accuracy_std"],
            "old_f1": score["old_f1"], "new_f1": score["new_f1"], "memory_rows": row["memory_rows"],
            "construction_seconds": row["construction_seconds"], "synthetic_undirected_edges": row["undirected_edges"],
            "result_family": "topology_node_classification", "protocol": "older five-stage frozen large reference; same Prompt Student",
            "student_seeds": ",".join(map(str, score["student_seeds"])), "sd_unit": "Student-seed SD",
            "source": source_rel(large_path), "source_kind": "raw_topology_results",
        })

    # The old link CSV omitted split_seed. Recover it from raw files, not row order.
    lp_raw = []
    for dataset in ("cora", "citeseer"):
        for suffix in ("20261011", "s29002", "s29003"):
            path = ROOT / f"results/uqt_lp_{dataset}_{suffix}.json"
            obj = read_json(path)
            for row in obj["rows"]:
                lp_raw.append({**row, **row["metrics"], **row["structure"],
                    "split_seed": obj["protocol"]["split_seed"], "source": source_rel(path),
                    "feature_only_auc": row["metrics_feature_only"]["auc"],
                    "feature_only_ap": row["metrics_feature_only"]["ap"]})
    lp = []
    for row in lp_raw:
        arm = {
            "unified_q_edge_free": "Unified-Q-EF",
            "unified_q": "Unified-Q-T",
            "unified_q_boundary": "Unified-Q-T-boundary",
            "boundary": "Unified-Q-T-boundary",
        }.get(row.get("arm", ""), row.get("arm", ""))
        lp.append({
            "dataset": row.get("dataset", ""),
            "dataset_display": DISPLAY.get(row.get("dataset", ""), row.get("dataset", "")),
            "order": row.get("order", ""),
            "split_seed": row["split_seed"],
            "stage": int(row["stage"]) if row.get("stage") is not None else None,
            "method": arm,
            "memory_rows": row["memory_rows"],
            "auc": number(row.get("auc")),
            "ap": number(row.get("ap")),
            "feature_only_auc": number(row.get("feature_only_auc")),
            "feature_only_ap": number(row.get("feature_only_ap")),
            "topology_positive_rate": number(row.get("topology_positive_rate")),
            "topology_negative_rate": number(row.get("topology_negative_rate")),
            "heldout_edge_recall": number(row.get("heldout_edge_recall")),
            "degree_pearson": number(row.get("degree_pearson")),
            "synthetic_undirected_edges": number(row.get("synthetic_undirected_edges")),
            "synthetic_directed_edge_entries": number(row.get("synthetic_directed_edge_entries")),
            "result_family": "topology_link_prediction",
            "protocol": "strict train-edges-only slot-level link probe; not trained GAE",
            "source": row["source"],
            "source_kind": "raw_topology_results",
        })
    resources = []
    for row in read_csv(TOPO / "topology_resources.csv"):
        arm = {
            "unified_q_edge_free": "Unified-Q-EF",
            "unified_q": "Unified-Q-T",
            "unified_q_boundary": "Unified-Q-T-boundary",
            "boundary": "Unified-Q-T-boundary",
        }.get(row.get("arm", ""), row.get("arm", ""))
        out = dict(row)
        out.update({
            "dataset_display": DISPLAY.get(row.get("dataset", ""), row.get("dataset", "")),
            "method": arm,
            "result_family": "topology_resources",
            "protocol": "synthetic topology resource audit",
            "source": source_rel(TOPO / "topology_resources.csv"),
            "source_kind": "topology_report",
        })
        for key in ("stage", "memory_rows", "directed_edge_entries", "undirected_edges", "max_degree"):
            if key in out:
                out[key] = number(out[key])
        for key in ("mean_degree", "construction_seconds"):
            if key in out:
                out[key] = number(out[key])
        resources.append(out)
    for row in read_json(large_path)["rows"]:
        resources.append({
            "dataset": "bookhis", "dataset_display": "BookHis", "order": "frozen large reference",
            "stage": row["stage"], "method": "Unified-Q-EF" if row["arm"] == "unified_q_edge_free" else "Unified-Q-T",
            "memory_rows": row["memory_rows"], "directed_edge_entries": row["directed_edge_entries"],
            "undirected_edges": row["undirected_edges"], "construction_seconds": row["construction_seconds"],
            "result_family": "topology_resources", "protocol": "older frozen large reference",
            "source": source_rel(large_path), "source_kind": "raw_topology_results",
        })
    return node, lp, resources


def diagnostics_rows() -> list[dict]:
    rows = read_csv(FINAL / "diagnostics.csv")
    for row in rows:
        row["dataset_display"] = DISPLAY.get(row.get("dataset", ""), row.get("dataset", ""))
        row["result_family"] = "q_and_construction_diagnostics"
        row["raw_source"] = row.get("source", "")
        row["source"] = source_rel(FINAL / "diagnostics.csv")
        row["source_kind"] = "final_evidence_diagnostics"
        for key, value in list(row.items()):
            if key not in {"category", "dataset", "dataset_display", "order_seed", "construction_seed", "arm", "stage", "source", "raw_source", "source_sha256", "assignment_parameters", "residual_parameters", "equal_parameter_budget", "parameterization", "result_family", "source_kind"}:
                row[key] = number(value)
    return rows


def rate_rows() -> list[dict]:
    rows = read_csv(RATE / "rate_sensitivity_summary.csv")
    for row in rows:
        dataset = row.get("dataset", "")
        canonical = {v: k for k, v in DISPLAY.items()}.get(dataset, dataset.lower())
        row["dataset"] = canonical
        row["dataset_display"] = DISPLAY.get(canonical, dataset)
        row["rate"] = number(row.get("rate"))
        for key in ("actual_rate", "final_memory", "final_f1", "f1_std_student", "accuracy", "accuracy_std_student"):
            row[key] = number(row.get(key))
        row["result_family"] = "rate_sensitivity"
        row["protocol"] = "fixed-rate report-development panel; no method selection"
        row["source"] = source_rel(RATE / "rate_sensitivity_summary.csv")
        row["source_kind"] = "rate_sensitivity_report"
    return rows


def joint_rows() -> list[dict]:
    text = (JOINT / "summary.md").read_text(encoding="utf-8")
    rows: list[dict] = []
    dev_pattern = re.compile(
        r"\| (arxiv|sportsfit|bookhis) \| ([0-9.]+) \| ([0-9.]+) \| ([+-]?[0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([+-]?[0-9.]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|"
    )
    dev_section = text.split("| Dataset | Dev joint F1", 1)[-1]
    dev_section = dev_section.split("Sealed-test readings below", 1)[0]
    for match in dev_pattern.finditer(dev_section):
        dataset, joint_f1, ef_f1, delta_f1, joint_acc, ef_acc, delta_acc, joint_old_new, ef_old_new, construct, peak = match.groups()
        rows.append({
            "dataset": dataset,
            "dataset_display": DISPLAY[dataset],
            "split": "report_dev",
            "method": "Joint-DM replacement",
            "reference_method": "Unified-Q-EF",
            "final_f1": number(joint_f1),
            "reference_f1": number(ef_f1),
            "delta_f1": number(delta_f1),
            "accuracy": number(joint_acc),
            "reference_accuracy": number(ef_acc),
            "delta_accuracy": number(delta_acc),
            "old_new": joint_old_new.strip(),
            "reference_old_new": ef_old_new.strip(),
            "construction_seconds_pair": construct.strip(),
            "peak_gib_pair": peak.strip(),
            "result_family": "joint_distribution_candidate",
            "protocol": "post-hoc report-development candidate; one matched cell; not promoted",
            "source": source_rel(JOINT / "summary.md"),
            "source_kind": "joint_distribution_audit",
        })
    sealed_pattern = re.compile(r"\| (arxiv|sportsfit|bookhis) \| ([0-9.]+) \| ([0-9.]+) \| ([+-]?[0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|")
    # The sealed table is distinguished by its section in the source report.
    sealed_section = text.split("| Dataset | Sealed joint F1", 1)[-1]
    sealed_section = sealed_section.split("All stages passed", 1)[0]
    for match in sealed_pattern.finditer(sealed_section):
        dataset, joint_f1, ef_f1, delta_f1, joint_acc, ef_acc = match.groups()
        rows.append({
            "dataset": dataset,
            "dataset_display": DISPLAY[dataset],
            "split": "sealed_test",
            "method": "Joint-DM replacement",
            "reference_method": "Unified-Q-EF",
            "final_f1": number(joint_f1),
            "reference_f1": number(ef_f1),
            "delta_f1": number(delta_f1),
            "accuracy": number(joint_acc),
            "reference_accuracy": number(ef_acc),
            "delta_accuracy": None,
            "result_family": "joint_distribution_candidate",
            "protocol": "post-hoc transparency only; sealed readings not used for selection",
            "source": source_rel(JOINT / "summary.md"),
            "source_kind": "joint_distribution_audit",
        })
    return rows


def method_map(baseline: list[dict], internal: list[dict]) -> list[dict]:
    rows = []
    path = MASTER / "method_adaptations.csv"
    if path.exists():
        rows.extend(read_csv(path))
        for row in rows:
            row["source"] = source_rel(path)
    for method in sorted({r["method"] for r in internal}):
        members = [r for r in internal if r["method"] == method]
        rows.append({
            "method": method,
            "original_method": method,
            "role": "internal_ablation_or_candidate",
            "fidelity": "internal control",
            "adaptation": " / ".join(sorted({r.get("adaptation", "internal experiment") for r in members})),
            "datasets_with_numbers": ",".join(sorted({r["dataset"] for r in members})),
            "official_condensation_reproduction": "False",
            "source": " | ".join(sorted({r["source"] for r in members})),
        })
    return rows


def paired_rows() -> list[dict]:
    path = FINAL / "pairs.csv"
    rows = read_csv(path)
    for row in rows:
        row["dataset_display"] = DISPLAY[row["dataset"]]
        row["left_method"] = ARM_NAMES.get(row["left"], row["left"])
        row["right_method"] = ARM_NAMES.get(row["right"], row["right"])
        row["source"] = source_rel(path)
        row["result_family"] = "paired_statistics"
        for key in ("cells", "mean_delta", "ci_low", "ci_high", "positive_cells", "positive_ratio",
                    "worst_delta", "worst_order", "worst_construction"):
            row[key] = number(row[key])
    return rows


def final_stage_rows(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    return [max(members, key=lambda r: r["stage"]) for members in groups.values()]


def diagnostic_summary(rows: list[dict]) -> list[dict]:
    finals = final_stage_rows(rows, ("category", "dataset", "arm", "order_seed", "construction_seed"))
    groups = defaultdict(list)
    for row in finals:
        groups[(row["category"], row["dataset"], row["arm"])].append(row)
    summaries = []
    for (category, dataset, arm), members in sorted(groups.items()):
        out = {"category": category, "dataset": dataset, "dataset_display": DISPLAY[dataset],
               "arm": arm, "method": ARM_NAMES.get(arm, arm), "cells": len(members),
               "scope": "unweighted cell mean of final-stage diagnostics; not cumulative cost or total parameter count"}
        for key in ("assignment_parameters", "residual_parameters", "q_gradient_norm", "residual_gradient_norm",
                    "unique_source_ratio", "unique_text_ratio", "within_memory_text_cosine",
                    "soft_hard_cosine_gap_mean", "soft_hard_cosine_gap_max", "support_top1_agreement",
                    "feature_normalized_row_entropy", "feature_hard_slot_coverage", "text_normalized_row_entropy",
                    "text_hard_slot_coverage", "feature_positive_mass_slot_coverage", "text_positive_mass_slot_coverage"):
            values = [number(row.get(key)) for row in members]
            values = [v for v in values if isinstance(v, (float, int))]
            out[key] = mean(values) if values else None
        out["source"] = source_rel(FINAL / "diagnostics.csv")
        summaries.append(out)
    return summaries


def link_summary(rows: list[dict]) -> list[dict]:
    finals = final_stage_rows(rows, ("dataset", "method", "order", "split_seed"))
    groups = defaultdict(list)
    for row in finals:
        groups[(row["dataset"], row["method"])].append(row)
    summaries = []
    for (dataset, method), members in sorted(groups.items()):
        out = {"dataset": dataset, "dataset_display": DISPLAY[dataset], "method": method,
               "n_splits": len(members), "split_seeds": ",".join(str(r["split_seed"]) for r in sorted(members, key=lambda r: r["split_seed"])),
               "sd_unit": "sample SD across independent edge splits; final stage"}
        for metric in ("auc", "ap", "heldout_edge_recall", "degree_pearson"):
            vals = [r[metric] for r in members if r[metric] is not None]
            out[metric] = mean(vals) if vals else None
            out[f"{metric}_sd"] = stdev(vals) if len(vals) > 1 else None
        out["source"] = " | ".join(sorted({r["source"] for r in members}))
        summaries.append(out)
    return summaries


def source_inventory(paths: list[Path]) -> list[dict]:
    inventory = {row["path"]: row for row in read_csv(MASTER / "source_inventory.csv")}
    for path in paths:
        key = source_rel(path)
        inventory.setdefault(key, {"path": key, "disposition": "included_in_aggregate", "note": "existing result integrated"})
    for key, row in inventory.items():
        path = ROOT / key
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            row["sha256"] = digest.hexdigest()
            row["bytes"] = path.stat().st_size
            row["availability"] = "local file verified"
        else:
            row["availability"] = "listed in source inventory; not present locally"
            row["bytes"] = number(row.get("bytes"))
    return list(inventory.values())


def fmt(value, digits=2):
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def metric_sd(row: dict, metric: str, sd_key: str, digits=2) -> str:
    value, sd = row.get(metric), row.get(sd_key)
    if value is None or value == "":
        return ""
    return fmt(value, digits) + (" ± " + fmt(sd, digits) if sd is not None and sd != "" else "")


def markdown_table(rows: list[dict], columns: list[tuple[str, str]], limit: int | None = None) -> list[str]:
    selected = rows[:limit] if limit else rows
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in selected:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if value is None or value == "":
                value = ""
            elif key in {"rate", "actual_rate", "positive_ratio"} and isinstance(value, (int, float)):
                value = f"{value:.2%}"
            elif key in {"auc", "ap", "heldout_edge_recall", "degree_pearson"} and isinstance(value, (int, float)):
                value = fmt(value, 4)
            elif key in {"memory_rows", "budget_rows", "full_raw_rows", "final_memory", "n_cells", "cells", "n_splits", "stage", "split_seed", "synthetic_undirected_edges"} and isinstance(value, (int, float)):
                value = f"{value:,.0f}"
            elif isinstance(value, float):
                value = fmt(value)
            values.append(str(value).replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline, internal = normalized_master_rows()
    extra_internal, extra_paths = attribution_rows()
    internal.extend(extra_internal)
    topo_node, topo_lp, topo_resources = topology_rows()
    diag = diagnostics_rows()
    diag_summary = diagnostic_summary(diag)
    lp_summary = link_summary(topo_lp)
    pairs = paired_rows()
    rates = rate_rows()
    joint = joint_rows()
    methods = method_map(baseline, internal)
    full_history = read_csv(MASTER / "full_history_comparison.csv")
    for row in full_history:
        row["dataset_display"] = DISPLAY.get(row.get("dataset", ""), row.get("dataset", ""))
        for key in row:
            if key not in {"dataset", "dataset_display", "split", "note"}:
                row[key] = number(row[key])
        row["result_family"] = "full_history_reference"
        row["source"] = source_rel(MASTER / "full_history_comparison.csv")
        row["source_kind"] = "master_full_history"
    integrated_paths = [
        MASTER / "baseline_results.csv", MASTER / "internal_candidates_and_ablations.csv",
        MASTER / "protocol_panels.csv", MASTER / "method_adaptations.csv", MASTER / "full_history_comparison.csv",
        ROOT / "results/uq_baseline_expansion_20261013/report_expansion_utc_20261014_021125_595059/matched.csv",
        ROOT / "results/uq_baseline_expansion_20261013/report_expansion_utc_20261014_021125_595059/sources.csv",
        ROOT / "results/uq_baseline_expansion_20261013/report_expansion_utc_20261014_021125_595059/independent_verification.json",
        ROOT / "results/uq_baseline_expansion_20261013/latest_report_receipt.json",
        FINAL / "summary.md", FINAL / "summary.csv", FINAL / "diagnostics.csv", FINAL / "pairs.csv",
        TOPO / "summary.md", TOPO / "node_classification.csv", TOPO / "link_prediction.csv", TOPO / "topology_resources.csv",
        RATE / "rate_sensitivity_summary.csv", RATE / "rate_sensitivity_summary.md", JOINT / "summary.md",
        ROOT / "results/unified_q_large_topology_bookhis_20261011.json",
    ] + extra_paths + [ROOT / row["source"] for row in topo_lp]
    sources = source_inventory(integrated_paths)
    panels = read_csv(MASTER / "protocol_panels.csv")
    panels.extend({"panel_id": code, "title": title, "note": note,
                   "student": "source-specific historical Student", "split": "report_dev",
                   "budget": "historical 5%-target schedule; source-specific"} for code, title, note in EXTRA_PANELS)
    assert len({r["record_id"] for r in baseline + internal}) == len(baseline) + len(internal)

    write_csv(OUT / "all_method_results.csv", baseline + internal)
    write_csv(OUT / "topology_node_classification.csv", topo_node)
    write_csv(OUT / "topology_link_prediction.csv", topo_lp)
    write_csv(OUT / "topology_resources.csv", topo_resources)
    write_csv(OUT / "q_diagnostics.csv", diag)
    write_csv(OUT / "q_diagnostic_summary.csv", diag_summary)
    write_csv(OUT / "paired_statistics.csv", pairs)
    write_csv(OUT / "link_prediction_summary.csv", lp_summary)
    write_csv(OUT / "protocol_panels.csv", panels)
    write_csv(OUT / "rate_sensitivity.csv", rates)
    write_csv(OUT / "joint_distribution_candidate.csv", joint)
    write_csv(OUT / "full_history.csv", full_history)
    write_csv(OUT / "method_map.csv", methods)
    write_csv(OUT / "source_inventory.csv", sources)

    bundle = {
        "all_method_results": baseline + internal,
        "topology_node_classification": topo_node,
        "topology_link_prediction": topo_lp,
        "topology_resources": topo_resources,
        "q_diagnostics": diag,
        "q_diagnostic_summary": diag_summary,
        "paired_statistics": pairs,
        "link_prediction_summary": lp_summary,
        "protocol_panels": panels,
        "rate_sensitivity": rates,
        "joint_distribution_candidate": joint,
        "full_history": full_history,
        "method_map": methods,
        "source_inventory": sources,
    }
    (OUT / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")

    expanded = [r for r in baseline if r.get("panel_id") == "00_expansion_sealed"]
    assert len(expanded) == 50 and all(sum(r["dataset"] == d for r in expanded) == 10 for d in ("arxiv", "sportsfit", "bookhis", "cora", "citeseer"))
    primary = [r for r in baseline if r.get("panel_id") == "01_sealed" and r.get("method") in {"Unified-Q-EF", "CaT-CCTP", "GECC-hard"}]
    summary_lines = [
        "# 全部实验结果总汇（2026-10-13）",
        "",
        "本报告把当前工程中已经完成并有数值记录的结果集中到一个入口。只读取已有结果，不启动新实验，也不覆盖 sealed 文件。",
        "",
        "## 读表规则",
        "",
        "- `00_expansion_sealed`：五图各十配置，同一类别顺序/构造种子、三个 Student seeds；仅是单 cell 横向比较，不是十篇官方论文复现。",
        "- `01_sealed`：三大图 EF/CaT/GECC，真实累计 5%，3 个类别顺序 × 3 个构造种子 × 3 个 Student seeds；是更强的稳定性确认。",
        "- TAGSAM bounded、STAD local Qwen 和 BookHis GECC-soft 在受限参考 panel；ArXiv/SportsFit GECC-soft 未完成且停止重试，不填零。",
        "- 旧 Student、旧调度、官方 checkpoint、简化适配和内部候选全部保留，但不能与主表合并排名。",
        "- `native-core adaptation` 表示保留原方法核心并做持续封装或共同 Student 适配，不等于原论文官方端到端复现。",
        "- 拓扑/链路预测、凝聚率敏感性和联合分布候选单独列出，避免把不同任务指标混成 F1/Accuracy。",
        "",
        "## 1. 五图同一 sealed cell：每图十配置",
        "",
    ]
    for row in baseline + internal:
        row["f1_display"] = metric_sd(row, "final_f1", "f1_sd")
        row["accuracy_display"] = metric_sd(row, "accuracy", "accuracy_sd")
    summary_lines += markdown_table(expanded, [
        ("dataset_display", "Dataset"), ("method", "Method"), ("final_f1", "Macro-F1 (%)"),
        ("accuracy", "Accuracy (%)"), ("f1_sd", "F1 SD"), ("accuracy_sd", "Acc SD"),
        ("n_cells", "Cells"), ("student", "Student"), ("fidelity", "Identity"), ("record_id", "ID"),
    ])
    summary_lines += ["", "该表每行是一个 order=9101/construction=271006 单元的三个 Student seeds 均值，表内 F1/Acc SD 也只对应 Student seeds。构造计算预算不同；PUMA 为 inspired reference，SFGC 为 bounded core。BookHis 的 GCond-x/SFGC 在单类末阶段无判别特征梯度，低分不能代指官方完整方法。", "",
        "### 三大图九 cell 稳定性（仅 EF/CaT/GECC）", ""]
    summary_lines += markdown_table(primary, [
        ("dataset_display", "Dataset"), ("method", "Method"), ("final_f1", "Macro-F1 (%)"),
        ("accuracy", "Accuracy (%)"), ("f1_sd", "Cell F1 SD"), ("accuracy_sd", "Cell Acc SD"),
        ("n_cells", "Cells"), ("student", "Student"), ("fidelity", "Identity"), ("record_id", "ID"),
    ])
    summary_lines += ["", "九 cell 中 Unified-Q-EF 在 ArXiv 为 52.91/71.15，SportsFit 为 80.08/85.52，BookHis 为 48.81/81.05。与 GECC-hard 的 F1 差值分别为 +0.22、-0.49、+3.47 个百分点；不能写成在所有数据集上稳定超过 GECC。", ""]
    primary_pairs = [r for r in pairs if r["category"] == "confirmation" and r["split"] == "sealed_test" and r["metric"] == "final_f1"]
    for row in primary_pairs:
        row["ci_display"] = f'[{row["ci_low"]:.2f}, {row["ci_high"]:.2f}]'
        row["positive_display"] = f'{row["positive_cells"]:.0f}/{row["cells"]:.0f}'
    summary_lines += ["### 配对统计", "", "每个 cell 先平均三个 Student seeds，再对相同顺序/构造种子的差值统计。95% CI 沿用原结果文件；不是把 27 次 Student 运行当作独立构造样本。", ""]
    summary_lines += markdown_table(primary_pairs, [("dataset_display", "Dataset"), ("right_method", "EF 对比对象"),
        ("mean_delta", "ΔF1 (pp)"), ("ci_display", "95% paired CI"), ("positive_display", "正收益 cells"), ("worst_delta", "Worst Δ (pp)")])

    summary_lines += ["## 2. Full-history 参考", "", "Full-history 允许访问全部历史 raw。下面使用原 report-dev 配对快照的分子和分母，不把最新 sealed 数值代入旧 full-history 结果。历史 ArXiv 为 6,773/118,539 = 5.71%，当前真实 5% 调度为 5,927 行，两者分开保留。", ""]
    summary_lines += markdown_table(full_history, [
        ("dataset_display", "Dataset"), ("budget_rows", "Synthetic rows"), ("full_raw_rows", "Raw rows"),
        ("actual_rate", "Rate"), ("unified_q_f1", "EF F1"), ("unified_q_accuracy", "EF Acc"),
        ("full_raw_f1", "Raw F1"), ("full_raw_accuracy", "Raw Acc"),
        ("f1_retention_pct", "F1 retention (%)"), ("accuracy_retention_pct", "Acc retention (%)"),
    ])
    summary_lines += ["", "保留率 = 凝聚结果 / 对应 full-history 结果 × 100%。超过 100% 表示该快照中压缩记忆训练的分数更高，不能单凭这个比值解释原因。", "",
                      "## 3. 按数据集的全部方法结果", "", "先列公开方法及参考，再单独列内部候选/消融。各 panel 的 Student、split 和调度分别说明；不同 panel 不合并排名。表中的 ± 可能是 Student-seed SD 或 cell SD，按下方统计单位读取。", ""]
    panel_lookup = {row["panel_id"]: row for row in panels}
    result_columns = [("record_id", "ID"), ("method", "Method"), ("readout", "Readout"),
                      ("f1_display", "F1 (%)"), ("accuracy_display", "Acc (%)"),
                      ("old_f1", "Old F1"), ("new_f1", "New F1"), ("n_cells", "Cells")]
    for dataset in DATASETS:
        summary_lines += [f"### {DISPLAY[dataset]}", ""]
        for family_rows, label in ((baseline, "公开基线、主方法及参考"), (internal, "内部候选、组件消融与归因对照")):
            dataset_rows = [r for r in family_rows if r.get("dataset") == dataset]
            if not dataset_rows:
                continue
            summary_lines += [f"#### {label}（{len(dataset_rows)} 行）", ""]
            for panel_id in sorted({r["panel_id"] for r in dataset_rows}):
                rows = sorted((r for r in dataset_rows if r["panel_id"] == panel_id), key=lambda r: (0 if r["method"] == "Unified-Q-EF" else 1, r["method"], str(r.get("readout", ""))))
                panel = panel_lookup[panel_id]
                summary_lines += [f'**{panel_id}：{panel["title"]}**', "", panel.get("note", ""), "",
                                  "Student：" + " / ".join(sorted({r.get("student", "") for r in rows})),
                                  "", "Split：" + " / ".join(sorted({r.get("split", "") for r in rows})) + "；预算：" + " / ".join(sorted({r.get("budget", "") for r in rows})),
                                  "", "统计单位：" + " / ".join(sorted({r.get("sd_unit", "未记录") or "未记录" for r in rows})), ""]
                summary_lines += markdown_table(rows, result_columns) + [""]

    summary_lines += ["## 4. 拓扑分支与链路预测", "", "拓扑结果来自独立扩展实验。链路预测是严格防泄漏的 slot-level link probe，不是训练式 GAE/GNN 复现。", "", "### 节点分类", ""]
    final_topo = final_stage_rows(topo_node, ("dataset", "order", "method"))
    summary_lines += markdown_table(final_topo, [("dataset_display", "Dataset"), ("order", "Order"), ("stage", "Stage"), ("method", "Method"), ("final_f1", "F1 (%)"), ("accuracy", "Acc (%)"), ("old_f1", "Old F1"), ("new_f1", "New F1")])
    summary_lines += ["", "### 链路预测", ""]
    for row in lp_summary:
        row["auc_display"] = metric_sd(row, "auc", "auc_sd", 4)
        row["ap_display"] = metric_sd(row, "ap", "ap_sd", 4)
    summary_lines += ["以下为最终阶段三个 edge split 的均值 ± 样本标准差；全部阶段和 split seed 在 CSV/Excel 中。", ""]
    summary_lines += markdown_table(lp_summary, [("dataset_display", "Dataset"), ("method", "Method"), ("auc_display", "AUC"), ("ap_display", "AP"), ("heldout_edge_recall", "Edge recall"), ("degree_pearson", "Degree Pearson"), ("split_seeds", "Split seeds")])
    summary_lines += ["", "拓扑版在 Cora 的最终链路预测 AUC 高于 edge-free；Citeseer 则不稳定。Unified-Q-T 目前应作为结构任务扩展，而不是替换 Unified-Q-EF 主方法。", ""]

    summary_lines += ["## 5. 凝聚率敏感性", "", "固定率 panel 是 report-development 证据，不用于 sealed 方法选择；SportsFit 10% 因 CUDA OOM 保留失败记录，没有用缩小模型替代。", ""]
    summary_lines += markdown_table(rates, [("dataset_display", "Dataset"), ("rate", "Target rate"), ("final_memory", "Memory rows"), ("method", "Method"), ("final_f1", "F1 (%)"), ("accuracy", "Acc (%)"), ("f1_std_student", "F1 Student SD"), ("accuracy_std_student", "Acc Student SD"), ("source_tier", "Source tier")])
    summary_lines += ["", "## 6. 组件消融与联合分布候选", "", "共享/独立 Q、no-residual、equal-parameter、soft-text-stored、Q-medoid 等内部结果在 `all_method_results.csv` 中保留。总体上，共享 Q 没有稳定优于独立 Q；因此它更适合作为统一接口和一致性设计，而不是已被证明的单独精度增益。联合分布匹配候选在 BookHis 退化，不升级为主方法。", ""]
    summary_lines += markdown_table(joint, [("dataset_display", "Dataset"), ("split", "Split"), ("method", "Candidate"), ("final_f1", "Candidate F1"), ("reference_f1", "EF F1"), ("delta_f1", "Delta F1"), ("accuracy", "Candidate Acc"), ("reference_accuracy", "EF Acc"), ("delta_accuracy", "Delta Acc")])
    summary_lines += ["", "### Q 与文本诊断", "", "按最终阶段、每个 cell 等权平均；覆盖率/熵是该阶段新增分配的诊断，不能解释为所有历史记忆的累计覆盖。文本多样性同时给出唯一文本比例和内部文本相似度。", ""]
    diag_columns = [("dataset_display", "Dataset"), ("method", "Control"), ("cells", "Cells"),
                    ("feature_hard_slot_coverage", "X hard coverage"), ("text_hard_slot_coverage", "Text hard coverage"),
                    ("feature_normalized_row_entropy", "X entropy"), ("text_normalized_row_entropy", "Text entropy"),
                    ("unique_text_ratio", "Unique text ratio"), ("within_memory_text_cosine", "Text cosine"),
                    ("soft_hard_cosine_gap_mean", "Soft/hard gap")]
    summary_lines += markdown_table(diag_summary, diag_columns)
    summary_lines += ["", "梯度尺度、分配参数量、residual 参数量、支持 top-1 一致性和构造成本保存在 `q_diagnostics.csv` 与 Excel 诊断页。独立 Q、等参数预算、无 residual 和 soft-only 存储对照均保留，不据单一诊断推断精度优势。", "",
                      "## 7. 方法名与适配对应", "", "下表只列公开方法和参考；同一论文的多种文本附着是适配变体，不计作多个独立强基线。内部组件的对应关系见 Excel 的 Method Map。", ""]
    external_methods = [r for r in methods if r["role"] != "internal_ablation_or_candidate"]
    summary_lines += markdown_table(external_methods, [("method", "当前表名"), ("original_method", "原方法"), ("fidelity", "实现身份"), ("adaptation", "做了什么改动"), ("datasets_with_numbers", "已有数值的数据集")])
    summary_lines += ["", "## 8. 文件与覆盖范围", "", f"- `all_method_results.csv`：{len(baseline)} 个主方法/公开基线/参考行，{len(internal)} 个内部候选/消融/归因行。",
                      f"- `paired_statistics.csv`：{len(pairs)} 个原始配对统计记录，保留 split、比较对象、指标、CI、正收益比例和 worst cell。",
                      "- `protocol_panels.csv`：实验分组、协议与统计单位。",
                      "- `topology_node_classification.csv`、`topology_link_prediction.csv`：拓扑任务全部阶段和真实 split seed。",
                      "- `q_diagnostics.csv`、`q_diagnostic_summary.csv`：逐阶段诊断与最终阶段摘要。",
                      "- `rate_sensitivity.csv`、`full_history.csv`：凝聚率实验与历史 full-history 配对参考。",
                      "- `method_map.csv`、`source_inventory.csv`：适配身份、来源路径、文件大小与 SHA-256。", "",
                      "本汇总以已有可核对数值为边界。未完成/OOM/未记录字段不补零；源清单中的重复导出和缺失数值另有 disposition。历史实验不升级为当前 sealed 结果，official checkpoint 评估不升级为 official condensation reproduction。", ""]
    (OUT / "all_results_report.md").write_text("\n".join(summary_lines), encoding="utf-8")

    manifest = {
        "scope": "existing-results-only",
        "counts": {
            "baseline_rows": len(baseline),
            "internal_rows": len(internal),
            "topology_node_rows": len(topo_node),
            "topology_link_rows": len(topo_lp),
            "topology_resource_rows": len(topo_resources),
            "diagnostic_rows": len(diag),
            "diagnostic_summary_rows": len(diag_summary),
            "paired_statistic_rows": len(pairs),
            "link_summary_rows": len(lp_summary),
            "protocol_panels": len(panels),
            "rate_rows": len(rates),
            "joint_rows": len(joint),
            "full_history_rows": len(full_history),
        },
        "dataset_counts": Counter(r.get("dataset", "") for r in baseline + internal),
        "outputs": [p.name for p in OUT.iterdir() if p.is_file()],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
