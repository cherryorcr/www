"""Summarize the frozen 3-order x 3-seed Unified-Q confirmation.

This report keeps construction cells as the statistical unit. Student seeds are
averaged inside a cell and are never counted as independent construction runs.
The GraphSAGE readout is evaluated from the same archived memories and is
reported separately from the GCN/SGC readouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = ROOT / "results/uq_text_path_20261008"
EXT_ROOT = ROOT / "results/uq_confirmation_extended_backup_20261009"
SAGE_ROOT = ROOT / "results/uq_sage_extended_backup_20261009"
DEFAULT_OUT = ROOT / "results/uq_text_path_extended_20261009"

DATASETS = ("wikics", "sportsfit", "bookhis")
ORDERS = {"wikics": (9201, 9202, 9203),
          "sportsfit": (9201, 9202, 9203),
          "bookhis": (9202, 9203, 9205)}
SEEDS = (271108, 271109, 271110)
ARMS = ("uq_qmedoid", "uq_dm_edgefree", "cat_cctp", "gecc_hard")
STUDENTS = ("prompt_gcn", "prompt_sgc", "prompt_sage")
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1")
NAMES = {
    "uq_qmedoid": "Q-medoid (internal one-factor control)",
    "uq_dm_edgefree": "In-loop CaT-Q (main)",
    "cat_cctp": "CaT + nearest5 (TAG adaptation)",
    "gecc_hard": "GECC-hard + TAG (adaptation)",
}
STUDENT_NAMES = {
    "prompt_gcn": "Prompt-GCN",
    "prompt_sgc": "Prompt-SGC",
    "prompt_sage": "Prompt-GraphSAGE",
}


def json_read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tag(dataset: str, order: int, seed: int) -> str:
    return f"{dataset}_o{order}_c{seed}"


def constructor_path(dataset: str, order: int, seed: int) -> Path:
    name = tag(dataset, order, seed)
    extended = EXT_ROOT / name / "results.json"
    if extended.exists():
        return extended
    return OLD_ROOT / f"confirm_{name}" / "results.json"


def sage_path(dataset: str, order: int, seed: int) -> Path:
    return SAGE_ROOT / tag(dataset, order, seed) / "results.json"


def expected_cells() -> list[tuple[str, int, int]]:
    return [(dataset, order, seed)
            for dataset in DATASETS for order in ORDERS[dataset] for seed in SEEDS]


def load_constructor_cells() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells = []
    failures = []
    for dataset, order, seed in expected_cells():
        path = constructor_path(dataset, order, seed)
        if not path.exists():
            failures.append({"cell": tag(dataset, order, seed), "error": "missing results.json"})
            continue
        try:
            value = json_read(path)
            if value.get("status") != "complete":
                raise ValueError(f"status={value.get('status')}")
            if len(value.get("stages", [])) != 20 or len(value.get("rows", [])) != 8:
                raise ValueError("expected 20 stage records and 8 GCN/SGC rows")
            cells.append(value)
        except Exception as error:  # noqa: BLE001 - report every failed cell
            failures.append({"cell": tag(dataset, order, seed), "error": repr(error)})
    return cells, {"expected": len(expected_cells()), "loaded": len(cells), "failures": failures}


def validate_constructor_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    expected = set(expected_cells())
    seen = set()
    common_signature = None
    dataset_contracts: dict[str, Any] = {}
    order_contracts: dict[tuple[str, int], Any] = {}
    checks = []
    for cell in cells:
        protocol = cell["protocol"]
        identity = (protocol["dataset"], protocol["order_seed"], protocol["construction_seed"])
        if identity not in expected:
            raise AssertionError(f"unplanned constructor cell: {identity}")
        if identity in seen:
            raise AssertionError(f"duplicate constructor cell: {identity}")
        seen.add(identity)
        if protocol["arms"] != list(ARMS):
            raise AssertionError(f"method arms differ in {identity}")
        if protocol["student_seeds"] != [150, 151, 152] or protocol["student_steps"] != 2048:
            raise AssertionError(f"Student schedule differs in {identity}")
        if protocol["student_batch"] != 24 or protocol["max_stages"] != 5:
            raise AssertionError(f"Student batch/stage schedule differs in {identity}")
        signature = {
            "source_hashes": protocol["source_hashes"],
            "runner_sha256": protocol["runner_sha256"],
            "students": protocol["students"],
            "student_batch": protocol["student_batch"],
        }
        if common_signature is None:
            common_signature = signature
        elif signature != common_signature:
            raise AssertionError(f"constructor code or Student drift in {identity}")
        dataset_contract = {
            "budgets": protocol["budgets"],
            "pool_mask_hash": protocol["pool_mask_hash"],
            "report_mask_hash": protocol["report_mask_hash"],
            "class_count": len({c for group in protocol["groups"] for c in group}),
        }
        previous = dataset_contracts.setdefault(protocol["dataset"], dataset_contract)
        if previous != dataset_contract:
            raise AssertionError(f"budget/split drift across orders in {identity}")
        order_key = (protocol["dataset"], protocol["order_seed"])
        order_contract = {
            "groups": protocol["groups"],
            "stage_id_hashes": protocol["stage_id_hashes"],
            "budgets": protocol["budgets"],
        }
        previous_order = order_contracts.setdefault(order_key, order_contract)
        if previous_order != order_contract:
            raise AssertionError(f"order drift across construction seeds in {identity}")
        for arm in ARMS:
            stages = sorted((v for v in cell["stages"] if v["arm"] == arm), key=lambda v: v["stage"])
            if [v["stage"] for v in stages] != list(range(5)):
                raise AssertionError(f"incomplete stage chain {identity}/{arm}")
            for stage in stages:
                if stage["memory_rows"] != protocol["budgets"][stage["stage"]]:
                    raise AssertionError(f"budget mismatch {identity}/{arm}/{stage['stage']}")
                for key in ("budget_ok", "prefix_ok", "finite_ok",
                            "current_delta_text_sources", "same_class_text_sources"):
                    if not stage["checks"].get(key, False):
                        raise AssertionError(f"protocol check {key} failed {identity}/{arm}")
                if stage["checks"].get("edge_free", True) is False:
                    raise AssertionError(f"non-edge-free stage {identity}/{arm}")
            rows = [v for v in cell["rows"] if v["arm"] == arm]
            if {v["student"] for v in rows} != {"prompt_gcn", "prompt_sgc"}:
                raise AssertionError(f"missing GCN/SGC row {identity}/{arm}")
            memory_hashes = {v["memory_hash"] for v in rows}
            if len(memory_hashes) != 1 or next(iter(memory_hashes)) != stages[-1]["memory_hash"]:
                raise AssertionError(f"GCN/SGC did not read identical memory {identity}/{arm}")
        checks.append({"cell": tag(*identity), "source": str(constructor_path(*identity)),
                       "stages": len(cell["stages"]), "rows": len(cell["rows"]),
                       "all_protocol_checks": True})
    missing = sorted(expected - seen)
    if missing:
        raise AssertionError(f"missing constructor cells: {missing}")
    return {"expected_cells": len(expected), "completed_cells": len(seen),
            "code_student_consistent": True, "budget_split_consistent": True,
            "missing_cells": [], "cell_checks": checks}


def load_and_validate_sage(cells: list[dict[str, Any]]) -> tuple[dict[tuple[str, int, int], dict[str, Any]], dict[str, Any]]:
    by_key = {(v["protocol"]["dataset"], v["protocol"]["order_seed"],
               v["protocol"]["construction_seed"]): v for v in cells}
    results: dict[tuple[str, int, int], dict[str, Any]] = {}
    checks = []
    failures = []
    for identity in expected_cells():
        path = sage_path(*identity)
        try:
            value = json_read(path)
            if value.get("status") != "complete" or len(value.get("rows", [])) != 4:
                raise ValueError("expected complete SAGE result with four method rows")
            protocol = value["protocol"]
            if (protocol["dataset"], protocol["order_seed"], protocol["construction_seed"]) != identity:
                raise ValueError("identity mismatch")
            if protocol["student"] ["name"] != "GraphTextSAGEStudent":
                raise ValueError("unexpected Student")
            if protocol["student_steps"] != 2048 or protocol["student_batch"] != 24:
                raise ValueError("SAGE schedule mismatch")
            constructor = by_key[identity]
            constructor_rows = {v["arm"]: v for v in constructor["rows"] if v["student"] == "prompt_gcn"}
            for row in value["rows"]:
                if row["student"] != "prompt_sage" or row["arm"] not in ARMS:
                    raise ValueError("unexpected SAGE row")
                if row["memory_hash"] != constructor_rows[row["arm"]]["memory_hash"]:
                    raise ValueError(f"SAGE memory hash differs for {row['arm']}")
                if not row.get("checks", {}).get("same_source_memory", False):
                    raise ValueError("source memory check failed")
            results[identity] = value
            checks.append({"cell": tag(*identity), "source": str(path), "rows": 4,
                           "same_memory_as_gcn": True, "all_protocol_checks": True})
        except Exception as error:  # noqa: BLE001
            failures.append({"cell": tag(*identity), "error": repr(error)})
    if failures:
        raise AssertionError(f"SAGE validation failures: {failures}")
    return results, {"expected_cells": len(expected_cells()), "completed_cells": len(results),
                     "cell_checks": checks, "failures": failures}


def row_for(cell: dict[str, Any], arm: str, student: str) -> dict[str, Any]:
    if student == "prompt_sage":
        raise KeyError("SAGE rows are stored separately")
    return next(v for v in cell["rows"] if v["arm"] == arm and v["student"] == student)


def score_for(cell: dict[str, Any], sage: dict[str, Any] | None, arm: str, student: str) -> dict[str, Any]:
    if student == "prompt_sage":
        return next(v for v in sage["rows"] if v["arm"] == arm and v["student"] == student)["score"]
    return row_for(cell, arm, student)["score"]


def stage_cost(cell: dict[str, Any], arm: str) -> float:
    return float(sum(v["construction_seconds"] for v in cell["stages"] if v["arm"] == arm))


def bootstrap_summary(values: dict[tuple[int, int], float], orders: tuple[int, ...], seeds: tuple[int, ...]) -> dict[str, Any]:
    matrix = np.asarray([[values[(order, seed)] for seed in seeds] for order in orders], dtype=float)
    rng = np.random.default_rng(20261009)
    order_draws = rng.integers(0, len(orders), size=(20000, len(orders)))
    seed_draws = rng.integers(0, len(seeds), size=(20000, len(seeds)))
    draws = matrix[order_draws[:, :, None], seed_draws[:, None, :]].mean(axis=(1, 2))
    return {
        "mean": float(matrix.mean()),
        "std": float(matrix.std(ddof=1)),
        "ci95_crossed_bootstrap": [float(v) for v in np.quantile(draws, [0.025, 0.975])],
        "positive_cells": int((matrix > 0).sum()),
        "cells": int(matrix.size),
        "positive_ratio": float((matrix > 0).mean()),
        "worst_cell": float(matrix.min()),
        "best_cell": float(matrix.max()),
        "per_order": {str(order): float(matrix[i].mean()) for i, order in enumerate(orders)},
        "per_construction_seed": {str(seed): float(matrix[:, i].mean()) for i, seed in enumerate(seeds)},
        "warning": "Exploratory crossed-factor bootstrap; three levels per factor, report-dev only.",
    }


def build_summary(cells: list[dict[str, Any]], sage: dict[tuple[str, int, int], dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    records = []
    aggregates: dict[str, dict[str, list[dict[str, Any]]]] = {}
    paired: dict[str, Any] = {}
    for cell in cells:
        protocol = cell["protocol"]
        identity = (protocol["dataset"], protocol["order_seed"], protocol["construction_seed"])
        for student in STUDENTS:
            for arm in ARMS:
                score = score_for(cell, sage.get(identity), arm, student)
                record = {
                    "dataset": identity[0], "order": identity[1], "construction_seed": identity[2],
                    "student": student, "arm": arm, "metrics": {m: float(score[m]) for m in METRICS},
                    "student_seconds": float(next(v for v in (sage[identity]["rows"] if student == "prompt_sage" else cell["rows"])
                                                   if v["arm"] == arm and (v["student"] == student))["student_seconds"]),
                    "construction_seconds": stage_cost(cell, arm),
                    "memory_rows": int(next(v for v in cell["rows"] if v["arm"] == arm)["memory_rows"]),
                    "unique_texts": float(next(v for v in (sage[identity]["rows"] if student == "prompt_sage" else cell["rows"])
                                                if v["arm"] == arm and v["student"] == student)["unique_texts"]),
                }
                records.append(record)
                aggregates.setdefault(identity[0], {}).setdefault(student, []).append(record)
    for dataset in DATASETS:
        for student in STUDENTS:
            for baseline in ("uq_qmedoid", "cat_cctp", "gecc_hard"):
                for metric in METRICS:
                    values = {}
                    for order in ORDERS[dataset]:
                        for seed in SEEDS:
                            main = next(v for v in records if v["dataset"] == dataset and v["order"] == order
                                        and v["construction_seed"] == seed and v["student"] == student
                                        and v["arm"] == "uq_dm_edgefree")
                            other = next(v for v in records if v["dataset"] == dataset and v["order"] == order
                                         and v["construction_seed"] == seed and v["student"] == student
                                         and v["arm"] == baseline)
                            values[(order, seed)] = main["metrics"][metric] - other["metrics"][metric]
                    paired[f"{dataset}/{student}/main_vs_{baseline}/{metric}"] = bootstrap_summary(
                        values, ORDERS[dataset], SEEDS)
    tables = {}
    for dataset in DATASETS:
        tables[dataset] = {}
        for student in STUDENTS:
            tables[dataset][student] = []
            rows = aggregates[dataset][student]
            for arm in ARMS:
                selected = [v for v in rows if v["arm"] == arm]
                item = {"arm": arm, "name": NAMES[arm], "cells": len(selected),
                        "memory_rows": selected[0]["memory_rows"],
                        "metrics": {},
                        "construction_seconds_mean": float(np.mean([v["construction_seconds"] for v in selected])),
                        "student_seconds_mean": float(np.mean([v["student_seconds"] for v in selected])),
                        "unique_texts_mean": float(np.mean([v["unique_texts"] for v in selected]))}
                for metric in METRICS:
                    values = [v["metrics"][metric] for v in selected]
                    item["metrics"][metric] = {"mean": float(np.mean(values)),
                                                "std": float(np.std(values, ddof=1))}
                tables[dataset][student].append(item)
    all_dataset = {}
    for student in STUDENTS:
        all_dataset[student] = []
        selected_student = [v for v in records if v["student"] == student]
        for arm in ARMS:
            selected = [v for v in selected_student if v["arm"] == arm]
            all_dataset[student].append({
                "arm": arm, "name": NAMES[arm], "cells": len(selected),
                "metrics": {metric: {"mean": float(np.mean([v["metrics"][metric] for v in selected])),
                                      "std": float(np.std([v["metrics"][metric] for v in selected], ddof=1))}
                             for metric in METRICS},
                "construction_seconds_mean": float(np.mean([v["construction_seconds"] for v in selected])),
                "student_seconds_mean": float(np.mean([v["student_seconds"] for v in selected])),
                "unique_texts_mean": float(np.mean([v["unique_texts"] for v in selected])),
            })
    return {
        "scope": "3 datasets x 3 feasible orders x 3 construction seeds; report-dev; final memory only",
        "protocol": {
            "constructor_cells": len(cells), "sage_cells": len(sage),
            "student_seeds_inside_cell": [150, 151, 152], "student_steps": 2048,
            "student_batch": 24, "memory_topology": "edge-free",
            "history_raw_access": "current delta plus own frozen synthetic prefix; interface audit",
            "sealed_test": False,
        },
        "audit": audit,
        "records": records,
        "tables": tables,
        "all_dataset": all_dataset,
        "paired_main_differences": paired,
        "interpretation": {
            "main_method": "uq_dm_edgefree",
            "external_baselines": ["cat_cctp", "gecc_hard"],
            "internal_control": "uq_qmedoid",
            "claim_boundary": "Do not claim SOTA or universal superiority from report-dev crossed-cell means.",
        },
    }


def write_markdown(out: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Unified-Q 3×3 独立确认汇总",
        "",
        "完整矩阵：3 个数据集 × 3 个可行类别顺序 × 3 个 construction seeds = 27 个构造 cell。",
        "每个 cell 内使用 Student seeds 150/151/152，2048 steps，batch 24；先在 cell 内平均，再跨 cell 汇总。",
        "评价为 final memory 的 report-dev，不是 sealed test；GraphSAGE 只读取与 GCN/SGC 完全相同的 memory。",
        "",
        "## 方法归类",
        "",
        "| 方法 | 类型 | 实际内容 |",
        "|---|---|---|",
        "| In-loop CaT-Q | 当前主方法 | learned class-constrained Q、in-loop DM/GM、bounded X residual、Q-conditioned real-text routing；edge-free |",
        "| Q-medoid | 内部单因素控制 | 只把硬文本材料化从 Q-max 改为 Q-weighted same-class medoid |",
        "| CaT + nearest5 | 外部机制适配 | CaT-style X condensation + current-delta graph-nearest5 text attachment |",
        "| GECC-hard + TAG | 外部机制适配 | GECC hard clustering with TAG propagation weights + current-delta nearest5 text attachment |",
        "",
    ]
    for dataset in DATASETS:
        lines += [f"## {dataset}", ""]
        for student in STUDENTS:
            lines += [f"### {STUDENT_NAMES[student]}", "",
                      "| 方法 | Final Macro-F1 | Accuracy | Old F1 | New F1 | 构造秒 | Student 秒 | 唯一文本数 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for row in summary["tables"][dataset][student]:
                m = row["metrics"]
                fmt = lambda metric: f"{m[metric]['mean']:.2f} ± {m[metric]['std']:.2f}"
                lines.append(f"| {row['name']} | {fmt('final_f1')} | {fmt('accuracy')} | {fmt('old_f1')} | "
                             f"{fmt('new_f1')} | {row['construction_seconds_mean']:.1f} | "
                             f"{row['student_seconds_mean']:.1f} | {row['unique_texts_mean']:.1f} |")
            lines.append("")
    lines += ["## 三图总体均值", ""]
    for student in STUDENTS:
        lines += [f"### {STUDENT_NAMES[student]}", "",
                  "| 方法 | Final Macro-F1 | Accuracy | Old F1 | New F1 |", "|---|---:|---:|---:|---:|"]
        for row in summary["all_dataset"][student]:
            m = row["metrics"]
            lines.append(f"| {row['name']} | {m['final_f1']['mean']:.2f} ± {m['final_f1']['std']:.2f} | "
                         f"{m['accuracy']['mean']:.2f} ± {m['accuracy']['std']:.2f} | "
                         f"{m['old_f1']['mean']:.2f} ± {m['old_f1']['std']:.2f} | "
                         f"{m['new_f1']['mean']:.2f} ± {m['new_f1']['std']:.2f} |")
        lines.append("")
    lines += ["## 主方法配对差值", "",
              "正值表示 In-loop CaT-Q 优于对照。配对单位为同一数据集、顺序和 construction seed；区间是 3×3 crossed bootstrap，仅作探索性描述。",
              "", "| 数据集 / Student | 对照 | F1 差值 | Accuracy 差值 | F1 95% CI | F1 正收益 | 最差 F1 cell |", "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        for student in STUDENTS:
            for baseline in ("uq_qmedoid", "cat_cctp", "gecc_hard"):
                f = summary["paired_main_differences"][f"{dataset}/{student}/main_vs_{baseline}/final_f1"]
                a = summary["paired_main_differences"][f"{dataset}/{student}/main_vs_{baseline}/accuracy"]
                low, high = f["ci95_crossed_bootstrap"]
                lines.append(f"| {dataset} / {STUDENT_NAMES[student]} | {NAMES[baseline]} | {f['mean']:+.2f} | "
                             f"{a['mean']:+.2f} | [{low:+.2f}, {high:+.2f}] | {f['positive_cells']}/{f['cells']} | {f['worst_cell']:+.2f} |")
    lines += ["", "## 客观边界", "",
              "- 这轮确认了协议、顺序和 construction seed 下的稳定性范围，但指标仍是 report-dev。",
              "- CaT/GECC 是论文机制的 TAG 适配，不是官方完整复现；不能把适配结果写成原论文结果。",
              "- Prompt-GraphSAGE 使用与 Prompt-GCN/SGC 相同的文本塔、损失和预算，但其图编码器是显式 self-loop mean SAGE，不是 canonical loop-free SAGE。",
              "- 本表不支持 SOTA 或跨 Student 普遍优越性；应同时报告主方法的正收益、负收益、最差 cell 和成本。",
              ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_summary(out: Path, summary: dict[str, Any]) -> None:
    colors = {"uq_qmedoid": "#9c6ade", "uq_dm_edgefree": "#c34462",
              "cat_cctp": "#286f9c", "gecc_hard": "#469b78"}
    labels = [NAMES[v].replace(" (internal one-factor control)", "").replace(" (main)", "")
              .replace(" (TAG adaptation)", "").replace(" (adaptation)", "") for v in ARMS]
    display = {"wikics": "WikiCS", "sportsfit": "SportsFit", "bookhis": "BookHis"}
    for metric, title, filename in (("final_f1", "Final Macro-F1 (%)", "final_f1.png"),
                                    ("accuracy", "Final Accuracy (%)", "accuracy.png")):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True, sharey=True)
        for i, dataset in enumerate(DATASETS):
            for j, student in enumerate(STUDENTS):
                ax = axes[i, j]
                table = summary["tables"][dataset][student]
                x = np.arange(len(ARMS))
                vals = [row["metrics"][metric]["mean"] for row in table]
                errs = [row["metrics"][metric]["std"] for row in table]
                bars = ax.bar(x, vals, yerr=errs, color=[colors[a] for a in ARMS], capsize=3)
                ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
                ax.set_title(f"{display[dataset]} | {STUDENT_NAMES[student]}")
                ax.set_ylim(0, 100)
                ax.grid(axis="y", alpha=.2)
                ax.set_axisbelow(True)
                for bar, value in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5,
                            f"{value:.1f}", ha="center", va="bottom", fontsize=8)
                if j == 0:
                    ax.set_ylabel(title)
        fig.suptitle(f"{title} | 27 construction cells; report-dev")
        fig.savefig(out / filename, dpi=170)
        plt.close(fig)
    for baseline, filename, title in (("cat_cctp", "delta_vs_cat.png", "Main minus CaT adaptation"),
                                      ("gecc_hard", "delta_vs_gecc.png", "Main minus GECC adaptation")):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True, sharey=True)
        for j, student in enumerate(STUDENTS):
            ax = axes[j]
            vals = []
            errs = []
            for dataset in DATASETS:
                item = summary["paired_main_differences"][f"{dataset}/{student}/main_vs_{baseline}/final_f1"]
                vals.append(item["mean"])
                errs.append(item["std"])
            bars = ax.bar(np.arange(3), vals, yerr=errs, color="#c34462", capsize=3)
            ax.axhline(0, color="#444", linewidth=.8)
            ax.set_xticks(np.arange(3), [display[d] for d in DATASETS])
            ax.set_title(STUDENT_NAMES[student])
            ax.grid(axis="y", alpha=.2)
            for bar, value in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, value + (0.15 if value >= 0 else -0.6),
                        f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
            if j == 0:
                ax.set_ylabel("F1 difference (percentage points)")
        fig.suptitle(f"{title} | positive favors In-loop CaT-Q")
        fig.savefig(out / filename, dpi=170)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cells, load_audit = load_constructor_cells()
    constructor_audit = validate_constructor_cells(cells)
    sage, sage_audit = load_and_validate_sage(cells)
    audit = {"load": load_audit, "constructor": constructor_audit, "sage": sage_audit,
             "all_complete": load_audit["loaded"] == 27 and len(sage) == 27}
    if not audit["all_complete"]:
        raise SystemExit(json.dumps(audit, indent=2))
    summary = build_summary(cells, sage, audit)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_markdown(args.out, summary)
    plot_summary(args.out, summary)
    print(json.dumps({"constructor_cells": len(cells), "sage_cells": len(sage),
                      "out": str(args.out), "all_complete": audit["all_complete"]}))


if __name__ == "__main__":
    main()
