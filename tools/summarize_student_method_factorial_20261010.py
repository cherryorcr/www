"""Summarize the 2 x 4 Student-by-constructor factorial replay.

The unit used for uncertainty is the category-order cell. Student seeds are
already averaged inside each cell and are not treated as independent orders.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/student_method_factorial_20261010/remote_results"
OUTPUT = ROOT / "results/student_method_factorial_20261010"
METHODS = ("uq_dm_edgefree", "gecc_hard", "cat_cctp", "adaptive_coherent")
METHOD_NAMES = {
    "uq_dm_edgefree": "Unified-Q (edge-free)",
    "gecc_hard": "GECC-hard TAG adapter",
    "cat_cctp": "CaT-CCTP adapter",
    "adaptive_coherent": "Adaptive coherent attribution",
}
STUDENTS = ("prompt", "legacy")
STUDENT_NAMES = {
    "prompt": "Prompt Student",
    "legacy": "Legacy Student",
}
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1")
T_CRIT_95_DF2 = 4.302652729


def ci95(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    margin = T_CRIT_95_DF2 * stdev(values) / math.sqrt(len(values))
    return (margin, margin)


def load_rows(input_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete input: {path}")
        if len(payload.get("rows", [])) != len(METHODS):
            raise ValueError(f"expected {len(METHODS)} rows: {path}")
        for row in payload["rows"]:
            base = {"dataset": payload["protocol"]["dataset"],
                    "order_seed": payload["protocol"]["order_seed"],
                    "arm": row["arm"],
                    "memory_rows": row["memory_rows"],
                    "memory_hash": row["memory_hash"]}
            for student_key, score_key in (("prompt", "student"),
                                            ("legacy", "legacy")):
                score = row[score_key]
                rows.append({**base, "student": student_key,
                             **{metric: score.get(metric) for metric in METRICS},
                             "student_source_reused": bool(score.get("source_reused", False))})
    if not rows:
        raise ValueError("no factorial results found")
    return rows


def summarize_cells(rows: list[dict]) -> list[dict]:
    output = []
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets:
        for arm in METHODS:
            for student in STUDENTS:
                cells = [row for row in rows if row["dataset"] == dataset
                         and row["arm"] == arm and row["student"] == student]
                if len(cells) != 3:
                    raise ValueError(f"expected 3 orders: {dataset} {arm} {student}")
                record = {"dataset": dataset, "arm": arm, "student": student,
                          "method": METHOD_NAMES[arm],
                          "student_name": STUDENT_NAMES[student],
                          "order_count": len(cells)}
                for metric in METRICS:
                    values = [float(cell[metric]) for cell in cells
                              if cell[metric] is not None]
                    record[f"{metric}_mean"] = mean(values)
                    record[f"{metric}_sd_order"] = stdev(values) if len(values) > 1 else 0.0
                    record[f"{metric}_orders"] = values
                record["orders"] = [cell["order_seed"] for cell in cells]
                output.append(record)
    return output


def paired_delta(records: list[dict], dataset: str, arm: str,
                 student: str, baseline: str = "cat_cctp",
                 metric: str = "final_f1") -> dict:
    current = {(r["dataset"], r["arm"], r["student"], order): value
               for r in records for order, value in zip(r["orders"], r[f"{metric}_orders"])}
    deltas = [current[(dataset, arm, student, order)]
              - current[(dataset, baseline, student, order)]
              for order in (9101, 9102, 9103)]
    margin, _ = ci95(deltas)
    return {"dataset": dataset, "arm": arm, "student": student,
            "baseline": baseline, "metric": metric,
            "delta_mean": mean(deltas), "delta_sd_order": stdev(deltas),
            "ci95_low": mean(deltas) - margin,
            "ci95_high": mean(deltas) + margin,
            "positive_orders": sum(value > 0 for value in deltas),
            "deltas": deltas}


def student_effect(records: list[dict], dataset: str, arm: str,
                   metric: str = "final_f1") -> dict:
    prompt = next(r for r in records if r["dataset"] == dataset and
                  r["arm"] == arm and r["student"] == "prompt")
    legacy = next(r for r in records if r["dataset"] == dataset and
                  r["arm"] == arm and r["student"] == "legacy")
    deltas = [a - b for a, b in zip(prompt[f"{metric}_orders"],
                                    legacy[f"{metric}_orders"])]
    margin, _ = ci95(deltas)
    return {"dataset": dataset, "arm": arm, "metric": metric,
            "prompt_minus_legacy_mean": mean(deltas),
            "ci95_low": mean(deltas) - margin,
            "ci95_high": mean(deltas) + margin,
            "deltas": deltas}


def interaction(records: list[dict], dataset: str, arm: str,
                baseline: str = "cat_cctp", metric: str = "final_f1") -> dict:
    prompt = paired_delta(records, dataset, arm, "prompt", baseline, metric)["deltas"]
    legacy = paired_delta(records, dataset, arm, "legacy", baseline, metric)["deltas"]
    values = [a - b for a, b in zip(prompt, legacy)]
    margin, _ = ci95(values)
    return {"dataset": dataset, "arm": arm, "baseline": baseline,
            "metric": metric, "interaction_mean": mean(values),
            "ci95_low": mean(values) - margin,
            "ci95_high": mean(values) + margin,
            "values": values}


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=True)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def table(records: list[dict], dataset: str, metric: str) -> str:
    lines = [f"### {dataset}：{metric}", "",
             "| 方法 | Prompt Student | Legacy Student | Prompt-Legacy |",
             "|---|---:|---:|---:|"]
    for arm in METHODS:
        values = {}
        for student in STUDENTS:
            row = next(r for r in records if r["dataset"] == dataset and
                       r["arm"] == arm and r["student"] == student)
            values[student] = row[f"{metric}_mean"]
        lines.append(f"| {METHOD_NAMES[arm]} | {values['prompt']:.2f} | "
                     f"{values['legacy']:.2f} | {values['prompt'] - values['legacy']:+.2f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    rows = load_rows(input_dir)
    records = summarize_cells(rows)
    deltas = []
    effects = []
    interactions = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for arm in METHODS:
            for student in STUDENTS:
                for metric in METRICS:
                    deltas.append(paired_delta(records, dataset, arm, student,
                                               metric=metric))
            effects.append(student_effect(records, dataset, arm))
        for arm in ("uq_dm_edgefree", "adaptive_coherent", "gecc_hard"):
            interactions.append(interaction(records, dataset, arm))

    summary = {
        "status": "complete",
        "name": "student-method-factorial-20261010-matched-operator",
        "unit_of_uncertainty": "three paired category-order means; Student seeds are internal repeats",
        "input_files": [str(path) for path in sorted(input_dir.glob("*.json"))],
        "cell_count": len(rows),
        "records": records,
        "method_deltas_vs_cat": deltas,
        "student_effects": effects,
        "constructor_prompt_vs_legacy_interactions": interactions,
        "interpretation_guard": [
            "Prompt Student and Legacy Student are different training/evaluation protocols; absolute scores are not directly interchangeable.",
            "Prompt Student rows for CaT, Adaptive coherent and Unified-Q reuse the verified frozen memory; GECC is reconstructed in memory under the current-delta-only adapter.",
            "The evaluation uses report-dev and transductive full-graph inference, not sealed test or strict unseen-class zero-shot.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    write_csv(output_dir / "cell_results.csv", rows)
    write_csv(output_dir / "method_means.csv", records)
    write_csv(output_dir / "method_deltas_vs_cat.csv", deltas)
    write_csv(output_dir / "student_effects.csv", effects)
    write_csv(output_dir / "constructor_interactions.csv", interactions)
    write_csv(output_dir / "unified_q_interactions.csv",
              [row for row in interactions if row["arm"] == "uq_dm_edgefree"])

    lines = [
        "# Student × 构造方法二因素实验汇总",
        "",
        "日期：2026-10-10。单位为百分比；最终阶段；评估集为 report-dev，不是 sealed test。",
        "",
        "## 实验目的",
        "",
        "本实验固定类别顺序、预算、构造输入边界、Student steps/batch/seeds 和推理算子，只改变构造方法与 Student 家族，区分共同的 Student 提升、方法本身的收益以及二者交互。三次 order 的均值和标准差以 order 为统计单位；Student seeds 只在每个 cell 内取均值。Legacy 与 Prompt 均使用同一个 data[real_adj] 推理算子；旧版不匹配算子结果另存为补充材料。",
        "",
        "## 协议",
        "",
        "| 项目 | 设置 |",
        "|---|---|",
        "| 数据集 | WikiCS、SportsFit、BookHis |",
        "| 类别顺序 | 9101、9102、9103 |",
        "| 构造方法 | Unified-Q edge-free、GECC-hard TAG adapter、CaT-CCTP adapter、Adaptive coherent attribution |",
        "| Student | Legacy row-wise InfoNCE；Prompt multi-positive graph-text Student |",
        "| Student seeds | 150、151、152 |",
        "| 每阶段训练 | 2048 steps，batch 24 |",
        "| 评估 | Final stage Macro-F1、Accuracy、Old F1、New F1；report-dev |",
        "| 区间 | 3 个配对 order 的描述性 t 区间（df=2，临界值 4.3027）；不是独立重复显著性检验 |",
        "| 历史 raw 访问 | 构造器只接收 current delta 和自身 frozen synthetic prefix |",
        "| 拓扑 | 本轮四种方法均为 edge-free/self-loop Student operator |",
        "",
        "## Final Macro-F1",
        "",
    ]
    for dataset in sorted({row["dataset"] for row in rows}):
        lines.append(table(records, dataset, "final_f1"))
        lines.append("")
    lines += ["## Final Accuracy", ""]
    for dataset in sorted({row["dataset"] for row in rows}):
        lines.append(table(records, dataset, "accuracy"))
        lines.append("")
    lines += [
        "## Unified-Q 相对 CaT-CCTP",
        "",
        "| 数据集 | Student | Δ Final F1 | 95% order CI | 正收益顺序 | Δ Accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in sorted({row["dataset"] for row in rows}):
        for student in STUDENTS:
            f1 = next(v for v in deltas if v["dataset"] == dataset and v["arm"] == "uq_dm_edgefree" and v["student"] == student and v["metric"] == "final_f1")
            # Accuracy is reported in the companion JSON/CSV; use a direct lookup.
            acc_values = paired_delta(records, dataset, "uq_dm_edgefree", student, metric="accuracy")
            lines.append(f"| {dataset} | {STUDENT_NAMES[student]} | {f1['delta_mean']:+.2f} | [{f1['ci95_low']:+.2f}, {f1['ci95_high']:+.2f}] | {f1['positive_orders']}/3 | {acc_values['delta_mean']:+.2f} |")
    lines += [
        "",
        "## Student 主效应",
        "",
        "| 数据集 | 方法 | Prompt−Legacy Final F1 | 95% order CI |",
        "|---|---|---:|---:|",
    ]
    for effect in effects:
        lines.append(f"| {effect['dataset']} | {METHOD_NAMES[effect['arm']]} | {effect['prompt_minus_legacy_mean']:+.2f} | [{effect['ci95_low']:+.2f}, {effect['ci95_high']:+.2f}] |")
    lines += [
        "",
        "## 二因素解释",
        "",
        "1. Prompt Student 对所有构造方法都带来共同的绝对分数变化，不能把 Prompt 相比 Legacy 的提升写成凝聚方法提升。",
        "2. Unified-Q 相对 CaT 的方法差异应在同一个 Student 内比较；Prompt 和 Legacy 下的差异不一致时，说明存在 Student×构造交互。",
        "3. 本实验不支持把 Unified-Q 写成稳定超越所有方法：它需要与 GECC、CaT 和 Adaptive coherent 分 Student 报告，并保留 order 方差。",
        "4. Adaptive coherent 在本表是 current-protocol fixed-X text attribution，不等同于历史完整 Adaptive coherent 构造；不应把它标成外部公开 baseline。",
        "5. 本轮只验证构造方法与 Student 协议的归因，不是严格 unseen-class zero-shot，也没有验证 link prediction 或跨架构泛化。",
        "",
        "## 文件",
        "",
        "- `cell_results.csv`：72 个 method×Student×order cell。",
        "- `method_means.csv`：按数据集、方法、Student 聚合的均值与 order SD。",
        "- `method_deltas_vs_cat.csv`：同一 Student 内相对 CaT 的配对差值。",
        "- `student_effects.csv`：Prompt−Legacy 的 Student 主效应。",
        "- `constructor_interactions.csv`：Unified-Q、Adaptive 和 GECC 相对 CaT 的 Student 交互项。",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output_dir), "cells": len(rows)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
