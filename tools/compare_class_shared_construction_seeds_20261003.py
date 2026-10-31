"""Compare class-shared payload across two construction seeds.

The comparison keeps the Student seeds and continual protocol fixed.  It is
intended to separate construction-seed sensitivity from Student-seed noise;
it does not test category-order generalization.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ARMS = {
    "cat_mmd_tag_cctp_top5": "CCTP top-5",
    "cat_mmd_tag_class_shared_tagsam": "class-shared real payload",
}
DATASETS = ("arxiv", "sportsfit")
METRICS = (
    ("final_f1", "Final F1"),
    ("final_accuracy", "Accuracy"),
    ("stage_avg_f1", "Stage-average F1"),
    ("old_f1", "Old F1"),
    ("new_f1", "New F1"),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_old_cells(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] not in DATASETS or row["arm"] not in ARMS:
                continue
            rows[(row["dataset"], row["arm"])] = {
                key: (None if row[key] == "" else float(row[key]))
                for key in (
                    "final_f1", "final_accuracy", "stage_avg_f1",
                    "old_f1", "new_f1", "construction_seconds",
                )
            }
    return rows


def result_overall(path: Path) -> dict:
    payload = read_json(path)
    return payload["overall"]


def result_stage_rows(path: Path) -> list[dict]:
    return read_json(path)["rows"]


def paired_student_deltas(path: Path) -> list[dict]:
    rows = result_stage_rows(path)
    final_scores = {
        row["arm"]: row["score"]["per_student"]
        for row in rows
    }
    cctp = final_scores["cat_mmd_tag_cctp_top5"]
    shared = final_scores["cat_mmd_tag_class_shared_tagsam"]
    if len(cctp) != len(shared):
        raise ValueError("paired Student seed counts differ")
    metric_map = {
        "final_f1": ("cumulative_macro_f1", "Final F1"),
        "accuracy": ("cumulative_accuracy", "Accuracy"),
        "stage_avg_f1": ("stage_avg_f1", "Stage-average F1"),
        "old_f1": ("old_mean_f1", "Old F1"),
        "new_f1": ("current_f1", "New F1"),
    }
    result = []
    for metric, (source, label) in metric_map.items():
        deltas = [float(b[source]) - float(a[source])
                  for a, b in zip(cctp, shared)]
        mean = sum(deltas) / len(deltas)
        variance = (sum((v - mean) ** 2 for v in deltas) /
                    (len(deltas) - 1)) if len(deltas) > 1 else 0.0
        std = math.sqrt(variance)
        half = 2.571 * std / math.sqrt(len(deltas))
        result.append({
            "metric": metric,
            "label": label,
            "n": len(deltas),
            "mean_delta": mean,
            "std_delta": std,
            "ci95_low": mean - half,
            "ci95_high": mean + half,
            "positive_seeds": sum(v > 0 for v in deltas),
            "deltas": deltas,
        })
    return result


def value(row: dict, key: str) -> float | None:
    raw = row.get(key)
    return None if raw is None else float(raw)


def fmt(raw: float | None) -> str:
    return "NA" if raw is None else f"{raw:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-summary-dir", type=Path,
        default=Path("results/class_shared_large_corrected_summary_20261003"),
    )
    parser.add_argument(
        "--new-root", type=Path,
        default=Path("results/class_shared_large_seed314159_20261003"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/class_shared_construction_seed_comparison_20261003"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    old = read_old_cells(args.old_summary_dir / "dataset_cells.csv")
    new_results = {
        dataset: args.new_root / f"{dataset}_base4" / "results.json"
        for dataset in DATASETS
    }

    comparison = []
    stage_comparison = []
    paired_student = {}
    for dataset in DATASETS:
        payload = {"dataset": dataset}
        new_overall = result_overall(new_results[dataset])
        paired_student[dataset] = paired_student_deltas(new_results[dataset])
        for arm, label in ARMS.items():
            old_row = old[(dataset, arm)]
            new_row = new_overall[arm]
            for key, _ in METRICS:
                payload[f"old_{arm}_{key}"] = value(old_row, key)
                payload[f"new_{arm}_{key}"] = value(new_row, key)
                if arm == "cat_mmd_tag_class_shared_tagsam":
                    continue
            payload[f"old_{arm}_construction_seconds"] = value(
                old_row, "construction_seconds")
            payload[f"new_{arm}_construction_seconds"] = value(
                new_row, "construction_seconds")
        for key, label in METRICS:
            old_cctp = payload[f"old_cat_mmd_tag_cctp_top5_{key}"]
            old_shared = payload[f"old_cat_mmd_tag_class_shared_tagsam_{key}"]
            new_cctp = payload[f"new_cat_mmd_tag_cctp_top5_{key}"]
            new_shared = payload[f"new_cat_mmd_tag_class_shared_tagsam_{key}"]
            payload[f"old_delta_{key}"] = (
                None if old_cctp is None or old_shared is None
                else old_shared - old_cctp)
            payload[f"new_delta_{key}"] = (
                None if new_cctp is None or new_shared is None
                else new_shared - new_cctp)
        comparison.append(payload)

        old_stage_path = args.old_summary_dir / "stage_rows.csv"
        old_stage = []
        with old_stage_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["dataset"] in (dataset,) and row["arm"] in ARMS:
                    old_stage.append(row)
        new_stage = result_stage_rows(new_results[dataset])
        by_key_old = {(int(r["stage"]), r["arm"]): r for r in old_stage}
        by_key_new = {(int(r["stage"]), r["arm"]): r for r in new_stage}
        for stage in sorted({key[0] for key in by_key_new}):
            old_a = by_key_old[(stage, "cat_mmd_tag_cctp_top5")]
            old_b = by_key_old[(stage, "cat_mmd_tag_class_shared_tagsam")]
            new_a = by_key_new[(stage, "cat_mmd_tag_cctp_top5")]["score"]
            new_b = by_key_new[(stage, "cat_mmd_tag_class_shared_tagsam")]["score"]
            stage_comparison.append({
                "dataset": dataset,
                "stage": stage,
                "old_cctp_f1": float(old_a["final_f1"]),
                "old_shared_f1": float(old_b["final_f1"]),
                "old_delta_f1": float(old_b["final_f1"]) - float(old_a["final_f1"]),
                "new_cctp_f1": float(new_a["final_f1"]),
                "new_shared_f1": float(new_b["final_f1"]),
                "new_delta_f1": float(new_b["final_f1"]) - float(new_a["final_f1"]),
                "old_cctp_accuracy": float(old_a["accuracy"]),
                "old_shared_accuracy": float(old_b["accuracy"]),
                "new_cctp_accuracy": float(new_a["accuracy"]),
                "new_shared_accuracy": float(new_b["accuracy"]),
            })

    with (args.output_dir / "seed_comparison.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = list(comparison[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)
    with (args.output_dir / "stage_comparison.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        fields = list(stage_comparison[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stage_comparison)

    report = {
        "old_construction_seed": 270927,
        "old_payload_seed": 271003,
        "new_construction_seed": 314159,
        "new_payload_seed": 314235,
        "student_seeds": [100, 101, 102, 103, 104, 105],
        "datasets": list(DATASETS),
        "comparison": comparison,
        "stage_comparison": stage_comparison,
        "paired_student": paired_student,
        "scope": [
            "The category order, base+4 protocol, memory rate and Student seeds are fixed.",
            "This is a construction-seed replication, not an independent category-order test.",
            "BookChild remains covered only by the original construction seed in this round.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    (args.output_dir / "paired_student.csv").write_text(
        "dataset,metric,n,mean_delta,std_delta,ci95_low,ci95_high,positive_seeds\n"
        + "\n".join(
            f"{dataset},{row['metric']},{row['n']},{row['mean_delta']:.8f},"
            f"{row['std_delta']:.8f},{row['ci95_low']:.8f},"
            f"{row['ci95_high']:.8f},{row['positive_seeds']}"
            for dataset, rows in paired_student.items()
            for row in rows
        ) + "\n", encoding="utf-8")

    lines = [
        "# Class-shared construction-seed replication",
        "",
        "固定 base+4、5% cumulative memory、类别顺序、Student seeds 100--105 和评估口径；只改变构造 seed。",
        "原配置：CaT seed 270927，payload seed 271003；复核配置：CaT seed 314159，payload seed 314235。",
        "该轮验证构造随机性，不等同于独立类别顺序泛化。",
        "",
        "## 1. Final comparison",
        "",
        "|Dataset|Metric|Original delta|Independent-seed delta|Original CCTP|Original class-shared|New CCTP|New class-shared|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        for key, label in METRICS:
            lines.append(
                f"|{row['dataset']}|{label}|{fmt(row[f'old_delta_{key}'])}|"
                f"{fmt(row[f'new_delta_{key}'])}|"
                f"{fmt(row[f'old_cat_mmd_tag_cctp_top5_{key}'])}|"
                f"{fmt(row[f'old_cat_mmd_tag_class_shared_tagsam_{key}'])}|"
                f"{fmt(row[f'new_cat_mmd_tag_cctp_top5_{key}'])}|"
                f"{fmt(row[f'new_cat_mmd_tag_class_shared_tagsam_{key}'])}|")

    lines += [
        "",
        "## 2. Stage-wise Final F1",
        "",
        "|Dataset|Stage|Original CCTP|Original class-shared|Original delta|New CCTP|New class-shared|New delta|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage_comparison:
        lines.append(
            f"|{row['dataset']}|{row['stage']}|{row['old_cctp_f1']:.2f}|"
            f"{row['old_shared_f1']:.2f}|{row['old_delta_f1']:+.2f}|"
            f"{row['new_cctp_f1']:.2f}|{row['new_shared_f1']:.2f}|"
            f"{row['new_delta_f1']:+.2f}|")

    lines += [
        "",
        "## 3. Paired Student-seed deltas for the independent construction seed",
        "",
        "区间使用 6 个配对 Student seeds 的 t(5) 近似，t(0.975,5)=2.571；正向 seeds 是 class-shared 高于 CCTP 的数量。",
        "",
        "|Dataset|Metric|Mean delta|Std|95% CI|Positive seeds|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, rows in paired_student.items():
        for row in rows:
            lines.append(
                f"|{dataset}|{row['label']}|{row['mean_delta']:+.2f}|"
                f"{row['std_delta']:.2f}|"
                f"[{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}]|"
                f"{row['positive_seeds']}/{row['n']}|")

    final_deltas = [row["new_delta_final_f1"] for row in comparison]
    lines += [
        "",
        "## 4. Interpretation",
        "",
        f"- New-seed Final F1 delta is positive on {sum(v > 0 for v in final_deltas)}/{len(final_deltas)} replicated datasets.",
        "- The construction-seed check therefore supports a non-zero signal, but not a universal stable gain: arXiv is small while SportsFit is large.",
        "- The arXiv New F1 decreases under the independent seed, so the observed gain should not be described as a universal plasticity improvement.",
        "- The next required check is an independent category order or a second construction seed on BookChild; this report alone is insufficient to freeze class-shared as a final method.",
        "",
        "## 5. Audit scope",
        "",
        "Both runs used current-delta-only construction, no report/test labels for construction, no old synthetic memory update, and no topology update in the class-shared arm.",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "datasets": list(DATASETS)}))


if __name__ == "__main__":
    main()
