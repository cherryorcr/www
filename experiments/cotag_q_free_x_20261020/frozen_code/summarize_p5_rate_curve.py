"""Summarize P5 rate-curve runs with budgets, effective batches, and costs.

The runner stores the configured batch size, while the student actually uses
``min(configured_batch, number_of_synthetic_rows)``.  This report keeps both
values so low-rate runs cannot be misread as matched-batch experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from p5_summary_fields import METRIC_DEFINITIONS, complete_sum, optional_seconds


def _rate_label(value: float) -> str:
    return f"{100.0 * float(value):.4g}%"


def _rows(root: Path) -> list[dict]:
    out: list[dict] = []
    for result_path in sorted(root.glob("**/results.json")):
        manifest = json.loads(result_path.read_text(encoding="utf-8"))
        dataset = str(manifest.get("dataset", result_path.parent.name))
        rate = float(manifest.get("rate", 0.0))
        budgets = [int(v) for v in manifest.get("budgets", [])]
        for item in manifest.get("results", []):
            stage = int(item["stage"])
            roots = int(item.get("roots", item.get("effective_budget", 0)))
            configured_batch = int(item.get("student_batch", manifest.get("student_batch", 0)))
            effective_batch = min(configured_batch, roots) if roots else 0
            seed_count = int(item.get("student_training_runs", len(item.get("student_seeds", []))))
            steps = int(item.get("student_steps", 0))
            out.append({
                "dataset": dataset,
                "rate": rate,
                "rate_label": _rate_label(rate),
                "budget_mode": manifest.get("budget_mode"),
                "split_mode": manifest.get("split_mode"),
                "n_stages": int(manifest.get("n_stages", 0)),
                "stage": stage,
                "stage_budget": budgets[stage] if stage < len(budgets) else roots,
                "roots": roots,
                "keep_old_rows": int(item.get("keep_old_rows", 0)),
                "new_rows": int(item.get("new_rows", 0)),
                "actual_new_rows": int(item.get("actual_new_rows", 0)),
                "effective_batch": effective_batch,
                "configured_batch": configured_batch,
                "student_steps": steps,
                "student_seeds": seed_count,
                "student_optimizer_steps": int(item.get("student_optimizer_steps_total", steps * seed_count)),
                "student_train_seconds": optional_seconds(item, "student_train_seconds"),
                "stage_build_seconds": optional_seconds(item, "stage_build_seconds"),
                "selection_runs": int(item.get("selection_training_runs", 0)),
                "selection_optimizer_steps": int(item.get("selection_optimizer_steps_total", 0)),
                "cum_dev_f1_mean": item.get("cum_dev_f1_mean"),
                "cum_dev_f1_std": item.get("cum_dev_f1_std"),
                "current_f1_mean": item.get("current_f1_mean"),
                "old_mean_f1_mean": item.get("old_mean_f1_mean"),
                "response_gain_extra_rows": int(item.get("response_gain_extra_rows", 0)),
                "source": str(result_path),
                "text_embedding_seconds": optional_seconds(manifest, "text_embedding_seconds"),
                "old_new_metric_scope": "task_slice",
                "prompt_scope": "all",
            })
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict]) -> None:
    groups: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["rate"]), []).append(row)
    lines = [
        "# P5 小图 Rate Curve 汇总",
        "",
        "指标为 report-dev Macro-F1（%）。`configured_batch` 是配置值，",
        "`effective_batch=min(configured_batch, roots)` 是 student 实际每步抽取的行数。",
        "累计 F1 在全部已见样本上计算；old/new 为 task-slice F1，不能与五层表的 pooled old/new 直接比较。",
        "old 对历史任务等权，不是对历史类别等权。未记录的成本显示为 `-`（CSV 空值），不按零计。",
        "",
        "## 阶段表",
        "",
        "| dataset | rate | stage | budget/rows | old keep | new rows | effective batch | steps | student opt steps | cum F1 | old task-slice F1 | new task-slice F1 | build s | train s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda v: (v["dataset"], v["rate"], v["stage"])):
        lines.append("| {dataset} | {rate} | {stage} | {roots} | {keep_old_rows} | {new_rows} | {effective_batch} | {student_steps} | {student_optimizer_steps} | {cum_dev_f1_mean} | {old_mean_f1_mean} | {current_f1_mean} | {stage_build_seconds} | {student_train_seconds} |".format(
            dataset=row["dataset"], rate=row["rate_label"], stage=row["stage"],
            roots=row["roots"], keep_old_rows=row["keep_old_rows"],
            new_rows=row["new_rows"], effective_batch=row["effective_batch"],
            student_steps=row["student_steps"],
            student_optimizer_steps=row["student_optimizer_steps"],
            cum_dev_f1_mean=_fmt(row["cum_dev_f1_mean"]),
            old_mean_f1_mean=_fmt(row["old_mean_f1_mean"]),
            current_f1_mean=_fmt(row["current_f1_mean"]),
            stage_build_seconds=_fmt(row["stage_build_seconds"]),
            student_train_seconds=_fmt(row["student_train_seconds"])))
    lines.extend(["", "## Rate 最终阶段", "",
                  "| dataset | rate | final rows | final cum F1 | final old task-slice F1 | final new task-slice F1 | total student opt steps | total train s |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for key, group in sorted(groups.items()):
        final = max(group, key=lambda v: v["stage"])
        lines.append("| {dataset} | {rate} | {roots} | {cum} | {old} | {new} | {steps} | {seconds} |".format(
            dataset=key[0], rate=final["rate_label"], roots=final["roots"],
            cum=_fmt(final["cum_dev_f1_mean"]),
            old=_fmt(final["old_mean_f1_mean"]),
            new=_fmt(final["current_f1_mean"]),
            steps=sum(v["student_optimizer_steps"] for v in group),
            seconds=_fmt(complete_sum(v["student_train_seconds"] for v in group))))
    lines.extend(["", "## 解释", "",
                  "1. 低 rate 的首阶段 memory 可能小于 batch=16；此时不能把它当作与 5%/10% 完全同 batch 的对照。",
                  "2. `student_optimizer_steps` 只统计 student 重训，不包括构造器和文本编码成本。",
                  "3. 最终 F1 需要与阶段曲线、old/new F1 和实际 memory rows 一起解释。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = _rows(root)
    _write_csv(output / "rate_stage_summary.csv", rows)
    final_rows = []
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["rate"]), []).append(row)
    for (dataset, rate), group in sorted(grouped.items()):
        final = max(group, key=lambda v: v["stage"])
        final_rows.append({
            "dataset": dataset,
            "rate": rate,
            "rate_label": final["rate_label"],
            "final_stage": final["stage"],
            "final_rows": final["roots"],
            "final_cum_dev_f1_mean": final["cum_dev_f1_mean"],
            "final_cum_dev_f1_std": final["cum_dev_f1_std"],
            "final_old_mean_f1_mean": final["old_mean_f1_mean"],
            "final_current_f1_mean": final["current_f1_mean"],
            "total_student_optimizer_steps": sum(v["student_optimizer_steps"] for v in group),
            "total_student_train_seconds": complete_sum(v["student_train_seconds"] for v in group),
            "total_stage_build_seconds": complete_sum(v["stage_build_seconds"] for v in group),
            "text_embedding_seconds": final["text_embedding_seconds"],
            "total_selection_optimizer_steps": sum(v["selection_optimizer_steps"] for v in group),
            "source": final["source"],
            "old_new_metric_scope": "task_slice",
            "prompt_scope": "all",
        })
    _write_csv(output / "rate_final_summary.csv", final_rows)
    _write_markdown(output / "P5_RATE_CURVE_SUMMARY_ZH.md", rows)
    (output / "summary_manifest.json").write_text(json.dumps({
        "source_root": str(root),
        "stage_rows": len(rows),
        "final_rows": len(final_rows),
        "effective_batch_rule": "min(configured student_batch, synthetic roots)",
        "metric_definitions": METRIC_DEFINITIONS,
        "missing_cost_policy": "null or blank means unrecorded; incomplete totals remain unknown",
        "files": ["rate_stage_summary.csv", "rate_final_summary.csv", "P5_RATE_CURVE_SUMMARY_ZH.md"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage_rows": len(rows), "final_rows": len(final_rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
