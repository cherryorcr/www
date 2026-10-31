"""Summarize arbitrary-stage P5 continual runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from p5_summary_fields import METRIC_DEFINITIONS, complete_sum, optional_seconds


def read_runs(root: Path) -> tuple[list[dict], list[dict]]:
    stage_rows: list[dict] = []
    run_rows: list[dict] = []
    for path in sorted(root.glob("**/results.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        dataset = str(manifest.get("dataset", path.parent.name))
        stages = manifest.get("results", [])
        for item in stages:
            roots = int(item.get("roots", item.get("effective_budget", 0)))
            seeds = int(item.get("student_training_runs", len(item.get("student_seeds", []))))
            stage_rows.append({
                "dataset": dataset,
                "stage": int(item["stage"]),
                "stage_classes": json.dumps(manifest["stage_classes"][int(item["stage"])], ensure_ascii=True),
                "source_rows": int(manifest["stage_train_counts"][int(item["stage"])]),
                "budget": int(manifest["budgets"][int(item["stage"])]),
                "memory_rows": roots,
                "keep_old_rows": int(item.get("keep_old_rows", 0)),
                "new_rows": int(item.get("new_rows", 0)),
                "effective_batch": min(int(item.get("student_batch", 0)), roots) if roots else 0,
                "student_steps": int(item.get("student_steps", 0)),
                "student_seeds": seeds,
                "student_optimizer_steps": int(item.get("student_optimizer_steps_total", 0)),
                "student_train_seconds": optional_seconds(item, "student_train_seconds"),
                "stage_build_seconds": optional_seconds(item, "stage_build_seconds"),
                "cum_dev_f1_mean": item.get("cum_dev_f1_mean"),
                "cum_dev_f1_std": item.get("cum_dev_f1_std"),
                "old_mean_f1_mean": item.get("old_mean_f1_mean"),
                "current_f1_mean": item.get("current_f1_mean"),
                "source": str(path),
                "text_embedding_seconds": optional_seconds(manifest, "text_embedding_seconds"),
                "old_new_metric_scope": "task_slice",
                "prompt_scope": "all",
            })
        if stages:
            final = max(stages, key=lambda x: int(x["stage"]))
            stage0 = min(stages, key=lambda x: int(x["stage"]))
            run_rows.append({
                "dataset": dataset,
                "n_stages": int(manifest.get("n_stages", len(stages))),
                "split_mode": manifest.get("split_mode"),
                "rate": float(manifest.get("rate", 0.0)),
                "stage_train_counts": json.dumps(manifest.get("stage_train_counts", [])),
                "budgets": json.dumps(manifest.get("budgets", [])),
                "stage_classes": json.dumps(manifest.get("stage_classes", []), ensure_ascii=True),
                "initial_f1": stage0.get("cum_dev_f1_mean"),
                "final_stage": int(final["stage"]),
                "final_rows": int(final.get("roots", final.get("effective_budget", 0))),
                "final_cum_dev_f1_mean": final.get("cum_dev_f1_mean"),
                "final_cum_dev_f1_std": final.get("cum_dev_f1_std"),
                "final_old_mean_f1_mean": final.get("old_mean_f1_mean"),
                "final_current_f1_mean": final.get("current_f1_mean"),
                "total_student_optimizer_steps": sum(int(x.get("student_optimizer_steps_total", 0)) for x in stages),
                "total_student_train_seconds": complete_sum(optional_seconds(x, "student_train_seconds") for x in stages),
                "total_stage_build_seconds": complete_sum(optional_seconds(x, "stage_build_seconds") for x in stages),
                "text_embedding_seconds": optional_seconds(manifest, "text_embedding_seconds"),
                "source": str(path),
                "old_new_metric_scope": "task_slice",
                "prompt_scope": "all",
            })
    return stage_rows, run_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_md(path: Path, stage_rows: list[dict], run_rows: list[dict]) -> None:
    lines = [
        "# P5 多阶段 / 顺序诊断报告",
        "",
        "协议：C0（5% fixed-rate、dispersion、Mean-X、joint_g5_c320 center-concat、",
        "frozen-old topology、cross-degree=1、response off、6 student seeds），",
        "阶段划分与阶段数以各运行的 manifest 和下表为准，报告指标为 report-dev Macro-F1（%）。",
        "累计 F1 在全部已见样本上计算；old/new 为 task-slice F1，不能与五层表的 pooled old/new 直接比较。",
        "old 对历史任务等权，不是对历史类别等权。未记录的成本显示为 `-`（CSV 空值），不按零计。",
        "",
        "## 最终结果",
        "",
        "| dataset | stages | stage split | initial F1 | final rows | final cumulative F1 | final old task-slice F1 | final new task-slice F1 | text s | build s | train s |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_rows:
        lines.append("| {dataset} | {n_stages} | {split_mode} | {initial_f1} | {final_rows} | {final_cum_dev_f1_mean} | {final_old_mean_f1_mean} | {final_current_f1_mean} | {text_embedding_seconds} | {total_stage_build_seconds} | {total_student_train_seconds} |".format(
            dataset=row["dataset"], n_stages=row["n_stages"], split_mode=row["split_mode"],
            initial_f1=fmt(row["initial_f1"]), final_rows=row["final_rows"],
            final_cum_dev_f1_mean=fmt(row["final_cum_dev_f1_mean"]),
            final_old_mean_f1_mean=fmt(row["final_old_mean_f1_mean"]),
            final_current_f1_mean=fmt(row["final_current_f1_mean"]),
            total_student_optimizer_steps=row["total_student_optimizer_steps"],
            total_stage_build_seconds=fmt(row["total_stage_build_seconds"]),
            text_embedding_seconds=fmt(row["text_embedding_seconds"]),
            total_student_train_seconds=fmt(row["total_student_train_seconds"])))
    lines.extend(["", "## 阶段轨迹", "",
                  "| dataset | stage | classes | source rows | budget/rows | old keep | new rows | cum F1 | old task-slice F1 | new task-slice F1 | student steps | build s | train s |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in sorted(stage_rows, key=lambda x: (x["dataset"], x["stage"])):
        lines.append("| {dataset} | {stage} | `{stage_classes}` | {source_rows} | {budget}/{memory_rows} | {keep_old_rows} | {new_rows} | {cum_dev_f1_mean} | {old_mean_f1_mean} | {current_f1_mean} | {student_steps} | {stage_build_seconds} | {student_train_seconds} |".format(
            dataset=row["dataset"], stage=row["stage"], stage_classes=row["stage_classes"],
            source_rows=row["source_rows"], budget=row["budget"], memory_rows=row["memory_rows"],
            keep_old_rows=row["keep_old_rows"], new_rows=row["new_rows"],
            cum_dev_f1_mean=fmt(row["cum_dev_f1_mean"]),
            old_mean_f1_mean=fmt(row["old_mean_f1_mean"]),
            current_f1_mean=fmt(row["current_f1_mean"]),
            student_steps=row["student_steps"],
            stage_build_seconds=fmt(row["stage_build_seconds"]),
            student_train_seconds=fmt(row["student_train_seconds"])))
    lines.extend(["", "## 结果解释", ""])
    for row in run_rows:
        initial = float(row["initial_f1"])
        final = float(row["final_cum_dev_f1_mean"])
        delta = final - initial
        lines.append(
            f"- {row['dataset']} 从 stage 0 的 {initial:.2f} 到最终 {final:.2f}，"
            f"变化 {delta:+.2f} pp；最终 task-slice old/new F1 为 "
            f"{fmt(row['final_old_mean_f1_mean'])}/{fmt(row['final_current_f1_mean'])}。"
        )
    lines.extend([
        "- 长链结果同时受类别块、每阶段 source rows、最后阶段新增 rows 和到达顺序影响，不能把最终回落单独归因于 fixed-rate。",
        "- 单类增量阶段的 current F1 只用于同一顺序内诊断，不能与多类阶段直接横向比较。",
        "- 本实验只诊断冻结 C0 的阶段/顺序表现，不把结果事后用于晋升 C1、response 或其它候选。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stages, runs = read_runs(root)
    write_csv(output / "longchain_stage_summary.csv", stages)
    write_csv(output / "longchain_final_summary.csv", runs)
    write_md(output / "P5_LONGCHAIN_SUMMARY_ZH.md", stages, runs)
    (output / "summary_manifest.json").write_text(json.dumps({
        "source_root": str(root), "stage_rows": len(stages), "run_rows": len(runs),
        "metric_definitions": METRIC_DEFINITIONS,
        "missing_cost_policy": "null or blank means unrecorded; incomplete totals remain unknown",
        "files": ["longchain_stage_summary.csv", "longchain_final_summary.csv", "P5_LONGCHAIN_SUMMARY_ZH.md"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage_rows": len(stages), "run_rows": len(runs), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
