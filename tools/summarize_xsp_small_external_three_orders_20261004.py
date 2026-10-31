"""Build a clean three-order small-graph comparison for XSP."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(r"D:/Aprojects/continue_tag_condensation")
EXTERNAL = Path(
    r"D:/Aprojects/tag_baseline/EXTERNAL_BASELINE_HANDOFF_PACKAGE_20261004/"
    r"EXTERNAL_BASELINE_HANDOFF_PACKAGE_20261004/docs/external_baseline_summary_20261002.json"
)
OUT = ROOT / "results/xsp_small_external_three_orders_20261004"
DOC = ROOT / "docs/current/XSP_SMALL_EXTERNAL_THREE_ORDER_COMPARISON_ZH_2026-10-04.md"
ORDERS = ("order19611", "order19612", "order19613")
METHODS = (
    "CaT-MMD-TAG",
    "GECC-hard-TAG",
    "k-center/FPS",
    "medoid",
    "XSP",
    "XSP + ST-Sinkhorn",
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def xsp_rows() -> list[dict]:
    rows = []
    for dataset in ("cora", "citeseer"):
        root = ROOT / "results" / f"xsp_small_external_strict_20261004_{dataset}" / dataset
        for order in ORDERS:
            result = load(root / order / "results.json")
            for row in result["rows"]:
                if int(row["stage"]) != 2:
                    continue
                score = row["aggregate"]
                method = "XSP + ST-Sinkhorn" if row["arm"] == "xsp_st" else "XSP"
                rows.append({
                    "classification": "new-method",
                    "method": method,
                    "dataset": dataset,
                    "order": order,
                    "stage": 2,
                    "macro_f1": float(score["mean_f1"]),
                    "accuracy": float(score["mean_accuracy"]),
                    "old_f1": float(score["old_f1"]),
                    "new_f1": float(score["new_f1"]),
                    "worst_stage_f1": float(min(score["per_stage_f1"])),
                    "memory_rows": int(row["rows"]),
                    "construction_seconds": float(row["construction_seconds"]),
                    "student_seconds": mean(float(v["student_seconds"]) for v in row["per_student"]),
                })
    return rows


def external_rows() -> list[dict]:
    rows = []
    for row in load(EXTERNAL)["rows"]:
        if row["method"] not in METHODS or row["method"] in ("XSP", "XSP + ST-Sinkhorn"):
            continue
        if row["order"] not in ORDERS or int(row["stage"]) != 2:
            continue
        rows.append({
            "classification": row.get("classification", "mechanism-adapted"),
            "method": row["method"],
            "dataset": row["dataset"],
            "order": row["order"],
            "stage": 2,
            "macro_f1": float(row["macro_f1"]),
            "accuracy": float(row["accuracy"]),
            "old_f1": None if row.get("old_f1") is None else float(row["old_f1"]),
            "new_f1": None if row.get("new_f1") is None else float(row["new_f1"]),
            "worst_stage_f1": None if row.get("worst_stage_f1") is None else float(row["worst_stage_f1"]),
            "memory_rows": int(row["memory_rows"]),
            "construction_seconds": row.get("construction_seconds"),
            "student_seconds": row.get("student_seconds"),
        })
    return rows


def fmt(value) -> str:
    return "NA" if value is None else f"{float(value):.2f}"


def average(values: list[dict], key: str):
    numbers = [float(row[key]) for row in values if row.get(key) is not None]
    return mean(numbers) if numbers else None


def deviation(values: list[dict], key: str):
    numbers = [float(row[key]) for row in values if row.get(key) is not None]
    return stdev(numbers) if len(numbers) > 1 else 0.0


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group, values in sorted(groups.items()):
        item = {key: value for key, value in zip(keys, group)}
        item.update({
            "n": len(values),
            "final_f1": average(values, "macro_f1"),
            "final_f1_std": deviation(values, "macro_f1"),
            "accuracy": average(values, "accuracy"),
            "old_f1": average(values, "old_f1"),
            "new_f1": average(values, "new_f1"),
            "worst_stage_f1": average(values, "worst_stage_f1"),
            "construction_seconds": average(values, "construction_seconds"),
        })
        result.append(item)
    return result


def deltas(rows: list[dict], candidate: str, baseline: str) -> list[dict]:
    index = {(row["dataset"], row["order"], row["method"]): row for row in rows}
    result = []
    for dataset in ("cora", "citeseer"):
        for order in ORDERS:
            candidate_row = index.get((dataset, order, candidate))
            baseline_row = index.get((dataset, order, baseline))
            if candidate_row is None or baseline_row is None:
                continue
            result.append({
                "candidate": candidate,
                "baseline": baseline,
                "dataset": dataset,
                "order": order,
                "delta_f1": candidate_row["macro_f1"] - baseline_row["macro_f1"],
                "delta_accuracy": candidate_row["accuracy"] - baseline_row["accuracy"],
                "delta_old_f1": candidate_row["old_f1"] - baseline_row["old_f1"],
                "delta_new_f1": candidate_row["new_f1"] - baseline_row["new_f1"],
            })
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = xsp_rows() + external_rows()
    rows.sort(key=lambda row: (row["dataset"], row["order"], row["method"]))
    dataset_aggregates = aggregate(rows, ("dataset", "method"))
    global_aggregates = aggregate(rows, ("method",))
    paired = []
    for candidate in ("XSP", "XSP + ST-Sinkhorn"):
        for baseline in ("CaT-MMD-TAG", "GECC-hard-TAG", "k-center/FPS", "medoid"):
            paired.extend(deltas(rows, candidate, baseline))

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "cell_results.csv", rows)
    write_csv(OUT / "dataset_aggregates.csv", dataset_aggregates)
    write_csv(OUT / "global_aggregates.csv", global_aggregates)
    write_csv(OUT / "paired_deltas.csv", paired)
    (OUT / "protocol.json").write_text(json.dumps({
        "datasets": ["cora", "citeseer"],
        "orders": list(ORDERS),
        "stages": 3,
        "memory": "5% cumulative fixed-rate",
        "construction_access": "current delta + synthetic parent only; no historical raw graph/text/edge",
        "student": "two-layer GCN + frozen MiniLM + text projection + bidirectional InfoNCE; scratch per stage",
        "student_seeds": [150, 151, 152, 153, 154, 155],
        "evaluation": "full real graph report-dev nodes with prompt-based node classification",
        "baseline_status": "mechanism-adapted, not native end-to-end reproductions",
    }, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        "# XSP 小图三共同顺序外部比较（2026-10-04）",
        "",
        "本报告把 XSP 与已完成外部适配的 baseline 放在三个共同类别顺序 `order19611/19612/19613` 上比较。这里的“三次划分”是三次预注册类别到达顺序；每个顺序内部仍是三阶段 class-incremental stream，不是三次重新访问历史 raw graph 的 offline 重凝聚。",
        "",
        "## 1. 冻结协议",
        "",
        "- 数据集：Cora、Citeseer。",
        "- 每个顺序：三阶段类别增量，5% cumulative fixed-rate memory。",
        "- 构造器：只接收当前 `delta_t` 和上一阶段 synthetic parent；历史 raw graph、raw text、raw edge 不可回读。",
        "- Student：原两层 GCN、冻结 MiniLM、text projection、双向 InfoNCE；每阶段从头训练；Student seeds `150--155`。",
        "- 评估：完整真实图的 report-dev 节点，最终阶段 Macro-F1、Accuracy、Old F1、New F1。",
        "- CaT、GECC、k-center、medoid 均为保留原机制后的 TAG/Student 适配，不宣称官方 native reproduction。",
        "",
        "## 2. 最终阶段逐顺序结果",
        "",
        "| Dataset | Order | Method | Final F1 | Accuracy | Old F1 | New F1 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['order']} | {row['method']} | {fmt(row['macro_f1'])} | {fmt(row['accuracy'])} | {fmt(row['old_f1'])} | {fmt(row['new_f1'])} |")

    lines += [
        "",
        "## 3. 按数据集三顺序汇总",
        "",
        "| Dataset | Method | N | Final F1 mean ± std | Accuracy | Old F1 | New F1 | Worst-stage F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_aggregates:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['n']} | {fmt(row['final_f1'])} ± {fmt(row['final_f1_std'])} | {fmt(row['accuracy'])} | {fmt(row['old_f1'])} | {fmt(row['new_f1'])} | {fmt(row['worst_stage_f1'])} |")

    lines += [
        "",
        "## 4. 两个数据集六个共同 cell 的总体汇总",
        "",
        "| Method | N | Final F1 mean ± std | Accuracy | Old F1 | New F1 | Construction s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_aggregates:
        lines.append(f"| {row['method']} | {row['n']} | {fmt(row['final_f1'])} ± {fmt(row['final_f1_std'])} | {fmt(row['accuracy'])} | {fmt(row['old_f1'])} | {fmt(row['new_f1'])} | {fmt(row['construction_seconds'])} |")

    lines += [
        "",
        "## 5. XSP 的配对差值",
        "",
        "| Candidate | Baseline | Common cells | Mean ΔF1 | Positive cells | Mean ΔAccuracy | Mean ΔOld F1 | Mean ΔNew F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in ("XSP", "XSP + ST-Sinkhorn"):
        for baseline in ("CaT-MMD-TAG", "GECC-hard-TAG", "k-center/FPS", "medoid"):
            values = [row for row in paired if row["candidate"] == candidate and row["baseline"] == baseline]
            if not values:
                continue
            avg = lambda key: mean(float(row[key]) for row in values)
            positives = sum(float(row["delta_f1"]) > 0 for row in values)
            lines.append(f"| {candidate} | {baseline} | {len(values)} | {avg('delta_f1'):.2f} | {positives}/{len(values)} | {avg('delta_accuracy'):.2f} | {avg('delta_old_f1'):.2f} | {avg('delta_new_f1'):.2f} |")

    lines += [
        "",
        "## 6. PUMA、TAGSAM 与 STAD 的处理边界",
        "",
        "PUMA 当前每个 dataset/order 有多个适配变体，不能把这些行简单平均后作为一个 baseline 排名；它们应在单独的 variant 表中报告。TAGSAM 与 STAD 的现有结果使用 Student seeds `100--105`，且顺序信息不完整，因此保留为 adapted/reference evidence，不混入本三顺序 strict ranking。",
        "",
        "## 7. 结论",
        "",
        "1. XSP 已经完成与外部机制适配 baseline 的同协议小图比较；它不是只在自己的 runner 中得到结果。",
        "2. XSP 相对 GECC、k-center 和 medoid 有稳定的竞争力，但相对 CaT 仍不是稳定正收益；因此论文中不能写成全面超过 CaT。",
        "3. XSP + ST-Sinkhorn 在这三个共同顺序上没有显示出比普通 XSP 更可靠的收益，暂时作为消融组件，而不是主方法升级。",
        "4. 三顺序表适合做第一版外部比较；正式论文主表仍应补充更多独立顺序或明确将覆盖不足的方法放入 supplementary/reference 层。",
        "",
        f"逐 cell、汇总和 paired delta 文件位于 `{OUT}`。",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DOC)


if __name__ == "__main__":
    main()
