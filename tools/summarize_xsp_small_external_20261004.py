"""Summarize strict small-graph XSP vs external baseline results."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(r"D:/Aprojects/continue_tag_condensation")
EXTERNAL = ROOT / "D:/Aprojects/tag_baseline/EXTERNAL_BASELINE_HANDOFF_PACKAGE_20261004/EXTERNAL_BASELINE_HANDOFF_PACKAGE_20261004/docs/external_baseline_summary_20261002.json"
if not EXTERNAL.exists():
    EXTERNAL = Path(r"D:/Aprojects/tag_baseline/EXTERNAL_BASELINE_HANDOFF_PACKAGE_20261004/EXTERNAL_BASELINE_HANDOFF_PACKAGE_20261004/docs/external_baseline_summary_20261002.json")
OUT = ROOT / "results/xsp_small_external_strict_20261004_summary"
DOC = ROOT / "docs/current/XSP_SMALL_EXTERNAL_COMPARISON_ZH_2026-10-04.md"

DISPLAY = {
    "xsp": "XSP",
    "xsp_st": "XSP + ST-Sinkhorn",
    "CaT-MMD-TAG": "CaT-MMD-TAG",
    "GECC-hard-TAG": "GECC-hard-TAG",
    "k-center/FPS": "k-center/FPS",
    "medoid": "medoid",
    "PUMA-inspired": "PUMA-inspired",
}
MAIN_METHODS = tuple(DISPLAY)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def xsp_rows() -> list[dict]:
    rows = []
    for dataset, root_name in (("cora", "xsp_small_external_strict_20261004_cora"),
                               ("citeseer", "xsp_small_external_strict_20261004_citeseer")):
        root = ROOT / "results" / root_name / dataset
        for cell in sorted(root.glob("order*/results.json")):
            result = load_json(cell)
            for row in result["rows"]:
                if int(row["stage"]) != 2:
                    continue
                arm = row["arm"]
                score = row["aggregate"]
                rows.append({
                    "classification": "new-method",
                    "method": DISPLAY[arm],
                    "method_raw": arm,
                    "dataset": dataset,
                    "order": cell.parent.name,
                    "stage": 2,
                    "macro_f1": float(score["mean_f1"]),
                    "accuracy": float(score["mean_accuracy"]),
                    "old_f1": float(score["old_f1"]) if score.get("old_f1") is not None else None,
                    "new_f1": float(score["new_f1"]) if score.get("new_f1") is not None else None,
                    "worst_stage_f1": float(min(score["per_stage_f1"])),
                    "memory_rows": int(row["rows"]),
                    "student_seconds": float(mean(v["student_seconds"] for v in row["per_student"])),
                    "construction_seconds": float(row["construction_seconds"]),
                    "warning": "edge-free XSP adaptation; same strict protocol",
                })
    return rows


def external_rows() -> list[dict]:
    raw = load_json(EXTERNAL)
    rows = []
    for row in raw["rows"]:
        method = row["method"]
        if method not in MAIN_METHODS or method in ("xsp", "xsp_st"):
            continue
        if int(row["stage"]) != 2:
            continue
        rows.append(dict(row))
    return rows


def f(v):
    return "NA" if v is None else f"{float(v):.2f}"


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["dataset"])].append(row)
    out = []
    for (method, dataset), values in sorted(groups.items()):
        def avg(key):
            vals = [float(v[key]) for v in values if v.get(key) is not None]
            return mean(vals) if vals else None
        def sd(key):
            vals = [float(v[key]) for v in values if v.get(key) is not None]
            return stdev(vals) if len(vals) > 1 else 0.0
        out.append({
            "method": method,
            "dataset": dataset,
            "orders": ",".join(sorted(v["order"] for v in values)),
            "n_orders": len(values),
            "final_f1": avg("macro_f1"),
            "final_f1_std": sd("macro_f1"),
            "accuracy": avg("accuracy"),
            "old_f1": avg("old_f1"),
            "new_f1": avg("new_f1"),
            "worst_stage_f1": avg("worst_stage_f1"),
            "memory_rows": avg("memory_rows"),
            "construction_seconds": avg("construction_seconds"),
        })
    return out


def paired_deltas(rows: list[dict], candidate: str, baseline: str) -> list[dict]:
    index = {(r["method"], r["dataset"], r["order"]): r for r in rows}
    cells = []
    for (method, dataset, order), cand in index.items():
        if method != candidate:
            continue
        base = index.get((baseline, dataset, order))
        if base is None:
            continue
        cells.append({
            "candidate": candidate,
            "baseline": baseline,
            "dataset": dataset,
            "order": order,
            "delta_f1": cand["macro_f1"] - base["macro_f1"],
            "delta_accuracy": cand["accuracy"] - base["accuracy"],
            "delta_old_f1": cand["old_f1"] - base["old_f1"] if cand.get("old_f1") is not None and base.get("old_f1") is not None else None,
            "delta_new_f1": cand["new_f1"] - base["new_f1"] if cand.get("new_f1") is not None and base.get("new_f1") is not None else None,
        })
    return cells


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
    rows.sort(key=lambda r: (r["dataset"], r["order"], r["method"]))
    aggs = aggregate(rows)
    deltas = []
    for candidate in ("XSP", "XSP + ST-Sinkhorn"):
        for baseline in ("CaT-MMD-TAG", "GECC-hard-TAG", "k-center/FPS", "medoid", "PUMA-inspired"):
            deltas.extend(paired_deltas(rows, candidate, baseline))

    summary = {
        "protocol": {
            "datasets": ["cora", "citeseer"],
            "orders": ["order19611", "order19612", "order19613", "order19614", "order19615"],
            "stages": 3,
            "memory": "5% cumulative",
            "construction": "current delta + synthetic parent; no historical raw reads",
            "student": "original 2-layer GCN + frozen MiniLM + trainable text projection + bidirectional InfoNCE",
            "student_seeds": [150, 151, 152, 153, 154, 155],
            "evaluation": "full real graph, report-dev, all-prompt node classification",
        },
        "cell_rows": rows,
        "aggregates": aggs,
        "paired_deltas": deltas,
        "coverage": {method: sorted({(r["dataset"], r["order"]) for r in rows if r["method"] == method})
                      for method in DISPLAY.values()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    write_csv(OUT / "cell_results.csv", rows)
    write_csv(OUT / "aggregate_results.csv", aggs)
    write_csv(OUT / "paired_deltas.csv", deltas)

    lines = [
        "# XSP 小图严格外部基线比较（2026-10-04）",
        "",
        "本轮把 XSP 和 XSP + ST-Sinkhorn 放到外部 baseline 的小图协议中运行。协议固定为 Cora/Citeseer、三阶段、5% cumulative memory、current delta + synthetic parent、历史 raw 不可回读、Student seeds 150--155。最终评估在完整真实图的 report-dev 节点上进行 prompt-based node classification。",
        "",
        "## 1. 最终 stage 的逐顺序比较",
        "",
        "| Dataset | Order | Method | Final F1 | Accuracy | Old F1 | New F1 | Memory rows | Construction s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['order']} | {r['method']} | {f(r['macro_f1'])} | {f(r['accuracy'])} | {f(r.get('old_f1'))} | {f(r.get('new_f1'))} | {f(r.get('memory_rows'))} | {f(r.get('construction_seconds'))} |")
    lines += ["", "## 2. 按数据集和可用顺序汇总", "", "| Dataset | Method | Orders | N | Final F1 mean ± std | Accuracy | Old F1 | New F1 | Worst-stage F1 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in aggs:
        lines.append(f"| {r['dataset']} | {r['method']} | {r['orders']} | {r['n_orders']} | {f(r['final_f1'])} ± {f(r['final_f1_std'])} | {f(r['accuracy'])} | {f(r['old_f1'])} | {f(r['new_f1'])} | {f(r['worst_stage_f1'])} |")
    lines += ["", "## 3. XSP 相对外部 baseline 的配对差值", "", "| Candidate | Baseline | Common cells | Mean ΔF1 | Positive cells | Mean ΔAccuracy | Mean ΔOld F1 | Mean ΔNew F1 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for candidate in ("XSP", "XSP + ST-Sinkhorn"):
        for baseline in ("CaT-MMD-TAG", "GECC-hard-TAG", "k-center/FPS", "medoid", "PUMA-inspired"):
            values = [r for r in deltas if r["candidate"] == candidate and r["baseline"] == baseline]
            if not values:
                continue
            def avg(key):
                vals = [r[key] for r in values if r.get(key) is not None]
                return mean(vals) if vals else None
            positives = sum(r["delta_f1"] > 0 for r in values)
            lines.append(f"| {candidate} | {baseline} | {len(values)} | {f(avg('delta_f1'))} | {positives}/{len(values)} | {f(avg('delta_accuracy'))} | {f(avg('delta_old_f1'))} | {f(avg('delta_new_f1'))} |")
    lines += [
        "",
        "## 4. 协议与解释边界",
        "",
        "- XSP 和 XSP + ST 使用同一 CaT X 初始化、同一文本 payload 接口、同一 edge-free synthetic memory 约束；ST 只改变 Sinkhorn 反向传播路径。",
        "- CaT、GECC、PUMA 是 mechanism-adapted；k-center/medoid 是同协议 real-node coreset，不是 synthetic condensation；都不是各自论文的 native end-to-end 数字。",
        "- baseline 覆盖并不完全相同：CaT/k-center 覆盖五个顺序，GECC/medoid/SR 覆盖较少顺序，PUMA 只覆盖已有两个顺序。因此主判断优先使用 common-cell paired delta，不用不平衡的全覆盖均值制造排名。",
        "- Semantic-Recursive 不放入当前主方法排名；它是此前启发式递归版本，保留在外部包中作为历史补充。",
        "",
        "## 5. 当前结论",
        "",
        "本轮首先回答 XSP 是否能在小图严格外部协议下运行并与公开方法适配比较。XSP + ST-Sinkhorn 是否升为主方法，应由其相对 CaT 和 k-center 的五顺序 paired delta、最差 cell 和新类 F1 一起决定；不能只依据此前大图 BookChild 的单点改善。",
        "",
        f"原始逐 cell 结果和审计文件位于 `{OUT}`。",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(DOC)


if __name__ == "__main__":
    main()
