"""Audit and summarize the unified ArXiv base+4 comparison."""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/cotag_q_arxiv_base4_matched_20261018"
OUTPUT = INPUT / "summary"
ORDERS = (9101, 9102, 9103)
SEEDS = (271006, 271108, 271109)
ARMS = ("uq_dm_edgefree", "cat_cctp", "gecc_hard")
DISPLAY = {
    "uq_dm_edgefree": "Unified-Q-EF",
    "cat_cctp": "CaT-CCTP",
    "gecc_hard": "GECC-hard-TAG",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_id(path: Path) -> tuple[int, int]:
    match = re.search(r"o(\d+)_c(\d+)", str(path))
    if not match:
        raise ValueError(f"cannot parse cell id from {path}")
    return int(match.group(1)), int(match.group(2))


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def fmt(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def ci95(values: list[float]) -> tuple[float | None, float | None]:
    """Paired t interval over the nine construction cells (df=8)."""
    if len(values) < 2:
        return None, None
    # t_{0.975,8}; the construction cell, not the Student seed, is the unit.
    radius = 2.306004135204166 * sample_std(values) / (len(values) ** 0.5)
    center = mean(values)
    return center - radius, center + radius


def load_cells() -> tuple[list[dict], list[str]]:
    cells: list[dict] = []
    problems: list[str] = []
    paths = sorted(INPUT.glob("o*_c*/matrix/results.json"))
    expected_cells = {(o, s) for o in ORDERS for s in SEEDS}
    observed_cells = set()
    for path in paths:
        if "_failure_" in path.parent.parent.name:
            continue
        order, seed = cell_id(path)
        observed_cells.add((order, seed))
        payload = read(path)
        protocol = payload.get("protocol", {})
        if payload.get("status") != "complete":
            problems.append(f"incomplete: {path}")
            continue
        if protocol.get("dataset") != "arxiv":
            problems.append(f"wrong dataset: {path}")
        if [len(group) for group in protocol.get("groups", [])] != [8, 8, 8, 8, 8]:
            problems.append(f"not base+4 groups: {path}")
        if protocol.get("arms") != list(ARMS):
            problems.append(f"arm mismatch: {path}")
        if protocol.get("order_seed") != order or protocol.get("construction_seed") != seed:
            problems.append(f"seed mismatch: {path}")
        if protocol.get("student_seeds") != [150, 151, 152] or protocol.get("student_steps") != 2048:
            problems.append(f"student mismatch: {path}")
        rows = payload.get("rows", [])
        if len(rows) != 15:
            problems.append(f"expected 15 rows, got {len(rows)}: {path}")
            continue
        by_arm = defaultdict(list)
        for row in rows:
            by_arm[row["arm"]].append(row)
            checks = row.get("checks", {})
            for key in ("budget_ok", "prefix_ok", "finite_ok"):
                if not checks.get(key):
                    problems.append(f"failed {key}: {path} arm={row['arm']} stage={row['stage']}")
        if set(by_arm) != set(ARMS) or any(len(v) != 5 for v in by_arm.values()):
            problems.append(f"arm/stage coverage mismatch: {path}")
            continue
        for arm in ARMS:
            rows_arm = sorted(by_arm[arm], key=lambda r: r["stage"])
            scores = [row["score"] for row in rows_arm]
            final = scores[-1]
            cells.append({
                "order_seed": order,
                "construction_seed": seed,
                "arm": arm,
                "final_f1": final["final_f1"],
                "final_f1_student_sd": final["final_f1_std"],
                "accuracy": final["accuracy"],
                "accuracy_student_sd": final["accuracy_std"],
                "old_f1": final["old_f1"],
                "new_f1": final["new_f1"],
                "stage_f1": mean(score["final_f1"] for score in scores),
                "stage_acc": mean(score["accuracy"] for score in scores),
                "worst_stage_f1": min(score["final_f1"] for score in scores),
                "construction_seconds": sum(row["construction_seconds"] for row in rows_arm),
                "student_seconds": sum(row["student_seconds"] for row in rows_arm),
                "peak_mb": max(row["construction_peak_mb"] or 0 for row in rows_arm),
                "source": str(path.relative_to(ROOT)),
            })
    missing = sorted(expected_cells - observed_cells)
    problems.extend(f"missing cell: order={o}, construction={s}" for o, s in missing)
    return cells, problems


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(cells: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = []
    for arm in ARMS:
        subset = [row for row in cells if row["arm"] == arm]
        if not subset:
            continue
        grouped.append({
            "arm": arm,
            "method": DISPLAY[arm],
            "n_cells": len(subset),
            "final_f1": mean(row["final_f1"] for row in subset),
            "final_f1_sd": sample_std([row["final_f1"] for row in subset]),
            "accuracy": mean(row["accuracy"] for row in subset),
            "accuracy_sd": sample_std([row["accuracy"] for row in subset]),
            "old_f1": mean(row["old_f1"] for row in subset),
            "new_f1": mean(row["new_f1"] for row in subset),
            "stage_f1": mean(row["stage_f1"] for row in subset),
            "stage_acc": mean(row["stage_acc"] for row in subset),
            "worst_stage_f1": mean(row["worst_stage_f1"] for row in subset),
            "construction_seconds": mean(row["construction_seconds"] for row in subset),
            "student_seconds": mean(row["student_seconds"] for row in subset),
            "peak_mb": mean(row["peak_mb"] for row in subset),
        })
    paired = []
    uq = {(r["order_seed"], r["construction_seed"]): r for r in cells if r["arm"] == "uq_dm_edgefree"}
    for baseline in ("cat_cctp", "gecc_hard"):
        base = {(r["order_seed"], r["construction_seed"]): r for r in cells if r["arm"] == baseline}
        deltas = [uq[key]["final_f1"] - base[key]["final_f1"] for key in sorted(uq) if key in base]
        acc_deltas = [uq[key]["accuracy"] - base[key]["accuracy"] for key in sorted(uq) if key in base]
        paired.append({
            "baseline": DISPLAY[baseline],
            "n_pairs": len(deltas),
            "f1_delta": mean(deltas) if deltas else None,
            "f1_delta_sd": sample_std(deltas),
            "f1_positive_cells": sum(value > 0 for value in deltas),
            "f1_positive_ratio": sum(value > 0 for value in deltas) / len(deltas),
            "f1_worst_cell": min(deltas) if deltas else None,
            "f1_ci_low": ci95(deltas)[0],
            "f1_ci_high": ci95(deltas)[1],
            "accuracy_delta": mean(acc_deltas) if acc_deltas else None,
            "accuracy_delta_sd": sample_std(acc_deltas),
            "accuracy_positive_cells": sum(value > 0 for value in acc_deltas),
            "accuracy_positive_ratio": sum(value > 0 for value in acc_deltas) / len(acc_deltas),
            "accuracy_worst_cell": min(acc_deltas) if acc_deltas else None,
            "accuracy_ci_low": ci95(acc_deltas)[0],
            "accuracy_ci_high": ci95(acc_deltas)[1],
        })
    return grouped, paired


def render(cells: list[dict], grouped: list[dict], paired: list[dict], problems: list[str]) -> str:
    lines = [
        "# ArXiv unified base+4 audit",
        "",
        "This report evaluates the same cumulative 5% protocol used for SportsFit and BookHis: five class increments with an 8/8/8/8/8 class schedule. The original three-stage ArXiv confirmation stream is kept separately.",
        "",
        "## Protocol",
        "",
        "- Construction cells: 3 class orders (9101/9102/9103) x 3 construction seeds (271006/271108/271109).",
        "- Student seeds: 150/151/152; 2,048 updates; common Prompt-GCN and frozen MiniLM.",
        "- Methods: Unified-Q-EF, CaT-CCTP, and GECC-hard-TAG; all are mechanism-adapted rows, not official end-to-end reproductions.",
        "- Access: current delta plus frozen synthetic prefix; cumulative 5% memory rate; report-dev evaluation only.",
        "",
        f"- Completed method cells: {len(cells) // 3} / 9.",
        f"- Audit problems: {len(problems)}.",
        "",
        "## Aggregate comparison",
        "",
        "Values are mean +/- sample SD across construction cells; Student seeds are averaged within each cell.",
        "",
        "| Method | Final Macro-F1 | Accuracy | Stage Macro-F1 | Stage Accuracy | Old F1 | New F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in grouped:
        lines.append(f"| {row['method']} | {fmt(row['final_f1'])} +/- {fmt(row['final_f1_sd'])} | {fmt(row['accuracy'])} +/- {fmt(row['accuracy_sd'])} | {fmt(row['stage_f1'])} | {fmt(row['stage_acc'])} | {fmt(row['old_f1'])} | {fmt(row['new_f1'])} |")
    lines += [
        "",
        "## Unified-Q-EF paired differences",
        "",
        "Positive values mean Unified-Q-EF is higher. Intervals are paired 95% t intervals over the nine construction cells (df=8); Student seeds are averaged within each cell.",
        "",
        "| Baseline | Pairs | F1 delta [95% CI] | Positive F1 | Worst F1 cell | Accuracy delta [95% CI] | Positive Accuracy | Worst Accuracy cell |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(f"| {row['baseline']} | {row['n_pairs']} | {fmt(row['f1_delta'])} [{fmt(row['f1_ci_low'])}, {fmt(row['f1_ci_high'])}] | {row['f1_positive_cells']}/9 ({row['f1_positive_ratio']:.0%}) | {fmt(row['f1_worst_cell'])} | {fmt(row['accuracy_delta'])} [{fmt(row['accuracy_ci_low'])}, {fmt(row['accuracy_ci_high'])}] | {row['accuracy_positive_cells']}/9 ({row['accuracy_positive_ratio']:.0%}) | {fmt(row['accuracy_worst_cell'])} |")
    lines += ["", "## Stage records", "", "| Order | Construction | Method | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
    by_cell = defaultdict(dict)
    for path in sorted(INPUT.glob("o*_c*/matrix/results.json")):
        order, seed = cell_id(path)
        payload = read(path)
        for row in payload.get("rows", []):
            if row["arm"] not in ARMS:
                continue
            by_cell[(order, seed, row["arm"])][row["stage"]] = row["score"]["final_f1"]
    for key in sorted(by_cell):
        order, seed, arm = key
        values = by_cell[key]
        rendered = " | ".join(f"{values.get(stage, float('nan')):.2f}" if stage in values else "--" for stage in range(5))
        lines.append(f"| {order} | {seed} | {DISPLAY[arm]} | {rendered} |")
    if problems:
        lines += ["", "## Audit problems", ""]
        lines.extend(f"- {problem}" for problem in problems)
    else:
        lines += ["", "## Audit result", "", "All nine cells are complete and satisfy the declared five-stage base+4 protocol."]
    lines += ["", "## Interpretation", "", "The base+4 panel is a protocol-unification and robustness panel. It should not overwrite the previously sealed three-stage ArXiv confirmation result; the two schedules answer different questions."]
    return "\n".join(lines) + "\n"


def plot(cells: list[dict], output: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return []
    paths = []
    grouped = defaultdict(list)
    for row in cells:
        grouped[row["arm"]].append(row)
    active_arms = [arm for arm in ARMS if grouped[arm]]
    if not active_arms:
        return []
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(active_arms))
    means = [mean(row["final_f1"] for row in grouped[arm]) for arm in active_arms]
    errors = [sample_std([row["final_f1"] for row in grouped[arm]]) for arm in active_arms]
    colors = {"uq_dm_edgefree": "#168aad", "cat_cctp": "#b7b7a4", "gecc_hard": "#6c757d"}
    ax.bar(x, means, yerr=errors, capsize=4, color=[colors[arm] for arm in active_arms])
    ax.set_xticks(x, [DISPLAY[arm] for arm in active_arms], rotation=15)
    ax.set_ylabel("Final Macro-F1 (%)")
    ax.set_title("ArXiv unified base+4 comparison")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    path = output / "final_f1.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for arm, color in zip(active_arms, ("#168aad", "#b7b7a4", "#6c757d")):
        values = defaultdict(list)
        for row in cells:
            if row["arm"] == arm:
                values[row["order_seed"]].append(row)
        stage_means = []
        # The aggregate cell table contains only final/stage means. Read stage data again.
        for stage in range(5):
            stage_values = []
            for result_path in sorted(INPUT.glob("o*_c*/matrix/results.json")):
                if "_failure_" in result_path.parent.parent.name:
                    continue
                payload = read(result_path)
                for result_row in payload.get("rows", []):
                    if result_row["arm"] == arm and result_row["stage"] == stage:
                        stage_values.append(result_row["score"]["final_f1"])
            stage_means.append(mean(stage_values) if stage_values else float("nan"))
        ax.plot(range(5), stage_means, marker="o", label=DISPLAY[arm], color=color)
    ax.set_xticks(range(5), ["0", "1", "2", "3", "4"])
    ax.set_xlabel("Continual stage")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("ArXiv base+4 stage trajectory")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    path = output / "stage_trajectory.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells, problems = load_cells()
    grouped, paired = aggregate(cells)
    write_csv(OUTPUT / "cells.csv", cells)
    write_csv(OUTPUT / "aggregate.csv", grouped)
    write_csv(OUTPUT / "paired_deltas.csv", paired)
    plots = plot(cells, OUTPUT)
    manifest = {"cells": cells, "aggregate": grouped, "paired": paired, "problems": problems, "plots": plots}
    (OUTPUT / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUTPUT / "summary.md").write_text(render(cells, grouped, paired, problems), encoding="utf-8")
    print(json.dumps({"completed_method_cells": len(cells), "problems": problems, "output": str(OUTPUT), "plots": plots}, indent=2))


if __name__ == "__main__":
    main()
