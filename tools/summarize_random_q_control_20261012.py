"""Summarize the literal random-Q attribution control without touching sealed tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "unified_q_overnight_20261012"
RAW = ROOT / "tmp_remote_results" / "random_q_control_20261012"


def load_random_rows():
    rows = []
    for dataset in ("sportsfit", "bookhis", "wikics"):
        path = RAW / f"{dataset}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        final = max(data["rows"], key=lambda row: int(row["stage"]))
        score = final["score"]
        rows.append({
            "dataset": dataset,
            "method": "random-Q",
            "order": 9101,
            "construction_seed": 271006,
            "final_f1": float(score["final_f1"]),
            "accuracy": float(score["accuracy"]),
            "old_f1": score.get("old_f1"),
            "new_f1": score.get("new_f1"),
            "stages": len(data["rows"]),
            "source": str(path),
        })
    return rows


def load_primary_rows():
    rows = []
    with (EVIDENCE / "construction_cells.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["order"] == "9101" and row["construction_seed"] == "271006":
                if row["method"] in {"Unified-Q-EF", "CaT-CCTP", "GECC-hard-TAG"}:
                    rows.append({
                        "dataset": row["dataset"],
                        "method": row["method"],
                        "order": int(row["order"]),
                        "construction_seed": int(row["construction_seed"]),
                        "final_f1": float(row["final_f1"]),
                        "accuracy": float(row["accuracy"]),
                        "old_f1": float(row["old_f1"]),
                        "new_f1": float(row["new_f1"]),
                        "stages": 5,
                        "source": row["source"],
                    })
    return rows


def main():
    rows = load_primary_rows() + load_random_rows()
    out_csv = EVIDENCE / "random_q_comparison.csv"
    fields = ["dataset", "method", "order", "construction_seed", "final_f1",
              "accuracy", "old_f1", "new_f1", "stages", "source"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Literal Random-Q Attribution Control (2026-10-12)",
        "",
        "This is a reference-order diagnostic, not a sealed primary comparison.",
        "The control keeps class legality and balanced Sinkhorn capacity, but replaces the structured feature/message/text initialization with random legal logits, freezes Q, and disables residual updates.",
        "It is therefore distinct from the existing random real-node coreset control.",
        "",
        "| Dataset | CaT-CCTP F1 / Acc | GECC-hard-TAG F1 / Acc | Unified-Q-EF F1 / Acc | random-Q F1 / Acc | UQ - random-Q F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    by = {}
    for row in rows:
        by.setdefault(row["dataset"], {})[row["method"]] = row
    for dataset in ("sportsfit", "bookhis", "wikics"):
        item = by[dataset]
        uq = item["Unified-Q-EF"]
        rq = item["random-Q"]
        def fmt(name):
            row = item[name]
            return f"{row['final_f1']:.2f} / {row['accuracy']:.2f}"
        lines.append(f"| {dataset} | {fmt('CaT-CCTP')} | {fmt('GECC-hard-TAG')} | {fmt('Unified-Q-EF')} | {fmt('random-Q')} | {uq['final_f1'] - rq['final_f1']:+.2f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The random-Q control is substantially below Unified-Q-EF on all three reference-order datasets. This supports the claim that the learned, structure/text-conditioned assignment is meaningful. It does not establish universal SOTA, and it should not be merged into the 3-order/3-construction-seed sealed stability table.",
        "",
        "The comparison is intentionally labeled `random-Q`, not `random coreset`: the latter selects real nodes and is a different control.",
        "",
    ]
    (EVIDENCE / "random_q_control.md").write_text("\n".join(lines), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        datasets = ["SportsFit", "BookHis", "WikiCS"]
        lookup = {row["dataset"]: row for row in load_random_rows()}
        primary = {row["dataset"]: row for row in load_primary_rows()}
        x = np.arange(len(datasets))
        width = 0.36
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        uq = [primary[d]["final_f1"] for d in ("sportsfit", "bookhis", "wikics")]
        rq = [lookup[d.lower()]["final_f1"] for d in datasets]
        ax.bar(x - width / 2, uq, width, label="Unified-Q-EF", color="#2f6f9f")
        ax.bar(x + width / 2, rq, width, label="random-Q", color="#b5b9c2")
        ax.set_xticks(x, datasets)
        ax.set_ylabel("Final Macro-F1 (%)")
        ax.set_title("Learned assignment versus literal random-Q")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(EVIDENCE / "random_q_comparison.png", dpi=180)
        plt.close(fig)
    except Exception:
        pass
    print((EVIDENCE / "random_q_control.md").as_posix())


if __name__ == "__main__":
    main()
