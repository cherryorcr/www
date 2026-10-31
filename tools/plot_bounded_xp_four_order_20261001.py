"""Plot the frozen four-order small-graph Final F1 comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "bounded_xp_final_evidence_20261001"
OUT = EVIDENCE / "bounded_xp_four_order_final_f1.png"


def main() -> None:
    payload = json.loads((EVIDENCE / "four_order_pooled_summary.json").read_text(encoding="utf-8"))
    methods = payload["protocol"]["methods"]
    orders = payload["protocol"]["orders"]
    labels = [order.replace("order", "") for order in orders]
    cells = {(row["dataset"], row["order"], row["method"]): row for row in payload["cells"]}
    colors = {
        "Bounded X+P": "#1f77b4",
        "CaT-MMD-TAG adapted": "#d62728",
        "GECC-hard-TAG adapted": "#2ca02c",
        "k-center/FPS": "#9467bd",
        "medoid": "#ff7f0e",
    }
    short = {
        "Bounded X+P": "X+P",
        "CaT-MMD-TAG adapted": "CaT",
        "GECC-hard-TAG adapted": "GECC",
        "k-center/FPS": "k-center",
        "medoid": "medoid",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, ("cora", "citeseer")):
        for method in methods:
            values = [cells[(dataset, order, method)]["final_f1"] for order in orders]
            ax.plot(labels, values, marker="o", linewidth=2, label=short[method], color=colors[method])
        ax.set_title(dataset.capitalize())
        ax.set_xlabel("Category order")
        ax.set_ylabel("Final Macro-F1" if dataset == "cora" else "")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(40, 65)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Bounded X+P and adapted baselines across four category orders", y=1.14)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
