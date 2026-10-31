"""Figures sourced from the audited external-baseline summary."""
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/external_continual_adapters_20260927/summary.json"
OUT = ROOT / "docs/paper/assets"
NAMES = {
    "semantic_recursive": "Semantic-Recursive",
    "cat_mmd_tag": "CaT-MMD-TAG*",
    "gecc_hard_tag": "GECC-hard-TAG*",
    "k_center_native": "k-center",
    "medoid_native": "medoid",
}


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    if data["fit_count"] != 360 or data["previous_sr_max_metric_error_pp"] != 0:
        raise ValueError("figure requires audited complete matrix and unchanged SR")
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    rows = data["overall"]
    colors = ["#007f78", "#b44d49", "#807e88", "#376aa6", "#819445"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9), dpi=220)
    for ax, metric, title in zip(axes, ("final_f1", "final_accuracy"),
                               ("Final Macro-F1", "Final Accuracy")):
        ax.barh(np.arange(5), [r[metric] for r in rows], color=colors, height=.62)
        ax.set_yticks(np.arange(5), [NAMES[r["arm"]] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 70)
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=.18)
        ax.set_xlabel("%")
        for y, r in enumerate(rows):
            ax.text(r[metric] + .8, y, f"{r[metric]:.2f}", va="center", fontsize=10)
    fig.suptitle("两图 × 两顺序的等权均值：主方法冻结，外部方法独立构图", fontsize=13)
    fig.text(.5, .01, "每个条件 6 个配对 Student seeds；* 表示公开机制的 TAG 适配，非官方完整复现。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94), w_pad=2)
    fig.savefig(OUT / "external_continual_20260927_means.png", bbox_inches="tight")
    plt.close(fig)

    cells = [(d, o) for d in ("cora", "citeseer") for o in ("order19611", "order19612")]
    baselines = [a for a in NAMES if a != "semantic_recursive"]
    pair = {(r["dataset"], r["order"], r["baseline"]): r for r in data["paired_cell_results"]}
    values = np.array([[pair[d, o, a]["conditional_seed_ci"]["mean"] for a in baselines]
                       for d, o in cells])
    fig, ax = plt.subplots(figsize=(9.4, 3.8), dpi=220)
    im = ax.imshow(values, cmap="RdBu", vmin=-11, vmax=11, aspect="auto")
    ax.set_xticks(range(4), [NAMES[a] for a in baselines])
    ax.set_yticks(range(4), [f"{d.title()} / {o[5:]}" for d, o in cells])
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f"{value:+.2f}", ha="center", va="center",
                color="white" if abs(value) > 6 else "#202124", fontsize=13)
    ax.set_title("逐条件 Final F1：Semantic-Recursive 减去外部基线", pad=13)
    ax.tick_params(length=0, pad=8)
    fig.colorbar(im, ax=ax, label="百分点；正值为 SR 较高", fraction=.04, pad=.04)
    fig.text(.5, .01, "固定 5% growing rate；同一条件内 rows / Student 相同，不同顺序的早期预算不同。",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUT / "external_continual_20260927_cell_deltas.png", bbox_inches="tight")
    plt.close(fig)
    print("Created two figures from audited summary.json")


if __name__ == "__main__":
    main()
