"""Plot portable summary JSON without importing training dependencies."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    args = ap.parse_args()
    source = Path(args.summary)
    rows = json.loads(source.read_text())["rows"]
    out = source.parent
    lookup = {(r["dataset"], r["arm"]): r for r in rows}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    styles = {"v4": ("#007f8b", "v4"), "k_center": ("#7a5195", "k-center"),
              "raw_long_selected": ("#555555", "Full data, selector checkpoint"),
              "v4_init_finetune": ("#bc5090", "v4 initialization + raw fine-tune"),
              "init_selfloop": ("#e49a00", "Synthetic self-loop control"),
              "tagsam_semantic": ("#d14836", "TAGSAM semantic adapter")}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, constrained_layout=True)
    for col, dataset in enumerate(("cora", "citeseer")):
        for row, scope in enumerate(("all", "seen")):
            axis = axes[row, col]
            for arm, (color, label) in styles.items():
                item = lookup[dataset, arm]
                axis.plot([1, 2, 3], [item[f"s{s}_{scope}"] for s in range(3)],
                          "o-", color=color, label=label, linewidth=1.6, markersize=4)
            axis.set_title(f"{dataset.capitalize()} | {scope}-class prompts")
            axis.set_xticks([1, 2, 3])
            axis.set_ylim(0, 100)
            axis.grid(axis="y", alpha=.18)
            if col == 0:
                axis.set_ylabel("Seen-sample Macro-F1 (%)")
            if row == 1:
                axis.set_xlabel("Stage")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"stage_curves.{suffix}", dpi=180)
    plt.close(fig)
    arms = ["v4", "dispersion", "concat64", "k_center", "raw_equal", "raw_long",
            "raw_long_selected", "raw_finetune", "v4_init_finetune", "init_selfloop",
            "tagsam_selfloop", "tagsam_semantic", "asm_rowcol_semantic"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes, ("cora", "citeseer")):
        items = [lookup[dataset, a] for a in arms]
        bars = axis.barh(range(len(items)), [r["final_mean"] for r in items],
                         xerr=[r["final_std"] for r in items],
                         color=["#007f8b" if a == "v4" else "#929a9e" for a in arms], capsize=2)
        for bar, item in zip(bars, items):
            axis.text(item["final_mean"] + item["final_std"] + .8,
                      bar.get_y() + bar.get_height() / 2, f'{item["final_mean"]:.2f}', va="center", fontsize=9)
        axis.set_title(dataset.capitalize())
        axis.set_yticks(range(len(arms)), arms)
        axis.set_xlabel("Final Macro-F1 (%) +/- student SD")
        axis.set_xlim(0, 75)
        axis.grid(axis="x", alpha=.18)
    axes[0].invert_yaxis()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"final_comparison.{suffix}", dpi=180)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
