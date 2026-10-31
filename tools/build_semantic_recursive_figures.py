from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

FONT = r"C:\Windows\Fonts\msyh.ttc"
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False


def add_box(ax, xy, text, color, width=2.25, height=0.8, fontsize=11):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.4, edgecolor="#46515c", facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
    return (x + width, y + height / 2)


def arrow(ax, start, end, color="#46515c", style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=16,
                                 linewidth=1.4, color=color))


def pipeline():
    fig, ax = plt.subplots(figsize=(13, 4.6), dpi=180)
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.text(0.2, 4.25, "Semantic-Recursive 的持续访问边界与递归追加流程", fontsize=16, weight="bold")

    add_box(ax, (0.25, 2.8), "历史 raw\n禁止回读", "#F6C7C7", width=2.0, height=0.9, fontsize=11)
    add_box(ax, (0.25, 1.05), "当前 raw delta_t\n可访问", "#B9DDF5", width=2.0, height=0.9, fontsize=11)
    add_box(ax, (2.8, 1.05), "图文联合\nsource slots", "#C7E8C4", width=2.1, height=0.9, fontsize=11)
    add_box(ax, (5.45, 1.05), "预算守恒\ntwo-prototype refinement", "#F7DDA6", width=2.35, height=0.9, fontsize=10.5)
    add_box(ax, (8.35, 1.05), "完整 payload\nsemantic assignment", "#DCC9F2", width=2.2, height=0.9, fontsize=10.5)
    add_box(ax, (11.0, 1.05), "append 到\nparent memory", "#F4C89C", width=1.95, height=0.9, fontsize=10.5)
    add_box(ax, (4.2, 2.8), "上一阶段\nsynthetic parent", "#C8E5F5", width=2.1, height=0.9, fontsize=11)
    add_box(ax, (7.05, 2.8), "固定 C0\ntopology", "#D9E1F2", width=1.9, height=0.9, fontsize=11)
    add_box(ax, (9.55, 2.8), "原始两层 GCN\n+ InfoNCE Student", "#F6D6A8", width=2.55, height=0.9, fontsize=10.5)

    arrow(ax, (2.25, 1.5), (2.8, 1.5))
    arrow(ax, (4.9, 1.5), (5.45, 1.5))
    arrow(ax, (7.8, 1.5), (8.35, 1.5))
    arrow(ax, (10.55, 1.5), (11.0, 1.5))
    arrow(ax, (6.3, 2.8), (6.3, 1.95), color="#4e79a7")
    arrow(ax, (5.25, 3.25), (7.05, 3.25), color="#4e79a7")
    arrow(ax, (8.95, 3.25), (9.55, 3.25), color="#4e79a7")
    ax.text(0.35, 0.35, "核心不变量：fixed-rate budget  |  recursive prefix  |  payload multiset  |  historical raw reads = 0",
            fontsize=10.5, color="#39424e")
    fig.tight_layout()
    fig.savefig(OUT / "semantic_recursive_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def result_summary():
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), dpi=180)
    fig.suptitle("Semantic-Recursive 的正向信号与顺序边界", fontsize=16, weight="bold")

    # Small unified result.
    ax = axes[0, 0]
    labels = ["Final Macro-F1", "Accuracy"]
    c0 = [52.09, 59.22]
    sr = [53.02, 59.59]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, c0, width=0.36, label="C0", color="#9aa5b1")
    ax.bar(x + 0.18, sr, width=0.36, label="Semantic-Recursive", color="#4e79a7")
    ax.set_xticks(x, labels)
    ax.set_title("小图统一结果（四个 dataset-order cell）")
    ax.set_ylabel("百分制")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # Large canonical incremental deltas.
    ax = axes[0, 1]
    labels = ["arXiv", "BookChild", "SportsFit"]
    vals = [0.637, 0.203, 1.735]
    colors = ["#4e79a7", "#59a14f", "#f28e2b"]
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("大图 canonical：相对 C0 的增量平均")
    ax.set_ylabel("Macro-F1 delta (pp)")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.06, f"{v:+.3f}", ha="center", fontsize=9)

    # Independent order deltas.
    ax = axes[1, 0]
    vals = [-0.373, 0.402, 0.030]
    ax.bar(labels, vals, color=["#d95f59", "#59a14f", "#bab0ab"])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("独立 order：相对 C0 的增量")
    ax.set_ylabel("Macro-F1 delta (pp)")
    ax.set_ylim(-0.5, 0.65)
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.06 if v >= 0 else -0.14), f"{v:+.3f}", ha="center", fontsize=9)

    # Old/new tradeoff on canonical order.
    ax = axes[1, 1]
    x = np.arange(3)
    old = [-0.157, -0.361, 2.158]
    new = [2.971, 2.241, 0.080]
    ax.bar(x - 0.18, old, width=0.36, label="old-class", color="#76b7b2")
    ax.bar(x + 0.18, new, width=0.36, label="new-class", color="#e15759")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_title("canonical：old/new trade-off")
    ax.set_ylabel("Macro-F1 delta (pp)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.88, wspace=0.28, hspace=0.52)
    fig.savefig(OUT / "semantic_recursive_result_summary.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    result_summary()
    print(OUT)
