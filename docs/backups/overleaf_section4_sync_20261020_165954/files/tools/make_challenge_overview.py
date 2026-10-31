from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

DATASETS = ["ArXiv", "SportsFit", "BookHis"]
DATASET_COLORS = ["#2878A5", "#D9782F", "#27846F"]
INK = "#26343D"
MUTED = "#5C6A73"
RULE = "#D5DEE3"


def rounded_box(ax, x, y, width, height, face, edge):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.75, edgecolor=edge, facecolor=face,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)


def arrow(ax, start, end, color="#687780"):
    ax.annotate(
        "", xy=end, xytext=start, xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": color,
                     "shrinkA": 0, "shrinkB": 0},
    )


def data_table(ax, header_y, row_ys, labels, rows):
    columns = [0.625, 0.785, 0.945]
    for x, name, color in zip(columns, DATASETS, DATASET_COLORS):
        ax.text(x, header_y, name, ha="center", va="center",
                fontsize=6.2, fontweight="bold", color=color,
                transform=ax.transAxes)
    ax.plot([0.035, 0.975], [header_y - 0.027, header_y - 0.027],
            color=RULE, lw=0.6, transform=ax.transAxes)
    for y, label, values in zip(row_ys, labels, rows):
        ax.text(0.045, y, label, ha="left", va="center", fontsize=6.15,
                color=INK, transform=ax.transAxes)
        for x, value in zip(columns, values):
            ax.text(x, y, value, ha="center", va="center", fontsize=6.25,
                    color=INK, transform=ax.transAxes)


def draw_history_challenge(ax):
    ax.text(0.035, 0.965, "(a) Historical graph signal disappears at release",
            fontsize=8.4, fontweight="bold", color=INK, va="top",
            transform=ax.transAxes)

    rounded_box(ax, 0.035, 0.805, 0.235, 0.102, "#F0F6F9", "#9AB8C7")
    # A small graph glyph makes the source structural, not a plain text batch.
    edges = [((0.056, 0.833), (0.079, 0.866)),
             ((0.079, 0.866), (0.104, 0.836)),
             ((0.056, 0.833), (0.104, 0.836))]
    for p1, p2 in edges:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#7F9EAE", lw=0.8,
                transform=ax.transAxes)
    for x, y in [(0.056, 0.833), (0.079, 0.866), (0.104, 0.836)]:
        ax.scatter([x], [y], s=10, color="#2878A5", zorder=4,
                   transform=ax.transAxes)
    ax.text(0.123, 0.868, r"Current $\Delta_t$", fontsize=6.35, fontweight="bold",
            color=INK, va="center", transform=ax.transAxes)
    ax.text(0.123, 0.833, "graph + text", fontsize=6.0, color=MUTED,
            va="center", transform=ax.transAxes)

    rounded_box(ax, 0.365, 0.805, 0.205, 0.102, "#F0F7F3", "#9BBEAD")
    for i in range(3):
        ax.add_patch(Rectangle((0.382 + i * 0.009, 0.828 + i * 0.008),
                               0.045, 0.047, linewidth=0.45,
                               edgecolor="#7DA691", facecolor="#D9EBE1",
                               transform=ax.transAxes))
    ax.text(0.438, 0.867, "Write memory", fontsize=6.25, fontweight="bold",
            color=INK, va="center", transform=ax.transAxes)
    ax.text(0.438, 0.833, "before release", fontsize=5.9, color=MUTED,
            va="center", transform=ax.transAxes)

    arrow(ax, (0.278, 0.856), (0.350, 0.856))
    ax.plot([0.625, 0.625], [0.790, 0.922], color="#BD5268", lw=0.9,
            linestyle=(0, (2, 2)), transform=ax.transAxes)
    ax.text(0.625, 0.922, "release", ha="center", va="bottom", fontsize=5.7,
            color="#A74459", transform=ax.transAxes)
    arrow(ax, (0.579, 0.856), (0.665, 0.856), color="#A74459")

    rounded_box(ax, 0.685, 0.805, 0.285, 0.102, "#FAF7F2", "#D0B999")
    ax.text(0.827, 0.871, r"Stage $t+1$", ha="center", fontsize=6.3,
            fontweight="bold", color=INK, va="center", transform=ax.transAxes)
    ax.text(0.827, 0.839, r"current $\Delta$ + frozen $M_t$", ha="center",
            fontsize=5.8, color=MUTED, va="center", transform=ax.transAxes)
    ax.text(0.827, 0.812, "old raw data unavailable", ha="center",
            fontsize=5.5, color="#A74459", va="center", transform=ax.transAxes)

    data_table(
        ax, 0.735, [0.681, 0.628],
        ["No graph match: Macro-F1 drop (pp)", "5% memory / full-history F1 (%)"],
        [["1.91", "14.74", "1.30"], ["99.04", "98.23", "103.33"]],
    )
    ax.plot([0.035, 0.975], [0.565, 0.565], color=RULE, lw=0.8,
            transform=ax.transAxes)


def draw_text_challenge(ax):
    ax.text(0.035, 0.515, "(b) A pooled feature is not a readable text",
            fontsize=8.4, fontweight="bold", color=INK, va="top",
            transform=ax.transAxes)

    rounded_box(ax, 0.035, 0.375, 0.235, 0.095, "#F7F8F9", "#B5C0C7")
    for i, y in enumerate([0.395, 0.420, 0.445]):
        ax.scatter([0.058], [y], s=9, color=DATASET_COLORS[i], zorder=4,
                   transform=ax.transAxes)
        ax.plot([0.070, 0.104], [y, y], color=DATASET_COLORS[i], lw=1.4,
                transform=ax.transAxes)
        ax.add_patch(Rectangle((0.112, y - 0.007), 0.034, 0.014,
                               linewidth=0.45, edgecolor=DATASET_COLORS[i],
                               facecolor="white", transform=ax.transAxes))
    ax.text(0.159, 0.446, "source pairs", fontsize=5.9, fontweight="bold",
            color=INK, va="center", transform=ax.transAxes)
    ax.text(0.159, 0.416, r"$(x_i, t_i)$", fontsize=5.8, color=MUTED,
            va="center", transform=ax.transAxes)

    arrow(ax, (0.278, 0.421), (0.350, 0.421))
    rounded_box(ax, 0.365, 0.375, 0.175, 0.095, "#F0F6F9", "#9AB8C7")
    ax.text(0.452, 0.451, "learn Q", ha="center", fontsize=6.0,
            fontweight="bold", color=INK, va="center", transform=ax.transAxes)
    for r in range(3):
        for c in range(3):
            ax.add_patch(Rectangle((0.407 + c * 0.018, 0.389 + r * 0.016),
                                   0.015, 0.013, linewidth=0.35,
                                   edgecolor="#90ADBA",
                                   facecolor=("#6C9FB5" if (r + c) % 2 == 0 else "#DDEAF0"),
                                   transform=ax.transAxes))

    arrow(ax, (0.548, 0.421), (0.616, 0.421))
    ax.plot([0.620, 0.640], [0.421, 0.421], color="#687780", lw=0.8,
            transform=ax.transAxes)
    ax.plot([0.640, 0.640], [0.393, 0.449], color="#687780", lw=0.8,
            transform=ax.transAxes)
    for y, label, edge, face in [
        (0.432, "pooled feature", "#8EB9A8", "#F0F7F3"),
        (0.378, "selected real text", "#D7B27D", "#FBF5EC"),
    ]:
        rounded_box(ax, 0.665, y, 0.305, 0.038, face, edge)
        ax.text(0.817, y + 0.019, label, ha="center", va="center",
                fontsize=5.9, color=INK, transform=ax.transAxes)
        arrow(ax, (0.640, y + 0.019), (0.657, y + 0.019))

    data_table(
        ax, 0.285, [0.231, 0.178],
        [r"Fixed $Q$: Macro-F1 drop (pp)", "Soft/hard text cosine gap"],
        [["12.76", "21.67", "8.91"], ["0.385", "0.204", "0.469"]],
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    fig = plt.figure(figsize=(3.35, 3.60), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_history_challenge(ax)
    draw_text_challenge(ax)
    fig.savefig(OUT / "challenge_overview.png", dpi=360, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / "challenge_overview.pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    main()
