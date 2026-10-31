"""Draw the method's source support, paired construction, and update paths."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from diagram_primitives import (
    BLUE, INK, LINE, MUTED, ORANGE, RED, TEAL,
    document, feature_text_rows, label, matrix, rect, save_figure,
)


def flow(ax, points, color=INK, kind="data", lw=0.9):
    styles = {"data": "-", "init": (0, (1.2, 2.1)),
              "update": (0, (3.1, 2.0)), "memory": "-"}
    width = 1.3 if kind == "memory" else lw
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color,
                lw=width, linestyle=styles[kind], zorder=1,
                solid_capstyle="round")
    ax.add_patch(patches.FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=6.8,
        color=color, linewidth=width, linestyle=styles[kind], zorder=2,
    ))


def note(ax, x, y, text, color=MUTED, size=7.1, **kwargs):
    return label(ax, x, y, text, size=size, color=color,
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
                 **kwargs)


def vector(ax, x, y, values, color=BLUE, width=0.43, height=0.14):
    rgb = np.asarray(colors.to_rgb(color))
    for i, value in enumerate(values):
        face = 1 - (0.15 + 0.72 * float(value)) * (1 - rgb)
        rect(ax, x + i * width / len(values), y, width / len(values),
             height, face, "white", 0.35)
    rect(ax, x, y, width, height, "none", color, 0.6)


def current_graph(ax):
    positions = np.array([[.28, 3.14], [.57, 3.32], [.78, 3.05],
                          [.25, 2.89], [.62, 2.82], [.98, 3.25]])
    for i, j in ((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (1, 5), (3, 4)):
        ax.plot(positions[[i, j], 0], positions[[i, j], 1],
                color=LINE, lw=0.85, zorder=0)
    for i, (x, y) in enumerate(positions):
        ax.add_patch(patches.Circle((x, y), 0.063,
                     facecolor=TEAL if i < 3 else BLUE,
                     edgecolor="white", lw=0.55, zorder=3))
        label(ax, x, y, str(i + 1), size=6.6, color="white", zorder=4)


def class_statistics(ax):
    # Schematic class means in a fixed projected feature space.
    for cx, cy, color in ((5.63, 2.98, BLUE), (6.16, 3.10, TEAL)):
        for dx, dy in ((-.10, .00), (.03, -.07), (.08, .08)):
            ax.scatter(cx + dx, cy + dy, s=7, c=color, alpha=.38,
                       edgecolors="none", zorder=3)
        ax.scatter(cx, cy, s=17, c=color, marker="o", zorder=4)
        ax.scatter(cx + .15, cy - .03, s=23, marker="s",
                   facecolors="white", edgecolors=color, linewidths=.8, zorder=4)
        flow(ax, [(cx + .13, cy - .03), (cx + .025, cy)], color=color, lw=.65)


def legend_item(ax, x, text, color, kind):
    flow(ax, [(x, .12), (x + .27, .12)], color=color, kind=kind, lw=.8)
    label(ax, x + .33, .12, text, size=7.0, color=MUTED, ha="left")


def check_bounds(fig, ax):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for text in ax.texts:
        box = text.get_window_extent(renderer)
        if box.x0 < canvas.x0 or box.x1 > canvas.x1 or box.y0 < canvas.y0 or box.y1 > canvas.y1:
            raise ValueError(f"Figure label outside canvas: {text.get_text()}")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })
    fig = plt.figure(figsize=(7.45, 3.95))
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 7.45), ylim=(0, 3.95))
    ax.axis("off")

    for x in (2.25, 5.02):
        ax.plot([x, x], [1.11, 3.60], color="#D5DCE0", lw=.7)
    label(ax, .14, 3.77, "(a) Support learning", size=9.4, weight="bold", ha="left")
    label(ax, 2.42, 3.77, "(b) Joint construction", size=9.4, weight="bold", ha="left")
    label(ax, 5.18, 3.77, "(c) Matching and update", size=9.4, weight="bold", ha="left")

    # (a) Three current-data views initialize the scores; Sinkhorn balances Q.
    current_graph(ax)
    label(ax, .60, 3.52, "Current graph", size=7.6, color=BLUE)
    for dx, dy in ((.00, .00), (.21, -.06), (.42, .03)):
        document(ax, 1.27 + dx, 2.99 + dy, .20, .28, ORANGE)
    label(ax, 1.58, 3.52, "Source texts", size=7.6, color=ORANGE)
    flow(ax, [(.47, 2.81), (.47, 2.66)], color=BLUE)
    flow(ax, [(.99, 2.99), (1.08, 2.88), (1.08, 2.66)], color=BLUE)
    flow(ax, [(1.60, 2.88), (1.73, 2.88), (1.73, 2.66)], color=ORANGE)
    vector(ax, .25, 2.49, [.2, .8, .5, .7])
    vector(ax, .86, 2.49, [.6, .5, .8, .3])
    vector(ax, 1.51, 2.49, [.8, .3, .5, .7], color=ORANGE)
    label(ax, .47, 2.36, "Feature", size=7.2, color=BLUE)
    label(ax, 1.08, 2.36, "Neighbors", size=7.2, color=BLUE)
    label(ax, 1.73, 2.36, "Text", size=7.2, color=ORANGE)
    note(ax, 1.73, 2.78, "encode", color=ORANGE, size=6.6)
    for x in (.47, 1.08, 1.73):
        ax.plot([x, x], [2.25, 2.19], color=MUTED, lw=.8, linestyle=(0, (1.2, 2.1)))
    ax.plot([.47, 1.73], [2.19, 2.19], color=MUTED, lw=.8, linestyle=(0, (1.2, 2.1)))
    flow(ax, [(.47, 2.19), (.47, 1.40)], color=MUTED, kind="init")
    note(ax, .47, 1.97, "Prototype\nsimilarity", size=7.3)
    note(ax, .47, 1.58, "initialize", size=7.0)
    rect(ax, .15, 1.13, .66, .25, "#FCF1F4", RED, .8)
    label(ax, .48, 1.255, r"Scores $Z$", size=7.8, color=RED)

    q = np.array([[.80, .20, 0, 0], [.45, .55, 0, 0], [.25, .75, 0, 0],
                  [0, 0, .70, .30], [0, 0, .55, .45], [0, 0, .25, .75]])
    assert np.allclose(q.sum(axis=1), 1)
    assert np.allclose(q.sum(axis=0), 1.5)
    matrix(ax, 1.20, 1.30, .84, .66, q,
           [TEAL] * 3 + [BLUE] * 3, mask=q > 0)
    rect(ax, 1.20, 1.30, .21, .66, "none", TEAL, 1.2)
    label(ax, 1.62, 2.08, r"Support $Q$", size=8.4, weight="bold", color=TEAL)
    flow(ax, [(.83, 1.255), (.94, 1.255), (.94, 1.76), (1.17, 1.76)], color=TEAL)
    note(ax, .94, 1.49, "Sinkhorn", size=6.7, rotation=90)

    # (b) One source column drives the two payloads. Example values are schematic.
    label(ax, 3.59, 3.54, "One slot, the same source weights", size=7.8, color=TEAL)
    flow(ax, [(2.07, 1.83), (2.33, 1.83), (2.33, 2.92), (2.42, 2.92)], color=TEAL)
    source_x = np.array([[.20, .85, .40, .65], [.70, .30, .55, .40], [.40, .50, .90, .20]])
    weights = q[:3, 0]
    pooled = weights @ source_x / weights.sum()
    residual = np.array([.02, -.01, .015, .005])
    assert np.linalg.norm(residual) <= .05 * np.linalg.norm(pooled)
    matrix(ax, 2.46, 2.67, .18, .49, weights[:, None], [TEAL] * 3)
    matrix(ax, 2.70, 2.67, .43, .49, source_x, [BLUE] * 3)
    label(ax, 2.55, 3.29, r"$q_j$", size=8.0, color=TEAL)
    label(ax, 2.92, 3.29, r"$X_t$", size=8.0, color=BLUE)
    for i, weight in enumerate(weights):
        label(ax, 2.55, 3.078 - .1633 * i, f"{weight:.2f}", size=5.9, color=INK)
    flow(ax, [(3.16, 2.92), (3.43, 2.92)], color=BLUE)
    note(ax, 3.30, 3.10, "pool", color=BLUE)
    vector(ax, 3.45, 2.85, pooled, width=.39)
    label(ax, 3.65, 2.70, "mean", size=7.1, color=BLUE)
    label(ax, 3.99, 2.92, "+", size=12, color=INK)
    flow(ax, [(4.10, 2.92), (4.30, 2.92)], color=BLUE)
    vector(ax, 4.32, 2.85, pooled + residual, width=.42)
    label(ax, 4.53, 2.70, r"Feature $x_j^+$", size=7.5, color=BLUE)
    rect(ax, 3.77, 3.12, .45, .24, "#FCF1F4", RED, .8)
    label(ax, 3.995, 3.24, r"$R$", size=8.8, color=RED)
    flow(ax, [(3.995, 3.10), (3.995, 3.015)], color=RED)

    # The documents use the same relative support, with the maximum selected.
    flow(ax, [(2.33, 2.06), (2.44, 2.06)], color=TEAL)
    for i, x in enumerate((2.50, 2.84, 3.18)):
        document(ax, x, 1.88, .21, .29, ORANGE)
        rect(ax, x, 2.23, .21 * weights[i] / weights.max(), .065, TEAL, "none", 0)
        label(ax, x + .105, 1.77, rf"$t_{i+1}$", size=7.4, color=ORANGE)
    rect(ax, 2.47, 1.85, .27, .35, "none", ORANGE, 1.1)
    flow(ax, [(3.45, 2.06), (4.25, 2.06)], color=ORANGE)
    note(ax, 3.86, 2.20, "select largest", color=ORANGE, size=7.1)
    document(ax, 4.29, 1.91, .25, .33, ORANGE)
    label(ax, 4.42, 1.77, r"Real text $t_1$", size=7.4, color=ORANGE)
    label(ax, 3.87, 1.92, "forward", size=7.0, color=ORANGE)
    for x in (2.605, 2.945, 3.285):
        ax.plot([x, x], [1.68, 1.64], color=ORANGE, lw=.7)
    ax.plot([2.605, 3.285], [1.64, 1.64], color=ORANGE, lw=.7)
    flow(ax, [(2.96, 1.64), (2.96, 1.48), (3.18, 1.48)], color=ORANGE)
    vector(ax, 3.21, 1.41, [.65, .4, .58, .55], color=ORANGE, width=.39)
    label(ax, 3.40, 1.25, "soft mean", size=7.3, color=ORANGE)
    flow(ax, [(5.12, 1.48), (3.65, 1.48)], color=RED, kind="update")
    note(ax, 4.40, 1.57, "text gradient", color=RED, size=7.1)
    flow(ax, [(3.18, 1.48), (2.88, 1.48), (2.88, 1.10), (1.62, 1.10), (1.62, 1.27)],
         color=RED, kind="update", lw=.8)
    note(ax, 2.24, 1.10, "soft backward", color=RED, size=7.0)

    # Candidate feature/text outputs are paired, then evaluated in panel (c).
    ax.plot([4.82, 4.88, 4.88, 4.82], [2.94, 2.94, 2.03, 2.03], color=TEAL, lw=.9)
    flow(ax, [(4.89, 2.56), (5.13, 2.56)], color=TEAL)
    note(ax, 4.51, 2.46, "evaluate", color=TEAL, size=6.9)

    # (c) Each loss has its own reference: class means, messages, or gradients.
    label(ax, 5.20, 3.29, "Class statistics", size=8.0, weight="bold", ha="left")
    class_statistics(ax)
    label(ax, 6.50, 3.04, "DM", size=8.0, color=BLUE)
    ax.scatter(5.35, 2.78, s=15, c=BLUE, marker="o")
    label(ax, 5.44, 2.78, "current", size=6.9, ha="left", color=MUTED)
    ax.scatter(6.13, 2.78, s=19, facecolors="white", edgecolors=BLUE, marker="s", linewidths=.8)
    label(ax, 6.23, 2.78, "memory", size=6.9, ha="left", color=MUTED)
    ax.plot([5.20, 7.05], [2.63, 2.63], color="#E0E5E8", lw=.65)

    label(ax, 5.20, 2.47, "Neighborhood targets", size=8.0, weight="bold", ha="left")
    vector(ax, 5.36, 2.17, [.6, .8, .4, .5], width=.43)
    vector(ax, 6.17, 2.17, [.57, .76, .45, .52], color=TEAL, width=.43)
    flow(ax, [(6.12, 2.24), (5.85, 2.24)], color=TEAL, lw=.8)
    label(ax, 5.57, 2.05, "pooled messages", size=6.7, color=MUTED)
    label(ax, 6.41, 2.05, "slot feature", size=6.7, color=MUTED)
    ax.plot([5.20, 7.05], [1.90, 1.90], color="#E0E5E8", lw=.65)

    label(ax, 5.20, 1.75, "Proxy gradients", size=8.0, weight="bold", ha="left")
    rect(ax, 5.32, 1.22, .82, .36, "#F4F6F7", LINE, .7)
    label(ax, 5.73, 1.40, "Fixed proxy", size=7.4)
    flow(ax, [(6.19, 1.39), (6.38, 1.39)], color=MUTED)
    flow(ax, [(6.44, 1.25), (6.70, 1.56)], color=BLUE, lw=1.0)
    flow(ax, [(6.44, 1.25), (6.83, 1.46)], color=TEAL, lw=1.0)
    ax.add_patch(patches.Arc((6.44, 1.25), .36, .36, theta1=28, theta2=50,
                            edgecolor=RED, lw=.8))
    label(ax, 6.89, 1.71, "GM", size=7.7, color=BLUE)
    flow(ax, [(5.30, 1.48), (5.14, 1.48)], color=RED, kind="update")

    # Explicit parameter-update loops avoid confusing proxy gradients with proxy training.
    ax.plot([7.10, 7.18, 7.18, 7.10], [3.17, 3.17, 1.25, 1.25], color=RED, lw=.85)
    flow(ax, [(7.18, 1.25), (7.18, .91), (.48, .91), (.48, 1.10)],
         color=RED, kind="update")
    note(ax, .48, .91, "update scores", color=RED, size=7.0)
    flow(ax, [(7.18, 3.17), (7.18, 3.46), (4.00, 3.46), (4.00, 3.38)],
         color=RED, kind="update")
    note(ax, 5.95, 3.46, "update residual", color=RED, size=7.0)

    # A separate stage strip shows the persistent result, not optimization intermediates.
    ax.plot([.14, 7.22], [.76, .76], color="#D5DCE0", lw=.7)
    label(ax, .16, .49, "After\noptimization", size=7.4, weight="bold", ha="left")
    feature_text_rows(ax, 1.23, .31, .71, .35, [BLUE, TEAL], gray=True)
    label(ax, 1.585, .23, r"$M_{t-1}$", size=7.8, color=MUTED)
    label(ax, 2.12, .49, "+", size=12, color=MUTED)
    feature_text_rows(ax, 2.31, .31, .71, .35, [BLUE, TEAL])
    label(ax, 2.665, .23, "new paired rows", size=7.1, color=TEAL)
    flow(ax, [(3.13, .49), (3.75, .49)], color=TEAL, kind="memory")
    note(ax, 3.45, .62, "append", color=TEAL, size=7.1)
    feature_text_rows(ax, 3.88, .29, .81, .40, [BLUE, TEAL, BLUE, TEAL])
    label(ax, 4.90, .49, r"$M_t$", size=8.4, color=TEAL)
    flow(ax, [(5.18, .49), (5.72, .49)], color=TEAL, kind="memory")
    label(ax, 5.86, .49, "Next stage / student training", size=7.6, ha="left")

    legend_item(ax, .30, "data", INK, "data")
    legend_item(ax, 1.49, "initialization", MUTED, "init")
    legend_item(ax, 3.18, "optimization feedback", RED, "update")
    legend_item(ax, 5.65, "saved memory", TEAL, "memory")
    check_bounds(fig, ax)
    save_figure(fig, Path(__file__).resolve().parents[1] / "figures", "method_overview")
    plt.close(fig)
    print("Saved three-panel method figure with explicit data and gradient paths.")


if __name__ == "__main__":
    main()
