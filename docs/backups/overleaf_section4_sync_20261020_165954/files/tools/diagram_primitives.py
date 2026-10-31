"""Shared drawing primitives for the manuscript's conceptual diagrams."""

import matplotlib.colors as colors
import matplotlib.patches as patches
import numpy as np


INK = "#27333D"
MUTED = "#64717A"
BLUE = "#287DA8"
ORANGE = "#C28035"
TEAL = "#23806C"
RED = "#B3465E"
GRAY = "#E8ECEF"
LINE = "#B9C2C9"


def label(ax, x, y, value, size=7.2, color=INK, weight="normal", ha="center", **kwargs):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", linespacing=1.15, **kwargs)


def rect(ax, x, y, width, height, face="white", edge=LINE, lw=0.65, **kwargs):
    item = patches.Rectangle((x, y), width, height, facecolor=face,
                             edgecolor=edge, linewidth=lw, **kwargs)
    ax.add_patch(item)
    return item


def arrow(ax, points, color=INK, dashed=False, lw=0.8, scale=7):
    style = "--" if dashed else "-"
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=color, lw=lw, linestyle=style, zorder=1)
    ax.add_patch(patches.FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=scale,
        color=color, linewidth=lw, linestyle=style, zorder=2))


def graph(ax, x, y, width=1, height=0.75, size=0.06, muted=False):
    pos = np.array([[0.08, 0.65], [0.45, 0.94], [0.38, 0.42],
                    [0.08, 0.06], [0.90, 0.08], [0.96, 0.64]])
    pos = pos * [width, height] + [x, y]
    for i, j in ((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (1, 5), (3, 4), (4, 5)):
        ax.plot(pos[[i, j], 0], pos[[i, j], 1], color=LINE, lw=0.85, zorder=0)
    for i, (px, py) in enumerate(pos):
        face = "#A9B2BA" if muted else BLUE if i < 3 else ORANGE
        ax.add_patch(patches.Circle((px, py), size, facecolor=face,
                                   edgecolor="white", lw=0.5, zorder=3))


def matrix(ax, x, y, width, height, values, row_colors=None, mask=None, numbers=False):
    values = np.asarray(values)
    nr, nc = values.shape
    for i in range(nr):
        for j in range(nc):
            px, py = x + j * width / nc, y + (nr - i - 1) * height / nr
            blocked = mask is not None and not mask[i, j]
            base = colors.to_rgb(row_colors[i] if row_colors else TEAL)
            strength = 0.13 + 0.73 * float(values[i, j])
            face = "#EDF0F2" if blocked else tuple(1 - strength * (1 - c) for c in base)
            rect(ax, px, py, width / nc, height / nr, face, "white", 0.4)
            if numbers:
                label(ax, px + width / nc / 2, py + height / nr / 2,
                      "0" if blocked else f"{values[i, j]:.2f}", size=5.6,
                      color="#9AA4AD" if blocked else INK)
    rect(ax, x, y, width, height, "none", LINE, 0.55)


def document(ax, x, y, width, height, color=BLUE):
    rect(ax, x, y, width, height, "white", color, 0.7)
    for offset, fraction in ((0.25, .70), (0.48, .70), (0.71, .48)):
        ax.plot([x + width * .14, x + width * fraction],
                [y + height * offset, y + height * offset], color=color, lw=0.7)


def feature_text_rows(ax, x, y, width, height, row_colors, gray=False, words=False):
    """A tiny table keeps feature, source-text, and label columns distinct."""
    n = len(row_colors)
    row_height = height / n
    for i, color in enumerate(row_colors):
        py = y + (n - i - 1) * row_height
        base = "#E6EAED" if gray else "#F0F7F5"
        edge = LINE if gray else "#9EBDB2"
        rect(ax, x, py, width, row_height, base, edge, 0.55)
        for j, alpha in enumerate((.75, .40, .62)):
            rgb = colors.to_rgb("#8F9BA5" if gray else color)
            face = tuple(1 - alpha * (1 - c) for c in rgb)
            rect(ax, x + .04 + j * width * .07, py + row_height * .23,
                 width * .055, row_height * .54, face, "none", 0)
        if words:
            label(ax, x + width * .30, py + row_height / 2,
                  "old text" if gray else "source text", size=6, ha="left", color=MUTED)
        else:
            for frac, length in ((.37, .31), (.67, .22)):
                ax.plot([x + width * .30, x + width * (.30 + length)],
                        [py + row_height * frac] * 2, color=LINE if gray else TEAL, lw=.7)
        ax.add_patch(patches.Circle((x + width * .93, py + row_height / 2),
                                   row_height * .14, facecolor="#9EA7AE" if gray else color,
                                   edgecolor="none"))


def save_figure(fig, outdir, stem):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", pad_inches=.025,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight", pad_inches=.025, dpi=300)
