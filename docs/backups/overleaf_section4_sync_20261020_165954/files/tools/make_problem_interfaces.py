"""Show the original tasks of named condensers and the proposed extension."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagram_primitives import (
    BLUE, LINE, MUTED, ORANGE, TEAL,
    arrow, document, feature_text_rows, graph, label, matrix, save_figure,
)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.0,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(3.40, 3.68))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 3.40), ylim=(0, 3.68))
    ax.axis("off")

    values = np.array([[.8, .3, .6], [.2, .7, .9]])

    label(ax, .03, 3.52, "(a) Continual graph condensation", size=7.6,
          weight="bold", ha="left")
    label(ax, .03, 3.32, "CaT; PUMA", size=7.0, color=MUTED, ha="left")
    graph(ax, .04, 2.89, .44, .28, .030)
    label(ax, .57, 3.02, "+", size=8)
    matrix(ax, .69, 2.88, .24, .29, values, [BLUE, ORANGE])
    arrow(ax, [(1.06, 3.02), (1.52, 3.02)])
    matrix(ax, 1.69, 2.86, .53, .32, values, [BLUE, ORANGE])
    label(ax, 2.43, 3.02, "features\n+ labels", size=7.1, ha="left")
    label(ax, .49, 2.70, "increment + memory", size=6.6, color=MUTED)
    label(ax, .03, 2.49, "No readable-text construction.", size=7.0,
          color=MUTED, ha="left")

    ax.plot([.03, 3.36], [2.36, 2.36], color=LINE, lw=.6)
    label(ax, .03, 2.18, "(b) Static TAG condensation", size=7.6,
          weight="bold", ha="left")
    label(ax, .03, 1.98, "TAGSAM; STAD", size=7.0, color=MUTED, ha="left")
    graph(ax, .04, 1.60, .44, .28, .030)
    label(ax, .57, 1.74, "+", size=8)
    document(ax, .70, 1.60, .20, .28)
    arrow(ax, [(1.06, 1.74), (1.52, 1.74)])
    feature_text_rows(ax, 1.69, 1.58, .61, .32, [BLUE, ORANGE])
    label(ax, 2.43, 1.74, "condensed\nTAG", size=7.1, ha="left")
    label(ax, .54, 1.40, "fixed source graph + text", size=6.5, color=MUTED)
    label(ax, .03, 1.20, "No continual update rule in the original task.", size=6.8,
          color=MUTED, ha="left")

    ax.plot([.03, 3.36], [1.07, 1.07], color=LINE, lw=.6)
    label(ax, .03, .89, "(c) CoTAG-Q", size=7.6, weight="bold", ha="left", color=TEAL)
    graph(ax, .04, .40, .44, .28, .030)
    label(ax, .57, .54, "+", size=8)
    document(ax, .70, .40, .20, .28)
    arrow(ax, [(1.06, .54), (1.36, .54)], color=TEAL)
    q = np.zeros((6, 4))
    q[:3, :2] = [[.8, .2], [.6, .4], [.1, .9]]
    q[3:, 2:] = [[.2, .8], [.4, .6], [.9, .1]]
    mask = np.zeros_like(q, dtype=bool)
    mask[:3, :2] = True
    mask[3:, 2:] = True
    matrix(ax, 1.44, .39, .38, .30, q,
           [BLUE] * 3 + [ORANGE] * 3, mask=mask)
    label(ax, 1.63, .80, r"learn $Q$", size=7.1, color=TEAL)
    arrow(ax, [(1.92, .54), (2.16, .54)], color=TEAL)
    feature_text_rows(ax, 2.25, .38, 1.08, .32, [BLUE, ORANGE])
    label(ax, .52, .23, "increment + text", size=6.6, color=MUTED)
    label(ax, 2.79, .23, "feature-text pairs", size=6.6, color=TEAL)
    label(ax, .03, .06, "Append new pairs; keep old memory unchanged.", size=6.8,
          color=TEAL, ha="left")

    fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.005)
    save_figure(fig, Path(__file__).resolve().parents[1] / "figures", "problem_interfaces")
    plt.close(fig)
    print("Saved named-method task comparison (no added text adapter).")


if __name__ == "__main__":
    main()
