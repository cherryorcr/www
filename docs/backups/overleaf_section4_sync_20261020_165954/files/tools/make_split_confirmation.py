"""Plot paired F1 effects as grouped bars for separate mask panels."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER = Path(__file__).resolve().parents[1]
PROJECT = PAPER.parent
PRIMARY = PROJECT / "results/uq_final_evidence_report_20261013_132047/pairs.csv"
ADDITIONAL = PROJECT / "results/uq_four_priority_20261017/analysis_fresh_pairs.csv"
DATASETS = (("arxiv", "ArXiv"), ("sportsfit", "SportsFit"), ("bookhis", "BookHis"))


def read_records(path, panel):
    selected = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if panel == "primary":
                if not (
                    row["category"] == "confirmation" and row["split"] == "sealed_test"
                    and row["left"] == "uq_dm_edgefree" and row["right"] == "gecc_hard"
                    and row["metric"] == "final_f1"
                ):
                    continue
                effect = {"mean": float(row["mean_delta"]), "low": float(row["ci_low"]),
                          "high": float(row["ci_high"]), "n": int(row["cells"])}
                expected_n = 9
            else:
                if not (row["baseline"] == "gecc_hard" and row["metric"] == "f1"):
                    continue
                effect = {"mean": float(row["mean"]), "low": float(row["ci95_low"]),
                          "high": float(row["ci95_high"]), "n": int(row["n"])}
                expected_n = 3
            key = row["dataset"]
            if key in selected or effect["n"] != expected_n:
                raise ValueError(f"Unexpected record count for {panel}/{key}")
            if not effect["low"] <= effect["mean"] <= effect["high"]:
                raise ValueError(f"Invalid interval for {panel}/{key}")
            selected[key] = {"dataset": key, "panel": panel, **effect}
    if selected.keys() != {name for name, _ in DATASETS}:
        raise ValueError(f"Missing or extra {panel} datasets: {selected.keys()}")
    return selected


def main():
    primary = read_records(PRIMARY, "primary")
    additional = read_records(ADDITIONAL, "additional")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.1,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.40, 2.45))
    fig.patch.set_facecolor("white")
    x = list(range(len(DATASETS)))
    width = .34
    colors = {"primary": "#287DA8", "additional": "#B9742D"}
    panels = ((primary, -width / 2, "Primary mask (9 cells)", colors["primary"], "\\\\"),
              (additional, width / 2, "Additional mask (3 cells)", colors["additional"], "//"))
    for records, offset, label, color, hatch in panels:
        means = [records[name]["mean"] for name, _ in DATASETS]
        lower = [[records[name]["mean"] - records[name]["low"] for name, _ in DATASETS]]
        upper = [[records[name]["high"] - records[name]["mean"] for name, _ in DATASETS]]
        bars = ax.bar([value + offset for value in x], means, width=width,
                      color=color, alpha=.9, edgecolor=color, hatch=hatch,
                      yerr=lower + upper, ecolor="#4A545C", capsize=2.2,
                      error_kw={"elinewidth": .75, "capthick": .75},
                      label=label, zorder=3)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean + (.22 if mean >= 0 else -.28), f"{mean:+.2f}",
                    ha="center", va="bottom" if mean >= 0 else "top",
                    fontsize=6.1, color="#263238")
    ax.axhline(0, color="#87929B", lw=.8, linestyle="--", zorder=2)
    ax.set(xticks=x, xticklabels=[label for _, label in DATASETS],
           ylim=(-3.4, 9.5), ylabel="F1 difference (points)",
           xlabel="CoTAG-Q minus GECC-hard-TAG")
    ax.grid(axis="y", color="#E2E6E9", lw=.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#B4BCC3")
    ax.spines["bottom"].set_color("#B4BCC3")
    ax.legend(loc="upper left", bbox_to_anchor=(-.01, 1.01),
              frameon=False, fontsize=6.7, ncol=2, borderaxespad=0,
              handletextpad=.45, columnspacing=.8)
    fig.subplots_adjust(left=.16, right=.98, bottom=.23, top=.80)
    outdir = PAPER / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "split_confirmation.pdf", bbox_inches="tight", pad_inches=.025,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(outdir / "split_confirmation.png", bbox_inches="tight", pad_inches=.025, dpi=300)
    plt.close(fig)

    # The main paper uses the replicated primary mask only.  The combined
    # figure above remains available as an audit artifact for the appendix.
    fig, ax = plt.subplots(figsize=(3.40, 2.45))
    fig.patch.set_facecolor("white")
    x = list(range(len(DATASETS)))
    means = [primary[name]["mean"] for name, _ in DATASETS]
    lower = [[primary[name]["mean"] - primary[name]["low"] for name, _ in DATASETS]]
    upper = [[primary[name]["high"] - primary[name]["mean"] for name, _ in DATASETS]]
    bars = ax.bar(x, means, width=.56, color="#287DA8", alpha=.9,
                  edgecolor="#287DA8", yerr=lower + upper, ecolor="#4A545C",
                  capsize=2.2, error_kw={"elinewidth": .75, "capthick": .75},
                  zorder=3)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + (.22 if mean >= 0 else -.28), f"{mean:+.2f}",
                ha="center", va="bottom" if mean >= 0 else "top",
                fontsize=6.1, color="#263238")
    ax.axhline(0, color="#87929B", lw=.8, linestyle="--", zorder=2)
    ax.set(xticks=x, xticklabels=[label for _, label in DATASETS],
           ylim=(-3.4, 4.9), ylabel="F1 difference (points)",
           xlabel="CoTAG-Q minus GECC-hard-TAG")
    ax.grid(axis="y", color="#E2E6E9", lw=.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#B4BCC3")
    ax.spines["bottom"].set_color("#B4BCC3")
    fig.subplots_adjust(left=.16, right=.98, bottom=.23, top=.96)
    fig.savefig(outdir / "primary_confirmation.pdf", bbox_inches="tight", pad_inches=.025,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(outdir / "primary_confirmation.png", bbox_inches="tight", pad_inches=.025, dpi=300)
    plt.close(fig)
    provenance = {
        "sources": [{"path": source.relative_to(PROJECT).as_posix(),
                     "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                    for source in (PRIMARY, ADDITIONAL)],
        "filters": {"primary": {"category": "confirmation", "split": "sealed_test",
                                "left": "uq_dm_edgefree", "right": "gecc_hard", "metric": "final_f1"},
                    "additional": {"baseline": "gecc_hard", "metric": "f1"}},
        "records": [*primary.values(), *additional.values()],
        "interpretation": (
            "Separate nominal paired 95% t intervals over cell means; each cell averages "
            "three Student seeds. Historical sealed_test denotes an inspected fixed mask, "
            "not a new blind test. The additional mask rebuilds the legal construction pool. "
            "The panels are not pooled and do not estimate post-selection uncertainty."
        ),
    }
    (outdir / "split_confirmation.source.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Saved paired mask comparison: 6 verified records, with source hashes.")


if __name__ == "__main__":
    main()
