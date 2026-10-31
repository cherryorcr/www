"""Audit and summarize paired P0/P1 cells without treating seeds as orders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics as st

from scipy.stats import t


PRIMARY = ("uq_dm_edgefree", "cat_cctp", "adaptive_coherent")
NAMES = {"uq_dm_edgefree": "In-loop CaT-Q", "cat_cctp": "CaT + nearest5",
         "adaptive_coherent": "Adaptive coherent", "cat_nearest1": "CaT X + nearest1",
         "cat_random5": "CaT X + random5", "uq_nearest1": "CaT-Q X + nearest1",
         "uq_nearest5": "CaT-Q X + nearest5"}


def interval(values):
    mean = st.mean(values)
    if len(values) < 2:
        return dict(mean=mean, sd=None, ci95=None, n=len(values))
    sd = st.stdev(values)
    radius = float(t.ppf(.975, len(values)-1)) * sd / len(values)**.5
    return dict(mean=mean, sd=sd, ci95=[mean-radius, mean+radius], n=len(values))


def load_cells(root):
    cells, incomplete = [], []
    for path in sorted(root.glob("*_o*/results.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "complete":
            incomplete.append(dict(path=str(path), status=result["status"], rows=len(result["rows"])))
            continue
        p = result["protocol"]
        if p["name"] != "unified-q-frozen-attribution-20261008":
            raise ValueError("unexpected protocol")
        expected = len(p["budgets"])
        rows = result["rows"]
        for arm in p["arms"]:
            stages = [r["stage"] for r in rows if r["arm"] == arm and r["student_family"] == "prompt"]
            if sorted(stages) != list(range(expected)):
                raise AssertionError(f"partial or duplicate prompt arm {path} {arm}")
        for row in rows:
            if not all(row["checks"].get(k) for k in ("budget_ok", "prefix_ok", "finite_ok")):
                raise AssertionError("invalid memory")
            payload = row["audit"]["payload_validation"]
            if not payload["current_delta_text_sources"] or not payload["same_class_text_sources"]:
                raise AssertionError("invalid source provenance")
            base = "cat_cctp" if row["arm"] in ("adaptive_coherent", "cat_nearest1", "cat_random5") else "uq_dm_edgefree" if row["arm"].startswith("uq_nearest") else None
            if base:
                reference = next(r for r in rows if r["arm"] == base and r["stage"] == row["stage"] and r["student_family"] == "prompt")
                if row["x_hash"] != reference["x_hash"] or row["edge_hash"] != reference["edge_hash"] or not row["checks"]["fixed_content_exact"]:
                    raise AssertionError("fixed-X control was not fixed")
            if row["student_family"] != "prompt":
                reference = next(r for r in rows if r["arm"] == row["arm"] and r["stage"] == row["stage"] and r["student_family"] == "prompt")
                if row["memory_hash"] != reference["memory_hash"]:
                    raise AssertionError("Student replay used different memory")
        cells.append(dict(path=str(path), **result))
    # Within a dataset/order all arms already share one protocol. Across order
    # cells keep the same masks, budgets, Student and source hashes.
    for dataset in sorted({c["protocol"]["dataset"] for c in cells}):
        records = [c["protocol"] for c in cells if c["protocol"]["dataset"] == dataset]
        for key in ("budgets", "pool_mask_hash", "report_mask_hash", "student", "student_seeds", "student_steps", "student_batch", "construction_seed", "q_config", "source_hashes", "runner_sha256"):
            if any(p[key] != records[0][key] for p in records):
                raise AssertionError(f"protocol mismatch across orders: {dataset} {key}")
    return cells, incomplete


def summarize(cells):
    main, paired, text, students = [], [], [], []
    for dataset in sorted({c["protocol"]["dataset"] for c in cells}):
        records = [c for c in cells if c["protocol"]["dataset"] == dataset]
        final_rows = [(c, r) for c in records for r in c["rows"] if r["stage"] == len(c["protocol"]["budgets"])-1]
        for arm in PRIMARY:
            scores = [r["score"] for c,r in final_rows if r["arm"] == arm and r["student_family"] == "prompt"]
            trajectories = [[r["score"] for r in c["rows"]
                             if r["arm"] == arm and r["student_family"] == "prompt"]
                            for c in records]
            stage_means = [st.mean(s["final_f1"] for s in chain) for chain in trajectories]
            stage_acc = [st.mean(s["accuracy"] for s in chain) for chain in trajectories]
            stage_minima = [min(s["final_f1"] for s in chain) for chain in trajectories]
            main.append(dict(dataset=dataset, arm=arm, orders=len(scores),
                f1=interval([s["final_f1"] for s in scores]),
                accuracy=interval([s["accuracy"] for s in scores]),
                old_f1=st.mean(s["old_f1"] for s in scores),
                new_f1=st.mean(s["new_f1"] for s in scores),
                stage_f1=interval(stage_means), stage_accuracy=interval(stage_acc),
                worst_stage_f1=interval(stage_minima),
                worst_observed_stage_f1=min(stage_minima),
                final_memory_rows=records[0]["protocol"]["budgets"][-1]))
        for a,b in (("uq_dm_edgefree","cat_cctp"),("adaptive_coherent","cat_cctp"),("uq_dm_edgefree","adaptive_coherent")):
            deltas, acc = [], []
            for c in records:
                stage = len(c["protocol"]["budgets"])-1
                rr = {r["arm"]: r["score"] for r in c["rows"] if r["stage"]==stage and r["student_family"]=="prompt"}
                deltas.append(rr[a]["final_f1"]-rr[b]["final_f1"])
                acc.append(rr[a]["accuracy"]-rr[b]["accuracy"])
            paired.append(dict(dataset=dataset, comparison=f"{a} - {b}", **interval(deltas),
                               accuracy_delta=st.mean(acc), positive=sum(v>0 for v in deltas), worst=min(deltas)))
        for c,r in final_rows:
            if c["protocol"]["order_seed"] != 9101:
                continue
            if r["student_family"] == "prompt":
                base = "uq_dm_edgefree" if r["arm"].startswith("uq_") else "cat_cctp"
                reference = next(v for v in c["rows"] if v["stage"]==r["stage"] and v["student_family"]=="prompt" and v["arm"]==base)
                by_seed = {v["seed"]: v for v in reference["score"]["per_student"]}
                seed_deltas = [v["final_f1"]-by_seed[v["seed"]]["final_f1"]
                               for v in r["score"]["per_student"]]
                text.append(dict(dataset=dataset, arm=r["arm"], base=base, **r["score"],
                    delta_f1=r["score"]["final_f1"]-reference["score"]["final_f1"],
                    delta_acc=r["score"]["accuracy"]-reference["score"]["accuracy"],
                    paired_student_delta_sd=st.stdev(seed_deltas) if len(seed_deltas)>1 else None,
                    stage_f1=st.mean(v["score"]["final_f1"] for v in c["rows"] if v["arm"]==r["arm"] and v["student_family"]=="prompt")))
            else:
                current = next(v for v in c["rows"] if v["stage"]==r["stage"] and v["student_family"]=="prompt" and v["arm"]==r["arm"])
                students.append(dict(dataset=dataset, arm=r["arm"], legacy=r["score"], prompt=current["score"],
                    delta_f1=current["score"]["final_f1"]-r["score"]["final_f1"],
                    delta_acc=current["score"]["accuracy"]-r["score"]["accuracy"], memory_hash=r["memory_hash"]))
    return dict(main=main, paired=paired, text=text, student_replay=students)


def render(summary, incomplete):
    lines = ["# P0/P1 Frozen-Memory Attribution", "",
             "All scores are report-dev percentages, not sealed-test results.",
             "P0: order means; P1/replay: reference order 9101 only. One construction seed (271006).",
             "Student seeds 150/151/152 are averaged within a cell, not counted as independent orders.",
             f"Incomplete cells excluded: {len(incomplete)}.", "", "## P0: Common Prompt Student", "",
             "| Dataset | Method | Orders | F1 mean +/- order SD | Acc mean +/- order SD | Old F1 | New F1 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    def fmt(v):
        return f"{v['mean']:.2f}" + (f" +/- {v['sd']:.2f}" if v["sd"] is not None else " (one order)")
    for r in summary["main"]:
        lines.append(f"| {r['dataset']} | {NAMES[r['arm']]} | {r['orders']} | {fmt(r['f1'])} | {fmt(r['accuracy'])} | {r['old_f1']:.2f} | {r['new_f1']:.2f} |")
    lines += ["", "### Stage Trajectories", "",
              "Equal-weight stage means are computed within each order, then summarized across orders. Worst-stage means use each order's minimum stage F1; worst observed is the minimum of those minima. All stage scores first average Student seeds. They are not individual-seed minima.", "",
              "| Dataset | Method | Stage F1 +/- order SD | Stage Acc +/- order SD | Worst-stage F1 mean | Worst observed stage F1 | Final memory rows |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary["main"]:
        lines.append(f"| {r['dataset']} | {NAMES[r['arm']]} | {fmt(r['stage_f1'])} | {fmt(r['stage_accuracy'])} | {r['worst_stage_f1']['mean']:.2f} | {r['worst_observed_stage_f1']:.2f} | {r['final_memory_rows']} |")
    lines += ["", "### Paired Order Differences", "", "| Dataset | Comparison | F1 delta [95% CI] | Acc delta | Positive orders | Worst F1 delta |", "|---|---|---:|---:|---:|---:|"]
    for r in summary["paired"]:
        ci = "not available" if r["ci95"] is None else f"{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}"
        lines.append(f"| {r['dataset']} | {r['comparison']} | {r['mean']:+.2f} [{ci}] | {r['accuracy_delta']:+.2f} | {r['positive']}/{r['n']} | {r['worst']:+.2f} |")
    lines += ["", "## P1: Fixed-X Text Attribution (One Order)", "",
              "Delta is relative to the indicated X family's native text. SD below is Student SD, not order/constructor SD.", "",
              "| Dataset | Method | F1 +/- Student SD | Acc +/- Student SD | Delta F1 | Delta Acc | Stage F1 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary["text"]:
        lines.append(f"| {r['dataset']} | {NAMES[r['arm']]} | {r['final_f1']:.2f} +/- {r['final_f1_std']:.2f} | {r['accuracy']:.2f} +/- {r['accuracy_std']:.2f} | {r['delta_f1']:+.2f} | {r['delta_acc']:+.2f} | {r['stage_f1']:.2f} |")
    lines += ["", "## Matched Student-Loss Replay (Same Final Memory)", "",
              "Same symmetric evaluation graph, labels, memory, initial architecture, seeds, batch and steps. This does not replay historical budgets or historical graph normalization.", "",
              "| Dataset | Method | Row-wise loss F1 / Acc | Prompt loss F1 / Acc | Delta F1 | Delta Acc |",
              "|---|---|---:|---:|---:|---:|"]
    for r in summary["student_replay"]:
        a,b = r["legacy"],r["prompt"]
        lines.append(f"| {r['dataset']} | {NAMES[r['arm']]} | {a['final_f1']:.2f} / {a['accuracy']:.2f} | {b['final_f1']:.2f} / {b['accuracy']:.2f} | {r['delta_f1']:+.2f} | {r['delta_acc']:+.2f} |")
    lines += ["", "## Scope", "",
              "Adaptive coherent is an internal historical candidate, not an external paper baseline. Text replacements are component attribution controls, not independently reoptimized continual algorithms. No main-method parameter was chosen using these results. P2-P5 are not marked completed by this report."]
    return "\n".join(lines) + "\n"


def plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    for key,title in (("f1","Macro-F1"),("accuracy","Accuracy")):
        fig, ax = plt.subplots(figsize=(9,4.5))
        datasets = sorted({r["dataset"] for r in summary["main"]})
        x = np.arange(len(datasets))
        for i,arm in enumerate(PRIMARY):
            rows = [next(r for r in summary["main"] if r["dataset"]==d and r["arm"]==arm) for d in datasets]
            errors = ([r[key]["sd"] for r in rows]
                      if all(r[key]["sd"] is not None for r in rows) else None)
            ax.bar(x+(i-1)*.25, [r[key]["mean"] for r in rows], .24,
                   yerr=errors, capsize=3,
                   label=NAMES[arm], color=("#187d73","#526272","#bf733d")[i])
        ax.set_xticks(x,datasets)
        ax.set_ylim(0,100)
        ax.set_ylabel(title + " (%)")
        complete_orders = all(r["orders"] >= 3 for r in summary["main"])
        ax.set_title("Same Prompt Student and budget; error bars: order SD" if complete_orders
                     else "Partial P0 results; independent orders still pending")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5,-.12), ncol=3)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(output/f"p0_{key}.png", dpi=170)
        plt.close(fig)
    datasets = sorted({r["dataset"] for r in summary["student_replay"]})
    if datasets:
        fig, axes = plt.subplots(1,len(datasets),figsize=(4.5*len(datasets),4.6),squeeze=False)
        labels = ["CaT-Q", "CaT\nnearest5", "Adaptive\ncoherent"]
        for ax,dataset in zip(axes[0],datasets):
            records = [next(r for r in summary["student_replay"] if r["dataset"]==dataset and r["arm"]==arm) for arm in PRIMARY]
            x = np.arange(3)
            for i,(key,label,color) in enumerate((("legacy","Row-wise InfoNCE","#6a7380"),("prompt","Prompt objective","#187d73"))):
                ax.bar(x+(i-.5)*.34,[r[key]["final_f1"] for r in records],.32,
                       color=color,label=label)
            ax.set_xticks(x,labels)
            ax.set_ylim(0,100)
            ax.set_title(dataset)
            ax.spines[["top","right"]].set_visible(False)
        axes[0][0].set_ylabel("Final Macro-F1 (%)")
        fig.suptitle("Same frozen memory; reference order; matched steps/seeds/operator")
        handles,labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=2,frameon=False)
        fig.tight_layout(rect=(0,.09,1,.94))
        fig.savefig(output/"p0_student_loss.png",dpi=170)
        plt.close(fig)
    datasets = sorted({r["dataset"] for r in summary["text"]})
    if datasets:
        fig, axes = plt.subplots(1,len(datasets),figsize=(4.4*len(datasets)+2,4.5),squeeze=False,sharey=True)
        controls = ["cat_nearest1","cat_random5","uq_nearest1","uq_nearest5"]
        for ax,dataset in zip(axes[0],datasets):
            rows = [next(r for r in summary["text"] if r["dataset"]==dataset and r["arm"]==arm) for arm in controls]
            for i,r in enumerate(rows):
                ax.errorbar(r["delta_f1"],i,xerr=r["paired_student_delta_sd"],
                            fmt="o",capsize=3,color="#187d73" if i<2 else "#b66a34")
            ax.axvline(0,color="#737373",linestyle="--",linewidth=1)
            ax.set_yticks(range(4),[NAMES[a] for a in controls])
            ax.set_title(dataset)
            ax.set_xlabel("F1 change from native payload (pp)")
            ax.spines[["top","right"]].set_visible(False)
        axes[0][0].invert_yaxis()
        fig.suptitle("Fixed X/text replacement; bars: paired Student SD, NOT order uncertainty")
        fig.tight_layout(rect=(0,0,1,.91))
        fig.savefig(output/"p1_text_attribution.png",dpi=170)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("results/unified_q_attribution_20261008"))
    args = p.parse_args()
    cells, incomplete = load_cells(args.root)
    known = {Path(c["path"]).parent.name for c in cells}
    known.update(Path(c["path"]).parent.name for c in incomplete)
    incomplete.extend(dict(path=str(args.root/f"{dataset}_o{order}"/"results.json"),
                           status="not_started", rows=0)
                      for dataset in ("wikics", "sportsfit", "bookhis")
                      for order in (9101,9102,9103)
                      if f"{dataset}_o{order}" not in known)
    if not cells:
        raise SystemExit("No complete cells; no final summary written")
    summary = summarize(cells)
    summary.update(complete_cells=len(cells), incomplete=incomplete,
                   sources=[c["path"] for c in cells])
    (args.root/"summary.json").write_text(json.dumps(summary,indent=2), encoding="utf-8")
    (args.root/"summary.md").write_text(render(summary,incomplete), encoding="utf-8")
    plots(summary,args.root)
    print(json.dumps(dict(complete_cells=len(cells), incomplete=len(incomplete))))


if __name__ == "__main__":
    main()
