"""Aggregate development cells without pooling Student seeds as constructions."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results/uq_gecc_gap_20261009"
NAMES = dict(uq_dm_edgefree="Original In-loop CaT-Q", uq_pair_mmd="Q + joint pair-MMD (not promoted)",
    uq_marginal_mmd="Q + marginal-MMD (control)", uq_shuffled_mmd="Q + shuffled-pair MMD (control)",
    uq_message="Message-space Q (not promoted)", uq_message_no_prop="Message-space Q, no propagation loss",
    cat_cctp="CaT + nearest5 (paper mechanism adapted)",
    gecc_hard="GECC-hard + TAG (paper mechanism adapted)", uq_nearest1="Original Q X + common nearest1",
    uq_nearest5="Original Q X + common nearest5", gecc_nearest1="GECC X + common nearest1",
    gecc_nearest5="GECC X + common nearest5 (near-tie implementation control)")
METRICS = ("final_f1", "accuracy", "old_f1", "new_f1")


def build(root):
    cells, rows, diagnostics, costs = [], {}, {}, []
    for path in sorted(root.glob("*/results.json")):
        cell = json.loads(path.read_text())
        if cell["status"] != "complete":
            continue
        p = cell["protocol"]
        if len(cell["stages"]) != 5 or p["student_steps"] != 2048 or p["student_seeds"] != [150,151,152]:
            raise AssertionError("incomplete stage chain or changed evaluation contract")
        cells.append(dict(path=str(path), dataset=p["dataset"], order=p["order_seed"],
            seed=p["construction_seed"], phase=p["phase"]))
        for row in cell["rows"]:
            score = row["score"]
            if [s["seed"] for s in score["per_student"]] != p["student_seeds"]:
                raise AssertionError("readout seeds mismatch")
            for metric in METRICS:
                if abs(np.mean([s[metric] for s in score["per_student"]]) - score[metric]) > 1e-7:
                    raise AssertionError("Student aggregation mismatch")
            old_n, new_n = sum(map(len, p["groups"][:-1])), len(p["groups"][-1])
            reconstructed = (old_n * score["old_f1"] + new_n * score["new_f1"]) / (old_n + new_n)
            if abs(reconstructed - score["final_f1"]) > 1e-6:
                raise AssertionError("Old/New F1 decomposition mismatch")
            key = (p["dataset"], p["order_seed"], p["construction_seed"], row["arm"], row["student"])
            if key in rows and (rows[key]["score"] != score or rows[key]["memory_hash"] != row["memory_hash"]):
                raise AssertionError("conflicting baseline reuse")
            rows[key] = row
        for stage in cell["stages"]:
            if not stage["checks"]["sinkhorn_feasible"] or not stage["checks"]["same_class_text_sources"]:
                raise AssertionError("constructor invariant failed")
            if p["phase"] == "p0":
                if set(stage["text_checks"]) != {"uq_nearest1", "uq_nearest5", "gecc_nearest1", "gecc_nearest5"}:
                    raise AssertionError("missing text control")
                for checks in stage["text_checks"].values():
                    if not all(checks[k] for k in ("fixed_content_exact", "budget_ok", "prefix_ok", "current_delta_text_sources", "same_class_text_sources")):
                        raise AssertionError("fixed-X control invariant failed")
            elif not stage["checks"]["prefix_ok"] or not stage["checks"]["budget_ok"]:
                raise AssertionError("candidate changed prefix or budget")
            diagnostics.setdefault((p["dataset"], p["phase"]), []).append(stage)
        costs.append(dict(dataset=p["dataset"], phase=p["phase"], seed=p["construction_seed"],
            construction_seconds=sum(s["construction_seconds"] for s in cell["stages"]),
            peak_mb=max(s["construction_peak_mb"] for s in cell["stages"]),
            warning="includes read-only diagnostics; not an exclusive-hardware efficiency benchmark"))
    summary = []
    for dataset, arm, student in sorted({(k[0], k[3], k[4]) for k in rows}):
        selected = [v for k,v in rows.items() if (k[0], k[3], k[4]) == (dataset, arm, student)]
        summary.append(dict(dataset=dataset, arm=arm, student=student, cells=len(selected),
            memory_rows=selected[0]["memory_rows"],
            **{metric: dict(mean=float(np.mean([v["score"][metric] for v in selected])),
                std=float(np.std([v["score"][metric] for v in selected], ddof=1)) if len(selected)>1 else None)
                for metric in METRICS}))
    pairs = []
    comparisons = [("uq_dm_edgefree", "gecc_hard"), ("uq_nearest1", "gecc_nearest1"),
                   ("uq_nearest5", "gecc_nearest5"), ("uq_message", "uq_dm_edgefree"),
                   ("uq_message", "gecc_hard"), ("uq_message_no_prop", "uq_message")]
    comparisons += [(a,b) for a in ("uq_pair_mmd","uq_marginal_mmd","uq_shuffled_mmd") for b in ("uq_dm_edgefree","gecc_hard")]
    comparisons += [("uq_pair_mmd","uq_marginal_mmd"),("uq_pair_mmd","uq_shuffled_mmd")]
    for candidate, base in comparisons:
        for dataset, student in sorted({(k[0], k[4]) for k in rows}):
            differences = []
            for key, value in rows.items():
                if key[0] != dataset or key[3:] != (candidate, student):
                    continue
                match = rows.get((*key[:3], base, student))
                if match:
                    differences.append(dict(order=key[1], seed=key[2],
                        **{m: value["score"][m]-match["score"][m] for m in METRICS}))
            if differences:
                pairs.append(dict(candidate=candidate, base=base, dataset=dataset, student=student,
                    cells=len(differences), values=differences,
                    **{m: dict(mean=float(np.mean([v[m] for v in differences])),
                        positive_cells=sum(v[m]>0 for v in differences), worst_cell=min(v[m] for v in differences))
                        for m in METRICS}))
    geometry = []
    for (dataset, phase), stages in sorted(diagnostics.items()):
        classes = [c for s in stages for c in s["diagnostic"]["classes"]]
        geometry.append(dict(dataset=dataset, phase=phase, stages=len(stages),
            mean_distance_over_bound=float(np.mean([c["distance_over_bound"] for c in classes])),
            classes_beyond_bound=sum(c["distance_over_bound"]>1 for c in classes), class_records=len(classes),
            raw_message_cosine=float(np.mean([c["raw_message_cosine"] for c in classes])),
            synthetic_message_mean_error=float(np.mean([c["synthetic_mean_error"] for c in classes])),
            synthetic_message_mean_cosine=float(np.mean([c["synthetic_message_cosine"] for c in classes])),
            residual_saturation_stage_mean=float(np.mean([s["diagnostic"]["residual_cap_saturation"] for s in stages])),
            within_variance_ratio=float(np.mean([c["synthetic_within_variance"]/c["message_within_variance"] for c in classes if c["message_within_variance"]>1e-12])),
            holdout_dm_stage_mean=float(np.mean([s["diagnostic"]["holdout_dm"] for s in stages])),
            holdout_dm_per_class_mean=float(np.mean([s["diagnostic"]["holdout_dm"]/len(s["diagnostic"]["classes"]) for s in stages])),
            replay_relative_error_max=max(s.get("replay_x_relative_error",0.) for s in stages),
            replay_changed_texts=sum(s.get("replay_changed_texts",0) for s in stages)))
    joint_diagnostics=[]
    for (dataset, phase), stages in sorted(diagnostics.items()):
        if not phase.startswith('p2-'):
            continue
        details=[s['audit']['joint_pair_distribution'] for s in stages]
        gradients=[r['joint_distribution_q_gradient_norm'] for s in stages for r in s['audit']['trace']
                   if r.get('joint_distribution_q_gradient_norm') is not None]
        joint_diagnostics.append(dict(dataset=dataset,phase=phase,stages=len(stages),
            stages_with_reduced_objective=sum(d['final_loss']<d['initial_loss'] for d in details),
            mean_initial_objective=float(np.mean([d['initial_loss'] for d in details])),
            mean_final_objective=float(np.mean([d['final_loss'] for d in details])),
            q_gradient_norm_min=min(gradients),q_gradient_norm_max=max(gradients)))
    upgrades=[]
    for candidate in ('uq_message','uq_pair_mmd'):
        relevant=[r for r in pairs if r['candidate']==candidate and r['base']=='uq_dm_edgefree']
        if len(relevant)==6 and all(r['cells']==2 for r in relevant):
            means={s:float(np.mean([r['final_f1']['mean'] for r in relevant if r['student']==s])) for s in ('prompt_gcn','prompt_sgc')}
            upgrades.append(dict(candidate=candidate,mean_f1_delta_by_student=means,
                positive_cells=sum(r['final_f1']['positive_cells'] for r in relevant),cells=12,
                no_large_dataset_mean_regression=all(r['final_f1']['mean']>=-1 for r in relevant),
                passes_minimum_development_gate=all(v>0 for v in means.values()) and all(r['final_f1']['mean']>=-1 for r in relevant)))
    execution=dict(completed_cells=len(cells),phase_counts=dict(Counter(c['phase'] for c in cells)),
        constructor_stage_calls=sum(len(v) for v in diagnostics.values()),
        new_student_trainings=sum(len(r['score']['per_student']) for r in rows.values() if not r.get('reused',False)),
        reused_student_trainings=sum(len(r['score']['per_student']) for r in rows.values() if r.get('reused',False)))
    return dict(scope="report-dev development, one order and two construction seeds per graph; Student seeds averaged within cell, not independent replicates; no strong CI claim", cells=cells,
                summary=summary, paired=pairs, diagnostics=geometry, joint_diagnostics=joint_diagnostics,
                costs=costs,execution=execution,development_gates=upgrades)


def write(result, root, plots=True):
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# GECC Gap Investigation", "", result["scope"], "",
             "External baselines: CaT/GECC adapted mechanisms. Nearest text and pooling controls are internal attribution experiments.", ""]
    for dataset in ("wikics", "sportsfit", "bookhis"):
        lines.extend(["## " + dataset, "", "| Method | Student | Cells | F1 mean +/- std | Accuracy mean +/- std | Old F1 | New F1 | Rows |", "|---|---|---:|---:|---:|---:|---:|---:|"])
        for arm in NAMES:
            for row in result["summary"]:
                if row["dataset"] != dataset or row["arm"] != arm:
                    continue
                def fmt(metric):
                    v=row[metric]
                    return f"{v['mean']:.2f}" + (f" +/- {v['std']:.2f}" if v['std'] is not None else "")
                lines.append(f"| {NAMES[arm]} | {row['student']} | {row['cells']} | {fmt('final_f1')} | {fmt('accuracy')} | {row['old_f1']['mean']:.2f} | {row['new_f1']['mean']:.2f} | {row['memory_rows']} |")
        lines.append("")
    lines.extend(["## Paired Development Differences", "", "| Dataset | Student | Candidate - Base | F1 delta pp | Positive cells | Worst delta | Accuracy delta pp |", "|---|---|---|---:|---:|---:|---:|"])
    for row in result["paired"]:
        f=row["final_f1"]
        lines.append(f"| {row['dataset']} | {row['student']} | {row['candidate']} - {row['base']} | {f['mean']:+.2f} | {f['positive_cells']}/{row['cells']} | {f['worst_cell']:+.2f} | {row['accuracy']['mean']:+.2f} |")
    lines.extend(["", "## Representation Diagnostics", "", "Saturation and replay DM describe auxiliary constructor replays; exact archived-X diagnostics are retained in each results.json. Mean-gap bounds alone do not imply a classification bound.", "", "| Dataset | Phase | Mean gap / bound | Raw-message cosine | Residual saturation | Holdout DM / class |", "|---|---|---:|---:|---:|---:|"])
    for row in result["diagnostics"]:
        lines.append(f"| {row['dataset']} | {row['phase']} | {row['mean_distance_over_bound']:.2f} | {row['raw_message_cosine']:.6f} | {row['residual_saturation_stage_mean']:.3f} | {row['holdout_dm_per_class_mean']:.6f} |")
    (root / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        arms = [a for a in ("uq_dm_edgefree", "uq_pair_mmd", "uq_message", "cat_cctp", "gecc_hard") if any(r['arm']==a for r in result['summary'])]
        for metric in ("final_f1", "accuracy"):
            fig, axes = plt.subplots(2,3,figsize=(15,7),layout="constrained")
            colors = ["#56616e", "#007a70", "#a46b2b", "#467d9d", "#8c539d"]
            for i, student in enumerate(("prompt_gcn", "prompt_sgc")):
                for j, dataset in enumerate(("wikics", "sportsfit", "bookhis")):
                    ax=axes[i,j]
                    matches=[next((r for r in result['summary'] if r['dataset']==dataset and r['student']==student and r['arm']==arm),None) for arm in arms]
                    values=[r[metric]['mean'] if r else np.nan for r in matches]
                    ax.bar(range(len(arms)),values,color=colors[:len(arms)],yerr=[r[metric]['std'] or 0 if r else 0 for r in matches],capsize=3)
                    ax.set_xticks(range(len(arms)),[a.replace('uq_dm_edgefree','Original Q').replace('uq_pair_mmd','Pair MMD').replace('uq_message','Message Q').replace('cat_cctp','CaT').replace('gecc_hard','GECC') for a in arms],rotation=20)
                    ax.set_title(dataset+' / '+student.replace('prompt_','').upper())
                    ax.set_ylabel('Macro-F1 (%)' if metric=='final_f1' else 'Accuracy (%)')
                    ax.set_ylim(0,100)
                    for x,v in enumerate(values):
                        if np.isfinite(v): ax.text(x,v+1.5,f'{v:.2f}',ha='center',fontsize=9)
            fig.suptitle('Development cells: 1 order x 2 construction seeds; report-dev')
            fig.savefig(root/(metric+'.png'),dpi=160)
            plt.close(fig)
        if any(r['arm']=='uq_pair_mmd' for r in result['summary']):
            fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained')
            modes=('uq_dm_edgefree','uq_pair_mmd','uq_marginal_mmd','uq_shuffled_mmd')
            for i,student in enumerate(('prompt_gcn','prompt_sgc')):
                for j,dataset in enumerate(('wikics','sportsfit','bookhis')):
                    ax=axes[i,j]
                    matches=[next((r for r in result['summary'] if (r['dataset'],r['student'],r['arm'])==(dataset,student,mode)),None) for mode in modes]
                    values=[r['final_f1']['mean'] if r else np.nan for r in matches]
                    ax.bar(range(4),values,color=['#56616e','#007a70','#b68c36','#936284'],
                           yerr=[r['final_f1']['std'] or 0 if r else 0 for r in matches],capsize=3)
                    ax.set_xticks(range(4),['Original Q','Paired','Marginal','Shuffled'],rotation=15)
                    ax.set_title(dataset+' / '+student.replace('prompt_','').upper())
                    ax.set_ylim(0,100)
                    ax.set_ylabel('Macro-F1 (%)')
                    for x,v in enumerate(values):
                        if np.isfinite(v): ax.text(x,v+1.5,f'{v:.2f}',ha='center',fontsize=9)
            fig.suptitle('Pairing attribution: same Q initialization, candidate texts and Student')
            fig.savefig(root/'pairing_controls.png',dpi=160)
            plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    result=build(args.root)
    write(result,args.root,not args.no_plots)
    print(json.dumps(dict(completed_cells=len(result['cells']), rows=len(result['summary']), paired=len(result['paired']))))
