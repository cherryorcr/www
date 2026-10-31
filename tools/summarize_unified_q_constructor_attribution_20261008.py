"""Paired constructor attribution; keep order and Student uncertainty separate."""
import argparse
import json
from pathlib import Path
import statistics as st

from summarize_unified_q_attribution_20261008 import interval

MAIN = "uq_dm_edgefree"
CORE = (MAIN,"no_node_text","no_residual","no_distribution","free_x")
NAMES = {MAIN:"Full CaT-Q","no_node_text":"No node payload in construction",
    "no_residual":"No X residual","no_distribution":"No distribution matching",
    "free_x":"Independent X, same conditional bound","no_text_init":"No text initialization",
    "no_payload_gm":"No payload pair term in GM","stop_text_gradient":"Stop direct text-to-Q gradient"}


def load(root):
    cells,incomplete = [],[]
    for dataset in ("wikics","sportsfit","bookhis"):
        for order in (9101,9102,9103):
            path = root/f"{dataset}_o{order}"/"results.json"
            r = json.loads(path.read_text()) if path.exists() else dict(status="not_started")
            if r["status"] != "complete":
                incomplete.append(dict(cell=path.parent.name,status=r["status"]))
                continue
            p,rows = r["protocol"],r["rows"]
            if p["name"] != "unified-q-constructor-attribution-20261008":
                raise ValueError("wrong experiment")
            expected = {(arm,stage,"native") for arm in p["arms"] for stage in range(5)}
            expected |= {(arm,4,"common_cat_payload") for arm in p["arms"]}
            keys = [(v["arm"],v["stage"],v["readout"]) for v in rows]
            if set(keys) != expected or len(keys) != len(expected):
                raise AssertionError("partial or duplicate chain")
            for v in rows:
                if not all(v["checks"][key] for key in ("budget_ok","prefix_ok","finite_ok")):
                    raise AssertionError("invalid memory")
                if sorted(s["seed"] for s in v["score"]["per_student"]) != sorted(p["student_seeds"]):
                    raise AssertionError("Student pairing changed")
            common = [v for v in rows if v["readout"]=="common_cat_payload"]
            if len({v["text_hash"] for v in common}) != 1:
                raise AssertionError("common payload differs across arms")
            for v in common:
                native = next(w for w in rows if w["arm"]==v["arm"] and w["stage"]==4 and w["readout"]=="native")
                if v["x_hash"] != native["x_hash"] or not v["checks"]["fixed_content_exact"]:
                    raise AssertionError("common payload control changed X")
            cells.append(r)
    for dataset in ("wikics","sportsfit","bookhis"):
        ps = [c["protocol"] for c in cells if c["protocol"]["dataset"]==dataset]
        for key in ("budgets","student","student_seeds","student_steps","student_batch",
                    "pool_mask_hash","report_mask_hash","construction_seed","q_config","source_hashes","runner_sha256"):
            if ps and any(p[key] != ps[0][key] for p in ps):
                raise AssertionError(f"protocol varies across orders: {key}")
    return cells,incomplete


def summary(cells):
    main,contrasts,text,factorial,per_order = [],[],[],[],[]
    for dataset in sorted({c["protocol"]["dataset"] for c in cells}):
        cc = [c for c in cells if c["protocol"]["dataset"]==dataset]
        for readout in ("native","common_cat_payload"):
            for arm in CORE:
                pairs = []
                for c in cc:
                    rows = {r["arm"]:r for r in c["rows"] if r["stage"]==4 and r["readout"]==readout}
                    pairs.append((rows[arm],rows[MAIN]))
                main.append(dict(dataset=dataset,readout=readout,arm=arm,
                    f1=interval([a["score"]["final_f1"] for a,b in pairs]),
                    accuracy=interval([a["score"]["accuracy"] for a,b in pairs]),
                    old_f1=st.mean(a["score"]["old_f1"] for a,b in pairs),
                    new_f1=st.mean(a["score"]["new_f1"] for a,b in pairs),
                    stage_f1=st.mean(st.mean(r["score"]["final_f1"] for r in c["rows"]
                        if r["arm"]==arm and r["readout"]=="native") for c in cc),
                    stage_accuracy=st.mean(st.mean(r["score"]["accuracy"] for r in c["rows"]
                        if r["arm"]==arm and r["readout"]=="native") for c in cc),
                    worst_stage_f1=st.mean(min(r["score"]["final_f1"] for r in c["rows"]
                        if r["arm"]==arm and r["readout"]=="native") for c in cc),
                    worst_order_f1=min(a["score"]["final_f1"] for a,b in pairs)))
                for c,(a,b) in zip(cc,pairs):
                    per_order.append(dict(dataset=dataset,order=c["protocol"]["order_seed"],
                        readout=readout,arm=arm,f1=a["score"]["final_f1"],accuracy=a["score"]["accuracy"],
                        f1_delta=a["score"]["final_f1"]-b["score"]["final_f1"],
                        accuracy_delta=a["score"]["accuracy"]-b["score"]["accuracy"]))
                if arm != MAIN:
                    ds = [a["score"]["final_f1"]-b["score"]["final_f1"] for a,b in pairs]
                    acc = [a["score"]["accuracy"]-b["score"]["accuracy"] for a,b in pairs]
                    contrasts.append(dict(dataset=dataset,readout=readout,arm=arm,**interval(ds),
                        accuracy_delta=st.mean(acc),accuracy_paired=interval(acc),
                        positive=sum(d>0 for d in ds),worst=min(ds)))
            ref = next((c for c in cc if c["protocol"]["order_seed"]==9101),None)
            if ref:
                rr = {r["arm"]:r["score"] for r in ref["rows"] if r["stage"]==4 and r["readout"]==readout}
                for arm in (MAIN,"no_text_init","no_payload_gm","no_node_text","stop_text_gradient"):
                    text.append(dict(dataset=dataset,readout=readout,arm=arm,**rr[arm],
                                     delta=rr[arm]["final_f1"]-rr[MAIN]["final_f1"]))
                effects = {
                    "init_with_pair": {MAIN:1,"no_text_init":-1},
                    "init_without_pair": {"no_payload_gm":1,"no_node_text":-1},
                    "pair_with_init": {MAIN:1,"no_payload_gm":-1},
                    "pair_without_init": {"no_text_init":1,"no_node_text":-1},
                    "init_pair_interaction": {MAIN:1,"no_text_init":-1,"no_payload_gm":-1,"no_node_text":1}}
                for name,weights in effects.items():
                    paired = {arm:{p["seed"]:p for p in rr[arm]["per_student"]} for arm in weights}
                    for metric in ("final_f1","accuracy"):
                        values = [sum(weight*paired[arm][seed][metric] for arm,weight in weights.items())
                                  for seed in ref["protocol"]["student_seeds"]]
                        factorial.append(dict(dataset=dataset,readout=readout,effect=name,metric=metric,
                            mean=st.mean(values),paired_student_sd=st.stdev(values) if len(values)>1 else None))
    replay,cost = [],[]
    for c in cells:
        p=c["protocol"]
        for arm in p["arms"]:
            native=[r for r in c["rows"] if r["arm"]==arm and r["readout"]=="native"]
            cost.append(dict(dataset=p["dataset"],order=p["order_seed"],arm=arm,
                construction_seconds=sum(r["construction_seconds"] for r in native),
                native_student_seconds=sum(r["student_seconds"] for r in native),
                final_memory_rows=native[-1]["memory_rows"]))
        full=[r for r in c["rows"] if r["arm"]==MAIN and r["readout"]=="native"]
        replay.append(dict(dataset=p["dataset"],order=p["order_seed"],
            max_block_x_drift=max(r["audit"]["unchanged_main_replay"]["relative_x_difference"] for r in full),
            changed_source_rows=sum(r["audit"]["unchanged_main_replay"]["text_source_rows_changed"] for r in full)))
    return dict(main=main,contrasts=contrasts,text_factorial=text,factorial_effects=factorial,
                replay=replay,cost=cost,per_order=per_order)


def render(s,incomplete):
    lines=["# Constructor-Side Attribution", "", "Report-dev only; one construction seed, three Student seeds averaged inside each order.",
        f"Incomplete cells: {len(incomplete)}. Partial results are not final.",
        "All deltas are control minus full, not gains from adding the component.", ""]
    def val(x):
        return f"{x['mean']:.2f} +/- {x['sd']:.2f}" if x["sd"] is not None else f"{x['mean']:.2f} (one order)"
    def ci_text(ci):
        return "unavailable" if ci is None else f"{ci[0]:+.2f}, {ci[1]:+.2f}"
    for readout,title in (("native","Native Payload"),("common_cat_payload","Identical CaT Payload: Final-Stage Diagnostic")):
        lines += [f"## {title}","","| Dataset | Method | Orders | F1 +/- order SD | Acc +/- order SD | Old F1 | New F1 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for r in s["main"]:
            if r["readout"]==readout:
                lines.append(f"| {r['dataset']} | {NAMES[r['arm']]} | {r['f1']['n']} | {val(r['f1'])} | {val(r['accuracy'])} | {r['old_f1']:.2f} | {r['new_f1']:.2f} |")
        lines += ["","| Dataset | Control - full | F1 delta [95% paired t CI] | Acc delta [95% paired t CI] | Positive orders | Worst F1 delta |","|---|---|---:|---:|---:|---:|"]
        for r in s["contrasts"]:
            if r["readout"]==readout:
                lines.append(f"| {r['dataset']} | {NAMES[r['arm']]} | {r['mean']:+.2f} [{ci_text(r['ci95'])}] | {r['accuracy_delta']:+.2f} [{ci_text(r['accuracy_paired']['ci95'])}] | {r['positive']}/{r['n']} | {r['worst']:+.2f} |")
    lines += ["","## Reference-Order Text Factorial","","Student SD below does not measure construction/order generalization.","",
        "| Dataset | Readout | Arm | F1 +/- Student SD | Accuracy +/- Student SD | Delta F1 |","|---|---|---|---:|---:|---:|"]
    for r in s["text_factorial"]:
        lines.append(f"| {r['dataset']} | {r['readout']} | {NAMES[r['arm']]} | {r['final_f1']:.2f} +/- {r['final_f1_std']:.2f} | {r['accuracy']:.2f} +/- {r['accuracy_std']:.2f} | {r['delta']:+.2f} |")
    lines += ["","### Conditional Text Effects","",
        "Here positive means adding the named factor helps. Interaction = full - no-init - no-pair + neither. These are reference-order effects, not multi-order confirmations. Student SD is conditional on this one constructed chain.","",
        "| Dataset | Readout | Effect | F1 pp +/- paired Student SD | Accuracy pp |","|---|---|---|---:|---:|"]
    for r in s["factorial_effects"]:
        if r["metric"] != "final_f1":
            continue
        a=next(v for v in s["factorial_effects"] if v["metric"]=="accuracy" and
               (v["dataset"],v["readout"],v["effect"])==(r["dataset"],r["readout"],r["effect"]))
        sd="n/a" if r["paired_student_sd"] is None else f"{r['paired_student_sd']:.2f}"
        lines.append(f"| {r['dataset']} | {r['readout']} | {r['effect']} | {r['mean']:+.2f} +/- {sd} | {a['mean']:+.2f} |")
    lines += ["","## Stage Averages and Cost","","Student time includes evaluation, three seeds and all five stages; shared GPUs do not establish exclusive-hardware speedups.","",
        "| Dataset | Method | Stage F1 | Stage Acc | Mean worst-stage F1 | Worst-order final F1 | Build seconds | Student seconds | Final rows |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in s["main"]:
        if r["readout"] != "native":
            continue
        costs=[c for c in s["cost"] if c["dataset"]==r["dataset"] and c["arm"]==r["arm"]]
        lines.append(f"| {r['dataset']} | {NAMES[r['arm']]} | {r['stage_f1']:.2f} | {r['stage_accuracy']:.2f} | {r['worst_stage_f1']:.2f} | {r['worst_order_f1']:.2f} | {st.mean(c['construction_seconds'] for c in costs):.1f} | {st.mean(c['native_student_seconds'] for c in costs):.1f} | {costs[0]['final_memory_rows']} |")
    lines += ["","## Per-Order Final Scores","",
        "| Dataset | Order | Readout | Method | F1 | Acc | Delta F1 | Delta Acc |",
        "|---|---:|---|---|---:|---:|---:|---:|"]
    for r in s["per_order"]:
        lines.append(f"| {r['dataset']} | {r['order']} | {r['readout']} | {NAMES[r['arm']]} | {r['f1']:.2f} | {r['accuracy']:.2f} | {r['f1_delta']:+.2f} | {r['accuracy_delta']:+.2f} |")
    lines += ["","## Limits","",
        "No-node-text retains class prompts and dataset X. Common payload does not feed back into construction. Free-X retains the same Q-conditioned bound/targets, so it is not unbounded free-X or a full CaT comparison.",
        "These are component controls, not external paper baselines. No significance/equivalence claim is inferred from a confidence interval crossing zero. New parameterization changes optimizer geometry; lower free-X scores would not prove intrinsic representational superiority of Q.",
        "Intervals use only three order means (one construction seed), without multiplicity correction. They describe limited order sensitivity, not a confirmatory population-wide significance test.",
        "Same-seed numerical replay drift is recorded in summary.json; full was reconstructed in this batch, not pooled as a new independent seed."]
    return "\n".join(lines)+"\n"


def plots(s,root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric,label in (("mean","Macro-F1"),("accuracy_delta","Accuracy")):
        fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=True)
        arms=CORE[1:]
        for ax,dataset in zip(axes,("wikics","sportsfit","bookhis")):
            for offset,(readout,color) in enumerate((("native","#187d73"),("common_cat_payload","#b56c35"))):
                for i,arm in enumerate(arms):
                    r=next((r for r in s["contrasts"] if (r["dataset"],r["readout"],r["arm"])==(dataset,readout,arm)),None)
                    if r:
                        ci=r["ci95"] if metric=="mean" else r["accuracy_paired"]["ci95"]
                        error=None if ci is None else [[max(0.,r[metric]-ci[0])],[max(0.,ci[1]-r[metric])]]
                        ax.errorbar(r[metric],i+(offset-.5)*.18,xerr=error,fmt="o",color=color,
                                    capsize=3,label=readout if i==0 else None)
            ax.axvline(0,color="#888888",linestyle="--",linewidth=1)
            ax.set_yticks(range(4),["No node payload","No residual","No DM","Independent X"])
            ax.set_title(dataset)
            ax.set_xlabel(f"{label}: control - full (pp)")
            ax.spines[["top","right"]].set_visible(False)
        axes[0].invert_yaxis()
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc="lower center",ncol=2,frameon=False)
        fig.suptitle("Constructor attribution; paired 95% t intervals across available orders")
        fig.tight_layout(rect=(0,.07,1,.93))
        fig.savefig(root/f"attribution_{'f1' if metric=='mean' else 'accuracy'}.png",dpi=170)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("results/unified_q_constructor_attribution_20261008_v2"))
    args=p.parse_args()
    cells,incomplete=load(args.root)
    if not cells:
        raise SystemExit("No complete cells; no summary written")
    s=summary(cells)
    s.update(complete_cells=len(cells),incomplete=incomplete)
    (args.root/"summary.json").write_text(json.dumps(s,indent=2))
    (args.root/"summary.md").write_text(render(s,incomplete),encoding="utf-8")
    plots(s,args.root)
    print(json.dumps(dict(complete=len(cells),incomplete=len(incomplete))))


if __name__ == "__main__":
    main()
