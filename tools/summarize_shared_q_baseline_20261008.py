"""Keep shared-Q ablation, handoff protocol and Student effects separate."""
import argparse
import copy
import json
from pathlib import Path
import statistics as st

from summarize_unified_q_attribution_20261008 import interval

MAIN = "uq_dm_edgefree"
DATASETS = ("wikics", "sportsfit", "bookhis")
METHODS = (MAIN, "cat_mmd_tag_cctp_top5", "cat_cctp_repaired", "gecc_hard_tag", "puma_full",
           "tagsam_selfloop", "random_real", "k_center", "medoid", "cat_text_center_legacy")
NAMES = {MAIN: "In-loop CaT-Q", "split_q": "Independent Qx / Qt",
    "cat_mmd_tag_cctp_top5": "CaT-MMD + CCTP nearest5",
    "cat_cctp_repaired": "CaT-MMD + CCTP nearest5 (text repaired)",
    "cat_text_center_legacy": "Historical CaT + text_center (mislabeled top5)",
    "gecc_hard_tag": "GECC-hard + TAG attachment",
    "puma_full": "PUMA-inspired TAG (rebuilt)",
    "tagsam_selfloop": "TAGSAM-response adapted",
    "random_real": "Random real coreset", "k_center": "K-center coreset",
    "medoid": "Medoid coreset"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def basic_checks(cell, expected):
    p, rows = cell["protocol"], cell["rows"]
    actual = [(r["arm"], r["stage"], r.get("readout", r.get("student"))) for r in rows]
    if set(actual) != expected or len(actual) != len(expected):
        raise AssertionError("partial or duplicate chain")
    if p["student_steps"] != 2048 or p["student_seeds"] != [150, 151, 152]:
        raise AssertionError("unexpected Student schedule")
    for r in rows:
        if not all(r["checks"].get(k) for k in ("budget_ok", "prefix_ok", "finite_ok")):
            raise AssertionError("invalid memory")
        if r["memory_rows"] != p["budgets"][r["stage"]]:
            raise AssertionError("row budget changed")
        seeds = [s["seed"] for s in r["score"]["per_student"]]
        if sorted(seeds) != sorted(p["student_seeds"]):
            raise AssertionError("missing or duplicate Student seed")
        for metric in ("final_f1", "accuracy"):
            if abs(st.mean(s[metric] for s in r["score"]["per_student"]) - r["score"][metric]) > 1e-8:
                raise AssertionError("pooled score differs from per-seed mean")


def audit_shared(c):
    p = c["protocol"]
    if p["name"] != "shared-Q-vs-independent-Q-20261008" or p["max_stages"] != 5:
        raise AssertionError("unexpected shared-Q experiment")
    expected = {(a, s, "native") for a in (MAIN, "split_q") for s in range(5)}
    expected |= {(a, 4, "common_cat_payload") for a in (MAIN, "split_q")}
    basic_checks(c, expected)
    common = [r for r in c["rows"] if r["readout"] == "common_cat_payload"]
    if len({r["common_payload_hash"] for r in common}) != 1 or not all(r["checks"]["fixed_content_exact"] for r in common):
        raise AssertionError("common payload control changed content")
    for r in c["rows"]:
        a = r["audit"]
        if not a["assignment_checks"]["sinkhorn_feasible"] or a["historical_raw_reads"] != 0:
            raise AssertionError("assignment/access constraint failed")
        if a["x_residual_ratio_max"] > p["q_config"]["trust_radius"] + 1e-6:
            raise AssertionError("X trust radius exceeded")
        payload = a["payload_validation"]
        if not payload["current_delta_text_sources"] or not payload["same_class_text_sources"] or payload["text_reencode_max_error"] > 1e-4:
            raise AssertionError("invalid text materialization")
        if r["arm"] == "split_q":
            for q in a["assignments"].values():
                if q["cross_class_mass"] or max(q["row_error"], q["column_relative_error"]) > 1.1e-4:
                    raise AssertionError("independent assignment infeasible")


def audit_handoff(c):
    p = c["protocol"]
    if p["name"] != "handoff-order-two-Student-paired-20261008" or p["max_stages"] != 5:
        raise AssertionError("unexpected handoff experiment")
    audit_student_pair(c)


def audit_student_pair(c):
    p = c["protocol"]
    basic_checks(c, {(a, s, t) for a in p["arms"] for s in range(5) for t in ("legacy", "prompt")})
    for a in p["arms"]:
        for stage in range(5):
            rows = [r for r in c["rows"] if r["arm"] == a and r["stage"] == stage]
            for key in ("memory_hash", "x_hash", "text_hash", "edge_hash", "labels_hash"):
                if len({r[key] for r in rows}) != 1:
                    raise AssertionError("Student crossover changed memory")
            if {r["student"]:r["score"]["student_batch"] for r in rows} != {"legacy":16,"prompt":24}:
                raise AssertionError("Student batch mismatch")


def handoff_path(root, dataset):
    verified = root / f"handoff_{dataset}_verified"
    return verified if (verified / "results.json").exists() else root / f"handoff_{dataset}"


def merge_text_repair(original, repair):
    if repair["protocol"]["name"] != "handoff-cctp-fixed-X-text-repair-20261008":
        raise AssertionError("unknown text repair")
    audit_student_pair(repair)
    for key in ("dataset", "groups", "budgets", "student_steps", "student_seeds",
                "pool_mask_hash", "report_mask_hash", "stage_id_hashes"):
        if original["protocol"][key] != repair["protocol"][key]:
            raise AssertionError(f"repair changed comparison protocol: {key}")
    for key in ("graphtext_prompt_student.py", "run_gate1_continual.py", "external_continual_mechanisms.py"):
        if original["protocol"]["sources"][key] != repair["protocol"]["sources"][key]:
            raise AssertionError("repair changed a frozen module")
    for row in repair["rows"]:
        source = next(r for r in original["rows"] if (r["arm"], r["stage"], r["student"]) ==
                      ("cat_text_center_legacy", row["stage"], row["student"]))
        if source["provenance"]["hash"] != row["provenance"]["source_memory_hash"]:
            raise AssertionError("repair refers to a different CaT memory")
        for key in ("x_hash", "labels_hash", "edge_hash", "memory_rows"):
            if source[key] != row[key]:
                raise AssertionError("text repair changed fixed content")
        if not row["checks"].get("fixed_content_exact"):
            raise AssertionError("missing fixed-content verification")
    out = copy.deepcopy(original)
    out["rows"].extend(repair["rows"])
    out["protocol"]["arms"].append("cat_cctp_repaired")
    out["summary_repair_protocol"] = repair["protocol"]
    return out


def mark_historical_cctp(cell):
    out = copy.deepcopy(cell)
    for row in out["rows"]:
        if row["arm"] == "cat_mmd_tag_cctp_top5":
            row["arm"] = "cat_text_center_legacy"
    out["protocol"]["arms"] = ["cat_text_center_legacy" if a == "cat_mmd_tag_cctp_top5" else a
                                for a in out["protocol"]["arms"]]
    return out


def load(root):
    shared, handoff, incomplete = [], [], []
    for dataset in DATASETS:
        for order in (9101, 9102, 9103):
            path = root / f"shared_{dataset}_o{order}/results.json"
            c = read(path) if path.exists() else dict(status="not_started")
            if c["status"] != "complete":
                incomplete.append(dict(cell=path.parent.name, status=c["status"]))
            else:
                audit_shared(c)
                shared.append(c)
        path = handoff_path(root, dataset) / "results.json"
        c = read(path) if path.exists() else dict(status="not_started")
        if c["status"] != "complete":
            incomplete.append(dict(cell=path.parent.name, status=c["status"]))
        else:
            audit_handoff(c)
            if dataset == "sportsfit":
                c = mark_historical_cctp(c)
                repair_path = root / "handoff_cctp_repaired_sportsfit/results.json"
                repair = read(repair_path) if repair_path.exists() else dict(status="not_started")
                if repair["status"] == "complete":
                    c = merge_text_repair(c, repair)
                else:
                    incomplete.append(dict(cell=repair_path.parent.name, status=repair["status"]))
            handoff.append(c)
    for dataset in DATASETS:
        ps = [c["protocol"] for c in shared if c["protocol"]["dataset"] == dataset]
        for key in ("budgets", "student", "student_steps", "student_seeds", "student_batch",
                    "pool_mask_hash", "report_mask_hash", "construction_seed", "q_config", "source_hashes", "runner_sha256"):
            if ps and any(p[key] != ps[0][key] for p in ps):
                raise AssertionError(f"shared-Q protocol varies across orders: {key}")
    return shared, handoff, incomplete


def compute(shared, handoff):
    scores, effects, details, costs, routing = [], [], [], [], []
    for dataset in DATASETS:
        cells = [c for c in shared if c["protocol"]["dataset"] == dataset]
        if not cells:
            continue
        for readout in ("native", "common_cat_payload"):
            for arm in (MAIN, "split_q"):
                finals = [next(r for r in c["rows"] if (r["arm"],r["stage"],r["readout"]) == (arm,4,readout)) for c in cells]
                scores.append(dict(dataset=dataset, arm=arm, readout=readout,
                    **{k:interval([r["score"][k] for r in finals]) for k in ("final_f1","accuracy","old_f1","new_f1")}))
                for c, r in zip(cells, finals):
                    details.append(dict(dataset=dataset, order=c["protocol"]["order_seed"], arm=arm, readout=readout,
                        **{k:r["score"][k] for k in ("final_f1","accuracy","old_f1","new_f1")}))
            for metric in ("final_f1", "accuracy"):
                values = []
                for c in cells:
                    rr = {r["arm"]:r for r in c["rows"] if r["stage"] == 4 and r["readout"] == readout}
                    values.append(rr[MAIN]["score"][metric] - rr["split_q"]["score"][metric])
                effects.append(dict(dataset=dataset, readout=readout, metric=metric, **interval(values),
                    positive=sum(v > 0 for v in values), worst=min(values)))
        for c in cells:
            for arm in (MAIN, "split_q"):
                native = [r for r in c["rows"] if r["arm"] == arm and r["readout"] == "native"]
                costs.append(dict(dataset=dataset, order=c["protocol"]["order_seed"], arm=arm,
                    stage_f1=st.mean(r["score"]["final_f1"] for r in native),
                    stage_accuracy=st.mean(r["score"]["accuracy"] for r in native),
                    worst_stage_f1=min(r["score"]["final_f1"] for r in native),
                    build_seconds=sum(r["construction_seconds"] for r in native),
                    student_seconds=sum(r["student_seconds"] for r in native), rows=native[-1]["memory_rows"]))
                if arm == "split_q":
                    for r in native:
                        a = r["audit"]
                        routing.append(dict(dataset=dataset, order=c["protocol"]["order_seed"], stage=r["stage"],
                            row_l1=a["qx_qt_row_l1_mean"], slot_top1_source_disagreement=a["qx_qt_top1_disagreement"],
                            qx_change=a["qx_changed_norm"], qt_change=a["qt_changed_norm"],
                            assignment_parameters=a["assignment_parameter_count"]))
    baselines, paired, transfers, replay = [], [], [], []
    for c in handoff:
        dataset = c["protocol"]["dataset"]
        for student in ("legacy", "prompt"):
            for arm in METHODS:
                native = sorted([r for r in c["rows"] if r["arm"] == arm and r["student"] == student], key=lambda r:r["stage"])
                if not native:
                    continue
                final = native[-1]
                baselines.append(dict(dataset=dataset, student=student, arm=arm, **final["score"],
                    category="historical_control" if arm == "cat_text_center_legacy" else "paired_method",
                    stage_f1=st.mean(r["score"]["final_f1"] for r in native),
                    stage_accuracy=st.mean(r["score"]["accuracy"] for r in native),
                    worst_stage_f1=min(r["score"]["final_f1"] for r in native),
                    memory_rows=final["memory_rows"], construction_seconds=sum(r["construction_seconds"] for r in native) if all(r["construction_seconds"] is not None for r in native) else None,
                    student_seconds=sum(r["student_seconds"] for r in native),
                    text_chars_max=final["text_chars_max"], text_chars_total=final["text_chars_total"], nonself_edges=final["nonself_edges"]))
                if student == "legacy":
                    for r in native:
                        if r.get("legacy_replay_f1_difference") is not None:
                            replay.append(dict(dataset=dataset,arm=arm,stage=r["stage"],differences=r["legacy_replay_f1_difference"]))
            finals = {r["arm"]:r["score"]["per_student"] for r in c["rows"] if r["student"] == student and r["stage"] == 4}
            main = {v["seed"]:v for v in finals[MAIN]}
            for arm, rows in finals.items():
                if arm == MAIN:
                    continue
                for metric in ("final_f1", "accuracy"):
                    values = [main[v["seed"]][metric] - v[metric] for v in rows]
                    paired.append(dict(dataset=dataset,student=student,arm=arm,metric=metric,**interval(values)))
        for arm in c["protocol"]["arms"]:
            finals = {r["student"]:r for r in c["rows"] if r["arm"] == arm and r["stage"] == 4}
            transfers.append(dict(dataset=dataset,arm=arm,
                f1_prompt_minus_legacy=finals["prompt"]["score"]["final_f1"]-finals["legacy"]["score"]["final_f1"],
                accuracy_prompt_minus_legacy=finals["prompt"]["score"]["accuracy"]-finals["legacy"]["score"]["accuracy"],
                memory_hash=finals["prompt"]["memory_hash"]))
    return dict(shared_scores=scores, shared_effects=effects, shared_per_order=details, shared_cost=costs,
                independent_routing=routing, baseline_scores=baselines, baseline_paired=paired,
                student_transfer=transfers, legacy_replay=replay)


def fmt(value):
    return f"{value['mean']:.2f} +/- {value['sd']:.2f}" if value["sd"] is not None else f"{value['mean']:.2f} (n=1)"


def legacy_shared_summary(root, shared):
    path=root/"shared_legacy_transfer.json"
    if not path.exists():
        return dict(status="not_started",scores=[],effects=[])
    result=read(path)
    if result["status"]!="complete":
        return dict(status=result["status"],completed=len(result["rows"]),scores=[],effects=[])
    expected={(d,o,a) for d in DATASETS for o in (9101,9102,9103) for a in (MAIN,"split_q")}
    keys=[(r["dataset"],r["order"],r["arm"]) for r in result["rows"]]
    if set(keys)!=expected or len(keys)!=len(expected):
        raise AssertionError("incomplete or duplicate legacy-Q evaluation")
    by_key={(c["protocol"]["dataset"],c["protocol"]["order_seed"]):c for c in shared}
    for r in result["rows"]:
        source=by_key[(r["dataset"],r["order"])]
        native=next(v for v in source["rows"] if (v["arm"],v["stage"],v["readout"])==(r["arm"],4,"native"))
        if r["memory_hash"]!=native["memory_hash"] or r["prompt_score"]!=native["score"]:
            raise AssertionError("legacy-Q evaluation changed the sealed memory")
        if sorted(v["seed"] for v in r["score"]["per_student"])!=[150,151,152]:
            raise AssertionError("legacy-Q seeds differ")
    scores,effects=[],[]
    for d in DATASETS:
        for a in (MAIN,"split_q"):
            rows=[r for r in result["rows"] if (r["dataset"],r["arm"])==(d,a)]
            scores.append(dict(dataset=d,arm=a,**{k:interval([r["score"][k] for r in rows]) for k in ("final_f1","accuracy")}))
        for metric in ("final_f1","accuracy"):
            deltas=[]
            for o in (9101,9102,9103):
                rr={r["arm"]:r["score"] for r in result["rows"] if (r["dataset"],r["order"])==(d,o)}
                deltas.append(rr[MAIN][metric]-rr["split_q"][metric])
            effects.append(dict(dataset=d,metric=metric,**interval(deltas),positive=sum(v>0 for v in deltas),worst=min(deltas)))
    return dict(status="complete",scores=scores,effects=effects,exploratory=True)


def render(s):
    lines = ["# Shared-Q and Dual-Student Results", "", f"Shared-Q cells: {s['shared_cells']}/9; handoff datasets: {s['handoff_datasets']}/3.",
        "Report-dev only. Incomplete cells are excluded and listed in summary.json. No test-based tuning.", "",
        "## A. Shared vs Independent Q (Internal Ablation)", "",
        "Three order means, each averaging Student seeds 150/151/152; one construction seed. SD/CI here measure order variability, not independent construction variability.", ""]
    for readout in ("native", "common_cat_payload"):
        lines += [f"### {readout}", "", "| Dataset | Method | Final F1 +/- order SD | Acc +/- order SD | Old F1 | New F1 |", "|---|---|---:|---:|---:|---:|"]
        for r in s["shared_scores"]:
            if r["readout"] == readout:
                lines.append(f"| {r['dataset']} | {NAMES[r['arm']]} | {fmt(r['final_f1'])} | {fmt(r['accuracy'])} | {r['old_f1']['mean']:.2f} | {r['new_f1']['mean']:.2f} |")
        lines += ["", "| Dataset | Metric | Shared minus split (pp) | 95% paired t CI | Positive orders | Worst delta |", "|---|---|---:|---:|---:|---:|"]
        for r in s["shared_effects"]:
            if r["readout"] == readout:
                ci = "n/a" if r["ci95"] is None else f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"
                lines.append(f"| {r['dataset']} | {r['metric']} | {r['mean']:+.2f} | {ci} | {r['positive']}/{r['n']} | {r['worst']:+.2f} |")
    lines += ["", "Common payload is a final-stage post-construction diagnostic. It does not change optimization. Split-Q retains the same joint GM objective; it is not a model with no multimodal coupling.", "",
        "### Cost and Stage Metrics", "", "| Dataset | Method | Stage F1 | Stage Acc | Mean worst-stage F1 | Build seconds | Student seconds | Rows |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for d in DATASETS:
        for arm in (MAIN,"split_q"):
            rows = [r for r in s["shared_cost"] if r["dataset"] == d and r["arm"] == arm]
            if rows:
                values = " | ".join(f"{st.mean(r[k] for r in rows):.2f}" for k in ("stage_f1","stage_accuracy","worst_stage_f1","build_seconds","student_seconds"))
                lines.append(f"| {d} | {NAMES[arm]} | {values} | {rows[0]['rows']} |")
    legacy=s["shared_legacy"]
    lines += ["", "### Exploratory Legacy Student Readout", "",
        f"Status: {legacy['status']}. Added after Prompt results were seen, not a preregistered confirmation. Final stage only, unchanged archived memory; no reoptimization.", ""]
    if legacy["status"]=="complete":
        lines += ["| Dataset | Method | Legacy F1 +/- order SD | Legacy Acc +/- order SD |", "|---|---|---:|---:|"]
        for r in legacy["scores"]:
            lines.append(f"| {r['dataset']} | {NAMES[r['arm']]} | {fmt(r['final_f1'])} | {fmt(r['accuracy'])} |")
        lines += ["", "| Dataset | Metric | Shared minus split | 95% paired t CI | Positive orders |", "|---|---|---:|---:|---:|"]
        for r in legacy["effects"]:
            lines.append(f"| {r['dataset']} | {r['metric']} | {r['mean']:+.2f} | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | {r['positive']}/{r['n']} |")
    lines += ["", "## B. Handoff Protocol: External Methods and Coresets", "",
        "Do NOT merge with A or the earlier main table: different class orders/budget paths. Same archived memory across Students; our method constructed once on these orders, never tuned to the readout Student.",
        "Equal rows, not equal text bytes or edge counts. Adapted paper mechanisms, NOT official complete pipelines. PUMA rebuilt because no checkpoint was saved. TAGSAM omitted where source order differs.",
        "SportsFit's archived arm named CCTP top5 was actually text_center. It is separated as a historical control; the corrected nearest5 keeps the archived CaT X exactly fixed and is evaluated under both Students.",
        "SD below is over three Student seeds CONDITIONAL on one constructed memory chain. Different archived construction seeds are disclosed in source manifests.", ""]
    for d in DATASETS:
        for student in ("legacy","prompt"):
            rows = [r for r in s["baseline_scores"] if r["dataset"] == d and r["student"] == student
                    and r["category"] != "historical_control"]
            if not rows:
                continue
            lines += [f"### {d} / {student}", "", "| Method | Final F1 +/- Student SD | Acc +/- Student SD | Stage F1 | Stage Acc | Old F1 | New F1 | Worst stage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for r in rows:
                values = " | ".join(f"{r[k]:.2f}" for k in ("stage_f1","stage_accuracy","old_f1","new_f1","worst_stage_f1"))
                lines.append(f"| {NAMES[r['arm']]} | {r['final_f1']:.2f} +/- {r['final_f1_std']:.2f} | {r['accuracy']:.2f} +/- {r['accuracy_std']:.2f} | {values} |")
    historical = [r for r in s["baseline_scores"] if r["category"] == "historical_control"]
    if historical:
        lines += ["", "### Historical Implementation Control (Not an External Baseline)", "",
                  "| Dataset | Student | Method | Final F1 | Accuracy |", "|---|---|---|---:|---:|"]
        for r in historical:
            lines.append(f"| {r['dataset']} | {r['student']} | {NAMES[r['arm']]} | {r['final_f1']:.2f} | {r['accuracy']:.2f} |")
    lines += ["", "### Stored Content and Evaluation Cost", "", "| Dataset | Method | Rows | Max text chars | Total text chars | Nonself directed edges | Build seconds | Legacy / Prompt Student seconds |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in s["baseline_scores"]:
        if r["student"] != "legacy":
            continue
        prompt = next(v for v in s["baseline_scores"] if (v["dataset"],v["arm"],v["student"]) == (r["dataset"],r["arm"],"prompt"))
        build = "unavailable" if r["construction_seconds"] is None else f"{r['construction_seconds']:.1f}"
        lines.append(f"| {r['dataset']} | {NAMES[r['arm']]} | {r['memory_rows']} | {r['text_chars_max']} | {r['text_chars_total']} | {r['nonself_edges']} | {build} | {r['student_seconds']:.1f} / {prompt['student_seconds']:.1f} |")
    lines += ["", "## Limits", "",
        "Legacy uses row-wise InfoNCE/batch16 and no memory class labels; Prompt uses multi-positive + prompt CE + margin/batch24 and memory labels. Both use two-layer GCN, but their normalization/training protocols differ. This is Student-protocol transfer, NOT cross-architecture evidence.",
        "All methods use text payload during training and class prompts at inference. No strict unseen-class or cross-dataset zero-shot claim follows from these results.",
        "Paired CIs in baseline_paired are over Student seeds only, not construction seeds/orders. Three-order shared-Q intervals are exploratory and uncorrected for multiple comparisons; a CI crossing zero is not proof of equivalence.",
        "Construction/evaluation timing is descriptive, includes encoder/inference costs and shared machine load. Archived baseline construction times are not exclusive-hardware speed measurements."]
    lines += ["", "Legacy replay diagnostics show sizeable same-seed variation for edged coresets, while the tested edge-free GECC repeats matched. Original and wrapper metrics agree on identical predictions. In the tested WikiCS K-center case, deterministic-algorithm flags did not remove CUDA sparse-training drift; repeated CUDA dense and CPU sparse training each matched exactly. This localizes a numerical-execution sensitivity, not a score-aggregation error or a universal claim about every sparse kernel. Diagnostic reruns are not substituted for favorable scores; see legacy_*diagnostic*.json. Baseline Student-seed SD does not isolate within-seed numerical variation."]
    return "\n".join(lines) + "\n"


def plots(s, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if s["shared_effects"]:
        fig, axes = plt.subplots(1,2,figsize=(11,4),sharey=True)
        for ax,metric,title in zip(axes,("final_f1","accuracy"),("Macro-F1","Accuracy")):
            for offset,(readout,color,label) in enumerate((("native","#137b73","Native payload"),("common_cat_payload","#b25a2b","Identical final payload"))):
                for i,d in enumerate(DATASETS):
                    r = next((r for r in s["shared_effects"] if (r["dataset"],r["metric"],r["readout"]) == (d,metric,readout)),None)
                    if r is None:
                        continue
                    ci=r["ci95"]
                    err=None if ci is None else [[r["mean"]-ci[0]],[ci[1]-r["mean"]]]
                    ax.errorbar(r["mean"],i+(offset-.5)*.18,xerr=err,fmt="o",capsize=3,color=color,label=label if i==0 else None)
            ax.axvline(0,color="#888888",ls="--",lw=1)
            ax.set_yticks(range(3),["WikiCS","SportsFit","BookHis"])
            ax.set_xlabel(f"{title}: shared minus split (pp)")
            ax.spines[["top","right"]].set_visible(False)
        axes[0].invert_yaxis()
        fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False)
        fig.suptitle("Shared-Q attribution: exploratory 95% paired intervals across orders")
        fig.tight_layout(rect=(0,.09,1,.93))
        fig.savefig(root/"shared_q_effect.png",dpi=180)
        plt.close(fig)
    if s["shared_legacy"]["status"] == "complete":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
        for ax, metric, title in zip(axes, ("final_f1", "accuracy"), ("Macro-F1", "Accuracy")):
            for offset, (student, color) in enumerate((("Legacy InfoNCE", "#c07034"), ("Prompt", "#187b76"))):
                source = s["shared_legacy"]["effects"] if offset == 0 else [r for r in s["shared_effects"] if r["readout"] == "native"]
                for i, d in enumerate(DATASETS):
                    r = next(v for v in source if v["dataset"] == d and v["metric"] == metric)
                    ci = r["ci95"]
                    ax.errorbar(r["mean"], i+(offset-.5)*.20,
                        xerr=[[r["mean"]-ci[0]], [ci[1]-r["mean"]]], fmt="o", capsize=3,
                        color=color, label=student if i == 0 else None)
            ax.axvline(0, color="#888888", ls="--", lw=1)
            ax.set_yticks(range(3), ["WikiCS", "SportsFit", "BookHis"])
            ax.set_xlabel(f"{title}: shared minus independent Q (pp)")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].invert_yaxis()
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
        fig.suptitle("Shared-Q effect depends on the Student protocol")
        fig.text(.5, .09, "Same archived memory; 3 order means; exploratory 95% paired intervals", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .14, 1, .94))
        fig.savefig(root/"shared_q_student_effect.png", dpi=180)
        plt.close(fig)
    for metric,title in (("final_f1","Macro-F1"),("accuracy","Accuracy")):
        if not s["baseline_scores"]:
            continue
        fig,axes=plt.subplots(1,3,figsize=(17,6.2),sharey=True)
        plot_methods = (MAIN, "cat_mmd_tag_cctp_top5", "gecc_hard_tag", "puma_full",
                        "tagsam_selfloop", "random_real", "k_center", "medoid")
        for ax,d in zip(axes,DATASETS):
            for offset,(student,color) in enumerate((("legacy","#c07034"),("prompt","#187b76"))):
                for i,a in enumerate(plot_methods):
                    if d == "sportsfit" and a == "cat_mmd_tag_cctp_top5":
                        a = "cat_cctp_repaired"
                    r=next((r for r in s["baseline_scores"] if (r["dataset"],r["arm"],r["student"])==(d,a,student)),None)
                    if r is not None:
                        ax.barh(i+(offset-.5)*.34,r[metric],height=.30,color=color,
                                xerr=r[metric+"_std"],label=student if i==0 else None)
                        ax.text(r[metric]+r[metric+"_std"]+1,i+(offset-.5)*.34,
                                f"{r[metric]:.1f}",va="center",fontsize=8)
            if d == "sportsfit":
                ax.text(3, plot_methods.index("tagsam_selfloop"), "Excluded: different order",
                        va="center", fontsize=8, color="#666666")
            ax.set_yticks(range(len(plot_methods)),[NAMES[a] for a in plot_methods],fontsize=9)
            ax.set_title(d)
            ax.set_xlim(0,103)
            ax.set_xlabel(f"Final {title} (%)")
            ax.spines[["top","right"]].set_visible(False)
        axes[0].invert_yaxis()
        fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False)
        fig.suptitle(f"Handoff-order {title}; same memory under two Students (3 evaluation seeds)")
        fig.text(.5, .075, "Error bars: Student-seed SD. SportsFit CaT: corrected nearest5 at fixed archived X.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0,.13,1,.94))
        fig.savefig(root/f"handoff_{metric}.png",dpi=170)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path("results/shared_q_baseline_20261008"))
    args=p.parse_args()
    shared,handoff,incomplete=load(args.root)
    s=compute(shared,handoff)
    s.update(shared_cells=len(shared),handoff_datasets=len(handoff),incomplete=incomplete,
             shared_legacy=legacy_shared_summary(args.root,shared))
    (args.root/"summary.json").write_text(json.dumps(s,indent=2,allow_nan=False),encoding="utf-8")
    (args.root/"summary.md").write_text(render(s),encoding="utf-8")
    plots(s,args.root)
    print(json.dumps(dict(shared=len(shared),handoff=len(handoff),incomplete=incomplete)))


if __name__ == "__main__":
    main()
