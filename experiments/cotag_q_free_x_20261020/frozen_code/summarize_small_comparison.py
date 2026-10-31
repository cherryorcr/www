"""Audit saved checkpoints and export small-graph comparison evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch

import run_small_comparison_audited as C


def effect(a, b):
    delta = np.array(a) - np.array(b)
    half = 2.570581835636 * delta.std(ddof=1) / math.sqrt(len(delta))
    return {"mean": float(delta.mean()), "ci_low": float(delta.mean() - half),
            "ci_high": float(delta.mean() + half), "positive_seeds": int((delta > 0).sum())}


def field(rows, scope="all", key="cumulative_f1", selected=False):
    return [r["selected_metrics" if selected else "metrics"][scope][key] for r in rows]


def audit_run(root, adapter):
    payload = json.loads((root / "results.json").read_text())
    m, rows = payload["manifest"], payload["results"]
    errors, checks = [], 0
    expected_arms = m["arms"]
    if len(rows) != 3 * len(expected_arms):
        errors.append("incomplete arm/stage matrix")
    if m["seeds"] != list(range(100, 106)):
        errors.append("incorrect student seed list")
    if m["test_labels_evaluated"] is not False:
        errors.append("test labels not sealed")
    data_file = C.U.DATA_CFG[m["dataset"]]["path"]
    if C.sha(data_file) != m.get("data_sha256", m.get("source_sha256")):
        errors.append("dataset hash differs")
    original = torch.load(data_file, map_location="cpu", weights_only=False)
    masks = C.P.split_masks_v4(torch.arange(len(original.x)), C.P.REPORT_DEV_MOD)
    source = {"x": original.x, "y": original.y, "texts": list(original.raw_texts),
              "edge_index": original.edge_index, "pool_mask": masks[3],
              "text_emb": torch.load(C.artifact_dir(m["dataset"], "v4") / "text_emb_pool_only.pt",
                                      map_location="cpu", weights_only=False)}
    stage_classes = [[0, 1], [2, 3], [4, 5, 6]] if m["dataset"] == "cora" else [[0, 1], [2, 3], [4, 5]]
    stage_ids = [torch.where(masks[3] & torch.isin(original.y, torch.tensor(cs)))[0] for cs in stage_classes]
    if adapter:
        for stage, ids in enumerate(stage_ids):
            teacher = torch.load(root / f"teacher_s{stage}.pt", map_location="cpu", weights_only=False)
            if not torch.equal(ids, teacher["source_ids"]):
                errors.append(f"teacher current-delta access differs: {stage}")
            if teacher["support_sha256"] != C.tensor_hash(C.raw_support(source, ids, stage)):
                errors.append(f"teacher support differs: {stage}")
    if not adapter:
        for name, digest in m["code_sha256"].items():
            if C.sha(Path(C.__file__).parent / name) != digest:
                errors.append(f"code hash differs: {name}")
    else:
        if C.sha(Path(C.__file__).with_name("run_tagsam_graphclip_comparison.py")) != m["runner_sha256"]:
            errors.append("adapter runner hash differs")
        if C.sha(C.__file__) != m["trainer_sha256"]:
            errors.append("adapter trainer hash differs")
    by_arm = {arm: sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["stage"])
              for arm in expected_arms}
    for arm, stages in by_arm.items():
        if [r["stage"] for r in stages] != [0, 1, 2]:
            errors.append(f"missing stage: {arm}")
            continue
        old = None
        for row in stages:
            stage = row["stage"]
            if arm.endswith("finetune"):
                mempath = (root / ("v4_s0_memory.pt" if arm.startswith("v4") else "raw_equal_s0_memory.pt")) if stage == 0 else None
                memory = (torch.load(mempath, map_location="cpu", weights_only=False) if mempath
                          else C.raw_support(source, stage_ids[stage], stage))
            else:
                mempath = root / f"{arm}_s{stage}_memory.pt"
                memory = torch.load(mempath, map_location="cpu", weights_only=False)
            if not arm.startswith("raw") and not arm.endswith("finetune") and row["rows"] != m["budgets"][stage]:
                errors.append(f"wrong budget: {arm}/{stage}")
            if memory is not None:
                if "source_ids" in memory:
                    ids = memory["source_ids"]
                    allowed = torch.cat(stage_ids[:stage + 1])
                    if len(ids) != len(torch.unique(ids)) or not torch.isin(ids, allowed).all():
                        errors.append(f"illegal real source IDs: {arm}/{stage}")
                if arm.startswith("raw") and not arm.endswith("finetune"):
                    expected_raw = C.raw_support(source, torch.cat(stage_ids[:stage + 1]), stage)
                    if C.tensor_hash(expected_raw) != C.tensor_hash(memory):
                        errors.append(f"raw training graph differs from legal induced graph: {arm}/{stage}")
                if not torch.isfinite(memory["x"]).all() or not torch.isfinite(memory["text_emb"]).all():
                    errors.append(f"nonfinite support: {arm}/{stage}")
                if arm not in ("full_history_syn", "recluster") and not arm.startswith("raw") and not arm.endswith("finetune"):
                    try:
                        C.assert_frozen(old, memory)
                    except AssertionError as e:
                        errors.append(f"{arm}/{stage}: {e}")
                old = memory
            scores = row["per_seed"]
            if [s["seed"] for s in scores] != m["seeds"]:
                errors.append(f"missing seed: {arm}/{stage}")
            for score in scores:
                seed = score["seed"]
                key = f"{arm}_s{stage}_seed{seed}"
                saved = torch.load(root / f"{key}.pt", map_location="cpu", weights_only=False)
                leaf = json.loads((root / f"{key}.json").read_text())
                # JSON converts integer class-map keys to strings; normalize
                # the checkpoint metadata before comparing serialized records.
                saved_result = json.loads(json.dumps(saved["result"]))
                if score != leaf or saved_result != score:
                    errors.append(f"checkpoint/summary disagreement: {key}")
                if C.tensor_hash(saved["state"]) != score["end_state_sha256"]:
                    errors.append(f"parameter hash differs: {key}")
                if memory is not None and C.tensor_hash(memory) != score["support_sha256"]:
                    errors.append(f"support hash differs: {key}")
                if arm.endswith("finetune") and stage > 0:
                    prev = json.loads((root / f"{arm}_s{stage - 1}_seed{seed}.json").read_text())
                    if prev["end_state_sha256"] != score["start_state_sha256"]:
                        errors.append(f"fine-tuning chain broken: {key}")
                if stage == 2 and score["metrics"]["all"] != score["metrics"]["seen"]:
                    errors.append(f"final prompt scopes differ: {key}")
                for scope in ("all", "seen"):
                    val = score["metrics"][scope]
                    if abs(np.mean(list(val["per_class_f1"].values())) - val["cumulative_f1"]) > 1e-9:
                        errors.append(f"cumulative/class aggregation differs: {key}/{scope}")
                expected_steps = 8192 if arm == "raw_long" else C.P.student_steps(m["budgets"][stage])
                if score["steps"] != expected_steps:
                    errors.append(f"wrong optimization budget: {key}")
                if arm == "raw_long":
                    best = max(score["selection_curve"], key=lambda s: s["selector_all_f1"])
                    if best["step"] != score["selected_steps"] or saved["selected_state"] is None:
                        errors.append(f"invalid selector checkpoint: {key}")
                checks += 1
    return payload, errors, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--plots", action="store_true",
                    help="Optional plots when matplotlib is installed; otherwise use the standalone renderer")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    all_rows, table_rows, paired, source_hashes, errors = {}, [], {}, {}, []
    count = 0
    for dataset in ("cora", "citeseer"):
        merged, payloads = {}, []
        for parent, adapter in ((Path(args.core), False), (Path(args.adapter), True)):
            root = parent / dataset
            p, err, checked = audit_run(root, adapter)
            count += checked
            errors.extend([f"{dataset}/{root.parent.name}: {e}" for e in err])
            source_hashes[str(root / "results.json")] = C.sha(root / "results.json")
            payloads.append(p)
            for row in p["results"]:
                merged.setdefault(row["arm"], []).append(row)
        all_rows[dataset] = merged
        paired[dataset] = {}
        baseline = field(merged["v4"][-1]["per_seed"])
        for arm, stages in merged.items():
            stages.sort(key=lambda r: r["stage"])
            variants = [(arm, False)] + ([("raw_long_selected", True)] if arm == "raw_long" else [])
            for label, selected in variants:
                final = stages[-1]
                values = field(final["per_seed"], selected=selected)
                mean_stage_by_seed = np.mean([field(r["per_seed"], selected=selected) for r in stages], axis=0)
                seen_mean = np.mean([field(r["per_seed"], "seen", selected=selected) for r in stages], axis=0)
                all_scores = [s for r in stages for s in r["per_seed"]]
                row = {"dataset": dataset, "arm": label, "final_mean": float(np.mean(values)),
                    "final_std": float(np.std(values)), "stage_average_all": float(np.mean(mean_stage_by_seed)),
                    "stage_average_seen": float(np.mean(seen_mean)),
                    "final_old": float(np.mean(field(final["per_seed"], key="old_f1", selected=selected))),
                    "final_new": float(np.mean(field(final["per_seed"], key="new_f1", selected=selected))),
                    "final_rows": final["rows"], "student_optimizer_steps": sum(s["steps"] for s in all_scores),
                    "student_seconds": sum(s["seconds"] for s in all_scores),
                    "peak_student_allocated_mb": max(s["peak_allocated_mb"] for s in all_scores)}
                for stage in range(3):
                    row[f"s{stage}_all"] = float(np.mean(field(stages[stage]["per_seed"], selected=selected)))
                    row[f"s{stage}_seen"] = float(np.mean(field(stages[stage]["per_seed"], "seen", selected=selected)))
                if selected:
                    row["selected_steps_by_stage"] = [[s["selected_steps"] for s in r["per_seed"]] for r in stages]
                table_rows.append(row)
                paired[dataset][label] = {"v4_minus_arm": effect(baseline, values)}
        expected = {"dispersion": 49.85181187054925 if dataset == "cora" else 59.17466849390389,
                    "v4": 49.85181187054925 if dataset == "cora" else 60.76531142197674,
                    "concat64": 50.63816182131276 if dataset == "cora" else 56.025257184774084}
        for arm, value in expected.items():
            if abs(np.mean(field(merged[arm][-1]["per_seed"])) - value) > 1e-7:
                errors.append(f"historical regression differs: {dataset}/{arm}")
        # Construction compute is separate from student training compute.
        core_audits = payloads[0]["audits"]
        for row in table_rows:
            if row["dataset"] != dataset:
                continue
            arm = row["arm"]
            row["archived_selector_fits"] = sum(
                a.get("selected_allocation", {}).get("selection_training_runs", 0)
                for a in core_audits.get(arm, []))
            row["archived_selector_steps"] = sum(
                a.get("selected_allocation", {}).get("selection_optimizer_steps_total", 0)
                for a in core_audits.get(arm, []))
            row["adapter_outer_steps"] = sum(a.get("optimizer_steps", 0)
                for a in payloads[1]["audits"] if a["arm"] == arm)
            row["adapter_inner_steps"] = sum(a.get("inner_steps_total", 0)
                for a in payloads[1]["audits"] if a["arm"] == arm)
            row["adapter_teacher_steps"] = 3 * 4096 if arm.startswith(("tagsam_", "asm_")) else 0
    verdict = {"integrity_passed": not errors, "errors": errors, "checked_student_fits": count,
               "test_labels_evaluated": False, "datasets": ["cora", "citeseer"],
               "source_results_sha256": source_hashes,
               "inference_scope": "one construction/order; six paired student initializations; dev only",
               "official_tagsam_reproduction": False, "large_graph_started": False}
    C.save_json(out / "acceptance.json", verdict)
    C.save_json(out / "runtime_environment.json", {
        "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__,
        "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "note": "Collected in the same tagsam environment after execution; GPU co-tenancy affects wall time."})
    C.save_json(out / "summary.json", {"rows": table_rows, "paired": paired})
    columns = sorted({k for r in table_rows for k in r})
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table_rows)
    lines = ["# Small Graph Comparison Tables", "", "Macro-F1 (%), report-dev, final-stage all/seen prompts coincide.", ""]
    for dataset in ("cora", "citeseer"):
        lines += [f"## {dataset}", "", "| Arm | Final mean +/- SD | Stage average (all / seen) | Old / New |", "|---|---:|---:|---:|"]
        for r in table_rows:
            if r["dataset"] == dataset:
                lines.append(f"| {r['arm']} | {r['final_mean']:.2f} +/- {r['final_std']:.2f} | {r['stage_average_all']:.2f} / {r['stage_average_seen']:.2f} | {r['final_old']:.2f} / {r['final_new']:.2f} |")
        lines += [""]
    (out / "tables.md").write_text("\n".join(lines))
    if errors:
        raise SystemExit(json.dumps(verdict))
    if not args.plots:
        print(json.dumps(verdict), flush=True)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    styles = {"v4": ("#007f8b", "v4"), "k_center": ("#7a5195", "k-center"),
              "raw_equal": ("#666666", "Full data, equal steps"),
              "raw_finetune": ("#bc5090", "Raw delta fine-tune"),
              "init_selfloop": ("#e49a00", "Synthetic self-loop control"),
              "tagsam_semantic": ("#d14836", "TAGSAM semantic adapter")}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, constrained_layout=True)
    for col, dataset in enumerate(("cora", "citeseer")):
        for row, scope in enumerate(("all", "seen")):
            axis = axes[row, col]
            for arm, (color, label) in styles.items():
                stages = all_rows[dataset][arm]
                means = [np.mean(field(r["per_seed"], scope)) for r in stages]
                axis.plot([1, 2, 3], means, "o-", color=color, label=label, linewidth=1.6, markersize=4)
            axis.set_title(f"{dataset.capitalize()} | {scope}-class prompts")
            axis.set_xticks([1, 2, 3])
            axis.set_ylim(0, 100)
            axis.grid(axis="y", alpha=.18)
            if col == 0:
                axis.set_ylabel("Seen-sample Macro-F1 (%)")
            if row == 1:
                axis.set_xlabel("Stage")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.savefig(out / "stage_curves.png", dpi=180)
    fig.savefig(out / "stage_curves.pdf")
    plt.close(fig)
    final_arms = ["v4", "dispersion", "concat64", "k_center", "raw_equal", "raw_long",
                  "raw_long_selected", "raw_finetune", "v4_init_finetune", "init_selfloop",
                  "tagsam_selfloop", "tagsam_semantic", "asm_rowcol_semantic"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)
    for axis, dataset in zip(axes, ("cora", "citeseer")):
        selected = [next(r for r in table_rows if r["dataset"] == dataset and r["arm"] == a) for a in final_arms]
        axis.barh(range(len(selected)), [r["final_mean"] for r in selected],
                  xerr=[r["final_std"] for r in selected], color=["#007f8b" if a == "v4" else "#929a9e" for a in final_arms],
                  capsize=2)
        axis.set_title(dataset.capitalize())
        axis.set_yticks(range(len(final_arms)), final_arms)
        axis.set_xlabel("Final Macro-F1 (%) +/- student SD")
        axis.set_xlim(0, 85)
        axis.grid(axis="x", alpha=.18)
    axes[0].invert_yaxis()
    fig.savefig(out / "final_comparison.png", dpi=180)
    fig.savefig(out / "final_comparison.pdf")
    print(json.dumps(verdict), flush=True)


if __name__ == "__main__":
    main()
