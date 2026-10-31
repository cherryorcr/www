"""Audit the five-level extension, including re-evaluation of saved models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import run_small_comparison_audited as C
import run_supervised_upper_bound_small as S
import run_tagsam_graphclip_static as T
import summarize_small_comparison as B
from small_comparison_metrics import evaluate


def normalized(value):
    return json.loads(json.dumps(value))


def audit_extension(root, dataset, device):
    errors, fits = [], 0
    data = T.load_data(dataset, str(Path(C.__file__).parent / "minilm_from_graphclip"), device)
    stage_ids, budgets = T.stage_inputs(data)

    def check(condition, message):
        if not condition:
            errors.append(message)

    payloads = {}
    for family, module in (("static", T), ("supervised", S)):
        path = root / family / dataset
        payload = json.loads((path / "results.json").read_text())
        payloads[family] = payload
        m = payload["manifest"]
        check(m["runner_sha256"] == C.sha(module.__file__), f"{family}: runner hash")
        check(m["source_sha256"] == C.sha(C.U.DATA_CFG[dataset]["path"]), f"{family}: source hash")
        check(m["test_labels_evaluated"] is False, f"{family}: test evaluated")
        check(m["report_dev_used_for_selection"] is False, f"{family}: report selection")
        dependencies = {Path(v.__file__).name: v.__file__ for v in (C, T.T, C.U, C.G, C.SCA, C.P)}
        for name, digest in m["dependencies"].items():
            check(C.sha(dependencies[name]) == digest, f"{family}: dependency {name}")
        seeds = m.get("student_seeds", m.get("seeds"))
        check(seeds == list(range(100, 106)), f"{family}: seed list")
        arms = m.get("arms", ["supervised_full_data_upper_bound"])
        expected = {(arm, stage) for arm in arms for stage in range(3)}
        check({(r["arm"], r["stage"]) for r in payload["results"]} == expected
              and len(payload["results"]) == len(expected), f"{family}: incomplete matrix")
        if family == "static":
            check(m["budgets"] == budgets, "static: budgets")
            _, official = T.T.official_loss(Path("/home/pc/research/TAGSAM_official"))
            check(official == m["official"], "static: official source provenance")
            check(m["text_cache_sha256"] == C.sha(C.artifact_dir(dataset, "v4") / "text_emb_pool_only.pt"),
                  "static: text cache hash")
        else:
            check(not set(m["selection_seeds"]) & set(seeds), "supervised: seed overlap")

        for row in payload["results"]:
            stage, arm = row["stage"], row["arm"]
            ids = torch.cat(stage_ids[:stage + 1])
            if family == "static":
                mempath = path / f"{arm}_s{stage}_memory.pt"
                memory = torch.load(mempath, map_location="cpu", weights_only=False)
                audit = json.loads((path / f"{arm}_s{stage}_construction.json").read_text())
                check(audit == payload["audits"][arm][stage], f"{arm}/{stage}: construction summary")
                check(C.sha(mempath) == audit["memory_file_sha256"], f"{arm}/{stage}: memory file hash")
                check(C.tensor_hash(memory) == audit["tensor_sha256"], f"{arm}/{stage}: memory tensor hash")
                check(len(memory["x"]) == budgets[stage], f"{arm}/{stage}: budget")
                check(torch.isfinite(memory["x"]).all().item(), f"{arm}/{stage}: nonfinite X")
                check(audit["source_ids_sha256"] == C.tensor_hash({"source_ids": ids}), f"{arm}/{stage}: sources")
                teacher = torch.load(path / f"teacher_s{stage}.pt", map_location="cpu", weights_only=False)
                check(teacher["support_sha256"] == C.tensor_hash(C.raw_support(data, ids, stage)),
                      f"{arm}/{stage}: full-history teacher support")
                check(teacher["steps"] == m["teacher_steps"], f"{arm}/{stage}: teacher steps")
                check(C.tensor_hash(teacher["state"]) == audit["teacher_state_sha256"], f"{arm}/{stage}: teacher hash")
                check(C.sha(path / f"teacher_s{stage}.pt") == audit["teacher_checkpoint_sha256"],
                      f"{arm}/{stage}: teacher file hash")
                initial = T.build_initial(data, ids, budgets[stage], stage, device)
                check(C.tensor_hash(initial) == audit["initial_tensor_sha256"], f"{arm}/{stage}: initialization replay")
                check(initial["texts"] == memory["texts"] and torch.equal(initial["text_emb"], memory["text_emb"]),
                      f"{arm}/{stage}: frozen text")
                expected_edges = C.SCA.memory_edges(initial, device=device) if arm.endswith("semantic") else torch.empty((2, 0), dtype=torch.long)
                check(torch.equal(expected_edges, memory["edge_index"]), f"{arm}/{stage}: frozen topology")
                check(len(audit["optimization"]["curve"]) == m["outer_steps"], f"{arm}/{stage}: outer budget")
            else:
                S.validate_train_ids(data, ids, stage)
                search = json.loads((path / f"search_s{stage}.json").read_text())
                check(len(search["grid"]) == len(m["learning_rates"]) * len(m["weight_decays"]),
                      f"supervised/{stage}: grid size")
                for trial_group in search["grid"]:
                    check([t["seed"] for t in trial_group["trials"]] == m["selection_seeds"],
                          f"supervised/{stage}: search seeds")
                    for trial in trial_group["trials"]:
                        key = f"search_s{stage}_lr{trial['lr']}_wd{trial['weight_decay']}_seed{trial['seed']}"
                        recovered = S.load_cached_fit(path / f"{key}.json", path / f"{key}.pt", ids)
                        check(recovered == trial and "metrics" not in trial and "selected_metrics" not in trial,
                              f"{key}: search reads report or differs")
                        best = max(trial["selection_curve"], key=lambda v: v["selector_all_f1"])
                        check(best["step"] == trial["selected_steps"] and best["selector_all_f1"] == trial["selector_best_f1"],
                              f"{key}: selection mismatch")
                        fits += 1
                    check(abs(trial_group["selector_mean"] - np.mean([t["selector_best_f1"] for t in trial_group["trials"]])) < 1e-10,
                          f"supervised/{stage}: grid mean")
                winner = max(search["grid"], key=lambda v: v["selector_mean"])
                check(all(row["selected_config"][k] == winner[k] for k in ("lr", "weight_decay")),
                      f"supervised/{stage}: selected grid config")

            check([r["seed"] for r in row["per_seed"]] == seeds, f"{arm}/{stage}: missing seed")
            for score in row["per_seed"]:
                key = f"{arm}_s{stage}_seed{score['seed']}" if family == "static" else f"supervised_s{stage}_seed{score['seed']}"
                checkpoint = torch.load(path / f"{key}.pt", map_location="cpu", weights_only=False)
                leaf = json.loads((path / f"{key}.json").read_text())
                check(normalized(checkpoint["result"]) == leaf == score, f"{key}: result disagreement")
                check(C.tensor_hash(checkpoint["state"]) == score["end_state_sha256"], f"{key}: state hash")
                if family == "static":
                    check(C.tensor_hash(memory) == score["support_sha256"], f"{key}: student support")
                    check(score["steps"] == C.P.student_steps(budgets[stage]), f"{key}: canonical steps")
                    model = C.G.CLIPStudent(data["x"].shape[1]).to(device)
                    model.load_state_dict(checkpoint["state"])
                    restored = evaluate(model, data, stage)
                    check(normalized(restored) == score["metrics"], f"{key}: restored metrics")
                else:
                    check(S.load_cached_fit(path / f"{key}.json", path / f"{key}.pt", ids) == score, f"{key}: provenance")
                    check(all(score[k] == row["selected_config"][k] for k in ("lr", "weight_decay")), f"{key}: config")
                    best = max(score["selection_curve"], key=lambda v: v["selector_all_f1"])
                    check(best["step"] == score["selected_steps"], f"{key}: selected step")
                    check(1 <= score["steps"] <= m["steps"], f"{key}: training cap")
                    model = S.SupervisedGCN(data["x"].shape[1], 384, int(data["y"].max()) + 1).to(device)
                    for state, metric in (("state", "metrics"), ("selected_state", "selected_metrics")):
                        model.load_state_dict(checkpoint[state])
                        restored = S.score_model(model, data, stage, "report_dev_mask", device)
                        check(normalized(restored) == score[metric], f"{key}: restored {metric}")
                fits += 1
                del model
            computed = C.aggregate(row["per_seed"]) if family == "static" else S.aggregate_selected(row["per_seed"])
            check(computed == row["metrics"], f"{arm}/{stage}: aggregate")
    return payloads, errors, fits


def summary_row(dataset, arm, stages, selected=False):
    stages = sorted(stages, key=lambda v: v["stage"])
    final = B.field(stages[-1]["per_seed"], selected=selected)
    return {"dataset": dataset, "arm": arm, "final_mean": float(np.mean(final)),
            "final_std": float(np.std(final)), "final_rows": stages[-1]["rows"],
            "stage_average_all": float(np.mean([B.field(s["per_seed"], selected=selected) for s in stages])),
            "stage_average_seen": float(np.mean([B.field(s["per_seed"], scope="seen", selected=selected) for s in stages])),
            "final_old": float(np.mean(B.field(stages[-1]["per_seed"], key="old_f1", selected=selected))),
            "final_new": float(np.mean(B.field(stages[-1]["per_seed"], key="new_f1", selected=selected))),
            "stage_f1": [float(np.mean(B.field(s["per_seed"], selected=selected))) for s in stages],
            "steps_total": sum(r["steps"] for s in stages for r in s["per_seed"]),
            "seconds_total": sum(r["seconds"] for s in stages for r in s["per_seed"]),
            "final_per_seed": final}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extension", required=True)
    ap.add_argument("--core", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, errors, sources, paired, checked = [], [], {}, {}, 0
    for dataset in ("cora", "citeseer"):
        core, err, count = B.audit_run(Path(args.core) / dataset, False)
        errors.extend(f"{dataset}/core: {e}" for e in err)
        checked += count
        extensions, err, count = audit_extension(Path(args.extension), dataset, args.device)
        errors.extend(f"{dataset}/extension: {e}" for e in err)
        checked += count
        records = core["results"] + extensions["static"]["results"]
        arms = ("v4", "dispersion", "full_history_syn", "recluster", "raw_equal", "raw_long",
                "raw_finetune", "v4_init_finetune", *T.STATIC_ARMS)
        for arm in arms:
            stages = [r for r in records if r["arm"] == arm]
            rows.append(summary_row(dataset, arm, stages))
            if arm == "raw_long":
                rows.append(summary_row(dataset, "raw_long_selected", stages, True))
        rows.append(summary_row(dataset, "supervised_selected", extensions["supervised"]["results"], True))
        baseline = next(r["final_per_seed"] for r in rows if r["dataset"] == dataset and r["arm"] == "v4")
        paired[dataset] = {r["arm"]: B.effect(baseline, r["final_per_seed"]) for r in rows
                           if r["dataset"] == dataset and r["arm"] != "supervised_selected"}
        for path in (Path(args.core) / dataset / "results.json",
                     Path(args.extension) / "static" / dataset / "results.json",
                     Path(args.extension) / "supervised" / dataset / "results.json"):
            sources[str(path)] = C.sha(path)
    C.save_json(out / "summary.json", {"rows": rows, "paired_v4_minus_arm": paired})
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    acceptance = {"checked_fits": checked, "errors": errors, "error_count": len(errors),
                  "sources": sources, "auditor_sha256": C.sha(__file__),
                  "extension_saved_models_reevaluated": 108,
                  "test_labels_evaluated": False,
                  "scope": "integrity and implementation checks; not a claim of SOTA or official reproduction"}
    C.save_json(out / "acceptance.json", acceptance)
    print(json.dumps(acceptance), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
