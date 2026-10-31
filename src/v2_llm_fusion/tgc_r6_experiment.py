"""Sealed experiment lifecycle with arbitrary evaluated stages and arms."""
import json
import os
from pathlib import Path

import torch
import run_tgc_r5_old_confirmation as O

R, T, D, B = O.R, O.T, O.D, O.B
read, load, save = O.read, O.load, O.save


def lock_protocol(path, spec):
    # Two GPU workers may initialize the same protocol at the same instant.
    if os.name == "posix":
        import fcntl
        with path.with_suffix(".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            R.stable_lock(path, spec)
    else:
        R.stable_lock(path, spec)


def configure(spec):
    D.ARMS, D.SEEDS, D.ORDERS = tuple(spec["arms"]), tuple(spec["report_seeds"]), tuple(spec["orders"])


def arms_at(spec, stage):
    return spec.get("stage_arms", {}).get(str(stage), spec["arms"])


def seal(root, spec):
    records = {}
    for dataset in spec["datasets"]:
        for order in spec["orders"]:
            out = root / dataset / order
            assert read(out / "prepared.json")["status"] == "prepared"
            for stage in spec["evaluation_stages"]:
                D.ARMS = tuple(arms_at(spec, stage))
                D.verify_decision(out, stage)
            for path in sorted(out.glob("*.json")):
                records[path.relative_to(root).as_posix()] = R.C.sha(path)
    R.stable_lock(root / "selection_seal.json", {"decisions": records,
                                                "protocol_sha256": R.C.sha(root / "protocol.json")})


def evaluate(data, order, args, spec):
    seal_hash = D.verify_seal(args.output)
    D.set_stream(data, order)
    out, results = args.output / data["dataset"] / order, []
    for stage in spec["evaluation_stages"]:
        D.ARMS = tuple(arms_at(spec, stage))
        D.verify_decision(out, stage)
        sources = {}
        for arm in arms_at(spec, stage):
            memory = load(out / f"{arm}_s{stage}_memory.pt")
            sha, rows = T.memory_hash(memory), []
            for seed in spec["report_seeds"]:
                if sha in sources:
                    row = B.alias_fit(sources[sha], arm, stage, seed, out)
                else:
                    row = R.fit(memory, data, stage, seed, arm, out, args.device, None)
                    row["selection_seal_sha256"] = seal_hash
                    save(out / f"{arm}_s{stage}_seed{seed}.json", row)
                rows.append(row)
            sources.setdefault(sha, arm)
            results.append({"stage": stage, "arm": arm, "metrics": R.C.aggregate(rows), "per_seed": rows})
            save(out / "progress.json", {"results": results})
            print(json.dumps({"event": "evaluated", "dataset": data["dataset"], "order": order,
                              "stage": stage, "arm": arm,
                              "f1": results[-1]["metrics"]["all"]["cumulative_f1"]["mean"]}), flush=True)
    save(out / "results.json", {"results": results})
    save(out / "completion.json", {"status": "complete", "fit_slots": len(results) * len(spec["report_seeds"]),
                                    "executed_fits": sum(v["executed_here"] for r in results for v in r["per_seed"])})


def audit(data, order, args, spec, validate):
    from audit_tgc_r2b_student_revision import independent_scores
    D.verify_seal(args.output)
    assert R.source_hashes() == spec["source_hashes"]
    ids, budgets = D.set_stream(data, order)
    out = args.output / data["dataset"] / order
    manifest = read(out / "manifest.json")
    assert manifest["stage_source_ids"] == [v.tolist() for v in ids]
    assert manifest["budgets"] == budgets
    assert manifest["source_dataset_sha256"] == R.C.sha(R.U.DATA_CFG[data["dataset"]]["path"])
    cache, maximum, checked = {}, 0., 0
    for stage in spec["evaluation_stages"]:
        D.ARMS = tuple(arms_at(spec, stage))
        D.verify_decision(out, stage)
        validate(data, out, stage, ids, budgets, args)
        for arm in arms_at(spec, stage):
            memory = load(out / f"{arm}_s{stage}_memory.pt")
            for seed in spec["report_seeds"]:
                row = read(out / f"{arm}_s{stage}_seed{seed}.json")
                checkpoint = out / f"{arm}_s{stage}_seed{seed}.pt"
                assert row["checkpoint_sha256"] == R.C.sha(checkpoint)
                assert row["support_memory_sha256"] == T.memory_hash(memory)
                assert row["steps"] == R.P.student_steps(budgets[stage]) and row["seed"] == seed
                key = (stage, row["checkpoint_sha256"])
                if key not in cache:
                    model = R.G.CLIPStudent(memory["x"].shape[1]).to(args.device)
                    model.load_state_dict(load(checkpoint)["state"])
                    cache[key] = independent_scores(model, data, stage)
                    del model
                for scope in ("all", "seen"):
                    for metric, value in cache[key][scope].items():
                        if value is not None:
                            maximum = max(maximum, abs(value - row["metrics"][scope][metric]))
                checked += 1
    assert checked == sum(len(arms_at(spec, stage)) for stage in spec["evaluation_stages"]) * len(spec["report_seeds"])
    assert maximum < 1e-7
    save(out / "audit.json", {"status": "passed", "fit_slots": checked, "unique_reload_scores": len(cache),
                               "maximum_metric_error_pp": maximum, "test_labels_evaluated": False})


def manifest(data, order, budgets, ids):
    return {"dataset": data["dataset"], "order": order, "stage_classes": data["stage_classes"],
            "budgets": budgets, "stage_source_ids": [v.tolist() for v in ids],
            "source_dataset_sha256": R.C.sha(R.U.DATA_CFG[data["dataset"]]["path"])}


def memory_checks(memory, rows):
    assert len(memory["x"]) == rows
    assert all(torch.isfinite(memory[k]).all() for k in ("x", "text_emb", "mass"))
    e = memory["edge_index"]
    assert e.shape[0] == 2 and (e >= 0).all() and (e < rows).all()
    pairs = set(map(tuple, e.T.tolist()))
    assert len(pairs) == e.shape[1] and all((j, i) in pairs for i, j in pairs)


def main(protocol, prepare, validate):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"))
    parser.add_argument("--phase", choices=("prepare", "seal", "evaluate", "audit"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    spec = protocol()
    configure(spec)
    lock_protocol(args.output / "protocol.json", spec)
    if args.phase == "seal":
        seal(args.output, spec)
        return
    if not args.dataset:
        parser.error("dataset required")
    if args.phase in ("evaluate", "audit"):
        D.verify_seal(args.output)
    cache = Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919") / f"{args.dataset}_r005/text_emb_pool_only.pt"
    data = R.R.load_data(args.dataset, args.minilm, args.device, cache)
    for order in spec["orders"]:
        if args.phase == "prepare":
            prepare(data, order, args, spec)
        elif args.phase == "evaluate":
            evaluate(data, order, args, spec)
        else:
            audit(data, order, args, spec, validate)
    print(f"TGC_R6_{args.phase.upper()}_COMPLETE", flush=True)
