"""Pre-update text-gradient diagnostics; no Student scoring or parameter search."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

import run_unified_q_attribution_20261008 as A
from unified_q_text_diagnostics import TextPathDiagnostics

R, L, V, M, U = A.R, A.L, A.V, A.M, A.UQ


def run(args):
    reference = L.read(args.reference)
    groups = V.choose_groups(reference, args.order_seed)
    data = L.load_large(args.dataset, args.minilm, args.text_cache, args.device,
        dict(groups=groups, report_dev_mod=reference.get("report_dev_mod", 3)))
    ids = L.stage_ids(data)
    budgets, _ = V.choose_budgets(reference, list(map(len, ids)), groups, "fixed_schedule", .05)
    cfg = R.q_config("uq_dm_edgefree", args.construction_seed, args.q_steps, 2048)
    protocol = dict(dataset=args.dataset, groups=groups, budgets=budgets,
        order_seed=args.order_seed, construction_seed=args.construction_seed, config=asdict(cfg),
        stage_id_hashes=[V.hash_tensor(i) for i in ids], pool_mask_hash=V.hash_tensor(data["pool_mask"]),
        report_mask_hash=V.hash_tensor(data["report_dev_mask"]), readout="none; mechanism only",
        historical_raw_access="only current delta plus own synthetic prefix; interface audit",
        source_hashes={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                       for m in (R, L, V, M, U)}, no_parameter_search=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.json").exists():
        raise ValueError("preserve existing diagnostic; use a fresh output")
    result = dict(protocol=protocol, status="running", stages=[])
    V.write_json(args.output / "protocol.json", protocol)
    V.write_json(args.output / "results.json", result)
    parent = None
    for stage, raw_ids in enumerate(ids[:args.max_stages]):
        delta = L.delta_view(data, raw_ids, stage)
        rows = budgets[stage] - (0 if parent is None else len(parent["x"]))
        labels = M.row_labels(M.ratio_allocation(delta["y"], rows))
        baseline, repeat = None, None
        if args.check_observer and stage == 0:
            with A.preserve_rng():
                baseline, _ = U.optimize_stage(delta, parent, labels, data["encoder"],
                    data["prompt_emb"], cfg, args.device, use_boundary=False, use_topology=False)
            with A.preserve_rng():
                repeat, _ = U.optimize_stage(delta, parent, labels, data["encoder"],
                    data["prompt_emb"], cfg, args.device, use_boundary=False, use_topology=False)
        diagnostic, measurements = TextPathDiagnostics(cfg.steps), {}
        V.sync(args.device)
        started = time.perf_counter()
        block, audit = U.optimize_stage(delta, parent, labels, data["encoder"], data["prompt_emb"],
            cfg, args.device, use_boundary=False, use_topology=False, optimization_callback=diagnostic,
            diagnostic_callback=lambda state: measurements.update(diagnostic.final(state)))
        V.sync(args.device)
        memory = U.append_memory(parent, block)
        checks = V.validate_memory(memory, parent, budgets[stage])
        checks.update(R.assignment_checks(audit))
        if not checks["sinkhorn_feasible"]:
            raise AssertionError("infeasible assignment")
        with A.preserve_rng():
            checks.update(A.validate_payload(block, delta, data["encoder"]))
        observer = None
        if baseline is not None:
            observer = dict(baseline_hash=V.memory_hash(baseline), observed_hash=V.memory_hash(block),
                x_max_error=float((baseline["x"] - block["x"]).abs().max()),
                off_off_x_max_error=float((baseline["x"] - repeat["x"]).abs().max()),
                relative_x_error=float((baseline["x"] - block["x"]).norm() / baseline["x"].norm().clamp_min(1e-12)),
                off_off_relative_x_error=float((baseline["x"] - repeat["x"]).norm() / baseline["x"].norm().clamp_min(1e-12)),
                off_off_sources_equal=baseline["text_source_ids"] == repeat["text_source_ids"],
                sources_equal=baseline["text_source_ids"] == block["text_source_ids"])
            observer["within_replay_envelope"] = observer["x_max_error"] <= 2 * observer["off_off_x_max_error"] + 1e-6
            if observer["relative_x_error"] > 1e-3 or not observer["sources_equal"]:
                V.write_json(args.output / "observer_failure.json", observer)
                raise AssertionError("observer replay has material drift; inspect off/off control")
        result["stages"].append(dict(stage=stage, rows=rows, raw_rows=len(raw_ids),
            checks=checks, observer=observer, memory_hash=V.memory_hash(memory),
            seconds=time.perf_counter() - started, audit=R.compact_audit(audit), measurements=measurements))
        V.write_json(args.output / "results.json", result)
        print(json.dumps(dict(dataset=args.dataset, stage=stage, event="diagnosed",
                              **measurements["final"]["macro"])), flush=True)
        parent = memory
    result["status"] = "complete" if args.max_stages >= len(ids) else "smoke_complete"
    V.write_json(args.output / "results.json", result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=("wikics", "sportsfit", "bookhis"))
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--order-seed", type=int, default=9101)
    p.add_argument("--construction-seed", type=int, default=271006)
    p.add_argument("--q-steps", type=int, default=30)
    p.add_argument("--max-stages", type=int, default=5)
    p.add_argument("--check-observer", action="store_true")
    torch.set_num_threads(2)
    run(p.parse_args())


if __name__ == "__main__":
    main()
