"""Paired native/common-text test of shared versus independent assignments."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

import run_unified_q_constructor_attribution_20261008 as B
import shared_q_controls as C

A, R, L, V, M, P, U = B.A, B.R, B.L, B.V, B.M, B.P, B.U


def run(args):
    source = json.loads((args.source / "results.json").read_text())
    source_protocol = source["protocol"]
    reference = L.read(args.reference)
    groups = V.choose_groups(reference, args.order_seed)
    data = L.load_large(args.dataset, args.minilm, args.text_cache, args.device,
                        dict(groups=groups, report_dev_mod=reference.get("report_dev_mod", 3)))
    ids = L.stage_ids(data)
    budgets, _ = V.choose_budgets(reference, list(map(len, ids)), groups, "fixed_schedule", .05)
    cfg = R.q_config("uq_dm_edgefree", 271006, args.q_steps, 2048)
    protocol = dict(name="shared-Q-vs-independent-Q-20261008", dataset=args.dataset,
        groups=groups, budgets=budgets, order_seed=args.order_seed, construction_seed=271006,
        student=P.model_config(), student_seeds=[150,151,152], student_steps=args.student_steps,
        student_batch=24, pool_mask_hash=V.hash_tensor(data["pool_mask"]),
        report_mask_hash=V.hash_tensor(data["report_dev_mask"]), stage_id_hashes=[V.hash_tensor(i) for i in ids],
        q_config=asdict(cfg), arms=["uq_dm_edgefree", "split_q"], max_stages=args.max_stages,
        source_hashes={Path(m.__file__).name: B.file_hash(m.__file__) for m in (R,L,V,M,P,U,A,C)},
        runner_sha256=B.file_hash(__file__), split_q_anchor="mean of feature/text anchors",
        hypothesis="shared assignment may help or hurt; no outcome-based selection",
        source_cell_sha256=B.file_hash(args.source / "results.json"),
        evaluation_split="report-dev, not sealed test", common_payload="archived same-slot CaT nearest5; final only")
    protocol = json.loads(json.dumps(protocol))
    if args.max_stages == 5:
        B.assert_source_protocol(source_protocol, protocol)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else dict(protocol=protocol, rows=[], status="running")
    if result["protocol"] != protocol:
        raise ValueError("changed experiment on resume")
    V.write_json(result_path, result)
    parents = {a: None for a in protocol["arms"]}
    common = None
    for stage, raw_ids in enumerate(ids[:args.max_stages]):
        delta = dict(L.delta_view(data, raw_ids, stage), prompt_emb=data["prompt_emb"])
        new_n = budgets[stage] - (budgets[stage-1] if stage else 0)
        labels = M.row_labels(M.ratio_allocation(delta["y"], new_n))
        common_block, _ = A.load_block(args.source, "cat_cctp", stage)
        common = A.append_block(common, common_block)
        for arm in protocol["arms"]:
            archive = output / f"block_{arm}_s{stage}.pt"
            if archive.exists():
                block, audit = A.load_block(output, arm, stage)
            else:
                V.sync(args.device)
                start = time.perf_counter()
                with A.preserve_rng():
                    if arm == "uq_dm_edgefree":
                        block, audit = U.optimize_stage(delta, parents[arm], labels, data["encoder"],
                            data["prompt_emb"], cfg, args.device, use_boundary=False, use_topology=False)
                    else:
                        block, audit = C.optimize(delta, parents[arm], labels, data["encoder"],
                            data["prompt_emb"], cfg, args.device)
                V.sync(args.device)
                audit["build_seconds"] = time.perf_counter() - start
                audit["assignment_checks"] = R.assignment_checks(audit)
                if not audit["assignment_checks"]["sinkhorn_feasible"]:
                    raise AssertionError("infeasible assignment")
                if arm == "split_q":
                    for value in audit["assignments"].values():
                        if value["cross_class_mass"] or max(value["row_error"], value["column_relative_error"]) > 1.1e-4:
                            raise AssertionError("independent text assignment violates constraints")
                with A.preserve_rng():
                    audit["payload_validation"] = A.validate_payload(block, delta, data["encoder"])
                A.archive_block(output, arm, stage, block, audit)
            memory = U.append_memory(parents[arm], block)
            checks = V.validate_memory(memory, parents[arm], budgets[stage])
            readouts = [("native", memory)]
            if stage == len(ids)-1:
                readouts.append(("common_cat_payload", B.fixed_payload(memory, common)))
            for readout, evaluated in readouts:
                saved = [r for r in result["rows"] if (r["arm"],r["stage"],r["readout"]) == (arm,stage,readout)]
                if saved:
                    if len(saved) != 1 or saved[0]["memory_hash"] != V.memory_hash(evaluated):
                        raise AssertionError("resume memory mismatch")
                    continue
                V.sync(args.device)
                start = time.perf_counter()
                with A.preserve_rng():
                    score = L.score(evaluated, data, stage, [150,151,152], args.device, args.student_steps, 24)
                V.sync(args.device)
                record = dict(arm=arm, stage=stage, readout=readout, score=score,
                    checks=checks, audit=R.compact_audit(audit), memory_rows=budgets[stage],
                    memory_hash=V.memory_hash(evaluated), construction_seconds=audit["build_seconds"],
                    student_seconds=time.perf_counter()-start)
                if readout != "native":
                    record["checks"] = dict(checks, **A.assert_fixed_content(memory, evaluated))
                    record["common_payload_hash"] = R.digest(common["text_emb"])
                result["rows"].append(record)
                V.write_json(result_path, result)
                print(json.dumps(dict(event="evaluated", dataset=args.dataset, order=args.order_seed,
                    arm=arm, stage=stage, readout=readout, f1=score["final_f1"], accuracy=score["accuracy"])), flush=True)
            parents[arm] = memory
    result["status"] = "complete" if args.max_stages == 5 else "smoke_complete"
    V.write_json(result_path, result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--order-seed", type=int, default=9101)
    p.add_argument("--q-steps", type=int, default=30)
    p.add_argument("--student-steps", type=int, default=2048)
    p.add_argument("--max-stages", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
