"""Closed-loop constructor attribution with native and fixed-payload readouts."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

import run_unified_q_attribution_20261008 as A
import unified_q_constructor_attribution as C

R, L, V, M, P, U = A.R, A.L, A.V, A.M, A.P, A.UQ


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_source_protocol(old, current):
    for key in ("dataset", "groups", "budgets", "order_seed", "construction_seed",
                "student", "student_seeds", "student_steps", "student_batch",
                "pool_mask_hash", "report_mask_hash", "stage_id_hashes", "q_config"):
        if old[key] != current[key]:
            raise AssertionError(f"source experiment protocol mismatch: {key}")
    for module in (R, L, V, M, P, U):
        name = Path(module.__file__).name
        if old["source_hashes"][name] != file_hash(module.__file__):
            raise AssertionError(f"frozen source changed: {name}")


def fixed_payload(memory, common):
    if not torch.equal(memory["labels"], common["labels"]):
        raise AssertionError("common payload requires identical slot labels/order")
    return A.replace_payload(memory, common, common["text_source_ids"])


def run(args):
    source = json.loads((args.source / "results.json").read_text(encoding="utf-8"))
    if source["status"] != "complete":
        raise ValueError("source P0 cell is incomplete")
    reference = L.read(args.reference)
    groups = V.choose_groups(reference, args.order_seed)
    data = L.load_large(args.dataset, args.minilm, args.text_cache, args.device,
                        dict(groups=groups, report_dev_mod=reference.get("report_dev_mod", 3)))
    ids = L.stage_ids(data)
    budgets, _ = V.choose_budgets(reference, list(map(len, ids)), groups, "fixed_schedule", .05)
    arms = list(C.CORE) + (list(C.TEXT) if args.order_seed == 9101 else [])
    cfg = R.q_config(C.MAIN, args.construction_seed, args.q_steps, 2048)
    protocol = dict(name="unified-q-constructor-attribution-20261008", dataset=args.dataset,
        groups=groups, budgets=budgets, order_seed=args.order_seed, construction_seed=args.construction_seed,
        student=P.model_config(), student_seeds=args.student_seeds, student_steps=args.student_steps,
        student_batch=24, pool_mask_hash=V.hash_tensor(data["pool_mask"]),
        report_mask_hash=V.hash_tensor(data["report_dev_mask"]), stage_id_hashes=[V.hash_tensor(i) for i in ids],
        q_config=asdict(cfg), arms=arms, max_stages=args.max_stages,
        control_configs={a: asdict(C.control_config(cfg, a)) for a in arms},
        evaluation_split="report-dev; no sealed test", access="current delta + own frozen synthetic prefix",
        topology="edge-free", no_node_text_scope="node strings/embeddings excluded from init/proxy losses; class prompts and supplied X retained",
        common_payload="exact archived CaT-nearest5 text by slot; final-stage posthoc only, never fed back",
        free_x_scope="same objective and conditional 5% feasible set; independent X parameter, not unbounded free X",
        source_cell_sha256=file_hash(args.source / "results.json"),
        source_hashes={Path(m.__file__).name: file_hash(m.__file__) for m in (R,L,V,M,P,U,A,C)},
        runner_sha256=file_hash(__file__))
    protocol = json.loads(json.dumps(protocol))
    assert_source_protocol(source["protocol"], protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else dict(protocol=protocol, rows=[], status="running")
    if result["protocol"] != protocol:
        raise ValueError("changed protocol/source on resume")
    V.write_json(args.output / "protocol.json", protocol)
    V.write_json(path, result)
    parents = {a: None for a in arms}
    common = None
    for stage, raw_ids in enumerate(ids[:args.max_stages]):
        delta = L.delta_view(data, raw_ids, stage)
        delta["prompt_emb"] = data["prompt_emb"]
        new_n = budgets[stage] - (budgets[stage-1] if stage else 0)
        labels = M.row_labels(M.ratio_allocation(delta["y"], new_n))
        common_block, _ = A.load_block(args.source, "cat_cctp", stage)
        common = A.append_block(common, common_block)
        previous_full, _ = A.load_block(args.source, C.MAIN, stage)
        for arm in arms:
            archive = args.output / f"block_{arm}_s{stage}.pt"
            if archive.exists():
                block, audit = A.load_block(args.output, arm, stage)
            else:
                V.sync(args.device)
                started = time.perf_counter()
                block, audit = C.optimize_control(delta, parents[arm], labels, data["encoder"],
                    data["prompt_emb"], cfg, args.device, arm)
                V.sync(args.device)
                audit["attribution_build_seconds"] = time.perf_counter() - started
                audit["assignment_checks"] = R.assignment_checks(audit)
                if not audit["assignment_checks"]["sinkhorn_feasible"]:
                    raise AssertionError("assignment tolerance failed")
                if audit["x_residual_ratio_max"] > cfg.trust_radius + 1e-5:
                    raise AssertionError("feature bound failed")
                if arm == "no_residual" and audit["x_residual_norm"] != 0:
                    raise AssertionError("disabled residual was updated")
                with A.preserve_rng():
                    audit["payload_validation"] = A.validate_payload(block, delta, data["encoder"])
                if arm == C.MAIN:
                    audit["unchanged_main_replay"] = dict(
                        previous_block_hash=V.memory_hash(previous_full),
                        relative_x_difference=float((block["x"]-previous_full["x"]).norm()/previous_full["x"].norm().clamp_min(1e-12)),
                        text_source_rows_changed=sum(a != b for a,b in zip(block["text_source_ids"],previous_full["text_source_ids"])))
                A.archive_block(args.output, arm, stage, block, audit)
            parent = parents[arm]
            memory = U.append_memory(parent, block)
            checks = V.validate_memory(memory, parent, budgets[stage])
            if memory["edge_index"].numel():
                raise AssertionError("unexpected synthetic edges")
            readouts = [("native", memory)]
            if stage == len(ids)-1:
                readouts.append(("common_cat_payload", fixed_payload(memory, common)))
            for readout, evaluated in readouts:
                saved = [r for r in result["rows"] if (r["arm"],r["stage"],r["readout"]) == (arm,stage,readout)]
                if saved:
                    if len(saved) != 1 or saved[0]["memory_hash"] != V.memory_hash(evaluated):
                        raise AssertionError("resume does not reconstruct evaluated memory")
                    continue
                V.sync(args.device)
                started = time.perf_counter()
                with A.preserve_rng():
                    score = L.score(evaluated, data, stage, args.student_seeds, args.device, args.student_steps, 24)
                V.sync(args.device)
                record = dict(arm=arm,stage=stage,readout=readout,score=score,checks=checks,
                    audit=R.compact_audit(audit),memory_rows=budgets[stage],new_rows=new_n,
                    memory_hash=V.memory_hash(evaluated),x_hash=R.digest(evaluated["x"]),
                    text_hash=R.digest(evaluated["text_emb"]),edge_hash=R.digest(evaluated["edge_index"]),
                    construction_seconds=audit["attribution_build_seconds"],student_seconds=time.perf_counter()-started)
                if readout == "common_cat_payload":
                    record["common_payload_hash"] = R.digest(common["text_emb"])
                    record["checks"] = dict(checks,**A.assert_fixed_content(memory,evaluated))
                result["rows"].append(record)
                V.write_json(path,result)
                print(json.dumps(dict(event="evaluated",dataset=args.dataset,order=args.order_seed,
                    arm=arm,stage=stage,readout=readout,f1=score["final_f1"],accuracy=score["accuracy"])),flush=True)
            parents[arm] = memory
    result["status"] = "complete" if args.max_stages >= len(ids) else "smoke_complete"
    V.write_json(path,result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",choices=("wikics","sportsfit","bookhis"),required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--text-cache",type=Path,required=True)
    p.add_argument("--minilm",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--order-seed",type=int,default=9101)
    p.add_argument("--construction-seed",type=int,default=271006)
    p.add_argument("--student-seeds",nargs="+",type=int,default=[150,151,152])
    p.add_argument("--student-steps",type=int,default=2048)
    p.add_argument("--q-steps",type=int,default=30)
    p.add_argument("--max-stages",type=int,default=5)
    args = p.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
