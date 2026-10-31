"""Independent final-memory confirmation of one hard-payload projection change."""
import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

import run_unified_q_attribution_20261008 as A
import graphtext_sgc_transfer as S
import unified_q_payload_projection as H

R, L, V, M, P, U = A.R, A.L, A.V, A.M, A.P, A.UQ
ARMS = ("uq_qmedoid", "uq_dm_edgefree", "cat_cctp", "gecc_hard")


def baseline_provenance(block, audit):
    # The existing external adapters store provenance in audit, not memory.
    sources = audit["text_source_ids"]
    if len(sources) != len(block["x"]):
        raise AssertionError("external text provenance length mismatch")
    return dict(block, text_source_ids=[list(v) for v in sources])


def config(seed, steps):
    base = R.q_config("uq_dm_edgefree", seed, steps, 2048)
    candidate = replace(base, text_materialization="q_text_medoid")
    assert {k for k in asdict(base) if asdict(base)[k] != asdict(candidate)[k]} == {"text_materialization"}
    return base, candidate


def save_memory(path, memory):
    if shutil.disk_usage(path.parent).free < .85 * 2**30:
        raise RuntimeError("disk guard; no data deleted")
    temp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temp, "wb", compresslevel=3) as stream:
        torch.save(memory, stream)
    temp.replace(path)


def load_memory(path):
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def run(args):
    reference = L.read(args.reference)
    groups = V.choose_groups(reference, args.order_seed)
    data = L.load_large(args.dataset, args.minilm, args.text_cache, args.device,
        dict(groups=groups, report_dev_mod=reference.get("report_dev_mod", 3)))
    ids = L.stage_ids(data)
    budgets, _ = V.choose_budgets(reference, list(map(len, ids)), groups, "fixed_schedule", .05)
    base, candidate = config(args.construction_seed, args.q_steps)
    for stage, raw in enumerate(ids):
        M.ratio_allocation(data["y"][raw], budgets[stage] - (budgets[stage - 1] if stage else 0))
    protocol = dict(name="uq-single-hard-projection-confirmation-v1", dataset=args.dataset,
        groups=groups, budgets=budgets, budget_mode="fixed_schedule", order_seed=args.order_seed,
        construction_seed=args.construction_seed, arms=ARMS, student_seeds=args.student_seeds,
        student_steps=args.student_steps, student_batch=24, max_stages=args.max_stages,
        students=dict(prompt_gcn=P.model_config(), prompt_sgc=S.model_config()),
        q_config=asdict(base), candidate_config=asdict(candidate),
        baseline_config=asdict(M.Config(seed=args.construction_seed, cat_steps=3000, text_sources=5, text_mode="nearest")),
        access="current delta and own frozen synthetic prefix; construction interface audit, not OS isolation",
        evaluation="final memory only, report-dev; not sealed test; original transductive graph",
        candidate_change="same-class Q-weighted cosine medoid at every hard forward AND final materialization",
        category=dict(uq_qmedoid="single-change candidate", uq_dm_edgefree="internal method anchor",
                      cat_cctp="CaT paper mechanism adapted", gecc_hard="GECC paper mechanism adapted"),
        pool_mask_hash=V.hash_tensor(data["pool_mask"]), report_mask_hash=V.hash_tensor(data["report_dev_mask"]),
        stage_id_hashes=[V.hash_tensor(i) for i in ids],
        source_hashes={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                       for m in (R, L, V, M, P, U, A, S, H)},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        official_reproduction=False, topology="edge-free", old_memory_updated=False,
        equal_budget="same rows and 320-character cap; not matched token counts or encoded information")
    protocol = json.loads(json.dumps(protocol))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    result = json.loads(path.read_text()) if path.exists() else dict(protocol=protocol, stages=[], rows=[], status="running")
    if result["protocol"] != protocol:
        raise ValueError("refusing to mix changed protocol/source on resume")
    V.write_json(args.output / "protocol.json", protocol)
    V.write_json(path, result)
    stages = min(args.max_stages, len(ids))
    for arm in ARMS:
        archive = args.output / f"final_{arm}.pt.gz"
        if archive.exists():
            memory = load_memory(archive)
            records = [v for v in result["stages"] if v["arm"] == arm]
            if len(records) != stages or records[-1]["memory_hash"] != V.memory_hash(memory):
                raise AssertionError("archive/progress mismatch")
        else:
            if any(r["arm"] == arm for r in result["stages"]):
                raise RuntimeError("partial constructor retained; use a new output, not silent rerun")
            parent = None
            for stage, raw in enumerate(ids[:stages]):
                delta = L.delta_view(data, raw, stage)
                rows = budgets[stage] - (len(parent["x"]) if parent else 0)
                labels = M.row_labels(M.ratio_allocation(delta["y"], rows))
                V.sync(args.device)
                torch.cuda.reset_peak_memory_stats(args.device) if args.device.startswith("cuda") else None
                started = time.perf_counter()
                if arm in ("cat_cctp", "gecc_hard"):
                    block, audit = R.make_baseline(delta, rows, arm, data["encoder"], args.construction_seed, args.device)
                    block = baseline_provenance(block, audit)
                else:
                    cfg = candidate if arm == "uq_qmedoid" else base
                    block, audit = U.optimize_stage(delta, parent, labels, data["encoder"], data["prompt_emb"],
                        cfg, args.device, use_boundary=False, use_topology=False)
                V.sync(args.device)
                seconds = time.perf_counter() - started
                peak = torch.cuda.max_memory_allocated(args.device) / 2**20 if args.device.startswith("cuda") else None
                memory = A.append_block(parent, block)
                checks = V.validate_memory(memory, parent, budgets[stage])
                if arm.startswith("uq_"):
                    checks.update(R.assignment_checks(audit))
                    if not checks["sinkhorn_feasible"] or audit["text_hard_forward_max_error"] > 1e-4:
                        raise AssertionError("assignment/materialization invariant failed")
                with A.preserve_rng():
                    checks.update(A.validate_payload(block, delta, data["encoder"]))
                if memory["edge_index"].numel():
                    raise AssertionError("unexpected learned synthetic edges")
                result["stages"].append(dict(arm=arm, stage=stage, rows=rows, memory_rows=len(memory["x"]),
                    raw_rows=len(raw), construction_seconds=seconds, construction_peak_mb=peak,
                    memory_hash=V.memory_hash(memory), checks=checks, audit=R.compact_audit(audit)))
                V.write_json(path, result)
                print(json.dumps(dict(event="constructed", dataset=args.dataset, order=args.order_seed,
                    seed=args.construction_seed, arm=arm, stage=stage, seconds=seconds)), flush=True)
                parent = memory
            save_memory(archive, memory)
        memory_hash = V.memory_hash(memory)
        for name, trainer in (("prompt_gcn", None), ("prompt_sgc", S.train_sgc_student)):
            previous = [r for r in result["rows"] if r["arm"] == arm and r["student"] == name]
            if previous:
                if len(previous) != 1 or previous[0]["memory_hash"] != memory_hash:
                    raise AssertionError("readout does not match archived memory")
                continue
            V.sync(args.device)
            started = time.perf_counter()
            with A.preserve_rng():
                score = L.score(memory, data, stages - 1, args.student_seeds, args.device, args.student_steps, 24, trainer=trainer)
            V.sync(args.device)
            if memory_hash != V.memory_hash(memory):
                raise AssertionError("readout mutated memory")
            result["rows"].append(dict(arm=arm, student=name, memory_hash=memory_hash,
                memory_rows=len(memory["x"]), archive_bytes=archive.stat().st_size,
                text_chars=sum(map(len, memory["texts"])), unique_texts=len(set(memory["texts"])),
                student_seconds=time.perf_counter() - started, score=score))
            V.write_json(path, result)
            print(json.dumps(dict(event="evaluated", dataset=args.dataset, order=args.order_seed,
                seed=args.construction_seed, arm=arm, student=name, f1=score["final_f1"], accuracy=score["accuracy"])), flush=True)
        del memory
    result["status"] = "complete" if stages == len(ids) else "smoke_complete"
    V.write_json(path, result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=("wikics", "sportsfit", "bookhis"))
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--order-seed", type=int, default=9201)
    p.add_argument("--construction-seed", type=int, default=271108)
    p.add_argument("--q-steps", type=int, default=30)
    p.add_argument("--max-stages", type=int, default=5)
    p.add_argument("--student-steps", type=int, default=2048)
    p.add_argument("--student-seeds", type=int, nargs="+", default=[150, 151, 152])
    torch.set_num_threads(2)
    run(p.parse_args())


if __name__ == "__main__":
    main()
