"""Current exact-5% feature-only-Q diagnostic.

This is a separate diagnostic batch. It does not overwrite the frozen primary
tables: Q is constructed without node-text cost or payload pair loss, then a
real current-delta text is materialized after Q has been learned. The same
Student evaluator and masks are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch


ROOT = Path("/home/pc/research/TAG_condensation")
FROZEN = ROOT / "experiments/uq_final_evidence_20261012/code"
sys.path.insert(0, str(FROZEN))
sys.path.insert(1, str(ROOT / "code/v2_llm_fusion"))

import run_unified_q_matrix_20261008 as R  # noqa: E402
import run_uq_final_confirmation_20261012 as E  # noqa: E402


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory_hash(memory: dict) -> str:
    return R.V.memory_hash(memory)


def run(args: argparse.Namespace) -> None:
    reference = R.L.read(args.reference)
    groups = R.V.choose_groups(reference, args.order_seed)
    data = R.L.load_large(args.dataset, args.minilm, args.text_cache, args.device,
                          dict(groups=groups, report_dev_mod=reference.get("report_dev_mod", 3)))
    ids = R.L.stage_ids(data)
    sizes = list(map(len, ids))
    budgets, floors = R.V.choose_budgets(reference, sizes, groups, "fixed_rate", .05)
    base = R.q_config("uq_dm_edgefree", args.construction_seed, 30, 4096)
    cfg = replace(base, text_weight=0.0, pair_weight=0.0)
    protocol = dict(
        name="uq-exact5-feature-only-q-posthoc-text-20261015-v1",
        dataset=args.dataset, order_seed=args.order_seed,
        construction_seed=args.construction_seed,
        intervention="text_weight=0 and payload pair_weight=0 during construction; real text materialized after learned Q",
        student_seeds=[150, 151, 152], student_steps=2048, student_batch=24,
        student=R.P.model_config(), groups=groups, budgets=budgets,
        stage_raw_rows=sizes, memory_rate=.05,
        actual_final_rate=budgets[-1] / sum(sizes), class_floor_adjustments=floors,
        budget_mode="fixed_rate_current_legal_pool",
        access="current-delta-only; own frozen synthetic prefix; zero historical raw reads",
        synthetic_topology="none; common full raw transductive inference graph",
        test_used_for_construction=False, method_selection_from_test=False,
        official_reproduction=False, q_config=vars(cfg),
        source_hashes={Path(m.__file__).name: file_hash(m.__file__)
                       for m in (R, R.L, R.M, R.P, R.UQ, R.V, E)},
        runner_hash=file_hash(__file__),
        pool_mask_hash=R.V.hash_tensor(data["pool_mask"]),
        report_mask_hash=R.V.hash_tensor(data["report_dev_mask"]),
        test_mask_hash=R.V.hash_tensor(data["test_mask"]),
        stage_id_hashes=[R.V.hash_tensor(v) for v in ids],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    path = args.output / "results.json"
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["protocol"] != protocol:
            raise RuntimeError("existing output protocol differs; use a fresh directory")
    else:
        result = {"protocol": protocol, "rows": [], "status": "running"}
    done = {int(row["stage"]) for row in result["rows"]}
    parent = None
    if done:
        raise RuntimeError("resume is intentionally disabled for this diagnostic; preserve partial output")
    for stage, raw_ids in enumerate(ids):
        delta = R.L.delta_view(data, raw_ids, stage)
        new_n = budgets[stage] - (0 if parent is None else len(parent["x"]))
        labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_n))
        R.V.sync(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        block, audit = R.UQ.optimize_stage(delta, parent, labels, data["encoder"],
                                           data["prompt_emb"], cfg, args.device,
                                           use_boundary=False, use_topology=False)
        R.V.sync(args.device)
        construction_seconds = time.perf_counter() - started
        peak_mb = torch.cuda.max_memory_allocated(args.device) / 2**20
        memory = R.L.append_edge_free(parent, block)
        checks = R.V.validate_memory(memory, parent, budgets[stage])
        checks.update(R.assignment_checks(audit))
        if not checks.get("sinkhorn_feasible", False):
            raise AssertionError("feature-only Q assignment is infeasible")
        if memory["edge_index"].numel():
            raise AssertionError("feature-only diagnostic acquired synthetic edges")
        scores = E.evaluate(memory, data, stage, [150, 151, 152], args.device)
        row = dict(stage=stage, raw_rows=len(raw_ids), new_rows=new_n,
                   memory_rows=budgets[stage], construction_seconds=construction_seconds,
                   construction_peak_mb=peak_mb, memory_hash=memory_hash(memory),
                   scores=scores, score=scores["report_dev"], checks=checks,
                   audit=R.compact_audit(audit), intervention=protocol["intervention"])
        result["rows"].append(row)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(dict(event="evaluated", dataset=args.dataset, stage=stage,
                              report_f1=scores["report_dev"]["final_f1"],
                              test_f1=scores["sealed_test"]["final_f1"])), flush=True)
        parent = memory
        del memory
        torch.cuda.empty_cache()
    result["status"] = "complete"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(event="complete", output=str(path))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("arxiv", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--order-seed", type=int, default=9101)
    parser.add_argument("--construction-seed", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)
