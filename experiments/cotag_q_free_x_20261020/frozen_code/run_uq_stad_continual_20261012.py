"""Current-delta-only STAD adaptation retaining its native topology branch."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import time

import torch

import run_uq_final_confirmation_20261012 as E
import uq_stad_native_core_20261012 as S

R = E.R


def run(args):
    reference = R.L.read(args.reference)
    groups = R.V.choose_groups(reference, args.order_seed)
    data = R.L.load_large(args.dataset, args.minilm, args.text_cache, args.device,
        dict(groups=groups, report_dev_mod=reference.get("report_dev_mod", 3)))
    ids = R.L.stage_ids(data)
    sizes = list(map(len, ids))
    budgets, floors = R.V.choose_budgets(reference, sizes, groups, "fixed_rate", .05)
    official = S.official_components()
    protocol = dict(name="stad-native-current-delta-20261012-v1", dataset=args.dataset,
        arm="stad_native_local", groups=groups, order_seed=args.order_seed,
        construction_seed=args.construction_seed, student_seeds=[150, 151, 152],
        student_steps=2048, student_batch=24, student=R.P.model_config(),
        budgets=budgets, class_floor_adjustments=floors, memory_rate=.05,
        actual_final_rate=budgets[-1] / sum(sizes), stage_raw_rows=sizes,
        pool_mask_hash=R.V.hash_tensor(data["pool_mask"]),
        report_mask_hash=R.V.hash_tensor(data["report_dev_mask"]),
        test_mask_hash=R.V.hash_tensor(data["test_mask"]),
        stage_id_hashes=[R.V.hash_tensor(values) for values in ids],
        source_hashes={Path(module.__file__).name: E.file_hash(module.__file__)
                       for module in (S, E, R, R.P, R.M, R.L, R.L.U, R.L.U.protocol_v3, R.V)},
        runner_hash=E.file_hash(__file__), official_source=official[-1],
        generator=json.loads((S.MODEL / "generator_provenance.json").read_text()),
        access="current delta only; own frozen synthetic prefix; no historical raw reads",
        semantic_core="common MiniLM; native mean pooling without L2 normalization",
        label_access="10% supervised + 10% internal validation; 80% pseudo-label pool",
        synthetic_topology="native weighted coarsening for candidate selection; binary support for common Student",
        candidate_selection="internal validation only; external report-dev/test excluded",
        common_text_cap=320, official_reproduction=False, method_selection_from_test=False)
    path = args.output / "results.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["protocol"] != protocol:
            raise ValueError("STAD protocol differs; preserve this directory and use a new one")
    else:
        result = dict(protocol=protocol, rows=[], status="running")
    S.save(args.output / "protocol.json", protocol)
    S.save(path, result)
    parent = None
    class_names = list(R.L.U.G.CLASS_NAMES)
    if len(class_names) != len(data["prompts"]):
        raise ValueError("STAD class names and common prompts differ")
    for stage, raw_ids in enumerate(ids):
        stage_dir = args.output / f"s{stage}"
        block_path = stage_dir / "block.pt.gz"
        audit_path = stage_dir / "construction.json"
        previous = next((row for row in result["rows"] if row["stage"] == stage), None)
        if previous:
            with gzip.open(block_path, "rb") as stream:
                block = torch.load(stream, map_location="cpu", weights_only=False)
            memory = S.append(parent, block)
            R.V.validate_memory(memory, parent, budgets[stage])
            if R.V.memory_hash(memory) != previous["memory_hash"]:
                raise ValueError("STAD resume artifact differs")
            parent = memory
            continue
        delta = R.L.delta_view(data, raw_ids, stage)
        new_rows = budgets[stage] - (0 if parent is None else len(parent["x"]))
        R.V.sync(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        block, audit = S.construct(delta, new_rows, data["encoder"], args.construction_seed,
            args.device, stage_dir, args.dataset, class_names, official)
        memory = S.append(parent, block)
        R.V.sync(args.device)
        seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(args.device) / 2**20
        checks = R.V.validate_memory(memory, parent, budgets[stage])
        if not audit["test_used_for_selection"] and not audit["report_dev_used_for_selection"]:
            checks["external_selection_labels_not_used"] = True
        else:
            raise AssertionError("STAD candidate selection exposed external labels")
        with gzip.open(block_path.with_suffix(".gz.tmp"), "wb", compresslevel=3) as stream:
            torch.save(block, stream)
        block_path.with_suffix(".gz.tmp").replace(block_path)
        S.save(audit_path, audit)
        print(json.dumps(dict(event="constructed", arm="stad_native_local", stage=stage,
                              seconds=seconds, peak_mb=peak, edges=memory["edge_index"].shape[1])), flush=True)
        started = time.perf_counter()
        scores = E.evaluate(memory, data, stage, [150, 151, 152], args.device)
        row = dict(arm="stad_native_local", stage=stage, raw_rows=len(raw_ids), new_rows=new_rows,
            memory_rows=len(memory["x"]), score=scores["report_dev"], scores=scores,
            construction_seconds=seconds, construction_peak_mb=peak,
            student_seconds=time.perf_counter() - started, checks=checks,
            memory_hash=R.V.memory_hash(memory), audit=audit)
        result["rows"].append(row)
        S.save(path, result)
        print(json.dumps(dict(event="evaluated", stage=stage,
            report_f1=scores["report_dev"]["final_f1"], test_f1=scores["sealed_test"]["final_f1"])), flush=True)
        parent = memory
        torch.cuda.empty_cache()
    result["status"] = "complete"
    S.save(path, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("arxiv", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, default=9101)
    parser.add_argument("--construction-seed", type=int, default=271006)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    run(args)
