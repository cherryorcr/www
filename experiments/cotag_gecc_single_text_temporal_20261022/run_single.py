"""Read-only feature reuse; change only GECC's text-source policy."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BIB = HERE.parent / "cotag_temporal_bibliographic_20261022"
PRIOR = HERE.parent / "cotag_fidelity_temporal_20261021"
sys.path.insert(0, str(BIB))
import runtime as B
from runtime import C, R, torch

sys.path.insert(0, str(HERE))
from controls import assert_numeric_equal, assert_frozen_prefix, assert_single_source, text_statistics

OUT = C.ROOT / "results/cotag_gecc_single_text_temporal_20261022"


def module_from(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = module_from("bibliographic_eval_saved_predictions", BIB / "run_temporal.py")
A = module_from("original_arxiv_temporal_loader", PRIOR / "run_temporal.py")


def source_run(dataset, seed):
    if dataset == "arxiv":
        return C.ROOT / f"results/cotag_fidelity_temporal_20261021/temporal/gecc_hard_c{seed}"
    return B.OUT / "runs" / dataset / f"gecc_hard_c{seed}"


def load_inputs(dataset, device):
    if dataset == "arxiv":
        data, arrival, ids, budgets, floors, source, stages = A.load(device)
        data["names"] = list(data["prompts"])
        return data, arrival, ids, budgets, floors, source, None, stages
    return B.load_data(dataset, device)


def run(args):
    C.disk_check()
    source_dir = source_run(args.dataset, args.seed)
    original = C.read(source_dir / "results.json")
    p = original["protocol"]
    if original["status"] != "complete" or len(original["rows"]) != 5:
        raise AssertionError("Incomplete original GECC chain")
    data, arrival, ids, budgets, floors, source, encoding, stages = load_inputs(args.dataset, args.device)
    expected = dict(budgets=budgets, class_floor_adjustments=floors, source=source,
        stage_id_hashes=[R.V.hash_tensor(i) for i in ids], pool_mask_hash=R.V.hash_tensor(data["pool_mask"]),
        report_mask_hash=R.V.hash_tensor(data["report_dev_mask"]), test_mask_hash=R.V.hash_tensor(data["test_mask"]))
    expected = json.loads(json.dumps(expected))
    for key, value in expected.items():
        if p[key] != value:
            raise AssertionError(f"Original protocol changed: {key}")
    if p["student_seeds"] != [150, 151, 152] or p["student_steps"] != 2048 or p["student_batch"] != 24:
        raise AssertionError("Student protocol differs")
    if p["source_hashes"] != C.source_hashes():
        raise AssertionError("Frozen construction/student sources changed")
    cfg5 = R.M.Config(seed=args.seed, text_sources=5, text_mode="nearest", text_chars=320)
    cfg1 = replace(cfg5, text_sources=1)
    directory = OUT / "runs" / args.dataset / f"gecc_nearest1_c{args.seed}"
    protocol = dict(name="temporal-gecc-fixed-x-nearest1-v1", dataset=args.dataset, arm="gecc_nearest1",
        construction_seed=args.seed, student_seeds=[150, 151, 152], student_steps=2048,
        student_batch=24, memory_rate=.05, student=R.P.model_config(), **expected,
        source_result_path=str(source_dir / "results.json"), source_result_sha256=C.sha(source_dir / "results.json"),
        source_protocol=p, source_hashes=C.source_hashes(), encoding=encoding,
        source_memory_hashes=[r["construction"]["memory_hash"] for r in original["rows"]],
        intervention="Reuse every saved GECC numeric tensor; nearest-5 to nearest-1 text, 320-character cap",
        clustering_rerun=False, historical_prefix="own single-source prefix stays exactly frozen",
        source_count=1, text_chars=320, source_count_reference=5,
        selection_space="float32 propagated features; same-class cosine; stable descending ordering",
        stages=stages, inference_modes=["arrived_snapshot", "current_snapshot"],
        method_selection_from_test=False, posthoc_sensitivity=True,
        test_split_source_name="historical_test" if args.dataset == "arxiv" else "test",
        runner_sha256=C.sha(__file__), controls_sha256=C.sha(HERE / "controls.py"),
        registration_sha256=C.sha(HERE / "REGISTRATION.md"), evaluator_sha256=C.sha(BIB / "run_temporal.py"),
        runtime_sha256=C.sha(BIB / "runtime.py"), arxiv_loader_sha256=C.sha(PRIOR / "run_temporal.py"),
        cost_scope="Text adaptation and student only; original clustering is reused, not remeasured")
    result = C.start_result(directory / "results.json", protocol)
    parent = None
    try:
        for stage, idx in enumerate(ids):
            folder = directory / f"s{stage}"
            old_row = original["rows"][stage]
            old = C.load_memory(source_dir / f"s{stage}/memory.pt.gz")
            if R.V.memory_hash(old) != old_row["construction"]["memory_hash"]:
                raise AssertionError("Original memory changed")
            keep = 0 if parent is None else len(parent["x"])
            memory_path, audit_path = folder / "memory.pt.gz", folder / "construction.json"
            saved = next((r for r in result["rows"] if r["stage"] == stage), None)
            if memory_path.exists() and audit_path.exists():
                memory, construction = C.load_memory(memory_path), C.read(audit_path)
                if R.V.memory_hash(memory) != construction["memory_hash"]:
                    raise AssertionError("New checkpoint changed")
            else:
                if memory_path.exists() or audit_path.exists():
                    raise RuntimeError("Incomplete checkpoint pair; preserve for inspection")
                access = B.validate_delta(idx, arrival, data["pool_mask"], stage)
                delta = R.L.delta_view(data, idx, stage)
                x, labels = old["x"][keep:], old["labels"][keep:]
                R.V.sync(args.device)
                torch.cuda.reset_peak_memory_stats(args.device)
                started = time.perf_counter()
                propagated = R.M.propagated_features(delta, cfg5.gecc_weights, args.device).cpu()
                R.V.sync(args.device)
                propagation_seconds = time.perf_counter()-started
                started = time.perf_counter()
                reproduced, audit5 = R.M.attach_text(delta, propagated, x, labels, data["encoder"], cfg5)
                if reproduced["texts"] != old["texts"][keep:]:
                    C.save(folder / "failed_nearest5_gate.json", dict(source=str(source_dir),
                        mismatch_rows=[i for i, (a, b) in enumerate(zip(reproduced["texts"], old["texts"][keep:])) if a != b]))
                    raise AssertionError("Nearest-5 reconstruction differs; do not silently change selection")
                max_error = float((reproduced["text_emb"]-old["text_emb"][keep:]).abs().max())
                if not torch.allclose(reproduced["text_emb"], old["text_emb"][keep:], atol=1e-6, rtol=1e-5):
                    raise AssertionError("Nearest-5 text encoding drift exceeds gate")
                R.V.sync(args.device)
                validation_seconds = time.perf_counter()-started
                started = time.perf_counter()
                block, audit1 = R.M.attach_text(delta, propagated, x, labels, data["encoder"], cfg1)
                block["labels"] = labels.clone()
                assert_single_source(audit1["text_source_ids"], audit5["text_source_ids"], idx.tolist())
                memory = R.L.append_edge_free(parent, block)
                assert_numeric_equal(memory, old)
                assert_frozen_prefix(memory, parent)
                R.V.sync(args.device)
                attachment_seconds = time.perf_counter()-started
                checks = R.V.validate_memory(memory, parent, budgets[stage])
                construction = dict(stage=stage, access=access, checks=checks,
                    source_memory_hash=old_row["construction"]["memory_hash"], memory_hash=R.V.memory_hash(memory),
                    memory_rows=len(memory["x"]), seconds=propagation_seconds+attachment_seconds,
                    propagation_seconds=propagation_seconds, validation_seconds=validation_seconds,
                    attachment_seconds=attachment_seconds,
                    peak_mb=torch.cuda.max_memory_allocated(args.device)/2**20,
                    reserved_peak_mb=torch.cuda.max_memory_reserved(args.device)/2**20,
                    resource_scope="Propagation/text validation/text attachment; excludes reused clustering",
                    audit=dict(text_mode="nearest", source_count=1, original_source_count=5,
                        original_nearest5_text_exact=True, original_text_emb_max_abs_error=max_error,
                        numeric_memory_exact=True, source_ids=audit1["text_source_ids"],
                        reference_source_ids=audit5["text_source_ids"],
                        new_text_statistics=text_statistics(block["texts"]),
                        reference_text_statistics=text_statistics(reproduced["texts"]),
                        all_memory_text_statistics=text_statistics(memory["texts"]),
                        changed_new_rows=sum(a != b for a, b in zip(block["texts"], reproduced["texts"])),
                        changed_cumulative_rows=sum(a != b for a, b in zip(memory["texts"], old["texts"]))))
                C.archive(memory_path, memory)
                C.save(audit_path, construction)
                del delta, propagated, reproduced, block
                C.emit(event="text_attached", dataset=args.dataset, seed=args.seed, stage=stage,
                       original_text_exact=True, numeric_memory_exact=True)
            assert_numeric_equal(memory, old)
            assert_frozen_prefix(memory, parent)
            if saved is None:
                scores = T.evaluate(memory, data, arrival, stage, args.device, folder)
                result["rows"].append(dict(stage=stage, construction=construction, scores=scores))
                C.save(directory / "results.json", result)
                C.emit(event="evaluated", dataset=args.dataset, seed=args.seed, stage=stage,
                       report=scores["prediction"]["arrived_snapshot"]["report_dev"]["cumulative"])
            elif saved["construction"]["memory_hash"] != construction["memory_hash"]:
                raise AssertionError("Saved score belongs to another memory")
            parent = memory
            torch.cuda.empty_cache()
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        C.save(directory / "results.json", result)
        raise
    C.save(directory / "results.json", result)
    C.emit(event="complete", dataset=args.dataset, seed=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("arxiv", "acm", "dblp"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    run(args)
