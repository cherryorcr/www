"""Reuse saved CoTAG memories and reconstruct a fixed GECC diagnostic."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("COTAG_ROOT", "/home/pc/research/TAG_condensation"))
sys.path.insert(0, str(HERE.parent / "cotag_q_free_x_20261020"))
import run as B

sys.path.insert(0, str(HERE))
from metrics import frequency_groups, metrics

R, E = B.R, B.E
FULL, GECC = "uq_dm_edgefree", "gecc_hard"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prefix(memory, rows):
    if memory["edge_index"].numel():
        raise ValueError("expected edge-free memory")
    out = {k: v[:rows].clone() for k, v in memory.items() if torch.is_tensor(v) and k != "edge_index"}
    out["edge_index"] = memory["edge_index"].clone()
    out["texts"] = list(memory["texts"][:rows])
    return out


def evaluate(memory, data, stage, device, directory, arm):
    prompt = data["prompt_emb"].float().to(device)
    y = data["y"].long().cpu().numpy()
    groups = data["stage_classes"]
    seen = [int(c) for group in groups[:stage + 1] for c in group]
    old, new = [int(c) for group in groups[:stage] for c in group], list(map(int, groups[stage]))
    outputs = []
    for seed in B.SEEDS:
        start = time.perf_counter()
        model = R.P.train_prompt_student(memory, prompt, seed, device, 2048, lr=.001, batch=24)
        with torch.no_grad():
            pred = model.logits(data["x_device"], data["real_adj"], prompt).argmax(1).cpu().numpy()
        R.V.sync(device)
        path = directory / f"pred_{arm}_s{stage}_u{seed}.npz"
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, predictions=pred.astype(np.int16))
        scores = {}
        for split, key in (("report_dev", "report_dev_mask"), ("sealed_test", "test_mask")):
            mask = data[key].cpu().numpy().astype(bool) & np.isin(y, seen)
            scores[split] = metrics(y[mask], pred[mask], seen, old, new, len(prompt))
        outputs.append(dict(seed=seed, seconds=time.perf_counter() - start,
                            prediction_file=path.name, prediction_sha256=digest(path), scores=scores))
        del model
    return outputs


def run_one(seed, output, device):
    path = output / "results.json"
    if output.exists():
        raise FileExistsError(f"preserve existing output: {output}")
    if shutil.disk_usage(ROOT).free < 8 * 2**30:
        raise RuntimeError("less than 8 GiB free disk")
    args = argparse.Namespace(dataset="bookhis", construction_seed=seed, order_seed=9101, device=device)
    primary_path, primary, data, ids, cfg, loaded = B.load(args)
    branch = "v1" if seed == 271006 else "v2"
    source = ROOT / f"results/cotag_q_free_x_20261020_{branch}/bookhis_o9101_c{seed}"
    reference = B.read(source / "results.json")
    memory_path = source / f"final_{FULL}.pt.gz"
    full = B.load_memory(memory_path)
    references = {r["stage"]: r for r in reference["rows"] if r["arm"] == FULL}
    for k in ("groups", "budgets", "stage_id_hashes", "pool_mask_hash", "report_mask_hash",
              "test_mask_hash", "student", "q_config", "student_steps", "student_seeds"):
        if reference["protocol"][k] != loaded[k]:
            raise AssertionError(f"saved CoTAG protocol mismatch: {k}")
    if reference["status"] != "complete" or len(references) != len(ids):
        raise AssertionError("incomplete CoTAG source")
    for stage, budget in enumerate(loaded["budgets"]):
        if R.V.memory_hash(prefix(full, budget)) != references[stage]["memory_hash"]:
            raise AssertionError(f"CoTAG prefix hash mismatch at stage {stage}")
    primary_rows = {(r["arm"], r["stage"]): r for r in primary["rows"]}
    protocol = dict(name="cotag-bookhis-benefit-20261022-v1", base_protocol=loaded,
                    source_result=str(source / "results.json"), source_sha256=digest(source / "results.json"),
                    memory_source=str(memory_path), memory_source_sha256=digest(memory_path),
                    primary_source=str(primary_path), primary_source_sha256=digest(primary_path),
                    registration_sha256=digest(HERE / "README.md"), runner_sha256=digest(__file__),
                    metrics_sha256=digest(HERE / "metrics.py"),
                    scope="post-hoc three-construction diagnostic, not replacement for primary scores")
    result = dict(protocol=protocol, status="running", rows=[])
    B.save(output / "protocol.json", protocol)
    B.save(path, result)
    y = data["y"].cpu().numpy()
    pool = data["pool_mask"].cpu().numpy().astype(bool)
    counts = np.bincount(y[pool], minlength=len(data["prompts"]))
    frequencies = frequency_groups(counts)
    np.savez_compressed(output / "evaluation_cohorts.npz", labels=y,
                        pool_mask=pool, report_dev_mask=data["report_dev_mask"].cpu().numpy(),
                        test_mask=data["test_mask"].cpu().numpy())
    metadata = [dict(class_id=c, prompt=data["prompts"][c], train_count=int(counts[c]),
                     frequency_group=frequencies[c],
                     introduced_stage=next(s for s, g in enumerate(data["stage_classes"]) if c in g))
                for c in range(len(counts))]
    B.save(output / "class_metadata.json", metadata)
    try:
        for arm in (FULL, GECC):
            parent = None
            for stage, raw_ids in enumerate(ids):
                start = time.perf_counter()
                if arm == FULL:
                    memory = prefix(full, loaded["budgets"][stage])
                    audit = dict(reused_saved_memory=True)
                else:
                    delta = R.L.delta_view(data, raw_ids, stage)
                    new_count = loaded["budgets"][stage] - (0 if parent is None else len(parent["x"]))
                    labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_count))
                    memory, audit = E.construct(arm, delta, parent, labels, cfg, data["encoder"],
                                                data["prompt_emb"], seed, device)
                    B.archive(output / f"stage_{stage}_{arm}.pt.gz", memory)
                checks = R.V.validate_memory(memory, parent, loaded["budgets"][stage])
                hash_value = R.V.memory_hash(memory)
                checks["primary_memory_matches"] = hash_value == primary_rows[(arm, stage)]["memory_hash"]
                checks["saved_cotag_stage_matches"] = (hash_value == references[stage]["memory_hash"]
                                                       if arm == FULL else None)
                quotas = np.bincount(memory["labels"].cpu().numpy(), minlength=len(metadata)).tolist()
                construction_seconds = time.perf_counter() - start
                students = evaluate(memory, data, stage, device, output, arm)
                row = dict(arm=arm, stage=stage, memory_hash=hash_value, checks=checks, quotas=quotas,
                           construction_seconds=construction_seconds, construction_audit=R.compact_audit(audit),
                           per_student=students, primary_scores=primary_rows[(arm, stage)]["scores"],
                           reused_memory_scores=references[stage]["scores"] if arm == FULL else None)
                result["rows"].append(row)
                B.save(path, result)
                B.emit(event="stage_complete", construction=seed, arm=arm, stage=stage,
                       test_f1=float(np.mean([r["scores"]["sealed_test"]["final_f1"] for r in students])),
                       primary_hash_matches=checks["primary_memory_matches"])
                parent = memory
            del parent, memory
            torch.cuda.empty_cache()
        result["status"] = "complete"
        B.save(path, result)
    except BaseException as error:
        result.update(status="failed", failure=dict(type=type(error).__name__, message=str(error),
                                                    traceback=traceback.format_exc()))
        B.save(path, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--construction-seeds", type=int, nargs="+", default=[271006, 271108, 271109])
    args = parser.parse_args()
    torch.set_num_threads(2)
    for seed in args.construction_seeds:
        run_one(seed, args.output / f"bookhis_o9101_c{seed}", args.device)
