"""Frozen shared-parent warm-prefix matrix with content-verified fit reuse."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import torch

import run_tgc_r2b_student_revision as B
import tgc_response_alignment as A

R, T, S = B.R, B.T, B.SR
WARMUPS, HORIZONS = (0, 64, 256), (1, 4)
ARMS = ("c0", "teacher") + tuple(f"{kind}_w{w}_k{k}" for w in WARMUPS
    for k in HORIZONS for kind in ("student", "random", "stop", "guarded"))
ROOT = Path(__file__).resolve().parents[2]
DOC = "docs/current/TGC_R2B2_RESPONSE_ALIGNMENT_PROTOCOL_ZH_2026-09-22.md"


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def source_hashes():
    files = [Path(m.__file__).resolve() for m in (R.C, R.G, R.P, R.U, R.SCA, R, T, S, A, B)]
    files += [Path(__file__).resolve(), ROOT / DOC]
    return {p.relative_to(ROOT).as_posix(): R.C.sha(p) for p in files}


def validate_data(data, manifest):
    if manifest["source_dataset_sha256"] != R.C.sha(R.U.DATA_CFG[data["dataset"]]["path"]):
        raise ValueError("source data changed")
    for field, mask in (("pool_sha256", "pool_mask"), ("report_mask_sha256", "report_dev_mask")):
        key = "pool" if mask == "pool_mask" else "report"
        if manifest[field] != R.C.tensor_hash({key: data[mask]}):
            raise ValueError(f"split changed: {mask}")
    code = Path(R.__file__).resolve().parent.parent
    for module in (R.C, R.G, R.P, R.U, R.SCA):
        path = Path(module.__file__).resolve()
        if manifest["protocol"]["source_hashes"][path.relative_to(code).as_posix()] != R.C.sha(path):
            raise ValueError(f"canonical source changed: {path.name}")


def reuse_fit(source, destination, memory, stage, seed, arm):
    row = B.read(source)
    checkpoint, target = source.with_suffix(".pt"), destination.with_suffix(".pt")
    if (row["support_memory_sha256"] != T.memory_hash(memory)
            or row["support_sha256"] != R.C.tensor_hash(memory)
            or row["stage"] != stage or row["seed"] != seed
            or row["steps"] != R.P.student_steps(len(memory["x"]))
            or row["effective_batch"] != min(R.P.STUDENT.batch, len(memory["x"]))
            or row["checkpoint_sha256"] != R.C.sha(checkpoint)):
        raise ValueError(f"invalid reused fit: {source}")
    if target.exists():
        if R.C.sha(target) != row["checkpoint_sha256"]:
            raise ValueError("existing target checkpoint differs")
    else:
        try:
            os.link(checkpoint, target)
        except OSError:
            shutil.copy2(checkpoint, target)
    row.update(arm=arm, executed_here=False, reused_from=str(source),
               reused_source_sha256=R.C.sha(source))
    B.save(destination, row)
    return row


def fit_record(record, memory, data, stage, seed, out, device, cache):
    arm = record["arm"]
    path = out / f"{arm}_s{stage}_seed{seed}.json"
    checkpoint = path.with_suffix(".pt")
    key = (T.memory_hash(memory), seed, R.P.student_steps(len(memory["x"])))
    if path.exists() and checkpoint.exists():
        row = B.read(path)
        if (row["support_memory_sha256"] != key[0] or row["seed"] != seed
                or row["stage"] != stage or row["steps"] != key[2]
                or row["checkpoint_sha256"] != R.C.sha(checkpoint)):
            raise ValueError(f"stale resumed fit: {path}")
    elif key in cache:
        row = reuse_fit(cache[key], path, memory, stage, seed, arm)
    elif record.get("reference_fit"):
        row = reuse_fit(Path(record["reference_fit"].format(seed=seed)), path, memory, stage, seed, arm)
    else:
        R.sync(device)
        started = time.perf_counter()
        row = R.C.fit(memory, seed, key[2], device, data, stage, checkpoint=checkpoint)
        row.update(arm=arm, stage=stage, seed=seed, executed_here=True,
                   support_memory_sha256=key[0], wall_fit_seconds=time.perf_counter() - started,
                   checkpoint_sha256=R.C.sha(checkpoint),
                   effective_batch=min(R.P.STUDENT.batch, len(memory["x"])))
        B.save(path, row)
    cache.setdefault(key, path)
    return row


def cpu_prefixes(prefixes):
    return {s: {**p, **{group: {k: v.detach().cpu() for k, v in p[group].items()}
                               for group in ("params", "first", "second")}}
            for s, p in prefixes.items()}


def construct_stage(data, stage, ids, out, reference, previous, device):
    old_path, base_path = reference / f"c0_s{stage-1}_memory.pt", reference / f"c0_s{stage}_memory.pt"
    old, base = load(old_path), load(base_path)
    support = T._concat_support(old, T.raw_delta_block(data, ids, stage))
    past = B.read(previous / f"s{stage}_decisions.json")
    if past["base_hash"] != T.memory_hash(base) or past["support_hash"] != T.memory_hash(support):
        raise ValueError("reference parent or current probe differs")
    editable, protected = past["editable_rows"], past["protected_rows"]
    old_records = {r["arm"]: r for r in past["arms"]}
    R.sync(device)
    started = time.perf_counter()
    preflight = [A.numerical_check(base, 7500, w, k, device) for w in WARMUPS for k in HORIZONS]
    R.sync(device)
    preflight_seconds = time.perf_counter() - started
    B.save(out / f"s{stage}_preflight.json", {"checks": preflight, "seconds": preflight_seconds})

    def record(arm, path, memory, audit, source=None):
        inv = S.invariants(base, memory, editable, len(old["x"]), protected, .1)
        return {"arm": arm, "memory_path": str(path), "memory_sha256": T.memory_hash(memory),
                "audit": audit, "invariants": inv, "reference_fit": source}

    records = [record("c0", base_path, base, {"kind": "c0"}, str(previous / f"c0_s{stage}_seed{{seed}}.json"))]
    teacher_path = previous / f"teacher_s{stage}_memory.pt"
    records.append(record("teacher", teacher_path, load(teacher_path), old_records["teacher"]["audit"],
                          str(previous / f"teacher_s{stage}_seed{{seed}}.json")))
    prefix_audits = {}
    for warmup in WARMUPS:
        if warmup:
            prefixes, prefix_audit = A.prepare_prefixes(base, warmup, device)
            path = out / f"prefix_w{warmup}_s{stage}.pt"
            torch.save(cpu_prefixes(prefixes), path)
            prefix_audit.update(checkpoint_path=str(path), checkpoint_sha256=R.C.sha(path))
            prefix_audits[str(warmup)] = prefix_audit
        for horizon in HORIZONS:
            suffix = f"w{warmup}_k{horizon}"
            name = f"student_{suffix}"
            if warmup == 0:
                path = previous / f"student_k{horizon}_s{stage}_memory.pt"
                candidate = load(path)
                opt = old_records[f"student_k{horizon}"]["audit"]["optimization"]
                stop_opt = old_records[f"stop_k{horizon}"]["audit"]["optimization"]
                stop = base
                random_path = previous / f"random_k{horizon}_s{stage}_memory.pt"
                random = load(random_path)
            else:
                candidate, opt = A.optimize(base, support, editable, stage, horizon, prefixes, device)
                stop, stop_opt = A.optimize(base, support, editable, stage, horizon, prefixes, device, stop=True)
                random = S.matched_random(base, candidate, editable, stage)
                path, random_path = out / f"{name}_s{stage}_memory.pt", out / f"random_{suffix}_s{stage}_memory.pt"
                torch.save(candidate, path)
                torch.save(random, random_path)
            if not torch.equal(stop["x"], base["x"]):
                raise AssertionError("stop-gradient changed memory")
            records.append(record(name, path, candidate, {"kind": "student", "warmup": warmup,
                "horizon": horizon, "optimization": opt},
                str(previous / f"student_k{horizon}_s{stage}_seed{{seed}}.json") if warmup == 0 else None))
            records.append(record(f"random_{suffix}", random_path, random,
                {"kind": "random", "warmup": warmup, "horizon": horizon},
                str(previous / f"random_k{horizon}_s{stage}_seed{{seed}}.json") if warmup == 0 else None))
            records.append(record(f"stop_{suffix}", base_path, base,
                {"kind": "stop", "warmup": warmup, "horizon": horizon, "optimization": stop_opt}))
            accepted = bool(opt["accepted"])
            records.append(record(f"guarded_{suffix}", path if accepted else base_path,
                candidate if accepted else base, {"kind": "guarded", "warmup": warmup,
                "horizon": horizon, "accepted": accepted, "selected_arm": name if accepted else "c0",
                "report_labels_used": False}))
        if warmup:
            del prefixes
    if tuple(r["arm"] for r in records) != ARMS:
        raise AssertionError("arm matrix differs")
    decision = {"dataset": data["dataset"], "stage": stage, "editable_rows": editable,
                "protected_rows": protected, "base_hash": T.memory_hash(base),
                "support_hash": T.memory_hash(support), "old_memory_hash": T.memory_hash(old),
                "previous_decision_sha256": R.C.sha(previous / f"s{stage}_decisions.json"),
                "prefix_audits": prefix_audits, "preflight_seconds": preflight_seconds,
                "report_labels_used": False, "historical_raw_reads": 0, "arms": records,
                "created_at": datetime.now(timezone.utc).isoformat()}
    B.save(out / f"s{stage}_decisions.json", decision)
    return decision


def run_cell(data, order, args):
    reference, previous = args.reference_root / data["dataset"] / order, args.previous_root / data["dataset"] / order
    manifest = B.read(reference / "manifest.json")
    validate_data(data, manifest)
    ids, budgets = R.A.set_stream(data, manifest["stage_classes"], .05)
    if [v.tolist() for v in ids] != manifest["stage_source_ids"] or list(budgets) != manifest["budgets"]:
        raise ValueError("stream changed")
    if not B.read(args.previous_root / "audit.json")["passed"]:
        raise ValueError("previous full audit required")
    out = args.output / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"name": "tgc-r2b2-fixed-prefix-v1", "dataset": data["dataset"], "order": order,
                "warmups": WARMUPS, "horizons": HORIZONS, "arms": ARMS,
                "stages": [1, 2], "report_seeds": B.REPORT_SEEDS, "outer": S.Config().__dict__,
                "rate": .05, "student": {"batch": R.P.STUDENT.batch, "lr": R.P.STUDENT.lr},
                "stage_classes": manifest["stage_classes"], "source_hashes": source_hashes(),
                "reference_manifest_sha256": R.C.sha(reference / "manifest.json"),
                "previous_audit_sha256": R.C.sha(args.previous_root / "audit.json"),
                "test_labels_evaluated": False, "selector_labels_used": False}
    R.stable_lock(out / "protocol.json", protocol)
    results = []
    for stage in (1, 2):
        path = out / f"s{stage}_decisions.json"
        decision = B.read(path) if path.exists() else construct_stage(data, stage, ids[stage], out, reference, previous, args.device)
        cache = {}
        for record in decision["arms"]:
            memory = load(record["memory_path"])
            if T.memory_hash(memory) != record["memory_sha256"]:
                raise ValueError("frozen decision memory changed")
            rows = [fit_record(record, memory, data, stage, seed, out, args.device, cache) for seed in B.REPORT_SEEDS]
            results.append({"arm": record["arm"], "stage": stage, "rows": len(memory["x"]),
                            "memory_sha256": record["memory_sha256"], "metrics": R.C.aggregate(rows),
                            "per_seed": rows, "audit": record["audit"]})
        B.save(out / "results.json", {"protocol": protocol, "results": results})
        print(json.dumps({"event": "completed_stage", "dataset": data["dataset"], "order": order,
                          "stage": stage, "slots": len(results) * len(B.REPORT_SEEDS)}), flush=True)
    B.save(out / "completion.json", {"status": "complete", "fit_slots": len(results) * len(B.REPORT_SEEDS),
        "executed_fits": sum(r["executed_here"] for v in results for r in v["per_seed"]),
        "completed_at": datetime.now(timezone.utc).isoformat()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--previous-root", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--text-cache-root", type=Path, default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    data = R.R.load_data(args.dataset, args.minilm, args.device,
                        args.text_cache_root / f"{args.dataset}_r005" / "text_emb_pool_only.pt")
    for order in B.ORDERS:
        run_cell(data, order, args)
    print("TGC_R2B2_ALIGNMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
