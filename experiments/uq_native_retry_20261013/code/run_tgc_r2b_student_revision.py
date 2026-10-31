"""Paired small-graph experiment for student-compatible historical-X revision."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import torch

import run_tgc_revision_gate as R
import tgc_revision_gate as T
import tgc_student_revision as SR


ORDERS = ("canonical", "order19101")
HORIZONS = (1, 4, 8)
REPORT_SEEDS = (100, 101, 102, 103, 104, 105)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    R.save_json(Path(path), value)


def clone(value):
    return T.clone_memory(value)


def replace_old_x(base, revised, old_n):
    result = clone(base)
    result["x"][:old_n] = revised["x"][:old_n]
    return result


def candidate_audit(base, candidate, rows, old_n, protected, radius):
    return SR.invariants(base, candidate, rows, old_n, protected, radius)


def fit_one(memory, data, stage, seed, arm, out, device, reference):
    row_path = out / f"{arm}_s{stage}_seed{seed}.json"
    checkpoint = out / f"{arm}_s{stage}_seed{seed}.pt"
    steps = R.P.student_steps(len(memory["x"]))
    support_hash = R.C.tensor_hash(memory)
    if row_path.exists() and checkpoint.exists():
        row = read(row_path)
        if (row["support_sha256"] != support_hash
                or int(row["steps"]) != int(steps)
                or R.C.sha(checkpoint) != row["checkpoint_sha256"]):
            raise ValueError(f"stale fit artifact: {row_path}")
        return row
    if arm == "c0":
        source_row = reference / f"c0_s{stage}_seed{seed}.json"
        source_pt = source_row.with_suffix(".pt")
        row = read(source_row)
        if row["support_sha256"] != support_hash or row["steps"] != steps:
            raise ValueError(f"C0 reference mismatch: {source_row}")
        shutil.copy2(source_pt, checkpoint)
        row.update(arm=arm, executed_here=False, reused_from=str(source_row),
                   reused_source_sha256=R.C.sha(source_row),
                   checkpoint_sha256=R.C.sha(checkpoint))
        save(row_path, row)
        return row
    started = time.perf_counter()
    row = R.C.fit(memory, seed, steps, device, data, stage, checkpoint=checkpoint)
    row.update(arm=arm, stage=int(stage), seed=int(seed),
               support_memory_sha256=T.memory_hash(memory),
               wall_fit_seconds=float(time.perf_counter() - started),
               effective_batch=min(int(R.P.STUDENT.batch), len(memory["x"])),
               executed_here=True, checkpoint_sha256=R.C.sha(checkpoint))
    save(row_path, row)
    return row


def make_teacher(old, delta, base, stage, device, cfg):
    revised, audit = T.revise_old_memory(old, delta, cfg, device, "gate")
    candidate = replace_old_x(base, revised, len(old["x"]))
    return candidate, audit


def run_cell(data, order, args, root, reference_root):
    reference = reference_root / data["dataset"] / order
    manifest = read(reference / "manifest.json")
    groups = manifest["stage_classes"]
    stage_ids, budgets = R.A.set_stream(data, groups, .05)
    out = root / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "name": "tgc-r2b-student-revision-v1",
        "dataset": data["dataset"], "order": order, "stages": [1, 2],
        "horizons": list(HORIZONS), "report_seeds": list(REPORT_SEEDS),
        "memory_rate": .05, "student_batch": int(R.P.STUDENT.batch),
        "student_lr": float(R.P.STUDENT.lr), "outer": SR.Config().__dict__,
        "revision": {"protected_fraction": .8, "quota": .1, "radius": .1},
        "history_access": "previous synthetic memory + current raw delta only",
        "labels_used_for_selection": False,
        "source_hashes": {name: R.C.sha(Path(__file__).with_name(name))
                          for name in ("run_tgc_r2b_student_revision.py",
                                       "tgc_student_revision.py", "tgc_revision_gate.py")},
    }
    R.stable_lock(out / "protocol.json", protocol)
    cell_results = []
    for stage in (1, 2):
        old = torch.load(reference / f"c0_s{stage - 1}_memory.pt",
                         map_location="cpu", weights_only=False)
        base = torch.load(reference / f"c0_s{stage}_memory.pt",
                          map_location="cpu", weights_only=False)
        delta = T.raw_delta_block(data, stage_ids[stage], stage)
        support = T._concat_support(old, delta)
        cfg = T.RevisionConfig(protected_fraction=.8, revision_fraction_of_old=.1,
                               revision_radius=.1, teacher_steps=128)
        teacher, teacher_audit = make_teacher(old, delta, base, stage, args.device, cfg)
        editable = teacher_audit.get("updated_rows", [])
        if not editable:
            editable = teacher_audit.get("eligible_rows", [])[:max(1, int(teacher_audit["revision_quota"]))]
        protected = teacher_audit.get("protected_rows", [])
        save(out / f"teacher_s{stage}_audit.json", teacher_audit)
        torch.save(teacher, out / f"teacher_s{stage}_memory.pt")
        records = [{"arm": "c0", "memory": base, "audit": {"kind": "c0"}}]
        records.append({"arm": "teacher", "memory": teacher,
                        "audit": {"kind": "teacher", "teacher": teacher_audit,
                                   "invariants": candidate_audit(base, teacher, editable,
                                                                  len(old["x"]), protected,
                                                                  .1)}})
        student_candidates = {}
        for horizon in HORIZONS:
            candidate, audit = SR.optimize(base, support, editable, stage, horizon,
                                           args.device, SR.Config())
            name = f"student_k{horizon}"
            inv = candidate_audit(base, candidate, editable, len(old["x"]), protected, .1)
            student_candidates[horizon] = candidate
            torch.save(candidate, out / f"{name}_s{stage}_memory.pt")
            save(out / f"{name}_s{stage}_audit.json", {"optimization": audit,
                "invariants": inv, "candidate_hash": T.memory_hash(candidate)})
            records.append({"arm": name, "memory": candidate,
                            "audit": {"kind": "student", "optimization": audit,
                                       "invariants": inv}})
            stop, stop_audit = SR.optimize(base, support, editable, stage, horizon,
                                           args.device, SR.Config(), stop=True)
            stop_name = f"stop_k{horizon}"
            stop_inv = candidate_audit(base, stop, editable, len(old["x"]), protected, .1)
            records.append({"arm": stop_name, "memory": stop,
                            "audit": {"kind": "stop", "optimization": stop_audit,
                                       "invariants": stop_inv}})
            random = SR.matched_random(base, candidate, editable, stage)
            random_name = f"random_k{horizon}"
            random_inv = candidate_audit(base, random, editable, len(old["x"]), protected, .1)
            torch.save(random, out / f"{random_name}_s{stage}_memory.pt")
            save(out / f"{random_name}_s{stage}_audit.json", {"kind": "random",
                "invariants": random_inv, "candidate_hash": T.memory_hash(random)})
            records.append({"arm": random_name, "memory": random,
                            "audit": {"kind": "random", "invariants": random_inv}})
            # The validation decision is made before report fits are read.
            guard_name = f"guarded_k{horizon}"
            guarded = candidate if audit["accepted"] else base
            guard_audit = {"kind": "guarded", "source": name,
                           "accepted": bool(audit["accepted"]),
                           "candidate_hash": T.memory_hash(candidate),
                           "selected_hash": T.memory_hash(guarded),
                           "report_labels_used": False}
            save(out / f"{guard_name}_s{stage}_decision.json", guard_audit)
            records.append({"arm": guard_name, "memory": guarded,
                            "audit": guard_audit})
        decision = {"dataset": data["dataset"], "order": order, "stage": stage,
                    "report_labels_used": False, "editable_rows": editable,
                    "protected_rows": protected, "teacher_hash": T.memory_hash(teacher),
                    "base_hash": T.memory_hash(base), "support_hash": T.memory_hash(support),
                    "arms": [{"arm": r["arm"], "memory_hash": T.memory_hash(r["memory"]),
                              "audit": r["audit"]} for r in records]}
        save(out / f"s{stage}_decisions.json", decision)
        for record in records:
            memory = record["memory"]
            for seed in REPORT_SEEDS:
                fit_one(memory, data, stage, seed, record["arm"], out, args.device, reference)
        for record in records:
            rows = [read(out / f"{record['arm']}_s{stage}_seed{seed}.json")
                    for seed in REPORT_SEEDS]
            cell_results.append({"arm": record["arm"], "stage": stage,
                                 "rows": len(record["memory"]["x"]),
                                 "metrics": R.C.aggregate(rows), "per_seed": rows,
                                 "memory_sha256": T.memory_hash(record["memory"]),
                                 "audit": record["audit"]})
        save(out / "results.json", {"protocol": protocol, "results": cell_results,
                                     "decisions": [f"s{s}_decisions.json" for s in (1, 2)]})
        print(json.dumps({"event": "completed_stage", "dataset": data["dataset"],
                          "order": order, "stage": stage}), flush=True)
    save(out / "completion.json", {"status": "complete", "fit_slots": len(cell_results) * 6,
                                   "completed_at": datetime.now(timezone.utc).isoformat()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--reference-root", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--text-cache-root", default="/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919")
    args = ap.parse_args()
    torch.set_num_threads(2)
    data = R.R.load_data(args.dataset, args.minilm, args.device,
                          Path(args.text_cache_root) / f"{args.dataset}_r005" / "text_emb_pool_only.pt")
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    for order in ORDERS:
        run_cell(data, order, args, root, args.reference_root)
    print("TGC_R2B_STUDENT_REVISION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
