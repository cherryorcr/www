"""Two-phase independent TGC-R3 experiment: seal all decisions, then report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import time

import torch

import tgc_selector_confirmation as M

R, T, S, A = M.R, M.T, M.S, M.A
DATASETS, ORDERS, ARMS, REPORT_SEEDS = tuple(M.GROUPS), M.ORDERS, M.ARMS, M.REPORT_SEEDS


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    R.save_json(Path(path), value)


def protocol():
    return json.loads(json.dumps({**M.frozen_protocol(), "source_hashes": R.source_hashes()}))


def cell_data(data, order):
    groups = M.GROUPS[data["dataset"]][order]
    if R.A.stage_groups(data["dataset"], 3, order) != groups:
        raise AssertionError("frozen class groups disagree with order seed")
    return R.A.set_stream(data, groups, .05)


def validate_decision(out, stage):
    d = read(out / f"s{stage}_decision.json")
    for arm in ARMS:
        path = out / f"{arm}_s{stage}_memory.pt"
        if R.C.sha(path) != d["memories"][arm]["file_sha256"]:
            raise ValueError(f"changed decision artifact: {path}")
        if T.memory_hash(load(path)) != d["memories"][arm]["memory_sha256"]:
            raise ValueError("decision memory hash mismatch")
    return d


def build_c0(data, ids, budgets, stage, out, device):
    path = out / f"c0_s{stage}_memory.pt"
    audit_path = out / f"c0_s{stage}_construction.json"
    old = load(out / f"c0_s{stage-1}_memory.pt") if stage else None
    parent = T.memory_hash(old) if old else None
    if audit_path.exists() and path.exists():
        audit = read(audit_path)
        memory = load(path)
        if (audit["memory_sha256"] != T.memory_hash(memory) or audit["old_hash"] != parent
                or audit["source_ids"] != ids.tolist() or len(memory["x"]) != budgets[stage]):
            raise ValueError("cached C0 construction mismatch")
        return memory, audit
    keep = budgets[stage-1] if stage else 0
    R.sync(device)
    start = time.perf_counter()
    block = R.U.build_stage_block(data, ids, budgets[stage] - keep, data["encoder"],
                                  stage, "joint_g5_c320", "concat", device,
                                  "dispersion", data["stage_classes"][stage])
    memory = R.SCA.append_memory(old, block, keep, topology_mode="frozen_old",
                                 cross_degree=1, refresh_mode="frozen_old", device=device)
    R.sync(device)
    if len(memory["x"]) != budgets[stage]:
        raise AssertionError("fresh C0 budget mismatch")
    if old is not None:
        R.C.assert_frozen(old, memory)
    audit = {"memory_sha256": T.memory_hash(memory), "old_hash": parent,
             "source_ids": ids.tolist(), "rows": len(memory["x"]),
             "new_rows": budgets[stage] - keep, "seconds": time.perf_counter() - start}
    torch.save(block, out / f"c0_s{stage}_block.pt")
    torch.save(memory, path)
    save(audit_path, audit)
    return memory, audit


def prepare(data, order, args):
    ids, budgets = cell_data(data, order)
    out = args.output / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": data["dataset"], "order": order,
                "stage_classes": data["stage_classes"], "budgets": budgets,
                "stage_source_ids": [x.tolist() for x in ids],
                "source_dataset_sha256": R.C.sha(R.U.DATA_CFG[data["dataset"]]["path"]),
                "pool_sha256": R.C.tensor_hash({"pool": data["pool_mask"]}),
                "report_mask_sha256": R.C.tensor_hash({"report": data["report_dev_mask"]})}
    R.stable_lock(out / "manifest.json", manifest)
    for stage in (0, 1, 2):
        if (out / f"s{stage}_decision.json").exists():
            validate_decision(out, stage)
            continue
        if (args.output / "selection_seal.json").exists():
            raise ValueError("cannot construct new decisions after report seal")
        base, construction = build_c0(data, ids[stage], budgets, stage, out, args.device)
        decision = {"dataset": data["dataset"], "order": order, "stage": stage,
                    "base_hash": T.memory_hash(base), "construction": construction,
                    "selector_labels_used": False, "historical_raw_reads": 0}
        if stage == 0:
            memories = {arm: base for arm in ARMS}
            decision.update(policies={p: {"accepted": False, "selected_arm": "c0"}
                                      for p in M.POLICIES}, editable_rows=[], protected_rows=[])
        else:
            old = load(out / f"c0_s{stage-1}_memory.pt")
            delta = T.raw_delta_block(data, ids[stage], stage)
            support = T._concat_support(old, delta)
            R.sync(args.device)
            start = time.perf_counter()
            revised, teacher_audit = T.revise_old_memory(old, delta, T.RevisionConfig(), args.device, "gate")
            R.sync(args.device)
            teacher_seconds = time.perf_counter() - start
            T.assert_revision_invariants(old, revised, teacher_audit, T.RevisionConfig())
            editable = teacher_audit["updated_rows"] or teacher_audit["eligible_rows"][:teacher_audit["revision_quota"]]
            protected = teacher_audit["protected_rows"]
            if not editable:
                raise ValueError("no editable rows in preregistered cell")
            teacher = T.clone_memory(base)
            teacher["x"][:len(old["x"])] = revised["x"]
            prefixes, prefix_audit = A.prepare_prefixes(base, M.WARMUP, args.device, M.DEFAULT_CFG)
            prefix_path = out / f"s{stage}_prefixes.pt"
            torch.save(prefixes, prefix_path)
            prefix_audit.update(checkpoint_file=prefix_path.name, checkpoint_sha256=R.C.sha(prefix_path))
            candidate, opt = A.optimize(base, support, editable, stage, 1, prefixes, args.device, M.DEFAULT_CFG)
            stop, stop_audit = A.optimize(base, support, editable, stage, 1, prefixes, args.device, M.DEFAULT_CFG, stop=True)
            if T.memory_hash(stop) != T.memory_hash(base) or stop_audit["nonzero_gradient_steps"]:
                raise AssertionError("stop-gradient is not an exact no-op")
            random = S.matched_random(base, candidate, editable, stage)
            curve = M.curve_scores(base, candidate, support, stage, args.device, out)
            decision.update(M.decision_summary(opt, curve))
            decision.update(old_hash=T.memory_hash(old), support_hash=T.memory_hash(support),
                            old_rows=len(old["x"]), editable_rows=editable, protected_rows=protected,
                            teacher_audit=teacher_audit, teacher_seconds=teacher_seconds,
                            prefix_audit=prefix_audit, optimization=opt, stop_audit=stop_audit, curve=curve)
            memories = {"c0": base, "teacher": teacher, "candidate": candidate, "random": random, "stop": stop}
            for policy in M.POLICIES:
                memories[policy] = memories[decision["policies"][policy]["selected_arm"]]
            decision["invariants"] = {arm: S.invariants(base, memory, editable, len(old["x"]), protected, M.DEFAULT_CFG.radius)
                                      for arm, memory in memories.items()}
        decision["memories"] = {}
        for arm in ARMS:
            path = out / f"{arm}_s{stage}_memory.pt"
            if arm != "c0":
                torch.save(memories[arm], path)
            decision["memories"][arm] = {"memory_sha256": T.memory_hash(memories[arm]),
                                          "file_sha256": R.C.sha(path), "file": path.name}
        decision["locked_at"] = datetime.now(timezone.utc).isoformat()
        save(out / f"s{stage}_decision.json", decision)
        print(json.dumps({"event": "decision_locked", "dataset": data["dataset"],
                          "order": order, "stage": stage, "policies": decision["policies"]}), flush=True)
    save(out / "prepared.json", {"status": "prepared", "report_fits_started": False})


def seal(root):
    entries = {}
    for dataset in DATASETS:
        for order in ORDERS:
            out = root / dataset / order
            if read(out / "prepared.json")["status"] != "prepared":
                raise ValueError("incomplete decision preparation")
            for stage in (0, 1, 2):
                validate_decision(out, stage)
                path = out / f"s{stage}_decision.json"
                entries[str(path.relative_to(root)).replace("\\", "/")] = R.C.sha(path)
    value = {"decision_hashes": entries, "protocol_sha256": R.C.sha(root / "protocol.json")}
    R.stable_lock(root / "selection_seal.json", value)
    return value


def verify_seal(root):
    saved = read(root / "selection_seal.json")
    if saved["protocol_sha256"] != R.C.sha(root / "protocol.json"):
        raise ValueError("sealed protocol changed")
    expected = {f"{d}/{o}/s{s}_decision.json" for d in DATASETS for o in ORDERS for s in (0, 1, 2)}
    if set(saved["decision_hashes"]) != expected:
        raise ValueError("decision seal matrix incomplete")
    for path, value in saved["decision_hashes"].items():
        if R.C.sha(root / path) != value:
            raise ValueError("selection changed after sealing")
    return R.C.sha(root / "selection_seal.json")


def alias_fit(source, arm, stage, seed, out):
    path = out / f"{source}_s{stage}_seed{seed}.json"
    row = read(path)
    target = out / f"{arm}_s{stage}_seed{seed}.json"
    shutil.copy2(path.with_suffix(".pt"), target.with_suffix(".pt"))
    row.update(arm=arm, executed_here=False, reused_from=str(path), reused_source_sha256=R.C.sha(path))
    save(target, row)
    return row


def evaluate(data, order, args):
    seal_hash = verify_seal(args.output)
    cell_data(data, order)
    out = args.output / data["dataset"] / order
    results = []
    for stage in (0, 1, 2):
        decision = validate_decision(out, stage)
        sources = {}
        for arm in ARMS:
            memory = load(out / f"{arm}_s{stage}_memory.pt")
            memory_hash = T.memory_hash(memory)
            rows = []
            for seed in REPORT_SEEDS:
                if memory_hash in sources:
                    row = alias_fit(sources[memory_hash], arm, stage, seed, out)
                else:
                    row = R.fit(memory, data, stage, seed, arm, out, args.device, None)
                    row.update(selection_seal_sha256=seal_hash)
                    save(out / f"{arm}_s{stage}_seed{seed}.json", row)
                rows.append(row)
            sources.setdefault(memory_hash, arm)
            results.append({"stage": stage, "arm": arm, "rows": len(memory["x"]),
                            "metrics": R.C.aggregate(rows), "per_seed": rows,
                            "memory_sha256": memory_hash})
            save(out / "progress.json", {"results": results})
            print(json.dumps({"event": "evaluated", "dataset": data["dataset"], "order": order,
                              "stage": stage, "arm": arm,
                              "f1": results[-1]["metrics"]["all"]["cumulative_f1"]["mean"]}), flush=True)
    save(out / "results.json", {"protocol": read(args.output / "protocol.json"),
                                "manifest": read(out / "manifest.json"), "results": results})
    save(out / "completion.json", {"status": "complete", "fit_slots": len(results) * len(REPORT_SEEDS),
                                   "executed_fits": sum(r["executed_here"] for item in results for r in item["per_seed"]),
                                   "completed_at": datetime.now(timezone.utc).isoformat()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--phase", choices=("prepare", "seal", "evaluate"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--text-cache-root", type=Path, default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    R.stable_lock(args.output / "protocol.json", protocol())
    if args.phase == "seal":
        seal(args.output)
    else:
        if not args.dataset:
            ap.error("--dataset required for prepare/evaluate")
        if args.phase == "evaluate":
            verify_seal(args.output)
        data = R.R.load_data(args.dataset, args.minilm, args.device,
                             args.text_cache_root / f"{args.dataset}_r005" / "text_emb_pool_only.pt")
        for order in ORDERS:
            (prepare if args.phase == "prepare" else evaluate)(data, order, args)
    print(f"TGC_R3_{args.phase.upper()}_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
