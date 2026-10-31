"""Fixed-parent construction repeats, sealed before common student scoring."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from dataclasses import asdict, replace
from pathlib import Path
import json
import time

import torch
import tgc_r6_experiment as E
import run_tgc_r5_joint as V
import tgc_r7_numerics as N

R, T, D = E.R, E.T, E.D
SOURCE = Path("/home/pc/research/TAG_condensation/results/tgc_r5_joint_20260923")
ARMS = ("c0", "r5_frozen", "native0", "native1", "native2", "stable0", "stable1", "stable2", "seed1", "seed2")


def protocol():
    return json.loads(json.dumps({
        "name": "tgc-r7-fixed-parent-stability-v1", "datasets": ["cora", "citeseer"],
        "orders": V.ORDERS, "report_seeds": V.SEEDS, "evaluation_stages": [2], "arms": ARMS,
        "class_groups": {d: {o: R.A.stage_groups(d, 3, o) for o in V.ORDERS} for d in V.DATASETS},
        "assignment": asdict(V.CFG), "teacher": asdict(V.TEACHER), "rate": .05,
        "fixed_inputs": "sealed R5 recursive s1, current legal delta, C0 block and source partition",
        "repeats": "3 identical seeds native; 3 identical seeds deterministic; 2 alternative teacher-seed groups deterministic",
        "alternative_teacher_seeds": [[8460, 8461], [8470, 8471]],
        "determinism_scope": "construction only; common student unchanged; CUBLAS workspace common to all arms",
        "trace": "state and gradient hash every teacher update; P hash every assignment update; no RNG consumption",
        "selection": "none; all repeats scored, no best-realization selection",
        "source_hashes": R.source_hashes(), "test_labels_used": False, "report_before_seal": False}))


def prepare(data, order, args, spec):
    ids, budgets = D.set_stream(data, order)
    out = args.output / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    R.stable_lock(out / "manifest.json", E.manifest(data, order, budgets, ids))
    if (out / "s2_decision.json").exists():
        D.verify_decision(out, 2)
        return
    if (args.output / "selection_seal.json").exists():
        raise RuntimeError("cannot prepare after seal")
    source = SOURCE / data["dataset"] / order
    old = E.load(source / "recursive_s1_memory.pt")
    c0 = E.load(source / "c0_s2_memory.pt")
    base = V.append_parent(old, E.load(source / "c0_s2_block.pt"), args.device)
    q = D.current_assignment(data, ids[2], c0, len(old["x"]), 2, args.device)
    delta = T.raw_delta_block(data, ids[2], 2)
    memories = {"c0": c0, "r5_frozen": E.load(source / "recursive_s2_memory.pt")}
    audits, ps = {}, {}
    for arm in ARMS[2:]:
        cfg = V.TEACHER
        if arm.startswith("seed"):
            start = 8450 + 10 * int(arm[-1])
            cfg = replace(cfg, teacher_seeds=(start, start + 1))
        R.sync(args.device)
        started = time.perf_counter()
        memory, p, audit = N.construct(old, delta, base, q, data["y"][ids[2]], data["encoder"],
                                       cfg, args.device, V.CFG, stable=not arm.startswith("native"))
        R.sync(args.device)
        audit.update(seconds=time.perf_counter()-started, parent_hash=T.memory_hash(old),
                     payload=R.payload(memory), teacher_seeds=cfg.teacher_seeds)
        memories[arm], audits[arm], ps[arm] = memory, audit, p
        E.save(out / f"{arm}_s2_build_audit.json", audit)
        torch.save(p, out / f"{arm}_s2_assignment.pt")
        print(json.dumps({"event": "repeat_constructed", "dataset": data["dataset"], "order": order,
                          "arm": arm, "seconds": audit["seconds"]}), flush=True)
    comparisons = {}
    for group in ("native", "stable"):
        for i in (1, 2):
            a, b = group + "0", group + str(i)
            comparisons[b + "_vs_" + a] = N.compare(memories[a], memories[b], ps[a], ps[b], audits[a], audits[b])
    a, b = "native0", "stable0"
    comparisons[b + "_vs_" + a] = N.compare(memories[a], memories[b], ps[a], ps[b], audits[a], audits[b])
    E.save(out / "construction_comparisons.json", comparisons)
    E.save(out / "s2_decision.json", {"stage": 2, "report_labels_used": False,
                                      "memories": {a: D.save_memory(out, a, 2, m) for a, m in memories.items()}})
    E.save(out / "prepared.json", {"status": "prepared"})


def validate(data, out, stage, ids, budgets, args):
    old = E.load(SOURCE / data["dataset"] / out.name / "recursive_s1_memory.pt")
    for arm in ARMS:
        memory = E.load(out / f"{arm}_s2_memory.pt")
        E.memory_checks(memory, budgets[2])
        if arm not in ("c0", "r5_frozen"):
            R.C.assert_frozen(old, memory)
            p = E.load(out / f"{arm}_s2_assignment.pt")
            audit = E.read(out / f"{arm}_s2_build_audit.json")
            assert N.fingerprint({"p": p}) == audit["assignment"]["assignment_sha256"]
            expected = p.double().T @ data["x"][ids[2]].double() / p.double().sum(0)[:, None]
            torch.testing.assert_close(memory["x"][len(old["x"]):].double(), expected, atol=2e-5, rtol=2e-5)
            assert not audit["report_labels_used"] and audit["historical_raw_reads"] == 0


if __name__ == "__main__":
    E.main(protocol, prepare, validate)
