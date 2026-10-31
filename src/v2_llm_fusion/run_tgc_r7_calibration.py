"""Deterministic dense-path repeats and coherent assignment calibration."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from dataclasses import asdict
from pathlib import Path
import json
import time

import torch
import tgc_r6_experiment as E
import run_tgc_r5_joint as V
import tgc_r7_stable as Q

R, T, D = E.R, E.T, E.D
SOURCE = Path("/home/pc/research/TAG_condensation/results/tgc_r5_joint_20260923")
ARMS = ("c0", "r5_frozen", "dense0", "dense1", "dense2", "alpha0", "alpha025", "alpha05")
ALPHAS = {"alpha0": 0., "alpha025": .25, "alpha05": .5}


def protocol():
    return json.loads(json.dumps({"name": "tgc-r7-calibration-dense-v1", "datasets": V.DATASETS,
        "orders": V.ORDERS, "report_seeds": V.SEEDS, "evaluation_stages": [2], "arms": ARMS,
        "class_groups": {d: {o: R.A.stage_groups(d, 3, o) for o in V.ORDERS} for d in V.DATASETS},
        "teacher": asdict(V.TEACHER), "assignment": asdict(V.CFG), "alphas": ALPHAS, "rate": .05,
        "scope": "fixed sealed R5 parent; dense only in construction; common sparse downstream student unchanged",
        "numerical_reason": "same adjacency/initialization/batches diverged at first teacher update even with deterministic flag",
        "probes": "two legal teachers at0/32/128;4 matched batch16 per checkpoint; source targets current delta and old synthetic",
        "candidate_selection": "none during prepare; all scored; metrics calibrated against complete student, not a deployed selector",
        "bounded_assignment": "P_alpha=(1-alpha)*P_initial_smooth+alpha*P_learned; coherent X/text/binary edges",
        "caveat": "alpha0 is smoothed assignment/top-membership text, not C0 center-text recipe",
        "source_hashes": R.source_hashes(), "test_labels_used": False, "report_before_seal": False}))


def prepare(data, order, args, spec):
    ids, budgets = D.set_stream(data, order)
    out = args.output / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    R.stable_lock(out / "manifest.json", E.manifest(data, order, budgets, ids))
    if (out / "s2_decision.json").exists():
        D.verify_decision(out, 2)
        return
    assert not (args.output / "selection_seal.json").exists()
    folder = SOURCE / data["dataset"] / order
    c0, old = E.load(folder / "c0_s2_memory.pt"), E.load(folder / "recursive_s1_memory.pt")
    base = V.append_parent(old, E.load(folder / "c0_s2_block.pt"), args.device)
    delta = T.raw_delta_block(data, ids[2], 2)
    q = D.current_assignment(data, ids[2], c0, len(old["x"]), 2, args.device)
    memories = {"c0": c0, "r5_frozen": E.load(folder / "recursive_s2_memory.pt")}
    audits, assignments = {}, {}
    with Q.N.deterministic(True):
        for index in range(3):
            started = time.perf_counter()
            ctx = Q.dense_context(old, delta, base, q, V.TEACHER, args.device, trace=True)
            p, p0, info = Q.learn(ctx, q, data["y"][ids[2]], len(old["x"]), V.CFG)
            memory, material = Q.materialize(ctx, base, delta, p, data["encoder"], V.CFG)
            R.sync(args.device)
            arm = f"dense{index}"
            memories[arm], assignments[arm] = memory, p
            audits[arm] = {"seconds": time.perf_counter()-started, "reference": ctx["audit"], "assignment": info,
                           "h_hash": Q.N.fingerprint(ctx["h"]), "materialization": material}
            if index == 0:
                ctx0, learned, initial = ctx, p, p0
        for arm, alpha in ALPHAS.items():
            p = (1-alpha)*initial + alpha*learned
            memories[arm], material = Q.materialize(ctx0, base, delta, p, data["encoder"], V.CFG)
            assignments[arm] = p
            audits[arm] = {"alpha": alpha, "materialization": material,
                           "assignment": {"assignment_sha256": R.C.tensor_hash({"p": p})}}
        targets = Q.response_targets(ctx0, q, len(base["x"]))
        for arm in ARMS:
            if arm == "c0":
                # Different historical output; report separately from fixed-parent alpha ranking.
                audits[arm] = {"reference_mismatch": True}
            audits.setdefault(arm, {})["response"] = Q.response_metrics(ctx0, memories[arm], targets)
            if arm in assignments:
                p = assignments[arm]
                u = Q.J.full_assignment(p.to(ctx0["h"].device), len(old["x"]))
                actual = Q.K.encode_models(ctx0["models"], memories[arm],
                    R.C.training_adjacency(memories[arm], "cpu").to_dense().to(args.device), args.device)
                diff = (ctx0["h"]-u@actual).square().sum(1)
                fraction = len(old["x"])/len(base["x"])
                audits[arm]["actual_reconstruction"] = float(fraction*diff[:len(old["x"])].mean()
                                                               +(1-fraction)*diff[len(old["x"]):].mean())
                torch.save(p, out / f"{arm}_s2_assignment.pt")
            audits[arm].update(payload=R.payload(memories[arm]), report_labels_used=False, historical_raw_reads=0)
            E.save(out / f"{arm}_s2_build_audit.json", audits[arm])
    comparisons = {f"dense{i}": {"same_memory": T.memory_hash(memories[f"dense{i}"]) == T.memory_hash(memories["dense0"]),
                        "same_p": torch.equal(assignments[f"dense{i}"], assignments["dense0"]),
                        "same_h": audits[f"dense{i}"]["h_hash"] == audits["dense0"]["h_hash"],
                        "same_teacher_traces": [t["trace"] for t in audits[f"dense{i}"]["reference"]["teachers"]]
                                                == [t["trace"] for t in audits["dense0"]["reference"]["teachers"]]}
                   for i in (1, 2)}
    E.save(out / "construction_comparisons.json", comparisons)
    E.save(out / "s2_decision.json", {"stage": 2, "report_labels_used": False,
               "memories": {a: D.save_memory(out, a, 2, memory) for a, memory in memories.items()}})
    E.save(out / "prepared.json", {"status": "prepared"})


def validate(data, out, stage, ids, budgets, args):
    old = E.load(SOURCE / data["dataset"] / out.name / "recursive_s1_memory.pt")
    for arm in ARMS:
        memory = E.load(out / f"{arm}_s2_memory.pt")
        E.memory_checks(memory, budgets[2])
        if arm not in ("c0", "r5_frozen"):
            R.C.assert_frozen(old, memory)
            p = E.load(out / f"{arm}_s2_assignment.pt")
            info = E.read(out / f"{arm}_s2_build_audit.json")
            assert R.C.tensor_hash({"p": p}) == info["assignment"]["assignment_sha256"]
            expected = p.double().T @ data["x"][ids[2]].double() / p.double().sum(0)[:, None]
            torch.testing.assert_close(memory["x"][len(old["x"]):].double(), expected, atol=2e-5, rtol=2e-5)
    for check in E.read(out / "construction_comparisons.json").values():
        assert all(check.values()), "dense construction is not reproducible; cannot treat as stable"


if __name__ == "__main__":
    E.main(protocol, prepare, validate)
