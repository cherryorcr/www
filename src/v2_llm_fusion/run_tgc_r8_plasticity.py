"""Frozen R8: deterministic sparse TGC-R and C0-anchored plasticity."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import torch

import tgc_r6_experiment as E
import run_tgc_r7_recursive as R7
import run_tgc_r5_joint as V
import tgc_r7_stable as Q
import tgc_r7_deterministic as F

R, T, D, B = E.R, E.T, E.D, E.B
DATASETS = ("cora", "citeseer")
ORDERS = ("order19600", "order19609", "order19610")
LAMBDAS = (0., .25, .5, 1.)
TEACHER_GROUPS = ((8800, 8801), (8810, 8811))
REPORT_SEEDS = tuple(range(220, 226))
EDGE_LIMIT = 65536
DET_ARMS = tuple(f"det_g{g}_l{int(lam*100):03d}" for g in range(2) for lam in LAMBDAS)
ARMS = ("c0", "k_center", "dense_g0") + DET_ARMS


def protocol():
    groups = {d: {o: R.A.stage_groups(d, 3, o) for o in ORDERS} for d in DATASETS}
    return json.loads(json.dumps({
        "name": "tgc-r8-deterministic-sparse-plasticity-v1", "datasets": DATASETS,
        "orders": ORDERS, "class_groups": groups, "evaluation_stages": [0, 1, 2],
        "report_seeds": REPORT_SEEDS, "arms": ARMS, "teacher_groups": TEACHER_GROUPS,
        "lambdas": LAMBDAS, "edge_limit": EDGE_LIMIT, "rate": .05,
        "hypothesis": "C0 anchor on new synthetic X restores new-class plasticity while retaining TGC text/topology",
        "anchor": "X_new=(1-lambda)*X_C0_new+lambda*X_TGC_new; text/topology/mass and old rows held at TGC values",
        "determinism": "sorted COO; deterministic segmented row and column reductions; custom adjoint; compare against R7 dense construction",
        "recursive": "each lambda and teacher-seed group reads its own previous synthetic memory; no historical raw replay",
        "dense_reference": "R7 dense constructor, teacher group 8800/8801, recursive, lambda=1",
        "student": "unchanged canonical two-layer GCN + frozen MiniLM inputs; batch16, Adam.001, max(50,min(2048,16K)), scratch",
        "primary": "final-stage all-prompt pooled report-dev Macro-F1; stage mean, old/new F1, worst order and costs reported",
        "controls": "C0, k-center real coreset, dense TGC-R, deterministic TGC-R lambda=1, anchored lambda=0/.25/.5",
        "selection": "all lambda values preregistered and reported; no per-order, seed, or teacher-group selection",
        "history_scan": "orders chosen after scanning available local manifests/protocols; independent from R7 orders19605/19606/19608",
        "test_labels_used": False, "report_before_seal": False, "historical_raw_reads": 0,
        "source_hashes": R.source_hashes()}))


def _append(parent, block, device):
    return R.SCA.append_memory(parent, block, len(parent["x"]), topology_mode="frozen_old",
                               cross_degree=1, refresh_mode="frozen_old", device=device)


def _arm_lambda(arm):
    _, group, value = arm.split("_")
    return int(group[1:]), int(value[1:]) / 100


def _det_branch(old, delta, base, q, labels, data, cfg, args, audit_dense=False):
    with Q.N.deterministic(True):
        candidates, p, audit = F.construct(old, delta, base, q, labels, data["encoder"],
                                           cfg, args.device, V.CFG, EDGE_LIMIT, audit_dense)
    return candidates, p, audit


def _dense_branch(old, delta, base, q, labels, data, cfg, args):
    with Q.N.deterministic(True):
        ctx = Q.dense_context(old, delta, base, q, cfg, args.device, keep_probes=False)
        p, _, assignment = Q.learn(ctx, q, labels, len(old["x"]), V.CFG)
        memory, material = Q.materialize(ctx, base, delta, p, data["encoder"], V.CFG)
    ctx["audit"]["h_hash"] = Q.N.fingerprint(ctx["h"])
    return memory, p, {"context": ctx["audit"], "assignment": assignment,
                                    "materialization": material, "seconds": ctx["audit"]["seconds"],
                                    "historical_raw_reads": 0, "report_labels_used": False}


def _build_kcenter(old, c0, data, ids, budgets, stage, args):
    R.sync(args.device)
    started = time.perf_counter()
    block_rows = budgets[stage] - (budgets[stage-1] if stage else 0)
    block, info = R.C.real_block(data, ids, block_rows, "k_center", stage, args.device)
    memory = R.SCA.append_memory(old, block, len(old["x"]) if old else 0,
                                 topology_mode="frozen_old", cross_degree=1,
                                 refresh_mode="frozen_old", device=args.device)
    R.sync(args.device)
    return memory, {**info, "seconds": time.perf_counter()-started,
                    "current_selected_ids": block["source_ids"].tolist(),
                    "real_coreset": True}


def prepare(data, order, args, spec):
    ids, budgets = D.set_stream(data, order)
    out = args.output / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    R.stable_lock(out / "manifest.json", E.manifest(data, order, budgets, ids))
    for stage in range(3):
        decision_path = out / f"s{stage}_decision.json"
        if decision_path.exists():
            D.verify_decision(out, stage)
            continue
        if (args.output / "selection_seal.json").exists():
            raise RuntimeError("cannot prepare after sealing")
        c0, c0_audit = B.build_c0(data, ids[stage], budgets, stage, out, args.device)
        memories, audits = {"c0": c0}, {"c0": c0_audit}
        previous = {arm: E.load(out / f"{arm}_s{stage-1}_memory.pt")
                    for arm in ARMS if stage and arm != "c0"}
        kcenter, ka = _build_kcenter(previous.get("k_center"), c0, data, ids[stage],
                                     budgets, stage, args)
        memories["k_center"], audits["k_center"] = kcenter, ka
        if not stage:
            for arm in ARMS:
                if arm not in memories:
                    memories[arm], audits[arm] = c0, {**c0_audit, "alias": "c0"}
        else:
            labels = data["y"][ids[stage]]
            delta = T.raw_delta_block(data, ids[stage], stage)
            q = D.current_assignment(data, ids[stage], c0, budgets[stage-1], stage, args.device)
            block = E.load(out / f"c0_s{stage}_block.pt")

            dense_old = previous["dense_g0"]
            dense_base = _append(dense_old, block, args.device)
            dense_cfg = replace(V.TEACHER, teacher_seeds=TEACHER_GROUPS[0])
            dense, dense_p, dense_audit = _dense_branch(dense_old, delta, dense_base, q,
                                                        labels, data, dense_cfg, args)
            memories["dense_g0"] = dense
            torch.save(dense_p, out / f"dense_g0_s{stage}_assignment.pt")
            audits["dense_g0"] = {**dense_audit, "parent_hash": T.memory_hash(dense_old),
                                  "current_delta_ids": ids[stage].tolist(), "lambda": 1.,
                                  "teacher_group": 0, "historical_raw_reads": 0}

            contexts = {}
            matched_stage1 = None
            for group_index, teacher_seeds in enumerate(TEACHER_GROUPS):
                for arm in (a for a in DET_ARMS if _arm_lambda(a)[0] == group_index):
                    lam = _arm_lambda(arm)[1]
                    old = previous[arm]
                    base = _append(old, block, args.device)
                    cfg = replace(V.TEACHER, teacher_seeds=teacher_seeds)
                    cache_key = (group_index, T.memory_hash(old))
                    if cache_key not in contexts:
                        do_dense_audit = (data["dataset"] == "cora" and order == "order19600"
                                          and stage == 1 and group_index == 0)
                        tgc, p, audit = _det_branch(old, delta, base, q, labels, data, cfg, args,
                                                    audit_dense=do_dense_audit)
                        contexts[cache_key] = (tgc, p, audit, T.memory_hash(base))
                        if data["dataset"] == "cora" and order == "order19600" and stage == 1 and group_index == 0:
                            matched_stage1 = (old, base, p, tgc, audit, cfg)
                    tgc, p, audit, base_hash = contexts[cache_key]
                    reference_path = out / f"{arm}_s{stage}_tgc_reference.pt"
                    if reference_path.exists():
                        existing = E.load(reference_path)
                        if T.memory_hash(existing) != T.memory_hash(tgc):
                            raise AssertionError("cached TGC reference changed")
                    else:
                        torch.save(tgc, reference_path)
                    anchored = F.apply_x_anchor(tgc, c0, budgets[stage-1], lam)
                    memories[arm] = anchored
                    torch.save(p, out / f"{arm}_s{stage}_assignment.pt")
                    audits[arm] = {**audit, "parent_hash": T.memory_hash(old),
                                   "base_hash": base_hash, "current_delta_ids": ids[stage].tolist(),
                                   "lambda": lam, "teacher_group": group_index,
                                   "assignment_sha256": R.C.tensor_hash({"p": p}),
                                   "tgc_reference_file": reference_path.name,
                                   "tgc_reference_hash": T.memory_hash(tgc),
                                   "historical_raw_reads": 0, "report_labels_used": False}
                    del p
            del contexts
            if data["dataset"] == "cora" and order == "order19600" and stage == 1:
                if matched_stage1 is None:
                    raise AssertionError("stage-1 deterministic replay probe was not built")
                old, base, p, memory, audit, cfg = matched_stage1
                repeated, p_repeat, repeated_audit = _det_branch(
                    old, delta, base, q, labels, data, cfg, args, audit_dense=False)
                repeat = {"same_assignment": torch.equal(p, p_repeat),
                          "same_memory": T.memory_hash(memory) == T.memory_hash(repeated),
                          "assignment_max_abs": float((p-p_repeat).abs().max()),
                          "x_max_abs": float((memory["x"]-repeated["x"]).abs().max()),
                          "h_hash_equal": audit["context"]["h_hash"] == repeated_audit["context"]["h_hash"],
                          "first_teacher_state_hashes_equal": [x["state_sha256"] for x in audit["context"]["teachers"]]
                          == [x["state_sha256"] for x in repeated_audit["context"]["teachers"]]}
                if not all(repeat[key] for key in ("same_assignment", "same_memory", "h_hash_equal",
                                                    "first_teacher_state_hashes_equal")):
                    raise AssertionError("deterministic full-construction replay diverged")
                E.save(out / "deterministic_replay_audit.json", repeat)

                dense_memory = memories["dense_g0"]
                dense_l1 = memories["det_g0_l100"]
                dense_p_path = out / "dense_g0_s1_assignment.pt"
                if not dense_p_path.exists():
                    torch.save(dense_p, dense_p_path)
                det_p = E.load(out / "det_g0_l100_s1_assignment.pt")
                sparse_vs_dense = {"assignment_max_abs": float((det_p-dense_p).abs().max()),
                                   "x_max_abs": float((dense_l1["x"]-dense_memory["x"]).abs().max()),
                                   "text_equal": dense_l1["texts"] == dense_memory["texts"],
                                   "edges_equal": torch.equal(dense_l1["edge_index"], dense_memory["edge_index"]),
                                   "dense_h_hash": dense_audit["context"]["h_hash"],
                                   "segmented_h_hash": audits["det_g0_l100"]["context"]["h_hash"],
                                   "same_h_bitwise": dense_audit["context"]["h_hash"]
                                   == audits["det_g0_l100"]["context"]["h_hash"],
                                   "deterministic_operator_audit": audits["det_g0_l000"]["context"].get("operator_audit")}
                E.save(out / "dense_vs_segmented_construction.json", sparse_vs_dense)

        for arm, memory in memories.items():
            audits[arm]["payload"] = R.payload(memory)
            audits[arm]["text"] = R.SCA.text_audit(memory["texts"], data["encoder"].tokenizer)
            E.save(out / f"{arm}_s{stage}_build_audit.json", audits[arm])
        decision = {"stage": stage, "report_labels_used": False,
                    "memories": {arm: D.save_memory(out, arm, stage, memory)
                                 for arm, memory in memories.items()}}
        E.save(decision_path, decision)
        E.save(out / "prepared.json", {"status": "prepared"})
        print(json.dumps({"event": "prepared", "dataset": data["dataset"],
                          "order": order, "stage": stage}), flush=True)


def validate(data, out, stage, ids, budgets, args):
    c0 = E.load(out / f"c0_s{stage}_memory.pt")
    for arm in ARMS:
        memory = E.load(out / f"{arm}_s{stage}_memory.pt")
        E.memory_checks(memory, budgets[stage])
        audit = E.read(out / f"{arm}_s{stage}_build_audit.json")
        if stage and arm not in ("c0", "k_center"):
            parent = E.load(out / f"{arm}_s{stage-1}_memory.pt")
            R.C.assert_frozen(parent, memory)
            old_n = len(parent["x"])
            if arm == "dense_g0":
                continue
            group, lam = _arm_lambda(arm)
            c0_x, tgc_x = c0["x"][old_n:], None
            sibling = f"det_g{group}_l100"
            reference = E.load(out / f"{sibling}_s{stage}_memory.pt")
            reference = E.load(out / audit["tgc_reference_file"])
            assert T.memory_hash(reference) == audit["tgc_reference_hash"]
            expected_parent = reference["x"][:old_n]
            torch.testing.assert_close(memory["x"][:old_n], expected_parent, atol=0, rtol=0)
            tgc_x = reference["x"][old_n:]
            expected = (1-lam)*c0_x + lam*tgc_x
            torch.testing.assert_close(memory["x"][old_n:], expected, atol=0, rtol=0)
            for key in ("text_emb", "mass", "depth", "stage", "edge_index"):
                assert torch.equal(memory[key], reference[key]), f"anchoring changed {key}"
            assert memory["texts"] == reference["texts"]
            assert audit["historical_raw_reads"] == 0 and not audit["report_labels_used"]
        elif stage and arm == "k_center":
            selected = torch.tensor(audit["current_selected_ids"])
            assert set(selected.tolist()) <= set(ids[stage].tolist())
            torch.testing.assert_close(memory["x"][budgets[stage-1]:],
                                       data["x"][selected].float(), atol=0, rtol=0)
    # At stage 1 all lambda arms share the aliased C0 parent, so their
    # TGC text/topology should match.  From stage 2 onward each arm has its
    # own anchored parent; cross-lambda text/topology equality is no longer
    # a valid invariant and is checked per arm against its saved TGC reference.
    if stage == 1:
        for group in range(2):
            siblings = [E.load(out / f"det_g{group}_l{int(lam*100):03d}_s{stage}_memory.pt")
                        for lam in LAMBDAS]
            ref = siblings[-1]
            for memory in siblings[:-1]:
                assert torch.equal(memory["text_emb"], ref["text_emb"])
                assert memory["texts"] == ref["texts"]
                assert torch.equal(memory["edge_index"], ref["edge_index"])


if __name__ == "__main__":
    E.main(protocol, prepare, validate)
