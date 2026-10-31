"""One coherent assignment bound with full recursive histories and seed controls."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from dataclasses import asdict, replace
from pathlib import Path
import json
import time

import torch
import tgc_r6_experiment as E
import run_tgc_r5_joint as V
import tgc_r7_stable as Q

R, T, D, B = E.R, E.T, E.D, E.B
CONFIRM = os.environ.get("TGC_R7_BATCH", "development") == "confirmation"
HISTORY = Path(__file__).resolve().parents[2] / "results/tgc_r7_preparation_20260923/history_selection.json"
TEACHER_GROUPS = ((8700, 8701), (8710, 8711)) if CONFIRM else ((8400, 8401),)
ARMS = ("c0", "k_center") if CONFIRM else ("c0",)
ARMS += tuple(f"{kind}{i}" for kind in ("original", "dense", "bounded") for i in range(len(TEACHER_GROUPS)))


def protocol():
    history = E.read(HISTORY) if CONFIRM else None
    orders = history["orders"] if CONFIRM else V.ORDERS
    groups = {d: {o: R.A.stage_groups(d, 3, o) for o in orders} for d in V.DATASETS}
    if CONFIRM:
        assert groups == history["class_groups"]
    return json.loads(json.dumps({"name": "tgc-r7-recursive-" + ("confirmation" if CONFIRM else "development") + "-v1",
        "datasets": V.DATASETS, "orders": orders, "class_groups": groups,
        "report_seeds": list(range(160, 166)) if CONFIRM else V.SEEDS,
        "evaluation_stages": [0, 1, 2], "arms": ARMS, "teacher_groups": TEACHER_GROUPS,
        "teacher": asdict(V.TEACHER), "assignment": asdict(V.CFG), "rate": .05,
        "bounded_alpha": .5, "candidate": "P=.5*smoothed_initial+.5*optimized_P; coherent X/text/edges; own history",
        "controls": "original sparse R5 and deterministic dense equivalent; all teacher seeds matched by group",
        "student": "unchanged sparse GCN,batch16,Adam.001,max(50,min(2048,16K)),scratch",
        "inference": "common full-real-graph transductive, all-class prompt competition, report-dev seen nodes",
        "primary": "final Macro-F1 averaged over construction groups then paired student seeds and equally weighted orders",
        "cost": "record teacher/assignment/materialization + common C0 block; excludes tracing/response probes and cold cache",
        "history_audit": {k: v for k, v in history.items() if k != "files"} if history else None,
        "selection": "alpha .5 fixed before calibration scores; no per-cell or seed selection",
        "test_labels_used": False, "report_before_seal": False, "historical_raw_reads": 0,
        "source_hashes": R.source_hashes()}))


def prepare(data, order, args, spec):
    ids, budgets = D.set_stream(data, order)
    out = args.output / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    R.stable_lock(out / "manifest.json", E.manifest(data, order, budgets, ids))
    for stage in range(3):
        if (out / f"s{stage}_decision.json").exists():
            D.verify_decision(out, stage)
            continue
        assert not (args.output / "selection_seal.json").exists()
        c0, ca = B.build_c0(data, ids[stage], budgets, stage, out, args.device)
        memories, audits = {"c0": c0}, {"c0": ca}
        if stage:
            block = E.load(out / f"c0_s{stage}_block.pt")
            delta = T.raw_delta_block(data, ids[stage], stage)
            q = D.current_assignment(data, ids[stage], c0, budgets[stage-1], stage, args.device)
            torch.save(q, out / f"s{stage}_initial_assignment.pt")
        for arm in ARMS[1:]:
            old = E.load(out / f"{arm}_s{stage-1}_memory.pt") if stage else None
            if arm == "k_center":
                started = time.perf_counter()
                keep = budgets[stage-1] if stage else 0
                block_real, info = R.C.real_block(data, ids[stage], budgets[stage]-keep, arm, stage, args.device)
                memory = R.SCA.append_memory(old, block_real, keep, topology_mode="frozen_old", cross_degree=1,
                                             refresh_mode="frozen_old", device=args.device)
                R.sync(args.device)
                audits[arm] = {**info, "seconds": time.perf_counter()-started,
                               "current_selected_ids": block_real["source_ids"].tolist(), "real_coreset": True}
            elif not stage:
                memory, audits[arm] = c0, {**ca, "alias": "c0"}
            else:
                cfg = replace(V.TEACHER, teacher_seeds=TEACHER_GROUPS[int(arm[-1])])
                started = time.perf_counter()
                base = V.append_parent(old, block, args.device)
                if arm.startswith("original"):
                    candidates, p, info = Q.J.construct(old, delta, base, q, data["y"][ids[stage]],
                                                        data["encoder"], cfg, args.device, V.CFG)
                    memory = candidates["joint"]
                else:
                    with Q.N.deterministic(True):
                        ctx = Q.dense_context(old, delta, base, q, cfg, args.device, keep_probes=False)
                        learned, initial, assn = Q.learn(ctx, q, data["y"][ids[stage]], len(old["x"]), V.CFG)
                        p = .5*(initial+learned) if arm.startswith("bounded") else learned
                        memory, material = Q.materialize(ctx, base, delta, p, data["encoder"], V.CFG)
                        info = {"teachers": ctx["audit"]["teachers"], "context": ctx["audit"],
                                "assignment": {**assn, "assignment_sha256": R.C.tensor_hash({"p": p})},
                                "materialization": material, "alpha": .5 if arm.startswith("bounded") else 1.,
                                "h_hash": Q.N.fingerprint(ctx["h"])}
                        torch.save({"initial": initial, "learned": learned}, out / f"{arm}_s{stage}_endpoints.pt")
                        del ctx
                R.sync(args.device)
                audits[arm] = {**info, "seconds": time.perf_counter()-started, "c0_block_seconds": ca["seconds"],
                               "parent_hash": T.memory_hash(old), "teacher_seeds": cfg.teacher_seeds,
                               "current_delta_ids": ids[stage].tolist(), "historical_raw_reads": 0, "report_labels_used": False}
                torch.save(p, out / f"{arm}_s{stage}_assignment.pt")
            memories[arm] = memory
        for arm, memory in memories.items():
            audits[arm].update(payload=R.payload(memory), text=R.SCA.text_audit(memory["texts"], data["encoder"].tokenizer))
            E.save(out / f"{arm}_s{stage}_build_audit.json", audits[arm])
        E.save(out / f"s{stage}_decision.json", {"stage": stage, "report_labels_used": False,
                   "memories": {a: D.save_memory(out, a, stage, memory) for a, memory in memories.items()}})
        print(json.dumps({"event": "prepared", "dataset": data["dataset"], "order": order, "stage": stage}), flush=True)
    E.save(out / "prepared.json", {"status": "prepared"})


def validate(data, out, stage, ids, budgets, args):
    for arm in ARMS:
        memory = E.load(out / f"{arm}_s{stage}_memory.pt")
        E.memory_checks(memory, budgets[stage])
        if stage:
            old = E.load(out / f"{arm}_s{stage-1}_memory.pt")
            R.C.assert_frozen(old, memory)
        audit = E.read(out / f"{arm}_s{stage}_build_audit.json")
        n = budgets[stage-1] if stage else 0
        if arm == "k_center":
            chosen = torch.tensor(audit["current_selected_ids"])
            assert len(chosen.unique()) == len(chosen) and set(chosen.tolist()) <= set(ids[stage].tolist())
            torch.testing.assert_close(memory["x"][n:], data["x"][chosen].float(), atol=0, rtol=0)
        if stage and arm not in ("c0", "k_center"):
            p = E.load(out / f"{arm}_s{stage}_assignment.pt")
            assert R.C.tensor_hash({"p": p}) == audit["assignment"]["assignment_sha256"]
            assert audit["parent_hash"] == T.memory_hash(old) and audit["current_delta_ids"] == ids[stage].tolist()
            torch.testing.assert_close(p.sum(1), torch.ones(len(p)), atol=2e-6, rtol=1e-5)
            q = E.load(out / f"s{stage}_initial_assignment.pt")
            classes = Q.J.row_classes(q, data["y"][ids[stage]], p.shape[1])
            assert int(torch.count_nonzero(p[data["y"][ids[stage]][:, None] != classes[None, :]])) == 0
            expected = p.double().T @ data["x"][ids[stage]].double() / p.double().sum(0)[:, None]
            torch.testing.assert_close(memory["x"][n:].double(), expected, atol=2e-5, rtol=2e-5)
            effective = 1 / (p/p.sum(0)).square().sum(0)
            assert float(effective.min()) > 1.01, "candidate collapsed into real-node selection"
            if arm.startswith("bounded"):
                ends = E.load(out / f"{arm}_s{stage}_endpoints.pt")
                torch.testing.assert_close(p, .5*(ends["initial"]+ends["learned"]), atol=0, rtol=0)
            texts = []
            for row in range(p.shape[1]):
                members = torch.where(p[:, row] > 0)[0]
                chosen = members[torch.argsort(p[members, row], descending=True, stable=True)[:V.CFG.text_sources]]
                per = V.CFG.text_chars // len(chosen)
                texts.append(" ".join(str(data["texts"][int(ids[stage][i])])[:per] for i in chosen)[:V.CFG.text_chars])
            assert texts == memory["texts"][n:]
            torch.testing.assert_close(data["encoder"].encode(texts).cpu(), memory["text_emb"][n:], atol=2e-6, rtol=2e-5)
            base = V.append_parent(old, E.load(out / f"c0_s{stage}_block.pt"), args.device)
            support = Q.M.lifted_support(old, T.raw_delta_block(data, ids[stage], stage), base, q)
            u = Q.J.full_assignment(p.double(), n)
            coarse = u.T @ torch.sparse.mm(Q.J.raw_adjacency(support, "cpu").double(), u)
            edges = Q.J.edges_from_projection(base, n, coarse)
            assert set(map(tuple, edges.T.tolist())) == set(map(tuple, memory["edge_index"].T.tolist()))


if __name__ == "__main__":
    E.main(protocol, prepare, validate)
