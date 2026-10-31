"""Unmodified recursive R5 on three reserved partitions with strong coresets."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import time

import torch
import tgc_r6_experiment as E
import run_tgc_r5_joint as V

R, T, D, B, J = E.R, E.T, E.D, E.B, V.J
ARMS = ("c0", "recursive", "k_center", "kmeans_medoid")
ORDERS = ("order19601", "order19602", "order19603")


def protocol():
    return json.loads(json.dumps({
        "name": "tgc-r6-unmodified-r5-confirmation-v1", "datasets": ["cora", "citeseer"],
        "orders": ORDERS, "evaluation_stages": [0, 1, 2], "report_seeds": list(range(140, 146)), "arms": ARMS,
        "class_groups": {d: {o: R.A.stage_groups(d, 3, o) for o in ORDERS} for d in ("cora", "citeseer")},
        "assignment": asdict(V.CFG), "teacher": asdict(V.TEACHER), "rate": .05,
        "construction_recipe": "R5 unchanged including teacher seeds8400/8401; report seeds independent",
        "recursive": "own prior memory; c0 at s0; no old raw replay",
        "baselines": "existing delta joint-space k-center and kmeans medoid; real coresets explicitly labeled",
        "fairness": "same rows/student/batch16/lr.001/steps; actual text bytes/tokens separately reported, not equal bytes",
        "primary": "final all-prompt pooled report-dev Macro-F1 vs c0; stage mean and old/new reported",
        "progression": {"mean_final_min_pp": .5, "stage_mean_min_pp": 0., "worst_final_min_pp": -2.,
                        "worst_old_min_pp": -2., "min_cells_ge_minus1": 4, "new_mean_min_pp": 0.},
        "baseline_criterion": "report paired final/stage mean vs both coresets; no stable-superiority claim unless independent evidence supports it",
        "history_audit": "214 local manifest/protocol/config JSON; actual partitions unseen within this scan",
        "test_labels_used": False, "report_before_seal": False,
        "access": "logical old synthetic/current raw only; full offline fixture physically loaded",
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
        c0, build = B.build_c0(data, ids[stage], budgets, stage, out, args.device)
        memories, audits = {"c0": c0}, {"c0": build}
        if stage == 0:
            memories["recursive"] = c0
            audits["recursive"] = {**build, "alias": "c0"}
        else:
            old = E.load(out / f"recursive_s{stage-1}_memory.pt")
            block = E.load(out / f"c0_s{stage}_block.pt")
            base = V.append_parent(old, block, args.device)
            delta = T.raw_delta_block(data, ids[stage], stage)
            q = D.current_assignment(data, ids[stage], c0, len(old["x"]), stage, args.device)
            candidates, p, audit = J.construct(old, delta, base, q, data["y"][ids[stage]], data["encoder"],
                                               V.TEACHER, args.device, V.CFG)
            torch.save(p, out / f"recursive_s{stage}_assignment.pt")
            memories["recursive"] = candidates["joint"]
            audits["recursive"] = {**audit, "parent_hash": T.memory_hash(old), "c0_block_seconds": build["seconds"]}
        for arm in ("k_center", "kmeans_medoid"):
            old = E.load(out / f"{arm}_s{stage-1}_memory.pt") if stage else None
            keep = budgets[stage-1] if stage else 0
            R.sync(args.device)
            started = time.perf_counter()
            block, details = R.C.real_block(data, ids[stage], budgets[stage] - keep, arm, stage, args.device)
            memory = R.SCA.append_memory(old, block, keep, topology_mode="frozen_old", cross_degree=1,
                                         refresh_mode="frozen_old", device=args.device)
            R.sync(args.device)
            memories[arm] = memory
            audits[arm] = {**details, "seconds": time.perf_counter() - started,
                           "current_selected_ids": block["source_ids"].tolist(),
                           "parent_hash": T.memory_hash(old) if old else None, "real_coreset": True}
        decision = {"stage": stage, "report_labels_used": False, "arms": {}}
        for arm, memory in memories.items():
            audits[arm]["payload"] = R.payload(memory)
            audits[arm]["text"] = R.SCA.text_audit(memory["texts"], data["encoder"].tokenizer)
            E.save(out / f"{arm}_s{stage}_build_audit.json", audits[arm])
            decision["arms"][arm] = {"build": f"{arm}_s{stage}_build_audit.json"}
        decision["memories"] = {a: D.save_memory(out, a, stage, memories[a]) for a in ARMS}
        decision["locked_at"] = datetime.now(timezone.utc).isoformat()
        E.save(out / f"s{stage}_decision.json", decision)
        print(json.dumps({"event": "prepared", "dataset": data["dataset"], "order": order, "stage": stage}), flush=True)
    E.save(out / "prepared.json", {"status": "prepared"})


def validate(data, out, stage, ids, budgets, args):
    for arm in ARMS:
        memory = E.load(out / f"{arm}_s{stage}_memory.pt")
        E.memory_checks(memory, budgets[stage])
        if stage:
            parent = E.load(out / f"{arm}_s{stage-1}_memory.pt")
            R.C.assert_frozen(parent, memory)
        audit = E.read(out / f"{arm}_s{stage}_build_audit.json")
        n = budgets[stage-1] if stage else 0
        if arm in ("k_center", "kmeans_medoid"):
            chosen = torch.tensor(audit["current_selected_ids"])
            assert len(chosen.unique()) == len(chosen) and set(chosen.tolist()) <= set(ids[stage].tolist())
            torch.testing.assert_close(memory["x"][n:], data["x"][chosen].float(), rtol=0, atol=0)
            assert memory["texts"][n:] == [data["texts"][int(i)] for i in chosen]
        if arm == "recursive" and stage:
            p = E.load(out / f"recursive_s{stage}_assignment.pt")
            assert R.C.tensor_hash({"p": p}) == audit["assignment"]["assignment_sha256"]
            torch.testing.assert_close(p.sum(1), torch.ones(len(p)), atol=2e-6, rtol=1e-5)
            expected = (p.double().T @ data["x"][ids[stage]].double()) / p.double().sum(0)[:, None]
            torch.testing.assert_close(memory["x"][n:].double(), expected, atol=2e-5, rtol=2e-5)
            encoded = data["encoder"].encode(memory["texts"][n:]).cpu()
            torch.testing.assert_close(memory["text_emb"][n:], encoded, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    E.main(protocol, prepare, validate)
