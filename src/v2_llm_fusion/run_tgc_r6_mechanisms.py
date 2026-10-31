"""R5 parent factorial, teacher/assignment controls and actual-output diagnostics."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time

import torch
import tgc_r6_experiment as E
import tgc_r6_mechanisms as K
import run_tgc_r5_joint as V

R, T, D, J = E.R, E.T, E.D, V.J
SOURCE = Path("/home/pc/research/TAG_condensation/results/tgc_r5_joint_20260923")
BITS = tuple("".join(v) for v in itertools.product("01", repeat=3))
ARMS = ("c0",) + tuple("parent_" + b for b in BITS) + (
    "reference0_output1", "reference1_output0", "geometry", "random_teacher", "fixed_assignment",
    "decoupled64", "decoupled128", "symmetric_fixed_parent", "symmetric_recursive",
    "pooled_text", "weighted_edges", "pooled_weighted", "frozen_joint", "frozen_recursive")
S1_ARMS = ("c0", "parent_111", "symmetric_recursive")
CFG = V.CFG


def protocol():
    return json.loads(json.dumps({
        "name": "tgc-r6-mechanisms-v2-numerical-amendment", "datasets": ["cora", "citeseer"],
        "orders": ["order19341", "order19342"], "evaluation_stages": [1, 2],
        "stage_arms": {"1": S1_ARMS, "2": ARMS}, "arms": ARMS, "report_seeds": list(range(100, 106)),
        "class_groups": {d: {o: R.A.stage_groups(d, 3, o) for o in ("order19341", "order19342")} for d in ("cora", "citeseer")},
        "assignment": asdict(CFG), "teacher": asdict(V.TEACHER), "source_root": str(SOURCE), "rate": .05,
        "parent_factorial": "s2 X/mass, text/text_emb, old edges independently from frozen C0(0)/recursive(1); 8 arms",
        "reference_cross": "teacher H and legal support adjacency use indicated parent; output stores other indicated parent; diagnostic only",
        "geometry": "unit-normalized raw X and MiniLM text concatenation divided by sqrt2; same objective/optimizer",
        "random_teacher": "same two architectures and initial seeds; zero teacher updates",
        "fixed_assignment": "exact R5 step-zero smoothed assignment, same materialization; no optimization",
        "decoupled": "content minimizes reconstruction+regularizers; graph minimizes message+regularizers; same initialization; 64+64 and128+128 updates",
        "symmetric": "replace random-walk propagation loss with symmetric GCN normalization; all other constants unchanged",
        "materialization": "2x2 actual snippet-reencode vs normalized source embedding pooling, binary vs projected weights on same selected nonself edges",
        "weighted": "same GCN symmetric operator; explicit selfloop weights from coarse projection; edge count comparison excludes selfloops",
        "pooled": "embedding-only diagnostic, not materialized text method",
        "numerical_amendment": "v1 stopped before scoring: same-seed GPU teacher/assignment reruns not bitwise reproducible; freshly regenerated endpoints are scored, with frozen R5 joint/recursive controls; no candidates selected by closeness or score",
        "primary_metric": "all-prompt pooled report-dev Macro-F1; s2 mechanism matrix, s1 for recursive control and symmetric rollout",
        "development_only": True, "selection_by_report": False, "test_labels_used": False,
        "student": {"batch": 16, "lr": .001, "steps": "max(50,min(2048,16*rows))"},
        "source_hashes": R.source_hashes()}))


def prepare(data, order, args, spec):
    ids, budgets = D.set_stream(data, order)
    out = args.output / data["dataset"] / order
    source = SOURCE / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    input_names = [f"{arm}_s{s}_memory.pt" for arm in ("c0", "recursive") for s in range(3)]
    input_names += [f"c0_s{s}_block.pt" for s in (1, 2)]
    input_names += ["joint_s2_memory.pt"]
    manifest = E.manifest(data, order, budgets, ids)
    manifest["source_artifact_hashes"] = {name: R.C.sha(source / name) for name in input_names}
    R.stable_lock(out / "manifest.json", manifest)
    if all((out / f"s{s}_decision.json").exists() for s in (1, 2)):
        for s in (1, 2):
            D.ARMS = tuple(E.arms_at(spec, s))
            D.verify_decision(out, s)
        return
    assert not (args.output / "selection_seal.json").exists()
    c0 = E.load(source / "c0_s2_memory.pt")
    block = E.load(source / "c0_s2_block.pt")
    old0 = E.load(source / "c0_s1_memory.pt")
    old1 = E.load(source / "recursive_s1_memory.pt")
    delta = T.raw_delta_block(data, ids[2], 2)
    labels, old_n = data["y"][ids[2]], len(old0["x"])
    q = D.current_assignment(data, ids[2], c0, old_n, 2, args.device)
    memories, audits, assignments, contexts = {"c0": c0}, {}, {}, {}

    def finish(arm, base, ctx, p, coarse, assign_audit, started, parent_bits, **extra):
        memory, mat_audit = K.materialize(base, delta, p, coarse, data["encoder"], CFG, **extra)
        R.sync(args.device)
        audits[arm] = {"parent_bits": parent_bits, "reference": ctx["audit"], "assignment": assign_audit,
                       "materialization": mat_audit, "seconds": time.perf_counter() - started,
                       "gap": K.gap(ctx, memory, p, args.device), "current_delta_ids": ids[2].tolist(),
                       "parent_hash": T.memory_hash(parents[parent_bits]) if parent_bits in parents else None}
        memories[arm], assignments[arm] = memory, p
        return memory

    parents = {b: K.blend_parent(old0, old1, b) for b in BITS}
    bases = {b: V.append_parent(parent, block, args.device) for b, parent in parents.items()}
    fits = {}
    for bits in BITS:
        started = time.perf_counter()
        ctx = K.context(parents[bits], delta, bases[bits], q, V.TEACHER, args.device)
        contexts[bits] = ctx
        fit = K.learn(q, labels, ctx, old_n, CFG)
        fits[bits] = fit
        finish("parent_" + bits, bases[bits], ctx, *fit, started, bits)
    # GPU reconstruction is a new realization, not the historical frozen artifact.
    for bits, endpoint in (("000", "joint"), ("111", "recursive")):
        fresh = memories["parent_" + bits]
        frozen = E.load(source / f"{endpoint}_s2_memory.pt")
        memories["frozen_" + endpoint] = frozen
        audits["frozen_" + endpoint] = {"source": str(source / f"{endpoint}_s2_memory.pt"), "seconds": 0.}
        drift = {"same_memory_hash": T.memory_hash(fresh) == T.memory_hash(frozen),
                 "max_abs": {key: float((fresh[key]-frozen[key]).abs().max()) for key in ("x", "text_emb", "mass")},
                 "changed_text_rows": sum(a != b for a, b in zip(fresh["texts"], frozen["texts"])),
                 "changed_directed_edge_entries": len(set(map(tuple, fresh["edge_index"].T.tolist())) ^ set(map(tuple, frozen["edge_index"].T.tolist())))}
        audits["parent_" + bits]["fresh_vs_frozen"] = drift
    for reference, output in (("000", "111"), ("111", "000")):
        started = time.perf_counter()
        arm = "reference" + reference[0] + "_output" + output[0]
        finish(arm, bases[output], contexts[reference], *fits[reference], started, output)
    for kind, arm in (("geometry", "geometry"), ("random", "random_teacher")):
        started = time.perf_counter()
        ctx = K.context(old1, delta, bases["111"], q, V.TEACHER, args.device, kind)
        fit = K.learn(q, labels, ctx, old_n, CFG)
        finish(arm, bases["111"], ctx, *fit, started, "111")
    ctx = contexts["111"]
    started = time.perf_counter()
    finish("fixed_assignment", bases["111"], ctx,
           *K.learn(q, labels, ctx, old_n, replace(CFG, steps=0)), started, "111")
    for steps in (64, 128):
        started = time.perf_counter()
        content_fit = K.learn(q, labels, ctx, old_n, replace(CFG, steps=steps, message=0.))
        topo_fit = K.learn(q, labels, ctx, old_n, replace(CFG, steps=steps), reconstruction_weight=0.)
        arm = f"decoupled{steps}"
        finish(arm, bases["111"], ctx, *content_fit, started, "111", p_topology=topo_fit[0], coarse_topology=topo_fit[1])
        audits[arm]["topology_optimization"] = topo_fit[2]
        torch.save(topo_fit[0], out / f"{arm}_s2_topology_assignment.pt")
    started = time.perf_counter()
    finish("symmetric_fixed_parent", bases["111"], ctx,
           *K.learn(q, labels, ctx, old_n, CFG, "symmetric"), started, "111")
    p, coarse, _ = fits["111"]
    for arm in ("pooled_text", "weighted_edges", "pooled_weighted"):
        started = time.perf_counter()
        memory = T.clone_memory(memories["parent_111"])
        if "pooled" in arm:
            memory = K.pooled_text_variant(memory, delta, p)
        if "weighted" in arm:
            memory = K.weighted_variant(memory, coarse)
        memories[arm], assignments[arm] = memory, p
        audits[arm] = {"parent_bits": "111", "parent_hash": T.memory_hash(old1),
                       "reference": ctx["audit"], "variant_seconds": time.perf_counter()-started,
                       "gap": K.gap(ctx, memory, p, args.device), "current_delta_ids": ids[2].tolist(),
                       "diagnostic_representation": arm}

    # Symmetric objective follows its own previous output, not a C0 shared parent.
    sym_old = E.load(source / "c0_s0_memory.pt")
    sym_first = None
    for stage in (1, 2):
        c0_stage = E.load(source / f"c0_s{stage}_memory.pt")
        sym_block = E.load(source / f"c0_s{stage}_block.pt")
        sym_base = V.append_parent(sym_old, sym_block, args.device)
        sym_delta = T.raw_delta_block(data, ids[stage], stage)
        sym_q = D.current_assignment(data, ids[stage], c0_stage, len(sym_old["x"]), stage, args.device)
        started = time.perf_counter()
        sym_ctx = K.context(sym_old, sym_delta, sym_base, sym_q, V.TEACHER, args.device)
        sym_p, sym_coarse, sym_audit = K.learn(sym_q, data["y"][ids[stage]], sym_ctx, len(sym_old["x"]), CFG, "symmetric")
        sym_memory, mat = K.materialize(sym_base, sym_delta, sym_p, sym_coarse, data["encoder"], CFG)
        sym_details = {"parent_hash": T.memory_hash(sym_old), "assignment": sym_audit, "materialization": mat,
                       "reference": sym_ctx["audit"], "seconds": time.perf_counter()-started,
                       "gap": K.gap(sym_ctx, sym_memory, sym_p, args.device), "current_delta_ids": ids[stage].tolist()}
        E.save(out / f"symmetric_recursive_s{stage}_build_audit.json", sym_details)
        torch.save(sym_p, out / f"symmetric_recursive_s{stage}_assignment.pt")
        if stage == 1:
            sym_first = sym_memory
        else:
            memories["symmetric_recursive"] = sym_memory
            audits["symmetric_recursive"] = sym_details
            assignments["symmetric_recursive"] = sym_p
        sym_old = sym_memory
    first = {"c0": E.load(source / "c0_s1_memory.pt"), "parent_111": old1, "symmetric_recursive": sym_first}
    E.save(out / "s1_decision.json", {"stage": 1, "memories": {a: D.save_memory(out, a, 1, first[a]) for a in S1_ARMS},
                                      "report_labels_used": False, "locked_at": datetime.now(timezone.utc).isoformat()})
    for arm in ARMS:
        audit = audits.get(arm, {"source": str(source / "c0_s2_memory.pt"), "seconds": 0.})
        audit["payload"] = R.payload(memories[arm])
        audit["text"] = R.SCA.text_audit(memories[arm]["texts"], data["encoder"].tokenizer)
        E.save(out / f"{arm}_s2_build_audit.json", audit)
        if arm in assignments:
            torch.save(assignments[arm], out / f"{arm}_s2_assignment.pt")
    E.save(out / "s2_decision.json", {"stage": 2, "memories": {a: D.save_memory(out, a, 2, memories[a]) for a in ARMS},
                                      "report_labels_used": False, "locked_at": datetime.now(timezone.utc).isoformat()})
    E.save(out / "prepared.json", {"status": "prepared"})
    print(json.dumps({"event": "prepared", "dataset": data["dataset"], "order": order}), flush=True)


def validate(data, out, stage, ids, budgets, args):
    source = SOURCE / data["dataset"] / out.name
    manifest = E.read(out / "manifest.json")
    for name, sha in manifest["source_artifact_hashes"].items():
        assert R.C.sha(source / name) == sha
    old0 = E.load(source / f"c0_s{stage-1}_memory.pt")
    old1 = E.load(source / f"recursive_s{stage-1}_memory.pt")
    old_n = len(old0["x"])
    for arm in (S1_ARMS if stage == 1 else ARMS):
        memory = E.load(out / f"{arm}_s{stage}_memory.pt")
        E.memory_checks(memory, budgets[stage])
        if arm.startswith("frozen_"):
            original = E.load(source / f"{arm.removeprefix('frozen_')}_s{stage}_memory.pt")
            assert T.memory_hash(memory) == T.memory_hash(original)
            continue
        if arm == "c0" or (stage == 1 and arm == "parent_111"):
            continue
        audit = E.read(out / f"{arm}_s{stage}_build_audit.json")
        if arm == "symmetric_recursive":
            parent = old0 if stage == 1 else E.load(out / "symmetric_recursive_s1_memory.pt")
        else:
            parent = K.blend_parent(old0, old1, audit["parent_bits"])
        assert T.memory_hash(parent) == audit["parent_hash"]
        for key in ("x", "text_emb", "mass", "stage"):
            assert torch.equal(memory[key][:old_n], parent[key])
        assert memory["texts"][:old_n] == parent["texts"]
        e = memory["edge_index"]
        historical = e[:, (e < old_n).all(0) & (e[0] != e[1])]
        assert torch.equal(historical, parent["edge_index"])
        c0 = E.load(source / f"c0_s{stage}_memory.pt")
        assert int((e[0] != e[1]).sum()) == c0["edge_index"].shape[1]
        p = E.load(out / f"{arm}_s{stage}_assignment.pt")
        assert p.shape == (len(ids[stage]), budgets[stage] - old_n)
        assert torch.isfinite(p).all() and (p >= 0).all()
        assignment_audit = audit.get("assignment")
        if assignment_audit is None:
            assignment_audit = E.read(out / "parent_111_s2_build_audit.json")["assignment"]
        assert R.C.tensor_hash({"p": p}) == assignment_audit["assignment_sha256"]
        classes = torch.tensor(assignment_audit["row_classes"])
        assert torch.count_nonzero(p[data["y"][ids[stage], None] != classes[None]]) == 0
        torch.testing.assert_close(p.sum(1), torch.ones(len(p)), atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(memory["mass"][old_n:], p.sum(0), atol=1e-5, rtol=1e-5)
        expect_x = (p.double().T @ data["x"][ids[stage]].double()) / p.double().sum(0)[:, None]
        torch.testing.assert_close(expect_x, memory["x"][old_n:].double(), atol=2e-5, rtol=2e-5)
        if "pooled" not in arm:
            expected_t = data["encoder"].encode(memory["texts"][old_n:]).cpu()
        else:
            expected_t = torch.nn.functional.normalize((p.T @ data["text_emb"][ids[stage]]) / p.sum(0)[:, None], dim=1)
        torch.testing.assert_close(expected_t, memory["text_emb"][old_n:], atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    E.main(protocol, prepare, validate)
