"""R12 class-preserving coverage floor built from sealed R10/R11 ingredients.

The experiment targets new-class plasticity without changing the student,
text budget, or topology operator.  A small class-preserving coverage floor is
mixed into the R11 anchor assignment before the shared X/text/topology
materialization.  No report F1 is used for construction or selection.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import torch

import tgc_r6_experiment as E
import run_tgc_r11_c0_anchor as R11
import run_tgc_r5_joint as V
import tgc_shared_assignment as J

R, T, D, B = E.R, E.T, E.D, E.B
SOURCE = R11.SOURCE
DATASETS = R11.DATASETS
ORDERS = R11.ORDERS
SEEDS = R11.SEEDS
ARMS = ("c0", "joint", "anchor025", "classfloor010", "classfloor025",
        "classfloor050", "randomfloor025")
FLOORS = {"classfloor010": 0.10, "classfloor025": 0.25,
          "classfloor050": 0.50, "randomfloor025": 0.25}
ANCHOR_ALPHA = 0.25
RANDOM_SEED_BASE = 12001


def protocol():
    return json.loads(json.dumps({
        "name": "tgc-r12-class-preserving-floor-v1",
        "datasets": DATASETS,
        "orders": ORDERS,
        "evaluation_stages": [0, 1, 2],
        "report_seeds": list(SEEDS),
        "arms": ARMS,
        "anchor_alpha": ANCHOR_ALPHA,
        "coverage_floors": FLOORS,
        "source_root": str(SOURCE),
        "source_protocol": "sealed R10 joint construction; R11 anchor ingredients; no report-F1 selection",
        "intervention": "mix a class-preserving coverage floor into R11 anchor025 assignment before materialization",
        "class_floor": "U_class[i,j]=1/number_of_rows_in_class(i) for allowed same-class synthetic rows",
        "random_control": "random row-wise same-class distributions with fixed deterministic seed, mixed at gamma=.25",
        "materialization": "same assignment for mean-X, top-5/320-char concat-reencode, projected binary topology",
        "primary": "classfloor025 versus anchor025 and C0; classfloor010/050 sensitivity; randomfloor025 control",
        "development_only": True,
        "independent_confirmation": False,
        "selection_by_report": False,
        "test_labels_used": False,
        "student": {"batch": 16, "lr": .001, "steps": "max(50,min(2048,16*rows))"},
        "source_hashes": R.source_hashes(),
    }))


def append_parent(parent, block, device):
    return V.append_parent(parent, block, device)


def class_uniform(q, labels, device):
    q, labels = q.to(device), labels.to(device)
    k = int(q.max()) + 1
    classes = J.row_classes(q, labels, k)
    allowed = labels[:, None] == classes[None, :]
    # Keep assignments on CPU like the sealed R10/R11 artifacts.  Materialization
    # explicitly moves them to the requested device, avoiding mixed-device blends.
    return (allowed.to(torch.float32) / allowed.sum(1, keepdim=True)).cpu(), allowed.cpu()


def random_class_distribution(allowed, seed):
    generator = torch.Generator(device=allowed.device)
    generator.manual_seed(int(seed))
    noise = torch.rand(allowed.shape, generator=generator, device=allowed.device)
    noise = noise * allowed.to(noise.dtype)
    return noise / noise.sum(1, keepdim=True).clamp_min(1e-12)


def candidate_assignment(arm, p_initial, p_joint, u_class, allowed,
                        dataset, order, stage, device):
    anchor = (1.0 - ANCHOR_ALPHA) * p_initial + ANCHOR_ALPHA * p_joint
    if arm == "anchor025":
        return anchor, {"base": "anchor025", "floor": 0.0, "random": False}
    if arm in ("classfloor010", "classfloor025", "classfloor050"):
        gamma = FLOORS[arm]
        return ((1.0 - gamma) * anchor + gamma * u_class,
                {"base": "anchor025", "floor": gamma, "random": False})
    if arm == "randomfloor025":
        gamma = FLOORS[arm]
        seed = RANDOM_SEED_BASE + sum(ord(c) for c in f"{dataset}/{order}/{stage}")
        random_u = random_class_distribution(allowed, seed)
        return ((1.0 - gamma) * anchor + gamma * random_u,
                {"base": "anchor025", "floor": gamma, "random": True,
                 "random_seed": seed})
    raise ValueError(f"unknown class-floor arm: {arm}")


def coarse_from_assignment(support, p, old_n, device):
    u = J.full_assignment(p.to(device), old_n)
    adjacency = J.raw_adjacency(support, device)
    return (u.T @ torch.sparse.mm(adjacency, u)).detach().cpu()


def source_files(source):
    return R11.source_files(source)


def prepare(data, order, args, spec):
    ids, budgets = D.set_stream(data, order)
    out = args.output / data["dataset"] / order
    source = SOURCE / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    manifest = E.manifest(data, order, budgets, ids)
    manifest["source_artifact_hashes"] = source_files(source)
    R.stable_lock(out / "manifest.json", manifest)
    if (out / "prepared.json").exists():
        return
    assert not (args.output / "selection_seal.json").exists()

    for stage in range(3):
        c0 = E.load(source / f"c0_s{stage}_memory.pt")
        joint_source = E.load(source / f"joint_s{stage}_memory.pt")
        memories = {"c0": c0}
        audits = {"c0": {"source": str(source / f"c0_s{stage}_memory.pt"),
                          "memory_sha256": T.memory_hash(c0), "alpha": None}}
        assignments = {}
        if stage == 0:
            for arm in ARMS[1:]:
                memories[arm] = T.clone_memory(c0)
                audits[arm] = {"source": str(source / f"c0_s{stage}_memory.pt"),
                               "memory_sha256": T.memory_hash(c0), "alpha": None,
                               "floor": 0.0, "random": False}
        else:
            old = E.load(source / f"c0_s{stage-1}_memory.pt")
            block = E.load(source / f"c0_s{stage}_block.pt")
            base = append_parent(old, block, args.device)
            delta = T.raw_delta_block(data, ids[stage], stage)
            labels = data["y"][ids[stage]]
            q = D.current_assignment(data, ids[stage], c0, len(old["x"]), stage, args.device)
            p_joint = E.load(source / f"shared_s{stage}_assignment.pt").float()
            p_initial = R11.initial_assignment(q, labels, V.CFG, args.device)
            u_class, allowed = class_uniform(q, labels, args.device)
            support = J.M.lifted_support(old, delta, base, q)

            for arm in ("anchor025", "classfloor010", "classfloor025",
                        "classfloor050", "randomfloor025"):
                p, floor_meta = candidate_assignment(
                    arm, p_initial, p_joint, u_class, allowed,
                    data["dataset"], order, stage, args.device)
                coarse = coarse_from_assignment(support, p, len(old["x"]), args.device)
                candidates, material = J.materialize(
                    base, delta, p, coarse, data["encoder"], V.CFG)
                memories[arm] = candidates["joint"]
                assignments[arm] = p
                audits[arm] = {
                    "parent_hash": T.memory_hash(old),
                    "base_hash": T.memory_hash(base),
                    "delta_hash": T.memory_hash(delta),
                    "alpha": ANCHOR_ALPHA,
                    "floor": floor_meta["floor"],
                    "random": floor_meta["random"],
                    "random_seed": floor_meta.get("random_seed"),
                    "p_initial_sha256": R.C.tensor_hash({"p": p_initial}),
                    "p_joint_sha256": R.C.tensor_hash({"p": p_joint}),
                    "p_anchor_sha256": R.C.tensor_hash({"p": (0.75 * p_initial + 0.25 * p_joint)}),
                    "u_class_sha256": R.C.tensor_hash({"u": u_class}),
                    "assignment_sha256": R.C.tensor_hash({"p": p}),
                    "p_joint_max_abs": float((p_joint - p_initial).abs().max()),
                    "p_anchor_max_abs": float(((0.75 * p_initial + 0.25 * p_joint) - p_initial).abs().max()),
                    "p_floor_max_abs": float((p - (0.75 * p_initial + 0.25 * p_joint)).abs().max()),
                    "class_allowed_nonzero": int(allowed.sum().item()),
                    "materialization": material,
                    "memory_sha256": T.memory_hash(memories[arm]),
                    "source_joint_memory_sha256": T.memory_hash(joint_source),
                    "historical_raw_reads": 0,
                    "report_labels_used": False,
                }
            assignments["joint"] = p_joint
            memories["joint"] = joint_source
            audits["joint"] = {
                "source": str(source / f"joint_s{stage}_memory.pt"),
                "assignment_source": str(source / f"shared_s{stage}_assignment.pt"),
                "assignment_sha256": R.C.tensor_hash({"p": p_joint}),
                "memory_sha256": T.memory_hash(joint_source),
                "alpha": 1.0,
                "floor": 0.0,
                "random": False,
                "historical_raw_reads": 0,
                "report_labels_used": False,
            }

        for arm, memory in memories.items():
            audits[arm]["payload"] = R.payload(memory)
            audits[arm]["text"] = R.SCA.text_audit(memory["texts"], data["encoder"].tokenizer)
            E.save(out / f"{arm}_s{stage}_build_audit.json", audits[arm])
            if arm in assignments:
                torch.save(assignments[arm], out / f"{arm}_s{stage}_assignment.pt")

        decision = {
            "stage": stage,
            "report_labels_used": False,
            "source_root": str(source),
            "memories": {a: D.save_memory(out, a, stage, memories[a]) for a in ARMS},
            "locked_at": datetime.now(timezone.utc).isoformat(),
        }
        E.save(out / f"s{stage}_decision.json", decision)
        print(json.dumps({"event": "prepared", "dataset": data["dataset"],
                          "order": order, "stage": stage}), flush=True)
    E.save(out / "prepared.json", {"status": "prepared"})


def validate(data, out, stage, ids, budgets, args):
    manifest = E.read(out / "manifest.json")
    source = SOURCE / data["dataset"] / out.name
    assert manifest["source_artifact_hashes"] == source_files(source)
    for arm in ARMS:
        memory = E.load(out / f"{arm}_s{stage}_memory.pt")
        E.memory_checks(memory, budgets[stage])
        if stage == 0:
            assert T.memory_hash(memory) == T.memory_hash(E.load(source / "c0_s0_memory.pt"))
            continue
        old = E.load(source / f"c0_s{stage-1}_memory.pt")
        for key in ("x", "text_emb", "mass", "depth", "stage"):
            assert torch.equal(memory[key][:len(old["x"])], old[key])
        assert memory["texts"][:len(old["x"])] == old["texts"]
        historical = memory["edge_index"][:, (memory["edge_index"] < len(old["x"])).all(0)]
        assert torch.equal(historical, old["edge_index"])
        audit = E.read(out / f"{arm}_s{stage}_build_audit.json")
        if arm == "c0":
            assert T.memory_hash(memory) == T.memory_hash(E.load(source / f"c0_s{stage}_memory.pt"))
            continue
        if arm == "joint":
            assert T.memory_hash(memory) == T.memory_hash(E.load(source / f"joint_s{stage}_memory.pt"))
            continue
        p = E.load(out / f"{arm}_s{stage}_assignment.pt").float()
        torch.testing.assert_close(p.sum(1), torch.ones(len(p)), atol=2e-6, rtol=1e-5)
        assert R.C.tensor_hash({"p": p}) == audit["assignment_sha256"]
        delta = T.raw_delta_block(data, ids[stage], stage)
        base = append_parent(old, E.load(source / f"c0_s{stage}_block.pt"), args.device)
        q = D.current_assignment(data, ids[stage], E.load(source / f"c0_s{stage}_memory.pt"),
                                 len(old["x"]), stage, args.device)
        p_joint = E.load(source / f"shared_s{stage}_assignment.pt").float()
        labels = data["y"][ids[stage]]
        p_initial = R11.initial_assignment(q, labels, V.CFG, args.device)
        u_class, allowed = class_uniform(q, labels, args.device)
        expected, _ = candidate_assignment(arm, p_initial, p_joint, u_class, allowed,
                                            data["dataset"], out.name, stage, args.device)
        torch.testing.assert_close(p, expected, atol=2e-6, rtol=1e-5)
        support = J.M.lifted_support(old, delta, base, q)
        coarse = coarse_from_assignment(support, p, len(old["x"]), args.device)
        n = len(old["x"])
        expected_x = (p.double().T @ data["x"][ids[stage]].double()) / p.double().sum(0)[:, None]
        torch.testing.assert_close(memory["x"][n:].double(), expected_x, atol=2e-5, rtol=2e-5)
        members_text = []
        for row in range(p.shape[1]):
            members = torch.where(p[:, row] > 0)[0]
            chosen = members[torch.argsort(p[members, row], descending=True, stable=True)[:V.CFG.text_sources]]
            per = max(1, V.CFG.text_chars // len(chosen))
            members_text.append(" ".join(str(data["texts"][int(ids[stage][i])])[:per] for i in chosen)[:V.CFG.text_chars])
        assert memory["texts"][n:] == members_text
        torch.testing.assert_close(data["encoder"].encode(members_text).cpu(), memory["text_emb"][n:], atol=2e-6, rtol=2e-5)
        expected_edges = J.edges_from_projection(base, n, coarse)
        assert set(map(tuple, expected_edges.T.tolist())) == set(map(tuple, memory["edge_index"].T.tolist()))


if __name__ == "__main__":
    E.main(protocol, prepare, validate)
