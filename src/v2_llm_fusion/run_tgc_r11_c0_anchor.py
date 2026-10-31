"""C0-anchored assignment interpolation built from sealed R10 outputs.

The intervention is deliberately post-optimization and pre-materialization:
P_alpha = (1-alpha) P_initial + alpha P_joint.  X, text, and binary topology
are all materialized from the same P_alpha, so this is not an incoherent
average of already materialized memories.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import tgc_r6_experiment as E
import run_tgc_r5_joint as V
import tgc_shared_assignment as J

R, T, D, B = E.R, E.T, E.D, E.B

SOURCE = Path("/home/pc/research/TAG_condensation/results/tgc_r10_joint_confirmation_20260923b")
DATASETS = ("cora", "citeseer")
ORDERS = ("order19601", "order19602", "order19603")
SEEDS = tuple(range(150, 156))
ARMS = ("c0", "p_initial", "joint", "anchor025", "anchor050", "anchor075")
ALPHAS = {"p_initial": 0.0, "anchor025": 0.25, "anchor050": 0.5, "anchor075": 0.75}


def protocol():
    return json.loads(json.dumps({
        "name": "tgc-r11-c0-anchor-v1",
        "datasets": DATASETS,
        "orders": ORDERS,
        "evaluation_stages": [0, 1, 2],
        "report_seeds": list(SEEDS),
        "arms": ARMS,
        "alphas": ALPHAS,
        "source_root": str(SOURCE),
        "source_protocol": "sealed R10 joint construction; no report-F1 selection",
        "intervention": "P_alpha=(1-alpha)P_initial+alpha P_joint before X/text/binary-topology materialization",
        "materialization": "same shared P_alpha for mean-X, top-5/320-char concat-reencode, projected binary topology",
        "primary": "anchor050 vs sealed joint and C0; anchor025/075 sensitivity",
        "development_only": True,
        "independent_confirmation": False,
        "selection_by_report": False,
        "test_labels_used": False,
        "student": {"batch": 16, "lr": .001, "steps": "max(50,min(2048,16*rows))"},
        "source_hashes": R.source_hashes(),
    }))


def append_parent(parent, block, device):
    return V.append_parent(parent, block, device)


def initial_assignment(q, labels, cfg, device):
    q, labels = q.to(device), labels.to(device)
    k = int(q.max()) + 1
    classes = J.row_classes(q, labels, k)
    allowed = labels[:, None] == classes[None, :]
    uniform = allowed.to(torch.float32) / allowed.sum(1, keepdim=True)
    return ((1 - cfg.initial_uniform) * F.one_hot(q, k).float()
            + cfg.initial_uniform * uniform).cpu()


def coarse_from_assignment(support, p, old_n, device):
    u = J.full_assignment(p.to(device), old_n)
    adjacency = J.raw_adjacency(support, device)
    return (u.T @ torch.sparse.mm(adjacency, u)).detach().cpu()


def source_files(source):
    names = []
    for stage in range(3):
        names += [f"c0_s{stage}_memory.pt", f"joint_s{stage}_memory.pt"]
        if stage:
            names += [f"c0_s{stage}_block.pt", f"shared_s{stage}_assignment.pt"]
    return {name: R.C.sha(source / name) for name in names}


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
                               "memory_sha256": T.memory_hash(c0), "alpha": ALPHAS.get(arm)}
        else:
            old = E.load(source / f"c0_s{stage-1}_memory.pt")
            block = E.load(source / f"c0_s{stage}_block.pt")
            base = append_parent(old, block, args.device)
            delta = T.raw_delta_block(data, ids[stage], stage)
            labels = data["y"][ids[stage]]
            q = D.current_assignment(data, ids[stage], c0, len(old["x"]), stage, args.device)
            p_joint = E.load(source / f"shared_s{stage}_assignment.pt").float()
            p_initial = initial_assignment(q, labels, V.CFG, args.device)
            support = J.M.lifted_support(old, delta, base, q)
            for arm in ("p_initial", "anchor025", "anchor050", "anchor075"):
                alpha = ALPHAS[arm]
                p = (1 - alpha) * p_initial + alpha * p_joint
                coarse = coarse_from_assignment(support, p, len(old["x"]), args.device)
                candidates, material = J.materialize(base, delta, p, coarse, data["encoder"], V.CFG)
                memory = candidates["joint"]
                memories[arm] = memory
                assignments[arm] = p
                audits[arm] = {
                    "parent_hash": T.memory_hash(old),
                    "base_hash": T.memory_hash(base),
                    "delta_hash": T.memory_hash(delta),
                    "alpha": alpha,
                    "p_initial_sha256": R.C.tensor_hash({"p": p_initial}),
                    "p_joint_sha256": R.C.tensor_hash({"p": p_joint}),
                    "assignment_sha256": R.C.tensor_hash({"p": p}),
                    "p_joint_max_abs": float((p_joint - p_initial).abs().max()),
                    "p_anchor_max_abs": float((p - p_initial).abs().max()),
                    "materialization": material,
                    "memory_sha256": T.memory_hash(memory),
                    "source_joint_memory_sha256": T.memory_hash(joint_source),
                    "historical_raw_reads": 0,
                    "report_labels_used": False,
                }
            memories["joint"] = joint_source
            assignments["joint"] = p_joint
            audits["joint"] = {
                "source": str(source / f"joint_s{stage}_memory.pt"),
                "assignment_source": str(source / f"shared_s{stage}_assignment.pt"),
                "assignment_sha256": R.C.tensor_hash({"p": p_joint}),
                "memory_sha256": T.memory_hash(joint_source),
                "alpha": 1.0,
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
