"""R14: disagreement-weighted plasticity correction on top of R11.

R11 is kept as the control.  The only new operation is a row-wise anchor:
rows whose joint assignment disagrees more with the C0 initialization retain
more of the teacher joint assignment, while stable rows keep the R11 anchor.
The same assignment materializes X, text, and topology, so the experiment
does not introduce a modality-specific patch.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import torch

import tgc_r6_experiment as E
import run_tgc_r11_c0_anchor as R11
import run_tgc_r5_joint as V
import tgc_shared_assignment as J

R, T, D, B = E.R, E.T, E.D, E.B

SOURCE = Path("/home/pc/research/TAG_condensation/results/tgc_r11_c0_anchor_20260923b")
BUILD_SOURCE = Path("/home/pc/research/TAG_condensation/results/tgc_r10_joint_confirmation_20260923b")
DATASETS = ("cora", "citeseer")
ORDERS = ("order19601", "order19602", "order19603")
SEEDS = tuple(range(150, 156))
BASE_ALPHA = 0.25
BETAS = {"plastic025": 0.25, "plastic050": 0.50}
MAX_ALPHA = 0.75
ARMS = ("c0", "anchor025", "joint", "plastic025", "plastic050")


def protocol():
    return json.loads(json.dumps({
        "name": "tgc-r14-r11-disagreement-plasticity-v1",
        "datasets": DATASETS,
        "orders": ORDERS,
        "evaluation_stages": [0, 1, 2],
        "report_seeds": list(SEEDS),
        "arms": ARMS,
        "base_alpha": BASE_ALPHA,
        "betas": BETAS,
        "max_alpha": MAX_ALPHA,
        "intervention": (
            "alpha_i=clip(0.25+beta*0.5*L1(P_joint_i-P_initial_i),0,0.75); "
            "P_i=(1-alpha_i)P_initial_i+alpha_i P_joint_i"
        ),
        "materialization": (
            "same row-wise assignment for mean-X, top-5/320-char concat-reencode, "
            "and projected binary topology"
        ),
        "source_root": str(SOURCE),
        "build_source_root": str(BUILD_SOURCE),
        "source_protocol": "sealed R11 C0/anchor025/joint/p_initial artifacts",
        "history_access": "old synthetic C0 memory plus current raw delta only",
        "historical_raw_reads": 0,
        "report_labels_used": False,
        "selection_by_report": False,
        "development_only": True,
        "independent_confirmation": False,
        "student": {"batch": 16, "lr": .001,
                    "steps": "max(50,min(2048,16*rows))"},
        "promotion_rule": {
            "mean_final_min_pp": .5,
            "stage_mean_min_pp": 0.0,
            "worst_final_min_pp": -2.0,
            "mean_new_min_pp": 0.0,
        },
        "source_hashes": R.source_hashes(),
    }))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save(path, value):
    E.save(Path(path), value)


def source_files(source):
    names = ["protocol.json", "selection_seal.json"]
    for stage in range(3):
        names.extend([
            f"c0_s{stage}_memory.pt",
            f"anchor025_s{stage}_memory.pt",
            f"joint_s{stage}_memory.pt",
        ])
        if stage:
            names.extend([
                f"p_initial_s{stage}_assignment.pt",
                f"anchor025_s{stage}_assignment.pt",
                f"joint_s{stage}_assignment.pt",
            ])
    hashes = {name: R.C.sha(source / name) for name in names
              if (source / name).exists()}
    build_source = BUILD_SOURCE / source.parent.name / source.name
    for stage in (1, 2):
        name = f"build/c0_s{stage}_block.pt"
        path = build_source / f"c0_s{stage}_block.pt"
        if path.exists():
            hashes[name] = R.C.sha(path)
    return hashes


def append_parent(parent, block, device):
    return V.append_parent(parent, block, device)


def coarse_from_assignment(support, p, old_n, device):
    u = J.full_assignment(p.to(device), old_n)
    adjacency = J.raw_adjacency(support, device)
    return (u.T @ torch.sparse.mm(adjacency, u)).detach().cpu()


def plastic_assignment(p_initial, p_joint, beta):
    if p_initial.shape != p_joint.shape:
        raise ValueError("R11 assignment shape mismatch")
    if not torch.isfinite(p_initial).all() or not torch.isfinite(p_joint).all():
        raise FloatingPointError("nonfinite R11 assignment")
    disagreement = 0.5 * (p_joint - p_initial).abs().sum(1)
    alpha = (BASE_ALPHA + beta * disagreement).clamp(0.0, MAX_ALPHA)
    p = (1.0 - alpha[:, None]) * p_initial + alpha[:, None] * p_joint
    torch.testing.assert_close(p.sum(1), torch.ones(len(p)), atol=2e-6, rtol=1e-5)
    return p, alpha, disagreement


def build_audit(memory, alpha, disagreement, beta, material):
    return {
        "alpha_base": BASE_ALPHA,
        "beta": beta,
        "alpha_max": MAX_ALPHA,
        "alpha_mean": float(alpha.mean()),
        "alpha_min": float(alpha.min()),
        "alpha_max_observed": float(alpha.max()),
        "disagreement_mean": float(disagreement.mean()),
        "disagreement_max": float(disagreement.max()),
        "memory_sha256": T.memory_hash(memory),
        "materialization": material,
        "content_and_topology_share_assignment": True,
        "historical_raw_reads": 0,
        "report_labels_used": False,
    }


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
        c0 = load(source / f"c0_s{stage}_memory.pt")
        anchor = load(source / f"anchor025_s{stage}_memory.pt")
        joint = load(source / f"joint_s{stage}_memory.pt")
        memories = {"c0": c0, "anchor025": anchor, "joint": joint}
        audits = {
            "c0": {"source": str(source / f"c0_s{stage}_memory.pt"),
                   "memory_sha256": T.memory_hash(c0), "historical_raw_reads": 0,
                   "report_labels_used": False},
            "anchor025": {"source": str(source / f"anchor025_s{stage}_memory.pt"),
                          "memory_sha256": T.memory_hash(anchor), "alpha": BASE_ALPHA,
                          "historical_raw_reads": 0, "report_labels_used": False},
            "joint": {"source": str(source / f"joint_s{stage}_memory.pt"),
                      "memory_sha256": T.memory_hash(joint), "alpha": 1.0,
                      "historical_raw_reads": 0, "report_labels_used": False},
        }
        assignments = {}
        if stage == 0:
            for arm in ("plastic025", "plastic050"):
                memories[arm] = c0
                audits[arm] = {"source": str(source / "c0_s0_memory.pt"),
                               "alias": "c0", "memory_sha256": T.memory_hash(c0),
                               "historical_raw_reads": 0, "report_labels_used": False}
        else:
            old = load(BUILD_SOURCE / data["dataset"] / order /
                       f"c0_s{stage-1}_memory.pt")
            block = load(BUILD_SOURCE / data["dataset"] / order /
                         f"c0_s{stage}_block.pt")
            base = append_parent(old, block, args.device)
            delta = T.raw_delta_block(data, ids[stage], stage)
            p_initial = load(source / f"p_initial_s{stage}_assignment.pt").float()
            p_joint = load(source / f"joint_s{stage}_assignment.pt").float()
            support = J.M.lifted_support(old, delta, base,
                                         D.current_assignment(data, ids[stage],
                                                              load(BUILD_SOURCE / data["dataset"] / order /
                                                                   f"c0_s{stage}_memory.pt"),
                                                              len(old["x"]), stage,
                                                              args.device))
            for arm, beta in BETAS.items():
                p, alpha, disagreement = plastic_assignment(p_initial, p_joint, beta)
                coarse = coarse_from_assignment(support, p, len(old["x"]), args.device)
                candidates, material = J.materialize(base, delta, p, coarse,
                                                     data["encoder"], V.CFG)
                memory = candidates["joint"]
                memories[arm] = memory
                assignments[arm] = p
                audits[arm] = build_audit(memory, alpha, disagreement, beta, material)
                audits[arm].update({
                    "parent_hash": T.memory_hash(old),
                    "base_hash": T.memory_hash(base),
                    "delta_hash": T.memory_hash(delta),
                    "p_initial_sha256": R.C.tensor_hash({"p": p_initial}),
                    "p_joint_sha256": R.C.tensor_hash({"p": p_joint}),
                    "assignment_sha256": R.C.tensor_hash({"p": p}),
                })

        for arm, memory in memories.items():
            audits[arm]["payload"] = R.payload(memory)
            audits[arm]["text"] = R.SCA.text_audit(
                memory["texts"], data["encoder"].tokenizer)
            save(out / f"{arm}_s{stage}_build_audit.json", audits[arm])
            if arm in assignments:
                torch.save(assignments[arm], out / f"{arm}_s{stage}_assignment.pt")

        decision = {
            "stage": stage,
            "report_labels_used": False,
            "historical_raw_reads": 0,
            "memories": {arm: D.save_memory(out, arm, stage, memories[arm])
                         for arm in ARMS},
            "locked_at": datetime.now(timezone.utc).isoformat(),
        }
        save(out / f"s{stage}_decision.json", decision)
        print(json.dumps({"event": "prepared", "dataset": data["dataset"],
                          "order": order, "stage": stage}), flush=True)
    save(out / "prepared.json", {"status": "prepared"})


def validate(data, out, stage, ids, budgets, args):
    manifest = E.read(out / "manifest.json")
    source = SOURCE / data["dataset"] / out.name
    assert manifest["source_artifact_hashes"] == source_files(source)
    for arm in ARMS:
        memory = E.load(out / f"{arm}_s{stage}_memory.pt")
        E.memory_checks(memory, budgets[stage])
        if stage == 0:
            assert T.memory_hash(memory) == T.memory_hash(load(source / "c0_s0_memory.pt"))
            continue
        old = load(BUILD_SOURCE / data["dataset"] / out.name /
                   f"c0_s{stage-1}_memory.pt")
        for key in ("x", "text_emb", "mass", "depth", "stage"):
            assert torch.equal(memory[key][:len(old["x"])], old[key])
        historical = memory["edge_index"][:, (memory["edge_index"] < len(old["x"])).all(0)]
        assert torch.equal(historical, old["edge_index"])
        if arm in ("c0", "anchor025", "joint"):
            assert T.memory_hash(memory) == T.memory_hash(load(source / f"{arm}_s{stage}_memory.pt"))
            continue
        p = load(out / f"{arm}_s{stage}_assignment.pt").float()
        audit = E.read(out / f"{arm}_s{stage}_build_audit.json")
        assert R.C.tensor_hash({"p": p}) == audit["assignment_sha256"]
        torch.testing.assert_close(p.sum(1), torch.ones(len(p)), atol=2e-6, rtol=1e-5)
        delta = T.raw_delta_block(data, ids[stage], stage)
        base = append_parent(old, load(BUILD_SOURCE / data["dataset"] / out.name /
                                       f"c0_s{stage}_block.pt"), args.device)
        q = D.current_assignment(data, ids[stage],
                                 load(BUILD_SOURCE / data["dataset"] / out.name /
                                      f"c0_s{stage}_memory.pt"),
                                 len(old["x"]), stage, args.device)
        support = J.M.lifted_support(old, delta, base, q)
        coarse = coarse_from_assignment(support, p, len(old["x"]), args.device)
        n = len(old["x"])
        expected_x = (p.double().T @ data["x"][ids[stage]].double()) / p.double().sum(0)[:, None]
        torch.testing.assert_close(memory["x"][n:].double(), expected_x, atol=2e-5, rtol=2e-5)
        chosen_text = []
        for row in range(p.shape[1]):
            members = torch.where(p[:, row] > 0)[0]
            chosen = members[torch.argsort(p[members, row], descending=True, stable=True)[:V.CFG.text_sources]]
            per = max(1, V.CFG.text_chars // len(chosen))
            chosen_text.append(" ".join(str(data["texts"][int(ids[stage][i])])[:per]
                                         for i in chosen)[:V.CFG.text_chars])
        assert memory["texts"][n:] == chosen_text
        torch.testing.assert_close(data["encoder"].encode(chosen_text).cpu(),
                                   memory["text_emb"][n:], atol=2e-6, rtol=2e-5)
        expected_edges = J.edges_from_projection(base, n, coarse)
        assert set(map(tuple, expected_edges.T.tolist())) == set(map(tuple, memory["edge_index"].T.tolist()))


if __name__ == "__main__":
    E.main(protocol, prepare, validate)
