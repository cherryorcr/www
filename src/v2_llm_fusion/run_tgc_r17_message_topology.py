"""R17: teacher-message-compatible topology gating.

This experiment keeps the sealed R11 anchor025 content and text unchanged.
Only the topology is reconsidered.  C0 is the scaffold; R11/C0 candidate
edges form the legal pool.  A small number of degree-preserving internal
2-switches and guarded cross-edge replacements are accepted only when the
teacher message target improves and the old-node message risk does not grow.
Any unsafe candidate falls back to C0 topology.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import tgc_r6_experiment as E
import tgc_relational_response as M
import tgc_shared_assignment as J
import run_tgc_r5_joint as V
import run_tgc_r11_c0_anchor as R11
import run_tgc_r16_topology_safe as Safe

R, T, D, B = E.R, E.T, E.D, E.B
SOURCE = Path("/home/pc/research/TAG_condensation/results/tgc_r11_c0_anchor_20260923b")
DATASETS = ("cora", "citeseer")
ORDERS = ("order19601", "order19602", "order19603")
SEEDS = tuple(range(150, 156))
ARMS = ("c0", "anchor025", "anchor_content_c0_topology",
        "message_gate_1", "message_gate_2")
TEACHER = replace(V.TEACHER, teacher_seeds=(9000, 9001))
MESSAGE_MARGIN = 1e-4
LOSS_MARGIN = 1e-5
CROSS_MARGIN = 0.01


def protocol():
    return json.loads(json.dumps({
        "name": "tgc-r17-teacher-message-topology-v1",
        "datasets": DATASETS, "orders": ORDERS,
        "evaluation_stages": [0, 1, 2], "report_seeds": list(SEEDS),
        "arms": ARMS, "source_root": str(SOURCE),
        "content": "sealed R11 anchor025; alpha_content=0.25",
        "text": "unchanged sealed R11 top-5/320-char concat-reencode",
        "topology": (
            "C0 scaffold; union of C0/R11 candidate edges; teacher-message "
            "compatibility score; max one or two internal swaps; guarded cross edges; C0 fallback"
        ),
        "teacher": {"seeds": list(TEACHER.teacher_seeds), "steps": TEACHER.teacher_steps,
                    "batch": TEACHER.teacher_batch, "sampler": "compact"},
        "gates": {"edge_score_margin": MESSAGE_MARGIN,
                  "cross_score_margin": CROSS_MARGIN,
                  "global_loss_margin": LOSS_MARGIN,
                  "max_internal_swaps": {"message_gate_1": 1, "message_gate_2": 2}},
        "constraints": {
            "old_old_edges": "immutable",
            "old_rows": "immutable",
            "new_new_edge_count": "C0",
            "new_internal_degree": "C0",
            "new_cross_degree": "C0",
            "old_cross_capacity": "C0-derived",
        },
        "history_access": "sealed C0/R11 synthetic memories plus current delta only; no historical raw graph",
        "historical_raw_reads": 0, "report_labels_used": False,
        "selection_by_report": False, "development_only": True,
        "independent_confirmation": False,
        "student": {"batch": 16, "lr": .001,
                    "steps": "max(50,min(2048,16*rows))"},
        "promotion_rule": {"mean_final_min_pp": .5, "stage_mean_min_pp": 0.0,
                            "worst_final_min_pp": -2.0, "mean_new_min_pp": 0.0,
                            "positive_final_cells": 4},
        "source_hashes": R.source_hashes(),
    }))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save(path, value):
    E.save(Path(path), value)


def source_files(source):
    names = ["protocol.json", "selection_seal.json"]
    for stage in range(3):
        names.extend([f"c0_s{stage}_memory.pt", f"anchor025_s{stage}_memory.pt"])
        if stage:
            names.append(f"anchor025_s{stage}_assignment.pt")
    return {name: R.C.sha(source / name) for name in names if (source / name).exists()}


def compact_teacher_targets(data, ids, source, stage, c0, old, device):
    """Recreate teacher messages from old synthetic rows and current raw delta."""
    # The sealed R11 directory stores the complete C0 stage memory rather than
    # a separate block artifact.  It is exactly the parent-append compact
    # support needed here, so using it avoids reconstructing a new topology.
    base = c0
    delta = T.raw_delta_block(data, ids[stage], stage)
    q = D.current_assignment(data, ids[stage], c0, len(old["x"]), stage, device)
    support = J.M.lifted_support(old, delta, base, q)
    h, teacher_audits = J.teacher_views(support, base, TEACHER, device)
    adjacency = M.adjacency(support, device)
    messages = torch.sparse.mm(adjacency, h)
    assignment = load(source / f"anchor025_s{stage}_assignment.pt").float().to(device)
    old_n = len(old["x"])
    mass = assignment.sum(0).clamp_min(1e-9)
    compact_h = torch.cat((h[:old_n], (assignment.T @ h[old_n:]) / mass[:, None]), dim=0)
    compact_m = torch.cat((messages[:old_n], (assignment.T @ messages[old_n:]) / mass[:, None]), dim=0)
    return compact_h.detach().cpu(), compact_m.detach().cpu(), {
        "teacher_support_sha256": T.memory_hash(support),
        "teacher_assignment_sha256": R.C.tensor_hash({"p": assignment.cpu()}),
        "teacher_audits": teacher_audits,
        "current_delta_ids": data["y"][ids[stage]].shape[0],
        "historical_raw_reads": 0,
    }


def edge_pairs(edges, old_n, total, internal):
    edges = edges.cpu()
    if internal:
        mask = (edges[0] >= old_n) & (edges[1] >= old_n)
    else:
        mask = ((edges[0] < old_n) & (edges[1] >= old_n) & (edges[1] < total))
    values = edges[:, mask]
    result = set()
    for left, right in values.T.tolist():
        if left == right:
            continue
        pair = tuple(sorted((int(left), int(right))))
        if internal or pair[0] < old_n:
            result.add(pair)
    return sorted(result)


def compact_edge_score(h, m):
    h = F.normalize(h.float(), dim=1)
    m = F.normalize(m.float(), dim=1)
    return 0.5 * (h @ m.T) + 0.5 * (m @ h.T)


def message_losses(edge_index, h, target, old_n):
    n = len(h)
    adj = M.weighted_adjacency(edge_index, n, torch.device("cpu"), dtype=h.dtype)
    predicted = torch.sparse.mm(adj, h)
    cosine = F.cosine_similarity(predicted, target, dim=1)
    old = 1. - cosine[:old_n].mean()
    new = 1. - cosine[old_n:].mean()
    total = 0.5 * (old + new)
    return {"total": float(total), "old": float(old), "new": float(new)}


def build_edges(old_edges, internal_pairs, cross_pairs):
    internal = torch.tensor(internal_pairs, dtype=torch.long).T if internal_pairs else torch.empty((2, 0), dtype=torch.long)
    cross = torch.tensor(cross_pairs, dtype=torch.long).T if cross_pairs else torch.empty((2, 0), dtype=torch.long)
    internal = torch.cat((internal, internal.flip(0)), dim=1)
    cross = torch.cat((cross, cross.flip(0)), dim=1)
    return torch.cat((old_edges.cpu(), internal, cross), dim=1)


def best_internal_switch(current, candidate_pool, scores, old_edges, cross, h, target, old_n, current_loss):
    best = None
    edges = sorted(current)
    for index, first in enumerate(edges):
        a, b = first
        for second in edges[index + 1:]:
            c, d = second
            if len({a, b, c, d}) < 4:
                continue
            old_score = float(scores[a, b] + scores[c, d])
            for proposed in (((a, c), (b, d)), ((a, d), (b, c))):
                proposed = tuple(sorted(tuple(sorted(pair)) for pair in proposed))
                if proposed[0] == proposed[1] or any(x == y for x, y in proposed):
                    continue
                if any(pair not in candidate_pool or pair in current for pair in proposed):
                    continue
                gain = float(scores[proposed[0]] + scores[proposed[1]] - old_score)
                if gain <= MESSAGE_MARGIN:
                    continue
                trial = set(current)
                trial.remove(first)
                trial.remove(second)
                trial.update(proposed)
                edge_index = build_edges(old_edges, sorted(trial), cross)
                loss = message_losses(edge_index, h, target, old_n)
                if loss["total"] >= current_loss["total"] - LOSS_MARGIN:
                    continue
                if loss["old"] > current_loss["old"] + LOSS_MARGIN:
                    continue
                key = (current_loss["total"] - loss["total"], gain,
                       tuple(-x for pair in proposed for x in pair))
                if best is None or key > best[0]:
                    best = (key, first, second, proposed, loss, gain)
    return best


def guarded_cross(current, c0_candidates, anchor_candidates, scores, old_edges,
                  internal, h, target, old_n, current_loss):
    c0_by_new = {new: old for old, new in c0_candidates}
    anchor_by_new = {new: old for old, new in anchor_candidates}
    old_load = {}
    for old, _ in c0_candidates:
        old_load[old] = old_load.get(old, 0) + 1
    capacity = max(1, max(old_load.values(), default=1))
    chosen = dict(c0_by_new)
    changed = []
    for new in sorted(c0_by_new):
        candidate = anchor_by_new.get(new)
        baseline = chosen[new]
        if candidate is None or candidate == baseline:
            continue
        gain = float(scores[candidate, new] - scores[baseline, new])
        if gain <= CROSS_MARGIN or old_load.get(candidate, 0) >= capacity:
            continue
        trial = dict(chosen)
        trial[new] = candidate
        trial_pairs = [(old, node) for node, old in sorted(trial.items())]
        edge_index = build_edges(old_edges, sorted(internal), trial_pairs)
        loss = message_losses(edge_index, h, target, old_n)
        if loss["total"] >= current_loss["total"] - LOSS_MARGIN:
            continue
        if loss["old"] > current_loss["old"] + LOSS_MARGIN:
            continue
        old_load[baseline] = max(0, old_load.get(baseline, 0) - 1)
        old_load[candidate] = old_load.get(candidate, 0) + 1
        chosen = trial
        current_loss = loss
        changed.append({"new": new, "from_old": baseline, "to_old": candidate,
                        "score_gain": gain, "loss": loss})
    return [(old, node) for node, old in sorted(chosen.items())], {
        "changed": changed, "changed_count": len(changed),
        "old_endpoint_capacity": capacity,
    }, current_loss


def message_edges(c0, anchor, old_n, h, target, max_swaps, allow_cross=True):
    total = len(c0["x"])
    scores = compact_edge_score(h, target)
    c0_internal = edge_pairs(c0["edge_index"], old_n, total, True)
    anchor_internal = edge_pairs(anchor["edge_index"], old_n, total, True)
    candidate_pool = set(c0_internal) | set(anchor_internal)
    c0_cross = edge_pairs(c0["edge_index"], old_n, total, False)
    anchor_cross = edge_pairs(anchor["edge_index"], old_n, total, False)
    old_edges = c0["edge_index"][:, (c0["edge_index"] < old_n).all(0)].cpu()
    current = set(c0_internal)
    cross = list(c0_cross)
    baseline = message_losses(c0["edge_index"], h, target, old_n)
    current_loss = dict(baseline)
    swaps = []
    for _ in range(max_swaps):
        best = best_internal_switch(current, candidate_pool, scores, old_edges,
                                    cross, h, target, old_n, current_loss)
        if best is None:
            break
        _, first, second, proposed, current_loss, gain = best
        current.remove(first)
        current.remove(second)
        current.update(proposed)
        swaps.append({"removed": [list(first), list(second)],
                      "added": [list(pair) for pair in proposed],
                      "score_gain": gain, "loss": current_loss})
    cross_audit = {"changed": [], "changed_count": 0, "old_endpoint_capacity": None}
    if allow_cross:
        cross, cross_audit, current_loss = guarded_cross(
            cross, c0_cross, anchor_cross, scores, old_edges, current,
            h, target, old_n, current_loss)
    result = build_edges(old_edges, sorted(current), cross)
    final_loss = message_losses(result, h, target, old_n)
    fallback = False
    if final_loss["total"] >= baseline["total"] - LOSS_MARGIN or final_loss["old"] > baseline["old"] + LOSS_MARGIN:
        result = c0["edge_index"].clone()
        final_loss = baseline
        fallback = True
        swaps = []
        cross_audit["changed"] = []
        cross_audit["changed_count"] = 0
    audit = {"mode": "message_gate", "max_internal_swaps": max_swaps,
             "internal_swaps": swaps, "internal_swap_count": len(swaps),
             "cross": cross_audit, "baseline_loss": baseline,
             "final_loss": final_loss, "fallback_to_c0": fallback,
             "message_margin": MESSAGE_MARGIN, "cross_margin": CROSS_MARGIN,
             "loss_margin": LOSS_MARGIN}
    return result, audit


def make_memories(c0, anchor, old_n, h, target):
    content_c0 = T.clone_memory(anchor)
    content_c0["edge_index"] = c0["edge_index"].clone()
    edge1, audit1 = message_edges(c0, anchor, old_n, h, target, 1, allow_cross=True)
    edge2, audit2 = message_edges(c0, anchor, old_n, h, target, 2, allow_cross=True)
    gate1, gate2 = T.clone_memory(anchor), T.clone_memory(anchor)
    gate1["edge_index"], gate2["edge_index"] = edge1, edge2
    return {
        "anchor_content_c0_topology": content_c0,
        "message_gate_1": gate1, "message_gate_2": gate2,
    }, {"message_gate_1": audit1, "message_gate_2": audit2}


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
        memories = {"c0": c0, "anchor025": anchor}
        audits = {
            "c0": {"source": str(source / f"c0_s{stage}_memory.pt"),
                   "memory_sha256": T.memory_hash(c0), "historical_raw_reads": 0,
                   "report_labels_used": False},
            "anchor025": {"source": str(source / f"anchor025_s{stage}_memory.pt"),
                          "memory_sha256": T.memory_hash(anchor), "alpha": .25,
                          "historical_raw_reads": 0, "report_labels_used": False},
        }
        if stage == 0:
            for arm in ARMS[2:]:
                memories[arm] = c0
                audits[arm] = {"alias": "c0", "memory_sha256": T.memory_hash(c0),
                               "historical_raw_reads": 0, "report_labels_used": False}
        else:
            old = load(source / f"c0_s{stage-1}_memory.pt")
            h, target, teacher_audit = compact_teacher_targets(
                data, ids, source, stage, c0, old, args.device)
            derived, topology_audits = make_memories(c0, anchor, len(old["x"]), h, target)
            memories.update(derived)
            for arm, memory in derived.items():
                audits[arm] = {
                    "content_source": "anchor025", "topology_source": "message-gated-c0-r11-candidates",
                    "memory_sha256": T.memory_hash(memory),
                    "teacher": teacher_audit,
                    "topology_audit": topology_audits.get(arm, {"mode": "c0"}),
                    "historical_raw_reads": 0, "report_labels_used": False,
                }
        for arm, memory in memories.items():
            audits[arm]["payload"] = R.payload(memory)
            audits[arm]["text"] = R.SCA.text_audit(memory["texts"], data["encoder"].tokenizer)
            save(out / f"{arm}_s{stage}_build_audit.json", audits[arm])
        decision = {
            "stage": stage, "report_labels_used": False, "historical_raw_reads": 0,
            "memories": {arm: D.save_memory(out, arm, stage, memories[arm]) for arm in ARMS},
            "locked_at": datetime.now(timezone.utc).isoformat(),
        }
        save(out / f"s{stage}_decision.json", decision)
        print(json.dumps({"event": "prepared", "dataset": data["dataset"],
                          "order": order, "stage": stage}), flush=True)
    save(out / "prepared.json", {"status": "prepared"})


def validate(data, out, stage, ids, budgets, args):
    source = SOURCE / data["dataset"] / out.name
    manifest = E.read(out / "manifest.json")
    assert manifest["source_artifact_hashes"] == source_files(source)
    c0 = load(source / f"c0_s{stage}_memory.pt")
    anchor = load(source / f"anchor025_s{stage}_memory.pt")
    for arm in ARMS:
        memory = E.load(out / f"{arm}_s{stage}_memory.pt")
        E.memory_checks(memory, budgets[stage])
        if stage == 0:
            assert T.memory_hash(memory) == T.memory_hash(c0)
            continue
        old = load(source / f"c0_s{stage-1}_memory.pt")
        for key in ("x", "text_emb", "mass", "depth", "stage"):
            assert torch.equal(memory[key][:len(old["x"])], old[key])
        assert memory["texts"][:len(old["x"])] == old["texts"]
        historical = memory["edge_index"][:, (memory["edge_index"] < len(old["x"])).all(0)]
        assert torch.equal(historical, old["edge_index"])
        if arm == "c0":
            assert T.memory_hash(memory) == T.memory_hash(c0)
        elif arm == "anchor025":
            assert T.memory_hash(memory) == T.memory_hash(anchor)
        else:
            for key in ("x", "text_emb", "mass"):
                assert torch.equal(memory[key], anchor[key])
            assert memory["texts"] == anchor["texts"]
            if arm == "anchor_content_c0_topology":
                assert torch.equal(memory["edge_index"], c0["edge_index"])
            else:
                audit = E.read(out / f"{arm}_s{stage}_build_audit.json")
                assert audit["topology_audit"]["mode"] == "message_gate"
                assert audit["teacher"]["historical_raw_reads"] == 0


if __name__ == "__main__":
    E.main(protocol, prepare, validate)
