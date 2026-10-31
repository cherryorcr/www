"""Topology-role-aware conserved semantic assignment.

This experiment keeps the original Student, C0 topology, text policy and
complete payload conservation fixed.  It compares the old semantic matching
with progressively stronger deterministic assignment costs:

* semantic only;
* semantic + C0 topology role;
* role matching with an identity anchor and per-class fallback;
* role matching with a normalized GCN-message proxy;
* message matching with a prompt-margin plasticity proxy;
* adaptive graph/text weighting in the same frozen assignment objective.

No report/test F1 is used during construction.  The only raw data accessed by
construction is the current legal delta; historical state comes from C0
synthetic memory.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import run_tgc_joint_student_response as V1
import run_tgc_r11_c0_anchor as R11
import run_tgc_two_prototype_role_permutation as Base
import run_tgc_two_prototype_smoke as B
import run_tgc_two_prototype_textcap_confirmation as C
import run_tgc_two_prototype_role_diagnostic as Role
import tgc_shared_assignment as J


PAYLOAD_KEYS = ("x", "text_emb", "texts", "mass", "depth", "stage")
ARMS = (
    "c0",
    "identity",
    "semantic_perm",
    "role_perm",
    "anchor_role_perm",
    "message_role_perm",
    "plasticity_role_perm",
    "adaptive_role_perm",
    "random_perm",
)


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def clone_block(block: dict) -> dict:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else list(value)
        for key, value in block.items()
    }


def normalize_matrix(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    return (value - value.mean()).div(value.std(unbiased=False).clamp_min(1e-6))


def role_tensor(roles: dict[int, dict[str, float]], slots: list[int]) -> torch.Tensor:
    keys = ("degree", "cross_degree", "new_new_degree",
            "old_neighbor_ratio", "old_endpoint_load")
    value = torch.tensor([[float(roles[slot][key]) for key in keys]
                          for slot in slots], dtype=torch.float32)
    return (value - value.mean(0)) / value.std(0, unbiased=False).clamp_min(1e-6)


def normalized_adjacency(memory: dict) -> torch.Tensor:
    """Build the exact dense symmetric-normalized C0 adjacency proxy."""
    n = int(len(memory["x"]))
    edge = memory["edge_index"].long().cpu()
    adj = torch.zeros((n, n), dtype=torch.float32)
    adj[edge[0], edge[1]] = 1.0
    adj.fill_diagonal_(1.0)
    degree = adj.sum(1).clamp_min(1.0)
    scale = degree.rsqrt()
    return scale[:, None] * adj * scale[None, :]


def proposal_payloads(data: dict, ids: torch.Tensor,
                      records: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    x = data["x"][ids].float()
    text = data["text_emb"][ids].float()
    px, pt = [], []
    for record in records:
        members = torch.tensor(record["source_positions"], dtype=torch.long)
        px.append(x[members].mean(0))
        pt.append(text[members].mean(0))
    return torch.stack(px), torch.stack(pt)


def message_costs(c0: dict, old_n: int, class_slots: list[int],
                  proposals: list[dict], proposal_x: torch.Tensor,
                  class_proto: torch.Tensor) -> torch.Tensor:
    """Approximate normalized GCN message compatibility for each target slot."""
    adj = normalized_adjacency(c0)
    full_x = c0["x"].float()
    costs = []
    for proposal_x_row in proposal_x:
        row = []
        for target in class_slots:
            node = old_n + int(target)
            baseline = adj[node] @ full_x
            replacement = baseline + adj[node, node] * (
                proposal_x_row - full_x[node])
            row.append(1.0 - F.cosine_similarity(
                F.normalize(replacement, dim=0),
                F.normalize(class_proto, dim=0), dim=0).item())
        costs.append(row)
    return torch.tensor(costs, dtype=torch.float32)


def plasticity_costs(data: dict, c0: dict, old_n: int,
                     class_id: int, class_slots: list[int],
                     proposal_x: torch.Tensor) -> torch.Tensor:
    """Prompt-margin risk after placing a proposal at a C0 slot."""
    prompt = data["prompt_emb"].float().cpu()
    if prompt.ndim != 2 or prompt.shape[1] != proposal_x.shape[1]:
        return torch.zeros((len(proposal_x), len(class_slots)))
    own = F.normalize(prompt[int(class_id)], dim=0)
    other = prompt[torch.arange(len(prompt)) != int(class_id)]
    other = F.normalize(other, dim=1) if len(other) else other
    adj = normalized_adjacency(c0)
    full_x = c0["x"].float()
    rows = []
    for x_row in proposal_x:
        values = []
        for target in class_slots:
            node = old_n + int(target)
            message = adj[node] @ full_x + adj[node, node] * (
                x_row - full_x[node])
            message = F.normalize(message, dim=0)
            own_score = float(torch.dot(message, own))
            other_score = float((other @ message).max()) if len(other) else -1.0
            values.append(other_score - own_score)
        rows.append(values)
    return torch.tensor(rows, dtype=torch.float32)


def adaptive_semantic_cost(proposal_x: torch.Tensor, proposal_t: torch.Tensor,
                           target_x: torch.Tensor,
                           target_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    px = F.normalize(proposal_x.float(), dim=1)
    pt = F.normalize(proposal_t.float(), dim=1)
    tx = F.normalize(target_x.float(), dim=1)
    tt = F.normalize(target_t.float(), dim=1)
    graph = torch.cdist(px, tx).square()
    text = torch.cdist(pt, tt).square()
    agreement = (px * pt).sum(1)
    alpha = (0.50 + 0.25 * agreement).clamp(0.25, 0.75)
    return alpha[:, None] * graph + (1.0 - alpha[:, None]) * text, alpha


def class_mapping(data: dict, ids: torch.Tensor, q: torch.Tensor,
                  row_classes: list[int], structural_records: list[dict],
                  c0: dict, old_n: int, class_id: int, mode: str,
                  random_seed: int) -> tuple[list[tuple[int, int]], dict]:
    slots = [i for i, value in enumerate(row_classes) if int(value) == class_id]
    changed = [i for i in slots if structural_records[i].get("changed")]
    if not changed:
        return [], {"class": int(class_id), "changed": 0, "fallback": False}

    source_x = data["x"][ids].float()
    source_t = data["text_emb"][ids].float()
    proposal_x, proposal_t = proposal_payloads(
        data, ids, [structural_records[i] for i in changed])
    target_positions = [torch.where(q == int(slot))[0] for slot in slots]
    target_x = torch.stack([source_x[pos].mean(0) for pos in target_positions])
    target_t = torch.stack([source_t[pos].mean(0) for pos in target_positions])
    sem, alpha = adaptive_semantic_cost(proposal_x, proposal_t,
                                         target_x, target_t)
    fixed_graph_text = torch.cdist(
        F.normalize(torch.cat((proposal_x, proposal_t), dim=1), dim=1),
        F.normalize(torch.cat((target_x, target_t), dim=1), dim=1)).square()

    roles = Role.node_roles(c0, old_n)
    slot_role = role_tensor(roles, [old_n + slot for slot in slots])
    proposal_roles = []
    for record in [structural_records[i] for i in changed]:
        positions = torch.tensor(record["source_positions"], dtype=torch.long)
        source_slots = q[positions].tolist()
        role_indices = [slots.index(int(source_slot))
                        for source_slot in source_slots]
        proposal_roles.append(slot_role[role_indices].mean(0))
    proposal_role = torch.stack(proposal_roles)
    role = torch.cdist(proposal_role, slot_role).square()
    proposal_mass = torch.tensor([
        max(1, int(structural_records[i].get("mass", 1))) for i in changed
    ], dtype=torch.float32)
    target_mass = c0["mass"][old_n + torch.tensor(slots)].float().clamp_min(1e-6)
    mass = torch.abs(torch.log((proposal_mass[:, None] + 1.0)
                               / (target_mass[None, :] + 1.0)))
    class_proto = proposal_x.mean(0)
    message = message_costs(c0, old_n, slots,
                            [structural_records[i] for i in changed],
                            proposal_x, class_proto)
    plastic = plasticity_costs(data, c0, old_n, class_id, slots, proposal_x)
    anchor = torch.tensor([
        [0.0 if source == target else 1.0 for target in slots]
        for source in changed
    ], dtype=torch.float32)

    if mode == "semantic_perm":
        cost = normalize_matrix(fixed_graph_text)
    elif mode == "role_perm":
        cost = (normalize_matrix(fixed_graph_text)
                + 0.50 * normalize_matrix(role)
                + 0.10 * normalize_matrix(mass))
    elif mode == "anchor_role_perm":
        cost = (normalize_matrix(fixed_graph_text)
                + 0.50 * normalize_matrix(role)
                + 0.10 * normalize_matrix(mass)
                + 0.25 * anchor)
    elif mode == "message_role_perm":
        cost = (normalize_matrix(fixed_graph_text)
                + 0.50 * normalize_matrix(role)
                + 0.10 * normalize_matrix(mass)
                + 0.50 * normalize_matrix(message)
                + 0.25 * anchor)
    elif mode == "plasticity_role_perm":
        cost = (normalize_matrix(fixed_graph_text)
                + 0.50 * normalize_matrix(role)
                + 0.10 * normalize_matrix(mass)
                + 0.50 * normalize_matrix(message)
                + 0.50 * normalize_matrix(plastic)
                + 0.25 * anchor)
    elif mode == "adaptive_role_perm":
        cost = (normalize_matrix(sem)
                + 0.50 * normalize_matrix(role)
                + 0.10 * normalize_matrix(mass)
                + 0.50 * normalize_matrix(message)
                + 0.50 * normalize_matrix(plastic)
                + 0.25 * anchor)
    elif mode == "random_perm":
        rng = random.Random(int(random_seed) + 7919 * int(class_id))
        targets = list(slots)
        rng.shuffle(targets)
        return list(zip(changed, targets[:len(changed)])), {
            "class": int(class_id), "changed": len(changed), "fallback": False,
            "mode": mode, "adaptive_alpha": [float(v) for v in alpha]
        }
    else:
        raise ValueError(f"unknown assignment mode: {mode}")

    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    selected = [(changed[int(row)], slots[int(col)]) for row, col in zip(rows, cols)]
    fallback = False
    if mode in {"anchor_role_perm", "message_role_perm",
                "plasticity_role_perm", "adaptive_role_perm"}:
        identity_cost = sum(float(cost[i, slots.index(source)])
                            for i, source in enumerate(changed))
        selected_cost = sum(float(cost[int(row), int(col)])
                            for row, col in zip(rows, cols))
        # A deterministic, F1-free C0 anchor: reject a class-level move unless
        # it improves the normalized proxy objective by at least 0.05.
        if selected_cost >= identity_cost - 0.05:
            selected = [(source, source) for source in changed]
            fallback = True
    return selected, {
        "class": int(class_id), "changed": len(changed), "fallback": fallback,
        "mode": mode, "selected_cost": float(sum(float(cost[r, c])
                                                     for r, c in zip(rows, cols))),
        "identity_cost": float(sum(float(cost[i, slots.index(source)])
                                    for i, source in enumerate(changed))),
        "adaptive_alpha": [float(v) for v in alpha],
        "cost_components": {
            "semantic_mean": float(fixed_graph_text.mean()),
            "role_mean": float(role.mean()),
            "mass_mean": float(mass.mean()),
            "message_mean": float(message.mean()),
            "plasticity_mean": float(plastic.mean()),
        },
    }


def complete_mapping(data: dict, ids: torch.Tensor, q: torch.Tensor,
                     row_classes: list[int], structural_records: list[dict],
                     c0: dict, old_n: int, mode: str,
                     random_seed: int) -> tuple[dict[int, int], dict]:
    mapping: dict[int, int] = {}
    classes = []
    for class_id in sorted(set(int(v) for v in row_classes)):
        slots = [i for i, value in enumerate(row_classes) if int(value) == class_id]
        pairs, audit = class_mapping(data, ids, q, row_classes,
                                     structural_records, c0, old_n,
                                     class_id, mode, random_seed)
        changed_sources = {source for source, _ in pairs}
        changed_targets = {target for _, target in pairs}
        for source, target in pairs:
            mapping[source] = target
        remaining_source = [slot for slot in slots if slot not in changed_sources]
        remaining_target = [slot for slot in slots if slot not in changed_targets]
        fixed = sorted(set(remaining_source).intersection(remaining_target))
        for slot in fixed:
            mapping[slot] = slot
        residual_source = [slot for slot in remaining_source if slot not in fixed]
        residual_target = [slot for slot in remaining_target if slot not in fixed]
        if len(residual_source) != len(residual_target):
            raise AssertionError("class residual mapping is not square")
        for source, target in zip(residual_source, residual_target):
            mapping[source] = target
        audit["class_slots"] = slots
        audit["moved_payloads"] = int(sum(source != target
                                           for source, target in mapping.items()
                                           if source in slots))
        classes.append(audit)
    if set(mapping) != set(range(len(row_classes))):
        raise AssertionError("mapping does not cover all rows")
    if len(set(mapping.values())) != len(row_classes):
        raise AssertionError("mapping is not bijective")
    return mapping, {"mode": mode, "classes": classes}


def memory_from_mapping(c0: dict, parent: dict, structural_block: dict,
                        records: list[dict], mapping: dict[int, int]) -> dict:
    block = clone_block(structural_block)
    perm_records = [None] * len(records)
    for source, target in mapping.items():
        for key in PAYLOAD_KEYS:
            if key == "texts":
                block[key][target] = structural_block[key][source]
            else:
                block[key][target] = structural_block[key][source]
        perm_records[target] = copy.deepcopy(records[source])
    if any(record is None for record in perm_records):
        raise AssertionError("permuted record missing")
    memory = Base.candidate_from_block(c0, parent, block)
    if not torch.equal(memory["edge_index"], c0["edge_index"]):
        raise AssertionError("topology changed")
    return memory, block, perm_records


def run(args: argparse.Namespace) -> None:
    source = Path(args.source_root)
    cell = source / args.dataset / args.order
    parent = load(cell / f"c0_s{args.stage - 1}_memory.pt")
    c0 = load(cell / f"c0_s{args.stage}_memory.pt")
    data = V1.load_data(args.dataset, args.minilm, args.device,
                        Path(args.cache_root))
    ids, budgets = R11.D.set_stream(data, args.order)
    old_n = len(parent["x"])
    q = R11.D.current_assignment(data, ids[args.stage], c0, old_n,
                                 args.stage, args.device)
    row_classes = B.class_rows_from_c0(q, data["y"][ids[args.stage]])
    c0_block = {
        key: c0[key][old_n:].clone() if isinstance(c0[key], torch.Tensor)
        else list(c0[key][old_n:]) for key in PAYLOAD_KEYS
    }
    structural_block, structural_audit = B.build_two_prototype_block(
        data, ids[args.stage], q, row_classes, args.stage, args.device, c0_block
    )
    structural_records = structural_audit["records"]
    out = Path(args.output) / f"{args.dataset}_{args.order}_s{args.stage}"
    if (out / "results.json").exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    memories = {"c0": c0, "identity": Base.candidate_from_block(
        c0, parent, structural_block)}
    audits = {
        "source": str(cell),
        "dataset": args.dataset, "order": args.order, "stage": int(args.stage),
        "budget": int(budgets[args.stage]), "old_rows": int(old_n),
        "new_rows": int(len(c0["x"]) - old_n),
        "historical_raw_reads": 0, "report_labels_used": False,
        "structural_audit": structural_audit, "mappings": {},
    }
    for mode in ("semantic_perm", "role_perm", "anchor_role_perm",
                 "message_role_perm", "plasticity_role_perm",
                 "adaptive_role_perm", "random_perm"):
        mapping, mapping_audit = complete_mapping(
            data, ids[args.stage], q, row_classes, structural_records,
            c0, old_n, mode, args.random_seed)
        memory, block, perm_records = memory_from_mapping(
            c0, parent, structural_block, structural_records, mapping)
        Base.assert_conserved(structural_block, block, structural_records,
                              perm_records, row_classes, c0, parent)
        name = mode
        memories[name] = memory
        audits["mappings"][name] = {
            "mapping": {str(k): int(v) for k, v in mapping.items()},
            "audit": mapping_audit,
            "payload_multiset_preserved": True,
            "same_c0_topology": True,
            "same_old_prefix": True,
            "source_coverage_preserved": True,
        }

    reports = {}
    for name, memory in memories.items():
        started = time.perf_counter()
        reports[name] = V1.report_candidate(
            memory, data, args.stage, list(args.report_seeds), args.device)
        reports[name]["wall_seconds_total"] = time.perf_counter() - started
        torch.save(memory, out / f"{name}_memory.pt")

    protocol = {
        "name": "tgc-topology-role-aware-conserved-assignment-v1",
        "dataset": args.dataset, "order": args.order, "stage": int(args.stage),
        "arms": list(ARMS), "report_seeds": list(args.report_seeds),
        "rate": 0.05,
        "content": "fixed two-prototype structural proposal; no P content regeneration",
        "topology": "exact C0 edge_index",
        "text": "top-5/320-char center-ranked sources, MiniLM re-encode",
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "assignment": {
            "semantic": "same-class joint graph-text cost",
            "role": ["degree", "cross_degree", "new_new_degree",
                     "old_neighbor_ratio", "old_endpoint_load"],
            "message": "C0 normalized one-hop message proxy",
            "plasticity": "current-class prompt margin proxy",
            "anchor": "per-class identity fallback if normalized objective gain < 0.05",
            "adaptive_alpha": "bounded per-proposal graph/text agreement weight [0.25,0.75]",
            "report_f1_selection": False,
        },
        "historical_raw_reads": 0,
        "test_labels_used_for_construction": False,
        "source_root": str(source),
    }
    save_json(out / "protocol.json", protocol)
    save_json(out / "build_audit.json", audits)
    save_json(out / "results.json", {"protocol": protocol,
                                      "audits": audits, "reports": reports})
    print(json.dumps({"event": "complete", "dataset": args.dataset,
                      "order": args.order, "stage": args.stage,
                      "output": str(out)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--random-seed", type=int, default=20260925)
    parser.add_argument("--report-seeds", nargs="+", type=int,
                        default=list(range(150, 156)))
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
