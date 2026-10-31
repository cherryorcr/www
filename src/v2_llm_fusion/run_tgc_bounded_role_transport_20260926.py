"""Bounded role-conditioned payload transport.

This is a conservative follow-up to ``role_perm``.  It keeps the same
two-prototype proposal and C0 topology, but only permits permutations among
the changed slots of the same class.  The assignment cost combines classwise
standardized semantic distance with an exact one/two-hop propagation-change
distance.  A permutation is accepted only when its total cost is strictly
lower than identity; otherwise the class falls back to identity.

No report F1, test labels, or historical raw graph is used during construction.
"""
from __future__ import annotations

import copy
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import run_tgc_role_aware_assignment_20260925 as R
import run_tgc_joint_student_response as V1
import run_tgc_r11_c0_anchor as R11
import run_tgc_two_prototype_role_permutation as Base
import run_tgc_two_prototype_smoke as B


ARMS = ("c0", "identity", "bounded_role_perm")


def zscore(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    return (value - value.mean()).div(value.std(unbiased=False).clamp_min(1e-6))


def exact_message_costs(c0: dict, old_n: int, source_slots: list[int],
                       proposal_x: torch.Tensor,
                       target_slots: list[int]) -> torch.Tensor:
    """Measure change in exact C0 one/two-hop message perturbation.

    For each proposal, compare placing its feature delta at the original
    changed slot versus another changed slot.  The comparison uses the exact
    dense symmetric-normalized C0 adjacency and its one/two-hop propagation,
    rather than degree-only role summaries.
    """
    adj = R.normalized_adjacency(c0)
    full_x = c0["x"].float()
    costs = torch.zeros((len(source_slots), len(target_slots)), dtype=torch.float32)
    for i, source in enumerate(source_slots):
        source_node = old_n + int(source)
        delta = proposal_x[i].float() - full_x[source_node]
        source_d1 = adj[:, source_node:source_node + 1] * delta[None, :]
        source_d2 = adj @ source_d1
        for j, target in enumerate(target_slots):
            target_node = old_n + int(target)
            target_d1 = adj[:, target_node:target_node + 1] * delta[None, :]
            target_d2 = adj @ target_d1
            d1 = (target_d1 - source_d1).square().mean().sqrt()
            d2 = (target_d2 - source_d2).square().mean().sqrt()
            costs[i, j] = d1 + d2
    return costs


def bounded_class_mapping(data: dict, ids: torch.Tensor, q: torch.Tensor,
                          row_classes: list[int], records: list[dict],
                          c0: dict, old_n: int, class_id: int) -> tuple[list[tuple[int, int]], dict]:
    slots = [i for i, value in enumerate(row_classes) if int(value) == class_id]
    changed = [i for i in slots if records[i].get("changed")]
    if len(changed) < 2:
        return [(slot, slot) for slot in changed], {
            "class": int(class_id), "changed": len(changed),
            "accepted": False, "reason": "fewer_than_two_changed_slots",
        }

    changed_records = [records[i] for i in changed]
    proposal_x, proposal_t = R.proposal_payloads(data, ids, changed_records)
    source_x = data["x"][ids].float()
    source_t = data["text_emb"][ids].float()
    target_positions = [torch.where(q == int(slot))[0] for slot in changed]
    target_x = torch.stack([source_x[pos].mean(0) for pos in target_positions])
    target_t = torch.stack([source_t[pos].mean(0) for pos in target_positions])
    proposal_joint = F.normalize(torch.cat((proposal_x, proposal_t), dim=1), dim=1)
    target_joint = F.normalize(torch.cat((target_x, target_t), dim=1), dim=1)
    semantic = torch.cdist(proposal_joint, target_joint).square()
    message = exact_message_costs(c0, old_n, changed, proposal_x, changed)
    cost = zscore(semantic) + zscore(message)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    selected_cost = sum(float(cost[int(r), int(c)]) for r, c in zip(rows, cols))
    identity_cost = sum(float(cost[i, i]) for i in range(len(changed)))
    accepted = selected_cost < identity_cost - 1e-8
    if accepted:
        pairs = [(changed[int(r)], changed[int(c)]) for r, c in zip(rows, cols)]
    else:
        pairs = [(slot, slot) for slot in changed]
    return pairs, {
        "class": int(class_id),
        "changed": len(changed),
        "accepted": bool(accepted),
        "source_slots": [int(v) for v in changed],
        "selected_cost": float(selected_cost),
        "identity_cost": float(identity_cost),
        "semantic_mean": float(semantic.mean()),
        "message_mean": float(message.mean()),
        "moved_payloads": int(sum(source != target for source, target in pairs)),
        "fallback": not accepted,
    }


def bounded_mapping(data: dict, ids: torch.Tensor, q: torch.Tensor,
                    row_classes: list[int], records: list[dict], c0: dict,
                    old_n: int) -> tuple[dict[int, int], dict]:
    mapping = {i: i for i in range(len(row_classes))}
    audits = []
    for class_id in sorted(set(int(v) for v in row_classes)):
        pairs, audit = bounded_class_mapping(
            data, ids, q, row_classes, records, c0, old_n, class_id,
        )
        for source, target in pairs:
            mapping[source] = target
        audits.append(audit)
    if set(mapping) != set(range(len(row_classes))):
        raise AssertionError("bounded mapping does not cover all rows")
    if len(set(mapping.values())) != len(row_classes):
        raise AssertionError("bounded mapping is not bijective")
    return mapping, {"mode": "bounded_role_perm", "classes": audits}


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    source = Path(args.source_root)
    cell = source / args.dataset / args.order
    parent = R.load(cell / f"c0_s{args.stage - 1}_memory.pt")
    c0 = R.load(cell / f"c0_s{args.stage}_memory.pt")
    data = V1.load_data(args.dataset, args.minilm, args.device,
                        Path(args.cache_root))
    ids, budgets = R11.D.set_stream(data, args.order)
    old_n = len(parent["x"])
    q = R11.D.current_assignment(data, ids[args.stage], c0, old_n,
                                 args.stage, args.device)
    row_classes = B.class_rows_from_c0(q, data["y"][ids[args.stage]])
    c0_block = {
        key: c0[key][old_n:].clone()
        if isinstance(c0[key], torch.Tensor) else list(c0[key][old_n:])
        for key in R.PAYLOAD_KEYS
    }
    structural_block, structural_audit = B.build_two_prototype_block(
        data, ids[args.stage], q, row_classes, args.stage, args.device, c0_block,
    )
    records = structural_audit["records"]
    output = Path(args.output) / f"{args.dataset}_{args.order}_s{args.stage}"
    if (output / "results.json").exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    identity = Base.candidate_from_block(c0, parent, structural_block)
    mapping, mapping_audit = bounded_mapping(
        data, ids[args.stage], q, row_classes, records, c0, old_n,
    )
    bounded, block, perm_records = R.memory_from_mapping(
        c0, parent, structural_block, records, mapping,
    )
    Base.assert_conserved(structural_block, block, records, perm_records,
                          row_classes, c0, parent)
    if not torch.equal(bounded["edge_index"], c0["edge_index"]):
        raise AssertionError("bounded method changed C0 topology")

    memories = {"c0": c0, "identity": identity,
                "bounded_role_perm": bounded}
    audits = {
        "source": str(cell),
        "dataset": args.dataset,
        "order": args.order,
        "stage": int(args.stage),
        "budget": int(budgets[args.stage]),
        "old_rows": int(old_n),
        "new_rows": int(len(c0["x"]) - old_n),
        "historical_raw_reads": 0,
        "report_labels_used": False,
        "structural_audit": structural_audit,
        "mapping_audit": mapping_audit,
        "same_c0_topology": True,
        "same_old_prefix": True,
        "payload_multiset_preserved": True,
    }
    for name, memory in memories.items():
        torch.save(memory, output / f"{name}_memory.pt")
    protocol = {
        "name": "tgc-bounded-role-conditioned-transport-v1",
        "dataset": args.dataset,
        "order": args.order,
        "stage": int(args.stage),
        "arms": list(ARMS),
        "content": "same frozen two-prototype proposal as role_perm",
        "topology": "exact C0 edge_index",
        "assignment": "changed-slot-only permutation; classwise z(semantic)+z(exact one/two-hop message)",
        "acceptance": "strictly lower total cost than identity; otherwise identity fallback",
        "report_f1_selection": False,
        "historical_raw_reads": 0,
        "source_root": str(source),
    }
    save_json(output / "protocol.json", protocol)
    save_json(output / "build_audit.json", audits)
    save_json(output / "results.json", {"protocol": protocol, "audits": audits})
    print(json.dumps({"event": "complete", "dataset": args.dataset,
                      "order": args.order, "stage": args.stage,
                      "output": str(output)}), flush=True)


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
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
