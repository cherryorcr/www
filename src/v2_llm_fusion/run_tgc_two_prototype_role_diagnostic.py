"""Diagnose whether two-prototype gains depend on the target C0 row role.

The source pieces, X/text values, total rows, and C0 topology are fixed.  Only
the C0 slots receiving the changed pieces are altered within each class.  This
is an attribution experiment, not a selector and not a new main method.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

import run_tgc_joint_student_response as V1
import run_tgc_r11_c0_anchor as R11
import run_tgc_two_prototype_smoke as B
import run_tgc_two_prototype_textcap_confirmation as C
import tgc_shared_assignment as J


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def node_roles(memory: dict, old_n: int) -> dict[int, dict[str, float]]:
    edge = memory["edge_index"].long()
    n = len(memory["x"])
    src, dst = edge[0], edge[1]
    degree = torch.bincount(src, minlength=n).float()
    old_neighbor = torch.zeros(n, dtype=torch.float32)
    new_neighbor = torch.zeros(n, dtype=torch.float32)
    old_endpoint_load = torch.bincount(src[(src < old_n) & (dst >= old_n)],
                                       minlength=n).float()
    for node in range(old_n, n):
        neigh = dst[src == node]
        old_neighbor[node] = float((neigh < old_n).sum())
        new_neighbor[node] = float((neigh >= old_n).sum())
    result = {}
    for node in range(old_n, n):
        d = max(1.0, float(degree[node]))
        neigh = dst[src == node]
        old_load = float(old_endpoint_load[neigh[neigh < old_n]].sum())
        result[node] = {
            "degree": float(degree[node]),
            "cross_degree": float(old_neighbor[node]),
            "new_new_degree": float(new_neighbor[node]),
            "old_neighbor_ratio": float(old_neighbor[node] / d),
            "old_endpoint_load": old_load,
        }
    return result


def role_order(slots: list[int], roles: dict[int, dict[str, float]], key: str,
               reverse: bool) -> list[int]:
    return sorted(
        slots,
        key=lambda slot: (roles[slot][key], slot),
        reverse=bool(reverse),
    )


def remap_changed_records(base_records: list[dict], structural_records: list[dict],
                          row_classes: list[int], roles: dict[int, dict[str, float]],
                          key: str | None, reverse: bool) -> tuple[list[dict], dict]:
    """Move the same per-class changed pieces to role-selected C0 slots."""
    records = [copy.deepcopy(record) for record in base_records]
    selected = []
    classes = sorted(set(int(v) for v in row_classes))
    for cls in classes:
        class_slots = [i for i, value in enumerate(row_classes) if int(value) == cls]
        changed = [r for r in structural_records
                   if r.get("changed") and int(r["class"]) == cls]
        if not changed or key is None:
            selected.append({"class": cls, "changed": len(changed),
                             "target_slots": [int(r.get("slot", -1)) for r in changed],
                             "mode": "matched"})
            continue
        # Keep the same proposal multiset in every arm.  The deterministic
        # order prevents source-piece order from becoming a hidden variable.
        proposals = sorted(
            (copy.deepcopy(r) for r in changed),
            key=lambda r: (int(r.get("mass", 0)),
                           tuple(int(v) for v in r.get("source_positions", []))),
        )
        targets = role_order(class_slots, roles, key, reverse)[:len(proposals)]
        for target, proposal in zip(targets, proposals):
            proposal["target_slot"] = int(target)
            records[int(target)] = proposal
        selected.append({
            "class": cls,
            "changed": len(changed),
            "target_slots": [int(v) for v in targets],
            "mode": f"{key}_{'high' if reverse else 'low'}",
        })
    return records, {"mode": "matched" if key is None else f"{key}_{'high' if reverse else 'low'}",
                     "classes": selected}


def role_matching_records(data: dict, ids: torch.Tensor, q: torch.Tensor,
                          row_classes: list[int], structural_records: list[dict],
                          c0_block: dict, local_roles: dict[int, dict[str, float]],
                          base_records: list[dict]) -> tuple[list[dict], dict]:
    """Match the same proposals to all same-class slots with a fixed cost."""
    joint_source = B.joint_points(data["x"][ids].float(),
                                  data["text_emb"][ids].float())
    slot_roles = torch.tensor(
        [[local_roles[s][key] for key in ("degree", "cross_degree",
                                          "new_new_degree", "old_neighbor_ratio",
                                          "old_endpoint_load")]
         for s in range(len(row_classes))], dtype=torch.float32
    )
    role_mean = slot_roles.mean(0)
    role_scale = slot_roles.std(0, unbiased=False).clamp_min(1e-6)
    normalized_roles = (slot_roles - role_mean) / role_scale
    records = [copy.deepcopy(record) for record in base_records]
    classes = sorted(set(int(v) for v in row_classes))
    audit_classes = []
    for cls in classes:
        class_slots = [i for i, value in enumerate(row_classes) if int(value) == cls]
        changed = [r for r in structural_records
                   if r.get("changed") and int(r["class"]) == cls]
        if not changed:
            continue
        proposals = sorted(
            (copy.deepcopy(r) for r in changed),
            key=lambda r: (int(r.get("mass", 0)),
                           tuple(int(v) for v in r.get("source_positions", []))),
        )
        proposal_centers = torch.stack([
            joint_source[torch.tensor(r["source_positions"], dtype=torch.long)].mean(0)
            for r in proposals
        ])
        target_centers = torch.stack([
            joint_source[torch.where(q == int(slot))[0]].mean(0)
            for slot in class_slots
        ])
        proposal_roles = []
        for record in proposals:
            source_slots = q[torch.tensor(record["source_positions"], dtype=torch.long)]
            proposal_roles.append(normalized_roles[source_slots].mean(0))
        proposal_roles = torch.stack(proposal_roles)
        sem_left = F.normalize(proposal_centers.float(), dim=1)
        sem_right = F.normalize(target_centers.float(), dim=1)
        semantic_cost = torch.cdist(sem_left, sem_right).square()
        role_cost = torch.cdist(proposal_roles, normalized_roles[class_slots]).square()
        target_mass = c0_block["mass"][class_slots].float().clamp_min(1e-6)
        proposal_mass = torch.tensor([max(1, int(r["mass"])) for r in proposals],
                                     dtype=torch.float32)
        mass_cost = torch.abs(torch.log((proposal_mass[:, None] + 1.0)
                                        / (target_mass[None, :] + 1.0)))
        cost = semantic_cost + 0.50 * role_cost + 0.10 * mass_cost
        proposal_order, target_order = linear_sum_assignment(cost.detach().cpu().numpy())
        targets = [class_slots[int(i)] for i in target_order]
        replacements = []
        for proposal_index, target_slot in zip(proposal_order, targets):
            proposal = copy.deepcopy(proposals[int(proposal_index)])
            proposal["target_slot"] = int(target_slot)
            records[int(target_slot)] = proposal
            replacements.append({
                "proposal_index": int(proposal_index),
                "target_slot": int(target_slot),
                "semantic_cost": float(semantic_cost[int(proposal_index), class_slots.index(target_slot)]),
                "role_cost": float(role_cost[int(proposal_index), class_slots.index(target_slot)]),
                "mass_cost": float(mass_cost[int(proposal_index), class_slots.index(target_slot)]),
            })
        audit_classes.append({
            "class": cls,
            "candidate_slots": [int(v) for v in class_slots],
            "target_slots": [int(v) for v in targets],
            "replacements": replacements,
            "cost_weights": {"semantic": 1.0, "role": 0.50, "mass": 0.10},
        })
    return records, {"mode": "role_match", "classes": audit_classes,
                     "role_features": ["degree", "cross_degree", "new_new_degree",
                                        "old_neighbor_ratio", "old_endpoint_load"]}


def build_block(data: dict, ids: torch.Tensor, c0: dict, parent: dict,
                q: torch.Tensor, stage: int, device: str, role_mode: str,
                roles: dict[int, dict[str, float]]) -> tuple[dict, dict]:
    old_n = len(parent["x"])
    new_rows = int(len(c0["x"]) - old_n)
    c0_block = {
        "x": c0["x"][old_n:].clone(),
        "text_emb": c0["text_emb"][old_n:].clone(),
        "texts": list(c0["texts"][old_n:]),
        "mass": c0["mass"][old_n:].clone(),
        "depth": c0["depth"][old_n:].clone(),
        "stage": c0["stage"][old_n:].clone(),
    }
    row_labels = B.class_rows_from_c0(q, data["y"][ids])
    local_roles = {slot: roles[old_n + slot] for slot in range(new_rows)}
    structural_block, structural_audit = B.build_two_prototype_block(
        data, ids, q, row_labels, stage, device, c0_block
    )
    structural_records = structural_audit["records"]
    base_records = C.records_from_assignment(q, new_rows)
    if role_mode == "matched":
        records = structural_records
        remap_audit = {"mode": "matched", "classes": []}
    elif role_mode == "role_match":
        records, remap_audit = role_matching_records(
            data, ids, q, row_labels, structural_records, c0_block,
            local_roles, base_records
        )
    else:
        key, direction = role_mode.rsplit("_", 1)
        if key not in {"degree", "cross_degree"} or direction not in {"low", "high"}:
            raise ValueError(f"unknown role mode: {role_mode}")
        records, remap_audit = remap_changed_records(
            base_records, structural_records, row_labels, local_roles,
            key, direction == "high"
        )
    block = C.materialize_records(
        data, ids, c0_block, records, data["encoder"], output_cap=320
    )
    audit = {
        "operator": "two_prototype_role_diagnostic",
        "role_mode": role_mode,
        "c0_row_classes": [int(v) for v in row_labels],
        "structural_operator": structural_audit["operator"],
        "structural_changed_rows": structural_audit["changed_rows"],
        "remap": remap_audit,
        "roles": {str(old_n + int(slot)): roles[old_n + int(slot)]
                   for slot in range(new_rows)},
        "records": records,
        "text_operator": "top-5 center-ranked snippets, explicit 320 chars, MiniLM re-encode",
        "x_operator": "Mean-X over the same source piece used by text",
        "topology": "sealed C0 copied bitwise",
    }
    return block, audit


def candidate_from_block(c0: dict, parent: dict, block: dict) -> dict:
    old_n = len(parent["x"])
    memory = J.T.clone_memory(c0)
    for key in ("x", "text_emb", "texts", "mass", "depth", "stage"):
        if isinstance(memory[key], torch.Tensor):
            memory[key][old_n:] = block[key]
        else:
            memory[key] = memory[key][:old_n] + list(block[key])
    if not torch.equal(memory["edge_index"], c0["edge_index"]):
        raise AssertionError("role diagnostic changed C0 topology")
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if not torch.equal(memory[key][:old_n], parent[key]):
            raise AssertionError("role diagnostic changed old prefix")
    if memory["texts"][:old_n] != parent["texts"]:
        raise AssertionError("role diagnostic changed old text prefix")
    return memory


def paired(candidate: dict, reference: dict) -> dict:
    return C.paired_delta(candidate, reference)


def run(args: argparse.Namespace) -> None:
    source = Path(args.source_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = V1.load_data(args.dataset, args.minilm, args.device, Path(args.cache_root))
    ids, budgets = R11.D.set_stream(data, args.order)
    stage = int(args.stage)
    cell = source / args.dataset / args.order
    parent = load(cell / f"c0_s{stage - 1}_memory.pt")
    c0 = load(cell / f"c0_s{stage}_memory.pt")
    old_n = len(parent["x"])
    q = R11.D.current_assignment(data, ids[stage], c0, old_n, stage, args.device)
    role_values = node_roles(c0, old_n)
    arms = ("sealed_c0", "c0_320", "matched", "role_match", "degree_low", "degree_high",
            "cross_degree_low", "cross_degree_high")
    new_rows = int(len(c0["x"]) - old_n)
    c0_block = {
        "x": c0["x"][old_n:].clone(),
        "text_emb": c0["text_emb"][old_n:].clone(),
        "texts": list(c0["texts"][old_n:]),
        "mass": c0["mass"][old_n:].clone(),
        "depth": c0["depth"][old_n:].clone(),
        "stage": c0["stage"][old_n:].clone(),
    }
    base_records = C.records_from_assignment(q, new_rows)
    c0_320_block = C.materialize_records(
        data, ids[stage], c0_block, base_records, data["encoder"], output_cap=320
    )
    memories = {
        "sealed_c0": c0,
        "c0_320": candidate_from_block(c0, parent, c0_320_block),
    }
    audits = {
        "sealed_c0": {"memory_sha256": J.T.memory_hash(c0),
                      "same_c0_topology": True, "same_old_prefix": True},
        "c0_320": {"memory_sha256": J.T.memory_hash(memories["c0_320"]),
                   "same_c0_topology": True, "same_old_prefix": True,
                   "text_cap": 320, "records": base_records},
    }
    for arm in arms[2:]:
        block, audit = build_block(
            data, ids[stage], c0, parent, q, stage, args.device, arm,
            role_values
        )
        memory = candidate_from_block(c0, parent, block)
        memories[arm] = memory
        audits[arm] = {
            "memory_sha256": J.T.memory_hash(memory),
            "same_c0_topology": True,
            "same_old_prefix": True,
            "build": audit,
        }
        torch.save(memory, out / f"{arm}_memory.pt")
    torch.save(c0, out / "sealed_c0_memory.pt")
    torch.save(memories["c0_320"], out / "c0_320_memory.pt")

    seeds = [int(v) for v in args.report_seeds]
    reports = {}
    for arm, memory in memories.items():
        reports[arm] = V1.report_candidate(memory, data, stage, seeds, args.device)
        row_classes = (audits["matched"]["build"]["c0_row_classes"]
                       if "build" in audits["matched"]
                       else B.class_rows_from_c0(q, data["y"][ids[stage]]))
        reports[arm]["margin"] = B.margin_report(
            memory, data, stage, old_n, row_classes, seeds, args.device
        )
    paired_results = {}
    for arm in arms[2:]:
        paired_results[f"{arm}_vs_c0_320"] = paired(reports[arm], reports["c0_320"])
        paired_results[f"{arm}_vs_matched"] = paired(reports[arm], reports["matched"])
    paired_results["sealed_c0_vs_c0_320"] = paired(
        reports["sealed_c0"], reports["c0_320"]
    )
    protocol = {
        "name": "tgc-two-prototype-role-diagnostic-v1",
        "dataset": args.dataset, "order": args.order, "stage": stage,
        "arms": list(arms), "role_modes": list(arms[2:]),
        "source_root": str(source), "old_rows": old_n,
        "new_rows": int(len(c0["x"]) - old_n), "budget": int(budgets[stage]),
        "text_cap": 320, "text_sources": 5,
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "report_seeds": seeds,
        "report_labels_used_for_selection": False,
        "historical_raw_reads": 0,
        "topology": "C0 edge_index copied bitwise for every arm",
        "role_definition": "C0 new-slot degree and direct old-neighbor degree",
    }
    save_json(out / "protocol.json", protocol)
    save_json(out / "build_audit.json", {
        "protocol": protocol,
        "parent_memory_sha256": J.T.memory_hash(parent),
        "sealed_c0_memory_sha256": J.T.memory_hash(c0),
        "audits": audits,
    })
    save_json(out / "results.json", {
        "protocol": protocol, "audits": audits,
        "reports": reports, "paired": paired_results,
    })
    print(json.dumps({"event": "complete", "dataset": args.dataset,
                      "order": args.order, "stage": stage,
                      "paired": paired_results}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report-seeds", nargs="+", type=int, default=list(range(150, 156)))
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)
    print("TGC_TWO_PROTOTYPE_ROLE_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
