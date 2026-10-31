"""Conserved-payload role permutation diagnostic for the two-prototype operator.

This runner fixes a flaw in the earlier role-matching diagnostic.  It first
constructs one localized split/merge block, then only permutes complete row
payloads inside each class.  No proposal is inserted without moving the row it
displaces, so source coverage, per-class mass, row count, X/text payloads and
the sealed C0 topology remain comparable across arms.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import run_tgc_joint_student_response as V1
import run_tgc_r11_c0_anchor as R11
import run_tgc_two_prototype_smoke as B
import run_tgc_two_prototype_textcap_confirmation as C
import run_tgc_two_prototype_role_diagnostic as D
import tgc_shared_assignment as J


PAYLOAD_KEYS = ("x", "text_emb", "texts", "mass", "depth", "stage")


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def row_digest(block: dict, slot: int) -> str:
    digest = hashlib.sha256()
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        value = block[key][slot]
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(str(value).encode("utf-8"))
    digest.update(str(block["texts"][slot]).encode("utf-8"))
    return digest.hexdigest()


def source_signature(records: list[dict], row_classes: list[int]) -> list[tuple[int, tuple[int, ...], int]]:
    return sorted(
        (int(row_classes[i]), tuple(sorted(int(v) for v in record["source_positions"])),
         int(record.get("mass", len(record["source_positions"]))))
        for i, record in enumerate(records)
    )


def normalize_roles(roles: dict[int, dict[str, float]], slots: list[int]) -> torch.Tensor:
    keys = ("degree", "cross_degree", "new_new_degree", "old_neighbor_ratio",
            "old_endpoint_load")
    value = torch.tensor([[float(roles[slot][key]) for key in keys] for slot in slots],
                         dtype=torch.float32)
    return (value - value.mean(0)) / value.std(0, unbiased=False).clamp_min(1e-6)


def class_assignment(data: dict, ids: torch.Tensor, q: torch.Tensor,
                     structural_records: list[dict], row_classes: list[int],
                     roles: dict[int, dict[str, float]], old_n: int,
                     class_id: int, mode: str, random_seed: int) -> list[tuple[int, int]]:
    """Return (structural source slot, target slot) for changed records."""
    class_slots = [i for i, value in enumerate(row_classes) if int(value) == class_id]
    changed_slots = [i for i in class_slots if structural_records[i].get("changed")]
    if not changed_slots:
        return []
    if mode == "identity":
        return [(slot, slot) for slot in changed_slots]

    source_x = data["x"][ids].float()
    source_text = data["text_emb"][ids].float()
    joint_source = B.joint_points(source_x, source_text)
    proposal_centers = torch.stack([
        joint_source[torch.tensor(structural_records[slot]["source_positions"],
                                   dtype=torch.long)].mean(0)
        for slot in changed_slots
    ])
    target_centers = torch.stack([
        joint_source[torch.where(q == int(slot))[0]].mean(0) for slot in class_slots
    ])
    left = F.normalize(proposal_centers.float(), dim=1)
    right = F.normalize(target_centers.float(), dim=1)
    semantic_cost = torch.cdist(left, right).square()
    if mode == "semantic_perm":
        cost = semantic_cost
    elif mode == "role_perm":
        source_roles = normalize_roles(roles, [old_n + slot for slot in class_slots])
        target_roles = source_roles
        proposal_roles = torch.stack([
            source_roles[class_slots.index(slot)] for slot in changed_slots
        ])
        role_cost = torch.cdist(proposal_roles, target_roles).square()
        proposal_mass = torch.tensor([
            max(1, int(structural_records[slot].get("mass", 1)))
            for slot in changed_slots
        ], dtype=torch.float32)
        target_mass = torch.tensor([
            max(1, int(structural_records[slot].get("mass", 1)))
            for slot in class_slots
        ], dtype=torch.float32)
        mass_cost = torch.abs(torch.log((proposal_mass[:, None] + 1.0)
                                        / (target_mass[None, :] + 1.0)))
        cost = semantic_cost + 0.50 * role_cost + 0.10 * mass_cost
    elif mode == "random_perm":
        rng = random.Random(int(random_seed) + 7919 * int(class_id))
        targets = list(class_slots)
        rng.shuffle(targets)
        return list(zip(changed_slots, targets[:len(changed_slots)]))
    else:
        raise ValueError(f"unknown permutation mode: {mode}")
    proposal_order, target_order = linear_sum_assignment(cost.detach().cpu().numpy())
    return [(changed_slots[int(proposal_index)], class_slots[int(target_index)])
            for proposal_index, target_index in zip(proposal_order, target_order)]


def complete_mapping(row_classes: list[int], structural_records: list[dict],
                     data: dict, ids: torch.Tensor, q: torch.Tensor,
                     roles: dict[int, dict[str, float]], old_n: int,
                     mode: str, random_seed: int) -> tuple[dict[int, int], dict]:
    """Build a class-wise bijection from structural source rows to target slots."""
    mapping: dict[int, int] = {}
    class_audit = []
    for class_id in sorted(set(int(v) for v in row_classes)):
        class_slots = [i for i, value in enumerate(row_classes) if int(value) == class_id]
        changed_pairs = class_assignment(
            data, ids, q, structural_records, row_classes, roles, old_n,
            class_id, mode, random_seed
        )
        source_changed = {source for source, _ in changed_pairs}
        target_changed = {target for _, target in changed_pairs}
        if len(source_changed) != len(target_changed):
            raise AssertionError("changed payload assignment is not injective")
        for source, target in changed_pairs:
            mapping[source] = target

        remaining_source = [slot for slot in class_slots if slot not in source_changed]
        remaining_target = [slot for slot in class_slots if slot not in target_changed]
        # Preserve as many unaffected rows in place as possible.  Any rows
        # displaced by the changed proposals are then paired deterministically.
        fixed = sorted(set(remaining_source).intersection(remaining_target))
        for slot in fixed:
            mapping[slot] = slot
        residual_source = [slot for slot in remaining_source if slot not in fixed]
        residual_target = [slot for slot in remaining_target if slot not in fixed]
        if len(residual_source) != len(residual_target):
            raise AssertionError("residual class mapping is not square")
        for source, target in zip(residual_source, residual_target):
            mapping[source] = target
        class_audit.append({
            "class": int(class_id),
            "class_slots": [int(v) for v in class_slots],
            "changed_source_slots": sorted(int(v) for v in source_changed),
            "changed_target_slots": sorted(int(v) for v in target_changed),
            "moved_payloads": int(sum(source != target for source, target in mapping.items()
                                       if source in class_slots)),
        })
    if set(mapping) != set(range(len(row_classes))):
        raise AssertionError("mapping does not cover every structural row")
    if len(set(mapping.values())) != len(row_classes):
        raise AssertionError("mapping targets are not bijective")
    return mapping, {"mode": mode, "classes": class_audit}


def permute_block(structural_block: dict, structural_records: list[dict],
                  row_classes: list[int], mapping: dict[int, int]) -> tuple[dict, list[dict]]:
    block = {
        "x": structural_block["x"].clone(),
        "text_emb": structural_block["text_emb"].clone(),
        "texts": list(structural_block["texts"]),
        "mass": structural_block["mass"].clone(),
        "depth": structural_block["depth"].clone(),
        "stage": structural_block["stage"].clone(),
    }
    records = [None] * len(structural_records)
    for source, target in mapping.items():
        for key in ("x", "text_emb", "mass", "depth", "stage"):
            block[key][target] = structural_block[key][source]
        block["texts"][target] = structural_block["texts"][source]
        records[target] = copy.deepcopy(structural_records[source])
    if any(record is None for record in records):
        raise AssertionError("permuted records contain an empty slot")
    return block, records


def assert_conserved(structural_block: dict, block: dict,
                     structural_records: list[dict], records: list[dict],
                     row_classes: list[int], c0: dict, parent: dict) -> None:
    if len(block["x"]) != len(structural_block["x"]):
        raise AssertionError("row count changed")
    if sorted(row_digest(structural_block, i) for i in range(len(block["x"]))) != \
            sorted(row_digest(block, i) for i in range(len(block["x"]))):
        raise AssertionError("complete row payload multiset changed")
    if source_signature(structural_records, row_classes) != source_signature(records, row_classes):
        raise AssertionError("source coverage or per-class mass changed")
    if not torch.equal(c0["edge_index"], c0["edge_index"]):
        raise AssertionError("unreachable topology assertion")
    old_n = len(parent["x"])
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if not torch.equal(c0[key][:old_n], parent[key]):
            raise AssertionError(f"C0 old prefix mismatch: {key}")
    if c0["texts"][:old_n] != parent["texts"]:
        raise AssertionError("C0 old text prefix mismatch")


def candidate_from_block(c0: dict, parent: dict, block: dict) -> dict:
    old_n = len(parent["x"])
    memory = J.T.clone_memory(c0)
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        memory[key][old_n:] = block[key]
    memory["texts"] = memory["texts"][:old_n] + list(block["texts"])
    if not torch.equal(memory["edge_index"], c0["edge_index"]):
        raise AssertionError("permutation changed C0 topology")
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if not torch.equal(memory[key][:old_n], parent[key]):
            raise AssertionError("permutation changed old prefix")
    if memory["texts"][:old_n] != parent["texts"]:
        raise AssertionError("permutation changed old text prefix")
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
    row_classes = B.class_rows_from_c0(q, data["y"][ids[stage]])
    role_values = D.node_roles(c0, old_n)
    new_rows = int(len(c0["x"]) - old_n)
    c0_block = {
        key: (c0[key][old_n:].clone() if isinstance(c0[key], torch.Tensor)
              else list(c0[key][old_n:]))
        for key in PAYLOAD_KEYS
    }
    c0_records = C.records_from_assignment(q, new_rows)
    c0_320_block = C.materialize_records(
        data, ids[stage], c0_block, c0_records, data["encoder"], output_cap=320
    )
    structural_block, structural_audit = B.build_two_prototype_block(
        data, ids[stage], q, row_classes, stage, args.device, c0_block
    )
    structural_records = structural_audit["records"]
    local_roles = {slot: role_values[old_n + slot] for slot in range(new_rows)}
    # complete_mapping uses block-local slot ids; its role vector is therefore
    # shifted back to the full-memory ids only at lookup time.
    role_values_local = {slot: local_roles[slot] for slot in range(new_rows)}

    memories = {
        "sealed_c0": c0,
        "c0_320": candidate_from_block(c0, parent, c0_320_block),
        "identity": candidate_from_block(c0, parent, structural_block),
    }
    audits = {
        "sealed_c0": {"memory_sha256": J.T.memory_hash(c0)},
        "c0_320": {"memory_sha256": J.T.memory_hash(memories["c0_320"]),
                   "source_records": c0_records},
        "identity": {"memory_sha256": J.T.memory_hash(memories["identity"]),
                      "structural": structural_audit},
    }
    for mode in ("semantic_perm", "role_perm", "random_perm"):
        mapping, mapping_audit = complete_mapping(
            row_classes, structural_records, data, ids[stage], q,
            role_values_local, 0, mode, args.random_seed
        )
        block, records = permute_block(structural_block, structural_records,
                                       row_classes, mapping)
        assert_conserved(structural_block, block, structural_records, records,
                         row_classes, c0, parent)
        memory = candidate_from_block(c0, parent, block)
        memories[mode] = memory
        audits[mode] = {
            "memory_sha256": J.T.memory_hash(memory),
            "mapping": {str(k): int(v) for k, v in mapping.items()},
            "mapping_audit": mapping_audit,
            "records": records,
            "source_coverage_preserved": True,
            "payload_multiset_preserved": True,
            "same_c0_topology": True,
            "same_old_prefix": True,
        }
        torch.save(memory, out / f"{mode}_memory.pt")
    torch.save(memories["sealed_c0"], out / "sealed_c0_memory.pt")
    torch.save(memories["c0_320"], out / "c0_320_memory.pt")
    torch.save(memories["identity"], out / "identity_memory.pt")

    seeds = [int(v) for v in args.report_seeds]
    reports = {
        arm: V1.report_candidate(memory, data, stage, seeds, args.device)
        for arm, memory in memories.items()
    }
    paired_results = {}
    for arm in ("identity", "semantic_perm", "role_perm", "random_perm"):
        paired_results[f"{arm}_vs_c0_320"] = paired(reports[arm], reports["c0_320"])
        paired_results[f"{arm}_vs_identity"] = paired(reports[arm], reports["identity"])
    protocol = {
        "name": "tgc-two-prototype-conserved-role-permutation-v1",
        "dataset": args.dataset, "order": args.order, "stage": stage,
        "arms": list(memories), "report_seeds": seeds,
        "text_cap": 320, "text_sources": 5,
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "historical_raw_reads": 0,
        "topology": "C0 edge_index copied bitwise for every arm",
        "permutation": "complete same-class bijection over structural row payloads",
        "selection_labels_used": False,
        "random_seed": int(args.random_seed),
    }
    save_json(out / "protocol.json", protocol)
    save_json(out / "build_audit.json", {
        "protocol": protocol,
        "parent_memory_sha256": J.T.memory_hash(parent),
        "sealed_c0_memory_sha256": J.T.memory_hash(c0),
        "audits": audits,
    })
    save_json(out / "results.json", {"protocol": protocol,
                                      "audits": audits,
                                      "reports": reports,
                                      "paired": paired_results})
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
    parser.add_argument("--random-seed", type=int, default=20260924)
    parser.add_argument("--report-seeds", nargs="+", type=int, default=list(range(150, 156)))
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)
    print("TGC_TWO_PROTOTYPE_ROLE_PERMUTATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
