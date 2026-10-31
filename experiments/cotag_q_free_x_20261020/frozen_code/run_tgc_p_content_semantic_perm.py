"""Combine P-guided content with conserved semantic slot reassignment.

This runner is deliberately narrower than the old R10/R11 joint operator:

* the sealed R11 anchor025 assignment supplies P weights for a localized
  split/merge proposal that regenerates X/text;
* C0 supplies the complete topology;
* the already sealed semantic/random permutation maps are applied to the
  complete new-block payload, not to individual fields;
* no report-F1 selector and no historical raw-graph read are introduced.

The existing permutation maps are reused rather than re-optimized here so
that this batch tests composition, not a second round of semantic-cost tuning.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import run_tgc_joint_student_response as V1
import run_tgc_r11_c0_anchor as R11
import run_tgc_two_prototype_smoke as Proto
import tgc_r6_experiment as E
import tgc_shared_assignment as J


R, T, D, B = E.R, E.T, E.D, E.B
PAYLOAD_KEYS = ("x", "text_emb", "texts", "mass", "depth", "stage")
ARMS = (
    "c0",
    "r11_content_c0_topology",
    "semantic_perm",
    "random_perm",
    "p_content_two_proto_identity",
    "p_content_semantic_recomputed",
    "p_content_random_perm",
)


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def clone_memory(memory: dict) -> dict:
    return T.clone_memory(memory)


def apply_mapping(block: dict, mapping: dict[int, int]) -> dict:
    n = len(block["x"])
    sources = set(int(k) for k in mapping)
    targets = set(int(v) for v in mapping.values())
    if sources != set(range(n)) or targets != set(range(n)):
        raise AssertionError("payload mapping is not a complete bijection")
    out = {
        "x": torch.empty_like(block["x"]),
        "text_emb": torch.empty_like(block["text_emb"]),
        "texts": [""] * n,
        "mass": torch.empty_like(block["mass"]),
        "depth": torch.empty_like(block["depth"]),
        "stage": torch.empty_like(block["stage"]),
    }
    for source, target in mapping.items():
        source, target = int(source), int(target)
        for key in ("x", "text_emb", "mass", "depth", "stage"):
            out[key][target] = block[key][source]
        out["texts"][target] = block["texts"][source]
    return out


def block(memory: dict, old_n: int) -> dict:
    return {
        key: (memory[key][old_n:].clone()
              if isinstance(memory[key], torch.Tensor)
              else list(memory[key][old_n:]))
        for key in PAYLOAD_KEYS
    }


def replace_block(parent: dict, topology: dict, content: dict) -> dict:
    old_n = len(parent["x"])
    memory = clone_memory(topology)
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        memory[key][old_n:] = content[key]
    memory["texts"] = list(memory["texts"][:old_n]) + list(content["texts"])
    return memory


def mapping_from_audit(path: Path, mode: str) -> dict[int, int]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    mapping = audit["audits"][mode]["mapping"]
    return {int(source): int(target) for source, target in mapping.items()}


def recompute_semantic_mapping(content_block: dict, c0_block: dict,
                               source_audit: dict) -> tuple[dict[int, int], dict]:
    """Recompute semantic placement for the actual P-generated payload.

    The sealed semantic permutation was scored on a different structural
    proposal.  Reusing that map for R11 content would confound composition
    with an obsolete cost matrix, so the same deterministic changed-slot
    pattern is retained while the semantic cost is recomputed from the P
    payload itself.
    """
    semantic = source_audit["audits"]["semantic_perm"]
    proposal_joint = Proto.joint_points(
        content_block["x"].float(), content_block["text_emb"].float())
    target_joint = Proto.joint_points(
        c0_block["x"].float(), c0_block["text_emb"].float())
    mapping: dict[int, int] = {}
    diagnostics = []
    for class_row in semantic["mapping_audit"]["classes"]:
        class_id = int(class_row["class"])
        class_slots = [int(v) for v in class_row["class_slots"]]
        changed_source = [int(v) for v in class_row["changed_source_slots"]]
        if not changed_source:
            for slot in class_slots:
                mapping[slot] = slot
            continue
        left = F.normalize(proposal_joint[changed_source], dim=1)
        right = F.normalize(target_joint[class_slots], dim=1)
        cost = torch.cdist(left, right).square()
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        changed_target = []
        for row, col in zip(rows, cols):
            source = changed_source[int(row)]
            target = class_slots[int(col)]
            mapping[source] = target
            changed_target.append(target)
        remaining_source = [slot for slot in class_slots if slot not in changed_source]
        remaining_target = [slot for slot in class_slots if slot not in changed_target]
        fixed = sorted(set(remaining_source).intersection(remaining_target))
        for slot in fixed:
            mapping[slot] = slot
        residual_source = [slot for slot in remaining_source if slot not in fixed]
        residual_target = [slot for slot in remaining_target if slot not in fixed]
        if len(residual_source) != len(residual_target):
            raise AssertionError("recomputed semantic residual is not square")
        for source, target in zip(residual_source, residual_target):
            mapping[source] = target
        diagnostics.append({
            "class": class_id,
            "changed_source_slots": changed_source,
            "changed_target_slots": changed_target,
            "cost_mean": float(cost.mean()),
            "cost_min": float(cost.min()),
            "cost_max": float(cost.max()),
        })
    n = len(content_block["x"])
    if set(mapping) != set(range(n)) or set(mapping.values()) != set(range(n)):
        raise AssertionError("recomputed semantic mapping is not bijective")
    return mapping, {"mode": "semantic_perm_recomputed_on_p_content",
                     "classes": diagnostics}


def audit_memory(memory: dict, c0: dict, parent: dict, expected_content: dict,
                 mapping: dict[int, int] | None) -> dict:
    old_n = len(parent["x"])
    if len(memory["x"]) != len(c0["x"]):
        raise AssertionError("row count changed")
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if not torch.equal(memory[key][:old_n], parent[key]):
            raise AssertionError(f"old prefix changed: {key}")
    if memory["texts"][:old_n] != parent["texts"]:
        raise AssertionError("old text prefix changed")
    if not torch.equal(memory["edge_index"], c0["edge_index"]):
        raise AssertionError("topology is not exact C0 topology")
    expected = sorted(T.memory_hash({
        key: expected_content[key][i:i + 1] if isinstance(expected_content[key], torch.Tensor)
        else [expected_content[key][i]]
        for key in PAYLOAD_KEYS
    }) for i in range(len(expected_content["x"])))
    actual = sorted(T.memory_hash({
        key: memory[key][old_n + i:old_n + i + 1]
        if isinstance(memory[key], torch.Tensor)
        else [memory[key][old_n + i]]
        for key in PAYLOAD_KEYS
    }) for i in range(len(expected_content["x"])))
    if expected != actual:
        raise AssertionError("new payload multiset changed")
    return {
        "rows": int(len(memory["x"])),
        "old_prefix_preserved": True,
        "same_c0_topology": True,
        "payload_multiset_preserved": True,
        "mapping_bijective": mapping is None or (
            set(mapping) == set(range(len(expected_content["x"]))) and
            set(mapping.values()) == set(range(len(expected_content["x"])))
        ),
    }


def load_mapping_memory(mapping_root: Path, dataset: str, order: str,
                        stage: int, mode: str) -> tuple[dict, dict]:
    cell = mapping_root / f"{dataset}_{order}_s{stage}"
    memory = load(cell / f"{mode}_memory.pt")
    audit = json.loads((cell / "build_audit.json").read_text(encoding="utf-8"))
    return memory, audit


def prepare_memories(args: argparse.Namespace, data: dict, order: str,
                     stage: int) -> tuple[dict[str, dict], dict]:
    r11_cell = Path(args.r11_source) / data["dataset"] / order
    c0 = load(r11_cell / f"c0_s{stage}_memory.pt")
    anchor = load(r11_cell / f"anchor025_s{stage}_memory.pt")
    parent = load(r11_cell / f"c0_s{stage - 1}_memory.pt") if stage > 0 else c0
    if stage == 0:
        return {arm: clone_memory(c0) for arm in ARMS}, {
            "stage": 0,
            "alias": "c0",
            "historical_raw_reads": 0,
        }

    old_n = len(parent["x"])
    r11_content = block(anchor, old_n)
    c0_topology = clone_memory(c0)
    r11_content_c0 = replace_block(parent, c0_topology, r11_content)
    ids, _ = R11.D.set_stream(data, order)
    q = R11.D.current_assignment(data, ids[stage], c0, old_n, stage, args.device)
    p_content = load(r11_cell / f"anchor025_s{stage}_assignment.pt").float()
    c0_block = block(c0, old_n)
    row_classes = Proto.class_rows_from_c0(q, data["y"][ids[stage]])
    p_structural_block, p_structural_audit = Proto.build_two_prototype_block(
        data, ids[stage], q, row_classes, stage, args.device, c0_block,
        content_assignment=p_content,
    )
    semantic_memory, semantic_audit = load_mapping_memory(
        Path(args.mapping_root_stage1 if stage == 1 else args.mapping_root_stage2),
        data["dataset"], order, stage, "semantic_perm")
    random_memory, random_audit = load_mapping_memory(
        Path(args.mapping_root_stage1 if stage == 1 else args.mapping_root_stage2),
        data["dataset"], order, stage, "random_perm")
    if not torch.equal(semantic_memory["edge_index"], c0["edge_index"]):
        raise AssertionError("sealed semantic topology does not match C0")
    if not torch.equal(random_memory["edge_index"], c0["edge_index"]):
        raise AssertionError("sealed random topology does not match C0")
    semantic_map = mapping_from_audit(
        Path(args.mapping_root_stage1 if stage == 1 else args.mapping_root_stage2)
        / f"{data['dataset']}_{order}_s{stage}" / "build_audit.json",
        "semantic_perm",
    )
    random_map = mapping_from_audit(
        Path(args.mapping_root_stage1 if stage == 1 else args.mapping_root_stage2)
        / f"{data['dataset']}_{order}_s{stage}" / "build_audit.json",
        "random_perm",
    )
    p_semantic_map, p_semantic_diag = recompute_semantic_mapping(
        p_structural_block, c0_block, semantic_audit)
    p_semantic_block = apply_mapping(p_structural_block, p_semantic_map)
    p_random_block = apply_mapping(p_structural_block, random_map)
    memories = {
        "c0": clone_memory(c0),
        "r11_content_c0_topology": r11_content_c0,
        "semantic_perm": clone_memory(semantic_memory),
        "random_perm": clone_memory(random_memory),
        "p_content_two_proto_identity": replace_block(parent, c0_topology, p_structural_block),
        "p_content_semantic_recomputed": replace_block(parent, c0_topology, p_semantic_block),
        "p_content_random_perm": replace_block(parent, c0_topology, p_random_block),
    }
    # All arms in this batch share the C0 topology and old prefix.  The two
    # sealed permutation memories are retained only as existing controls.
    audits = {
        "stage": stage,
        "parent_memory_sha256": T.memory_hash(parent),
        "c0_memory_sha256": T.memory_hash(c0),
        "r11_anchor_memory_sha256": T.memory_hash(anchor),
        "p_content_assignment_sha256": R.C.tensor_hash({"p": p_content}),
        "p_structural_audit": p_structural_audit,
        "semantic_source_memory_sha256": T.memory_hash(semantic_memory),
        "random_source_memory_sha256": T.memory_hash(random_memory),
        "semantic_mapping": semantic_map,
        "p_content_semantic_mapping": p_semantic_map,
        "p_content_semantic_mapping_diagnostics": p_semantic_diag,
        "random_mapping": random_map,
        "semantic_source_audit": semantic_audit,
        "random_source_audit": random_audit,
        "historical_raw_reads": 0,
        "report_labels_used": False,
        "selection_by_report": False,
        "candidate_audits": {},
    }
    for arm, memory in memories.items():
        expected_content = None
        if arm == "r11_content_c0_topology":
            expected_content = r11_content
        elif arm.startswith("p_content"):
            expected_content = p_structural_block
        if arm == "p_content_semantic_recomputed":
            mapping = p_semantic_map
        elif arm == "p_content_random_perm":
            mapping = random_map
        else:
            mapping = None
        if expected_content is not None:
            audits["candidate_audits"][arm] = audit_memory(
                memory, c0, parent, expected_content, mapping)
        else:
            if not torch.equal(memory["edge_index"], c0["edge_index"]):
                raise AssertionError(f"{arm} topology mismatch")
            audits["candidate_audits"][arm] = {
                "same_c0_topology": True,
                "old_prefix_preserved": True,
            }
    return memories, audits


def run_cell(args: argparse.Namespace, dataset: str, order: str) -> None:
    out = Path(args.output) / dataset / order
    if (out / "results.json").exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    data = V1.load_data(dataset, args.minilm, args.device, Path(args.cache_root))
    ids, budgets = R11.D.set_stream(data, order)
    reports = {}
    all_audits = {}
    for stage in [int(v) for v in args.stages]:
        memories, audits = prepare_memories(args, data, order, stage)
        all_audits[str(stage)] = audits
        for arm, memory in memories.items():
            torch.save(memory, out / f"{arm}_s{stage}_memory.pt")
            if stage == 0 and arm != "c0":
                reports[f"{arm}_s{stage}"] = {"alias": "c0_s0"}
            elif args.build_only:
                reports[f"{arm}_s{stage}"] = {"build_only": True}
            else:
                started = time.perf_counter()
                reports[f"{arm}_s{stage}"] = V1.report_candidate(
                    memory, data, stage, list(args.report_seeds), args.device)
                reports[f"{arm}_s{stage}"]["wall_seconds_total"] = time.perf_counter() - started
        save_json(out / f"s{stage}_build_audit.json", audits)
        print(json.dumps({"event": "stage_complete", "dataset": dataset,
                          "order": order, "stage": stage}), flush=True)
    protocol = {
        "name": "tgc-p-content-conserved-semantic-permutation-v1",
        "dataset": dataset,
        "order": order,
        "stages": [int(v) for v in args.stages],
        "arms": list(ARMS),
        "rate": 0.05,
        "content": "R11 anchor025 P-weighted localized split/merge X/text",
        "topology": "exact C0 edge_index",
        "permutation": "same-class complete payload mapping; semantic cost recomputed on P content",
        "text_budget": "top-5/320-char re-encoded inherited from R11",
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "report_seeds": [int(v) for v in args.report_seeds],
        "historical_raw_reads": 0,
        "selection_by_report": False,
        "test_labels_used_for_construction": False,
        "source": {
            "r11_source": str(args.r11_source),
            "mapping_root_stage1": str(args.mapping_root_stage1),
            "mapping_root_stage2": str(args.mapping_root_stage2),
        },
    }
    save_json(out / "protocol.json", protocol)
    save_json(out / "build_audit.json", all_audits)
    save_json(out / "results.json", {"protocol": protocol,
                                      "audits": all_audits,
                                      "reports": reports})
    print(json.dumps({"event": "complete", "dataset": dataset,
                      "order": order, "output": str(out)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--r11-source", required=True)
    parser.add_argument("--mapping-root-stage1", required=True)
    parser.add_argument("--mapping-root-stage2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stages", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--report-seeds", nargs="+", type=int, default=list(range(150, 156)))
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_cell(args, args.dataset, args.order)


if __name__ == "__main__":
    main()
