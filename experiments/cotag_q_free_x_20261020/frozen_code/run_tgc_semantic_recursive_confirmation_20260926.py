"""Build a true recursive semantic-permutation confirmation chain.

The frozen C0 topology and current-stage structural proposal remain fixed.
At stage 1 the semantic candidate starts from C0_s0; at stage 2 it starts
from the previously materialized semantic candidate, while historical raw
graphs remain inaccessible.  This runner writes a new immutable output root.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

import run_tgc_joint_student_response as V1
import run_tgc_r11_c0_anchor as R11
import run_tgc_role_aware_assignment_20260925 as Role
import run_tgc_two_prototype_smoke as Smoke
import tgc_shared_assignment as Shared
from semantic_payload_transport import apply_payload_permutation


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_from_parent(c0: dict, parent: dict, block: dict) -> dict:
    """Use C0's current topology with the supplied synthetic parent payload."""
    old_n = len(parent["x"])
    new_n = len(block["x"])
    if len(c0["x"]) != old_n + new_n:
        raise AssertionError("C0 and parent/block row counts are incompatible")
    memory = Shared.T.clone_memory(c0)
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        memory[key][:old_n] = parent[key]
        memory[key][old_n:] = block[key]
    memory["texts"] = list(parent["texts"]) + list(block["texts"])
    if not torch.equal(memory["edge_index"], c0["edge_index"]):
        raise AssertionError("recursive semantic candidate changed C0 topology")
    return memory


def row_digest(block: dict, index: int) -> str:
    digest = hashlib.sha256()
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        value = block[key][index].detach().cpu().contiguous()
        digest.update(value.numpy().tobytes())
    digest.update(str(block["texts"][index]).encode("utf-8"))
    return digest.hexdigest()


def assert_recursive_candidate(c0: dict, parent: dict, block: dict,
                               candidate: dict, records: list[dict],
                               perm_records: list[dict], row_classes: list[int]) -> None:
    old_n = len(parent["x"])
    if not torch.equal(candidate["edge_index"], c0["edge_index"]):
        raise AssertionError("topology changed")
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        if not torch.equal(candidate[key][:old_n], parent[key]):
            raise AssertionError(f"recursive parent prefix changed: {key}")
    if candidate["texts"][:old_n] != parent["texts"]:
        raise AssertionError("recursive parent text prefix changed")
    if sorted(row_digest(block, i) for i in range(len(block["x"]))) != \
            sorted(row_digest({key: candidate[key][old_n:] if isinstance(candidate[key], torch.Tensor)
                               else candidate[key][old_n:] for key in ("x", "text_emb", "mass", "depth", "stage", "texts")}, i)
                   for i in range(len(block["x"]))):
        raise AssertionError("semantic assignment did not conserve payload rows")
    source_signature = sorted(
        (int(row_classes[i]), tuple(sorted(int(v) for v in records[i]["source_positions"])),
         int(records[i].get("mass", len(records[i]["source_positions"]))))
        for i in range(len(records)))
    perm_signature = sorted(
        (int(row_classes[i]), tuple(sorted(int(v) for v in perm_records[i]["source_positions"])),
         int(perm_records[i].get("mass", len(perm_records[i]["source_positions"]))))
        for i in range(len(perm_records)))
    if source_signature != perm_signature:
        raise AssertionError("semantic assignment changed source coverage")


def parent_path(args: argparse.Namespace, cell: Path) -> Path:
    if args.stage == 1 or args.parent_arm == "c0":
        return cell / "c0_s0_memory.pt"
    return (Path(args.parent_root) /
            f"{args.dataset}_{args.order}_s{args.stage - 1}" /
            f"{args.parent_arm}_memory.pt")


def run(args: argparse.Namespace) -> None:
    source = Path(args.source_root)
    cell = source / args.dataset / args.order
    c0 = load(cell / f"c0_s{args.stage}_memory.pt")
    parent_file = parent_path(args, cell)
    parent = load(parent_file)
    old_n = len(parent["x"])
    data = V1.load_data(args.dataset, args.minilm, args.device,
                        Path(args.cache_root))
    ids, budgets = R11.D.set_stream(data, args.order)
    q = R11.D.current_assignment(data, ids[args.stage], c0, old_n,
                                 args.stage, args.device)
    row_classes = Smoke.class_rows_from_c0(q, data["y"][ids[args.stage]])
    c0_block = {
        key: (c0[key][old_n:].clone() if isinstance(c0[key], torch.Tensor)
              else list(c0[key][old_n:]))
        for key in Role.PAYLOAD_KEYS
    }
    structural_block, structural_audit = Smoke.build_two_prototype_block(
        data, ids[args.stage], q, row_classes, args.stage, args.device, c0_block
    )
    records = structural_audit["records"]
    out = Path(args.output_root) / f"{args.dataset}_{args.order}_s{args.stage}"
    if (out / "results.json").exists():
        raise FileExistsError(f"immutable output exists: {out}")
    out.mkdir(parents=True, exist_ok=True)

    mapping, mapping_audit = Role.complete_mapping(
        data, ids[args.stage], q, row_classes, records,
        c0, old_n, "semantic_perm", args.random_seed,
    )
    semantic_block = apply_payload_permutation(structural_block, mapping)
    semantic_records = [None] * len(records)
    for source, target in mapping.items():
        semantic_records[target] = copy.deepcopy(records[source])
    if any(record is None for record in semantic_records):
        raise AssertionError("semantic assignment did not cover every row")
    semantic_memory = candidate_from_parent(c0, parent, semantic_block)
    assert_recursive_candidate(c0, parent, structural_block,
                               semantic_memory, records, semantic_records,
                               row_classes)

    torch.save(c0, out / "c0_memory.pt")
    torch.save(semantic_memory, out / "semantic_perm_memory.pt")
    protocol = {
        "name": "tgc-semantic-permutation-recursive-confirmation-v1",
        "dataset": args.dataset,
        "order": args.order,
        "stage": int(args.stage),
        "parent_arm": args.parent_arm if args.stage > 1 else "c0",
        "parent_file": str(parent_file),
        "budget": int(budgets[args.stage]),
        "old_rows": int(old_n),
        "new_rows": int(len(c0["x"]) - old_n),
        "arms": ["c0", "semantic_perm"],
        "topology": "exact current-stage C0 edge_index",
        "content": "frozen two-prototype structural proposal",
        "assignment": "same-class graph-text semantic conserved payload assignment",
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "historical_raw_reads": 0,
        "report_labels_used_for_construction": False,
    }
    audit = {
        "source": str(cell),
        "parent_file": str(parent_file),
        "parent_arm": protocol["parent_arm"],
        "same_c0_topology": True,
        "same_parent_prefix": True,
        "payload_multiset_preserved": True,
        "structural_audit": structural_audit,
        "mapping": {str(k): int(v) for k, v in mapping.items()},
        "mapping_audit": mapping_audit,
        "memory_sha256": {
            "c0": file_sha256(out / "c0_memory.pt"),
            "semantic_perm": file_sha256(out / "semantic_perm_memory.pt"),
        },
    }
    save_json(out / "protocol.json", protocol)
    save_json(out / "build_audit.json", audit)
    save_json(out / "results.json", {"protocol": protocol, "audit": audit})
    print(json.dumps({"event": "recursive_complete", "cell": out.name,
                      "parent_arm": protocol["parent_arm"],
                      "output": str(out)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--parent-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parent-arm", default="semantic_perm")
    parser.add_argument("--random-seed", type=int, default=20260926)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)


if __name__ == "__main__":
    main()
