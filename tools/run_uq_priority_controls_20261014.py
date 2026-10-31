"""Small, auditable controls for the Unified-Q-EF priority experiments.

This runner deliberately does not touch the frozen primary tables.  It adds:
1) fixed-X text replacement on the archived primary final memory; and
2) a no-graph-message constructor control using the same exact-rate protocol.

The text controls are final-memory storage-only interventions.  They are not
claimed to be an end-to-end reconstruction of the historical text feedback
path when the archived primary cannot be replayed byte-for-byte.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path("/home/pc/research/TAG_condensation")
FROZEN = ROOT / "experiments/uq_final_evidence_20261012/code"
COMPONENT = ROOT / "experiments/uq_component_ablations_20261014/code"
sys.path.insert(0, str(FROZEN))
sys.path.insert(1, str(COMPONENT))
sys.path.insert(2, str(ROOT / "experiments/uq_native_retry_20261013/code"))
sys.path.insert(3, str(ROOT / "code/v2_llm_fusion"))

import run_unified_q_matrix_20261008 as R  # noqa: E402
import run_uq_final_confirmation_20261012 as E  # noqa: E402
import run_uq_component_ablations_20261014 as C  # noqa: E402


SEEDS = (271006, 271108, 271109)
STUDENT_SEEDS = (150, 151, 152)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_memory(path: Path) -> dict:
    with gzip.open(path, "rb") as stream:
        return torch.load(stream, map_location="cpu", weights_only=False)


def stage_block(memory: dict, ids: torch.Tensor) -> dict:
    ids = ids.long().cpu()
    block = {key: memory[key][ids].clone()
             for key in ("x", "text_emb", "labels", "mass", "depth", "stage")}
    block["texts"] = [memory["texts"][int(i)] for i in ids.tolist()]
    block["edge_index"] = torch.empty((2, 0), dtype=torch.long)
    return block


def replace_payloads(memory: dict, data: dict, ids: list[torch.Tensor], seed: int,
                     mode: str) -> tuple[dict, list[dict]]:
    """Replace only stored text fields while preserving every X/label/budget row."""
    text_emb = memory["text_emb"].clone()
    texts = list(memory["texts"])
    audits = []
    for stage, raw_ids in enumerate(ids):
        slot_ids = torch.where(memory["stage"].long() == int(stage))[0]
        block = stage_block(memory, slot_ids)
        delta = R.L.delta_view(data, raw_ids, stage)
        altered, audit = C.payload_block(
            block, delta, data["encoder"], seed, mode, R)
        text_emb[slot_ids] = altered["text_emb"].float()
        for local, global_id in enumerate(slot_ids.tolist()):
            texts[global_id] = altered["texts"][local]
        audits.append(dict(stage=stage, slots=len(slot_ids), **audit))
    variant = dict(memory, text_emb=text_emb, texts=texts)
    torch.testing.assert_close(variant["x"], memory["x"], rtol=0, atol=0)
    torch.testing.assert_close(variant["labels"], memory["labels"], rtol=0, atol=0)
    torch.testing.assert_close(variant["stage"], memory["stage"], rtol=0, atol=0)
    return variant, audits


def load_large(args, reference: dict) -> tuple[dict, list[torch.Tensor]]:
    groups = R.V.choose_groups(reference, args.order_seed)
    manifest = dict(groups=groups, report_dev_mod=reference.get("report_dev_mod", 3))
    data = R.L.load_large(args.dataset, args.minilm, args.text_cache, args.device, manifest)
    return data, R.L.stage_ids(data)


def primary_record(primary_path: Path) -> dict:
    payload = json.loads(primary_path.read_text(encoding="utf-8"))
    rows = [r for r in payload["rows"] if r["arm"] == "uq_dm_edgefree"]
    rows.sort(key=lambda r: r["stage"])
    if not rows:
        raise ValueError(f"primary rows missing: {primary_path}")
    return dict(protocol=payload["protocol"], final=rows[-1])


def load_or_reconstruct_primary(args, reference: dict, data: dict,
                                ids: list[torch.Tensor], seed: int,
                                primary: dict) -> tuple[dict, str, bool]:
    """Use the archived primary when available; otherwise replay without diagnostics."""
    primary_path = ROOT / "results/uq_final_evidence_20261012/confirmation" \
        / f"{args.dataset}_o{args.order_seed}_c{seed}" / "results.json"
    archive = primary_path.parent / "final_uq_dm_edgefree.pt.gz"
    if archive.exists():
        memory = load_memory(archive)
        source = "archived_primary_memory"
    else:
        groups = R.V.choose_groups(reference, args.order_seed)
        sizes = list(map(len, ids))
        budgets, _ = R.V.choose_budgets(reference, sizes, groups, "fixed_rate", .05)
        cfg = R.q_config("uq_dm_edgefree", seed, 30, 4096)
        parent = None
        for stage, raw_ids in enumerate(ids):
            delta = R.L.delta_view(data, raw_ids, stage)
            new_rows = budgets[stage] - (0 if parent is None else len(parent["x"]))
            labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_rows))
            block, _ = R.UQ.optimize_stage(
                delta, parent, labels, data["encoder"], data["prompt_emb"], cfg,
                args.device, use_boundary=False, use_topology=False)
            parent = R.L.append_edge_free(parent, block)
        memory = parent
        source = "frozen_source_replay_without_diagnostics"
    observed = R.V.memory_hash(memory)
    return memory, source, observed == primary["final"]["memory_hash"]


def run_payload_controls(args, reference: dict, data: dict,
                         ids: list[torch.Tensor], seed: int) -> dict:
    cell = f"{args.dataset}_o{args.order_seed}_c{seed}"
    primary_path = ROOT / "results/uq_final_evidence_20261012/confirmation" / cell / "results.json"
    primary = primary_record(primary_path)
    memory, memory_source, primary_hash_match = load_or_reconstruct_primary(
        args, reference, data, ids, seed, primary)
    replay_scores = (primary["final"]["scores"] if primary_hash_match
                     else E.evaluate(memory, data, len(ids) - 1,
                                     list(STUDENT_SEEDS), args.device))
    output: dict[str, Any] = dict(
        dataset=args.dataset, order_seed=args.order_seed, construction_seed=seed,
        protocol="final-memory fixed-X storage-only payload control",
        primary_source=str(primary_path.relative_to(ROOT)),
        primary_memory_source=memory_source,
        primary_frozen_hash=primary["final"]["memory_hash"],
        primary_hash_match=primary_hash_match,
        primary_memory_hash=R.V.memory_hash(memory),
        x_hash=R.V.hash_tensor(memory["x"]),
        rows=len(memory["x"]),
        controls=[],
    )
    output["controls"].append(dict(mode="primary", scores=replay_scores,
                                    frozen_scores=primary["final"]["scores"],
                                    text_unique_ratio=len(set(memory["texts"])) / len(memory["texts"])))
    for mode in ("uq_class_shared_payload", "uq_random5_payload"):
        variant, audits = replace_payloads(memory, data, ids, seed, mode)
        scores = E.evaluate(variant, data, len(ids) - 1, list(STUDENT_SEEDS), args.device)
        output["controls"].append(dict(
            mode=mode, scores=scores, audits=audits,
            memory_x_hash=R.V.hash_tensor(variant["x"]),
            memory_label_hash=R.V.hash_tensor(variant["labels"]),
            memory_text_hash=R.V.hash_tensor(variant["text_emb"]),
            text_unique_ratio=len(set(variant["texts"])) / len(variant["texts"]),
            x_unchanged=R.V.hash_tensor(variant["x"]) == output["x_hash"],
            labels_unchanged=R.V.hash_tensor(variant["labels"]) == R.V.hash_tensor(memory["labels"]),
            rows_unchanged=len(variant["x"]) == output["rows"],
        ))
    return output


def run_no_graph(args, reference: dict, data: dict,
                 ids: list[torch.Tensor], seed: int) -> dict:
    groups = R.V.choose_groups(reference, args.order_seed)
    sizes = list(map(len, ids))
    budgets, floors = R.V.choose_budgets(reference, sizes, groups, "fixed_rate", .05)
    base = R.q_config("uq_dm_edgefree", seed, 30, 4096)
    # Remove graph messages from both the structured Q initialization and the
    # propagation/distribution feedback.  The downstream Student still uses
    # the same full raw transductive evaluation graph as every other arm.
    cfg = base.__class__(**{**base.__dict__, "message_weight": 0.0,
                            "lambda_propagation": 0.0})
    parent = None
    rows = []
    for stage, raw_ids in enumerate(ids):
        delta = R.L.delta_view(data, raw_ids, stage)
        original_edges = int(delta["edge_index"].shape[1])
        delta["edge_index"] = torch.empty((2, 0), dtype=torch.long)
        delta["boundary_edge_index"] = torch.empty((2, 0), dtype=torch.long)
        new_rows = budgets[stage] - (0 if parent is None else len(parent["x"]))
        labels = R.M.row_labels(R.M.ratio_allocation(delta["y"], new_rows))
        block, audit = R.UQ.optimize_stage(
            delta, parent, labels, data["encoder"], data["prompt_emb"], cfg,
            args.device, use_boundary=False, use_topology=False)
        memory = R.L.append_edge_free(parent, block)
        checks = R.V.validate_memory(memory, parent, budgets[stage])
        checks.update(dict(original_raw_edges=original_edges, constructor_raw_edges=0,
                          message_weight=0.0, lambda_propagation=0.0,
                          graph_message_disabled=True))
        scores = E.evaluate(memory, data, stage, list(STUDENT_SEEDS), args.device)
        rows.append(dict(stage=stage, raw_rows=len(raw_ids), new_rows=new_rows,
                         memory_rows=len(memory["x"]), scores=scores, checks=checks,
                         audit=R.compact_audit(audit)))
        parent = memory
    return dict(dataset=args.dataset, order_seed=args.order_seed, construction_seed=seed,
                protocol="no-graph-message exact-5% constructor control",
                graph_input="raw edge_index replaced by empty graph during construction",
                downstream_graph="common full raw transductive evaluation graph",
                budgets=budgets, class_floor_adjustments=floors, rows=rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("arxiv", "sportsfit", "bookhis"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--minilm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--order-seed", type=int, default=9101)
    parser.add_argument("--construction-seed", type=int, required=True)
    parser.add_argument("--mode", choices=("payload", "no_graph", "both"), default="both")
    args = parser.parse_args()
    torch.set_num_threads(2)
    reference = R.L.read(args.reference)
    data, ids = load_large(args, reference)
    result = dict(dataset=args.dataset, order_seed=args.order_seed,
                  construction_seed=args.construction_seed)
    if args.mode in ("payload", "both"):
        result["payload"] = run_payload_controls(args, reference, data, ids, args.construction_seed)
    if args.mode in ("no_graph", "both"):
        result["no_graph"] = run_no_graph(args, reference, data, ids, args.construction_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(dict(event="complete", output=str(args.output),
                          dataset=args.dataset, seed=args.construction_seed)), flush=True)


if __name__ == "__main__":
    main()
