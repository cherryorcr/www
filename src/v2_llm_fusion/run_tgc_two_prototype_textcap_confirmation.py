"""Confirm the earlier two-prototype gain under matched text-cap controls.

The old high-score artifact used a legacy concatenation path whose nominal
320-character budget could produce 321--324 characters after joining five
64-character snippets.  This runner makes that behavior explicit and compares
it with a strict 320-character construction.  C0 and two-prototype memories
are rebuilt from the same sealed source assignment for each cap, so text
length is not confounded with the structural split/merge intervention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

import run_tgc_joint_student_response as V1
import run_tgc_r11_c0_anchor as R11
import run_tgc_two_prototype_smoke as B
import tgc_shared_assignment as J


SOURCE_SELECTION_CHARS = 320
TEXT_SOURCES = 5


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def clone_block(block: dict) -> dict:
    return {
        "x": block["x"].clone(),
        "text_emb": block["text_emb"].clone(),
        "texts": list(block["texts"]),
        "mass": block["mass"].clone(),
        "depth": block["depth"].clone(),
        "stage": block["stage"].clone(),
    }


def materialize_text(joint_source: torch.Tensor, members: torch.Tensor,
                     source_texts: list[str], encoder,
                     output_cap: int) -> tuple[str, torch.Tensor]:
    """Use the legacy 320 selection budget with an explicit output cap."""
    center = joint_source[members].mean(0, keepdim=True)
    distance = torch.cdist(joint_source[members], center).flatten()
    count = min(TEXT_SOURCES, len(members))
    chosen_local = torch.argsort(distance, stable=True)[:count]
    chosen = members[chosen_local]
    per = max(48, SOURCE_SELECTION_CHARS // max(1, len(chosen)))
    text = " ".join(str(source_texts[int(i)])[:per] for i in chosen)
    text = text[:int(output_cap)]
    return text, encoder.encode([text])[0].cpu()


def materialize_records(data: dict, ids: torch.Tensor,
                        c0_block: dict, records: list[dict],
                        encoder, output_cap: int) -> dict:
    """Materialize a block from source records under one text-cap policy."""
    block = clone_block(c0_block)
    source_x = data["x"][ids].float()
    source_text = data["text_emb"][ids].float()
    joint_source = B.joint_points(source_x, source_text)
    source_texts = [data["texts"][int(i)] for i in ids]
    for slot, record in enumerate(records):
        members = torch.tensor(record["source_positions"], dtype=torch.long)
        if len(members) == 0:
            raise AssertionError("empty source record")
        block["x"][slot] = source_x[members].mean(0).to(block["x"].dtype)
        text, embedding = materialize_text(
            joint_source, members, source_texts, encoder, output_cap
        )
        block["texts"][slot] = text
        block["text_emb"][slot] = embedding.to(block["text_emb"].dtype)
        block["mass"][slot] = float(len(members))
    return block


def records_from_assignment(q: torch.Tensor, rows: int) -> list[dict]:
    records = []
    for slot in range(int(rows)):
        members = torch.where(q == int(slot))[0]
        records.append({
            "source_positions": [int(v) for v in members.tolist()],
            "mass": int(len(members)),
            "changed": False,
        })
    return records


def paired_delta(candidate: dict, reference: dict) -> dict:
    values = [
        float(c["cumulative_macro_f1"] - r["cumulative_macro_f1"])
        for c, r in zip(candidate["per_student"], reference["per_student"])
    ]
    return {
        "per_seed_pp": values,
        "mean_pp": float(statistics.mean(values)),
        "std_pp": float(statistics.pstdev(values)),
        "median_pp": float(statistics.median(values)),
        "min_pp": float(min(values)),
        "max_pp": float(max(values)),
        "positive_seeds": int(sum(value > 0 for value in values)),
        "seed_count": len(values),
    }


def run(args: argparse.Namespace) -> None:
    source = Path(args.source_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = V1.load_data(args.dataset, args.minilm, args.device, Path(args.cache_root))
    ids, budgets = R11.D.set_stream(data, args.order)
    stage = int(args.stage)
    cell = source / args.dataset / args.order
    parent = B.load(cell / f"c0_s{stage - 1}_memory.pt")
    c0 = B.load(cell / f"c0_s{stage}_memory.pt")
    old_n = len(parent["x"])
    q = R11.D.current_assignment(data, ids[stage], c0, old_n, stage, args.device)
    new_rows = int(len(c0["x"]) - old_n)
    c0_block = {key: value[old_n:].clone() if isinstance(value, torch.Tensor)
                else list(value[old_n:])
                for key, value in {
                    "x": c0["x"], "text_emb": c0["text_emb"],
                    "texts": c0["texts"], "mass": c0["mass"],
                    "depth": c0["depth"], "stage": c0["stage"],
                }.items()}
    c0_records = records_from_assignment(q, new_rows)

    # The structural records are generated once and reused for both text caps.
    structural_block, structural_audit = B.build_two_prototype_block(
        data, ids[stage], q, B.class_rows_from_c0(q, data["y"][ids[stage]]),
        stage, args.device, c0_block
    )
    structural_records = structural_audit["records"]
    memories = {"sealed_c0": c0}
    build_audit = {
        "dataset": args.dataset,
        "order": args.order,
        "stage": stage,
        "source_root": str(source),
        "parent_memory_sha256": J.T.memory_hash(parent),
        "sealed_c0_memory_sha256": J.T.memory_hash(c0),
        "old_rows": old_n,
        "new_rows": new_rows,
        "budget": int(budgets[stage]),
        "historical_raw_reads": 0,
        "report_labels_used_for_construction": False,
        "selection": "same sealed C0 source assignment for all arms",
        "structural_operator": structural_audit["operator"],
        "structural_audit": structural_audit,
        "text_policy": {
            "source_selection_chars": SOURCE_SELECTION_CHARS,
            "text_sources": TEXT_SOURCES,
            "caps": [int(args.strict_cap), int(args.legacy_cap)],
            "legacy_cap_interpretation": "320 source budget, explicit 324 output cap",
        },
    }
    for label, cap, records in (
        ("c0_320", int(args.strict_cap), c0_records),
        ("two_320", int(args.strict_cap), structural_records),
        ("c0_324", int(args.legacy_cap), c0_records),
        ("two_324", int(args.legacy_cap), structural_records),
    ):
        block = materialize_records(data, ids[stage], c0_block, records,
                                    data["encoder"], cap)
        memory = B.candidate_from_block(c0, parent, block)
        if not torch.equal(memory["edge_index"], c0["edge_index"]):
            raise AssertionError(f"{label} changed C0 topology")
        if not torch.equal(memory["x"][:old_n], parent["x"]):
            raise AssertionError(f"{label} changed old prefix")
        if memory["texts"][:old_n] != parent["texts"]:
            raise AssertionError(f"{label} changed old text prefix")
        memories[label] = memory
        build_audit[label] = {
            "memory_sha256": J.T.memory_hash(memory),
            "text_lengths": [len(str(v)) for v in memory["texts"][old_n:]],
            "text_max": max(len(str(v)) for v in memory["texts"][old_n:]),
            "same_c0_topology": True,
            "same_old_prefix": True,
            "records": records,
        }

    seeds = [int(v) for v in args.report_seeds]
    reports = {}
    for name, memory in memories.items():
        reports[name] = V1.report_candidate(memory, data, stage, seeds, args.device)
        reports[name]["margin"] = B.margin_report(
            memory, data, stage, old_n, structural_audit["c0_row_classes"],
            seeds, args.device
        )

    paired = {
        "sealed_c0_vs_c0_320": paired_delta(reports["c0_320"], reports["sealed_c0"]),
        "two_320_vs_c0_320": paired_delta(reports["two_320"], reports["c0_320"]),
        "two_324_vs_c0_324": paired_delta(reports["two_324"], reports["c0_324"]),
        "two_324_vs_two_320": paired_delta(reports["two_324"], reports["two_320"]),
        "c0_324_vs_c0_320": paired_delta(reports["c0_324"], reports["c0_320"]),
    }
    save_json(out / "protocol.json", {
        "name": "tgc-two-prototype-textcap-confirmation-v1",
        "dataset": args.dataset,
        "order": args.order,
        "stage": stage,
        "arms": list(memories),
        "source_selection_chars": SOURCE_SELECTION_CHARS,
        "text_sources": TEXT_SOURCES,
        "strict_cap": int(args.strict_cap),
        "legacy_explicit_cap": int(args.legacy_cap),
        "structural_operator": structural_audit["operator"],
        "topology": "sealed C0 copied bitwise",
        "old_prefix": "sealed C0 parent copied bitwise",
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "report_seeds": seeds,
        "report_labels_used_for_selection": False,
    })
    for name, memory in memories.items():
        torch.save(memory, out / f"{name}_memory.pt")
    save_json(out / "build_audit.json", build_audit)
    save_json(out / "results.json", {
        "protocol": json.loads((out / "protocol.json").read_text(encoding="utf-8")),
        "audit": build_audit,
        "reports": reports,
        "paired": paired,
    })
    print(json.dumps({
        "event": "complete",
        "dataset": args.dataset,
        "order": args.order,
        "stage": stage,
        "paired": paired,
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict-cap", type=int, default=320)
    parser.add_argument("--legacy-cap", type=int, default=324)
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report-seeds", nargs="+", type=int, default=list(range(150, 156)))
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)
    print("TGC_TWO_PROTOTYPE_TEXTCAP_CONFIRMATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
