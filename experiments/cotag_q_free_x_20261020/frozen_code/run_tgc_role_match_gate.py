"""Evaluate a bounded synthetic-margin fallback for role-aware matching.

The gate is decided with independent Student seeds on synthetic memory only.
It never reads report labels or report F1.  The selected memory is then
evaluated with a disjoint report-seed group.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_tgc_joint_student_response as V1
import run_tgc_r11_c0_anchor as R11
import run_tgc_two_prototype_role_diagnostic as D
import run_tgc_two_prototype_smoke as B
import run_tgc_two_prototype_textcap_confirmation as C
import tgc_shared_assignment as J


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def synthetic_margin_only(memory: dict, data: dict, row_classes: list[int],
                          seeds: list[int], device: str) -> dict:
    adj = V1.U.G.norm_adj(memory["edge_index"], len(memory["x"]), device)
    prompt = data["prompt_emb"].to(device)
    row_labels = torch.tensor(row_classes, dtype=torch.long)
    values = []
    for seed in seeds:
        steps = V1.U.protocol_v3.student_steps(len(memory["x"]))
        model = V1.U.G.train_student(
            V1.U.G.memory_to_trainable(memory), int(seed), device, steps,
            batch=V1.U.protocol_v3.STUDENT.batch
        )
        with torch.no_grad():
            logits = model.logits(memory["x"].to(device), adj, prompt).cpu()
        new_ids = torch.arange(len(memory["x"]) - len(row_classes),
                               len(memory["x"]))
        correct = logits[new_ids, row_labels]
        other = logits[new_ids].clone()
        other[torch.arange(len(new_ids)), row_labels] = -torch.inf
        values.append(float((correct - other.max(1).values).mean()))
        del model
    return {
        "gate_seeds": [int(v) for v in seeds],
        "per_seed_synthetic_margin": values,
        "synthetic_margin_mean": float(sum(values) / len(values)),
    }


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
    new_rows = int(len(c0["x"]) - old_n)
    role_values = D.node_roles(c0, old_n)
    c0_block = {
        "x": c0["x"][old_n:].clone(),
        "text_emb": c0["text_emb"][old_n:].clone(),
        "texts": list(c0["texts"][old_n:]),
        "mass": c0["mass"][old_n:].clone(),
        "depth": c0["depth"][old_n:].clone(),
        "stage": c0["stage"][old_n:].clone(),
    }
    base_records = C.records_from_assignment(q, new_rows)
    c0_320 = D.candidate_from_block(
        c0, parent,
        C.materialize_records(data, ids[stage], c0_block, base_records,
                              data["encoder"], output_cap=320)
    )
    row_classes = B.class_rows_from_c0(q, data["y"][ids[stage]])
    memories = {"c0_320": c0_320}
    audits = {"c0_320": {"memory_sha256": J.T.memory_hash(c0_320)}}
    for arm in ("matched", "role_match"):
        block, audit = D.build_block(
            data, ids[stage], c0, parent, q, stage, args.device, arm,
            role_values
        )
        memories[arm] = D.candidate_from_block(c0, parent, block)
        audits[arm] = {
            "memory_sha256": J.T.memory_hash(memories[arm]),
            "build": audit,
        }
    gate = {
        arm: synthetic_margin_only(memories[arm], data, row_classes,
                                   [int(v) for v in args.gate_seeds], args.device)
        for arm in memories
    }
    score_order = sorted(
        memories,
        key=lambda arm: gate[arm]["synthetic_margin_mean"],
        reverse=True,
    )
    selected = score_order[0]
    reports = {
        arm: V1.report_candidate(memories[arm], data, stage,
                                 [int(v) for v in args.report_seeds], args.device)
        for arm in memories
    }
    selected_report = reports[selected]
    paired = {
        f"{arm}_vs_c0_320": C.paired_delta(reports[arm], reports["c0_320"])
        for arm in ("matched", "role_match")
    }
    paired["selected_vs_c0_320"] = C.paired_delta(selected_report,
                                                    reports["c0_320"])
    protocol = {
        "name": "tgc-role-match-synthetic-margin-gate-v1",
        "dataset": args.dataset, "order": args.order, "stage": stage,
        "arms": list(memories), "selected_arm": selected,
        "gate_seeds": [int(v) for v in args.gate_seeds],
        "report_seeds": [int(v) for v in args.report_seeds],
        "selection_signal": "synthetic margin only; no report/test labels",
        "text_cap": 320, "text_sources": 5,
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "topology": "C0 edge_index copied bitwise",
        "historical_raw_reads": 0,
        "report_labels_used_for_selection": False,
        "budget": int(budgets[stage]),
    }
    for arm, memory in memories.items():
        torch.save(memory, out / f"{arm}_memory.pt")
    save_json(out / "protocol.json", protocol)
    save_json(out / "selection.json", {
        "protocol": protocol, "gate": gate, "score_order": score_order,
        "selected": selected, "audits": audits,
    })
    save_json(out / "results.json", {
        "protocol": protocol, "gate": gate, "selection": selected,
        "reports": reports, "paired": paired, "audits": audits,
    })
    print(json.dumps({"event": "complete", "dataset": args.dataset,
                      "order": args.order, "stage": stage,
                      "selected": selected, "gate": gate,
                      "selected_vs_c0_320": paired["selected_vs_c0_320"]}), flush=True)


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
    parser.add_argument("--gate-seeds", nargs="+", type=int, default=[300, 301])
    parser.add_argument("--report-seeds", nargs="+", type=int, default=list(range(150, 156)))
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)
    print("TGC_ROLE_MATCH_GATE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
