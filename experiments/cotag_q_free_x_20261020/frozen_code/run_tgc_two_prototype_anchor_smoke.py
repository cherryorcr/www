"""C0-anchored follow-up for the localized two-prototype construction.

The two-prototype smoke showed a useful but unstable signal: changing a few
new-row contents while retaining the C0 topology helped one Cora condition,
but damaged old-class retention on Citeseer and another Cora order.  This
runner tests a single controlled hypothesis without a report-F1 selector:
the same prototype update should be applied less aggressively to rows with
larger C0 propagation load.

Only the incoming block is changed.  The old prefix, row budget, topology,
student, batch, and steps remain identical to the two-prototype control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_tgc_joint_student_response as V1
import run_tgc_r11_c0_anchor as R11
import run_tgc_two_prototype_smoke as B
import tgc_shared_assignment as J


TEXT_CHARS = 320


def topology_risk(c0: dict, old_n: int) -> dict:
    """Return an auditable degree-based risk profile for the new C0 rows.

    C0's binary topology is the only structural information used here.  A
    row with a larger degree participates in more normalized message updates,
    so changing its content can affect more downstream rows.  The score is
    normalized by the maximum new-row degree within this cell and is not
    learned from report labels.
    """
    edges = c0["edge_index"].long().cpu()
    n = len(c0["x"])
    degree = torch.bincount(edges.flatten(), minlength=n).float()
    new_degree = degree[old_n:]
    scale = new_degree.max().clamp_min(1.0)
    normalized = (new_degree / scale).clamp(0.0, 1.0)
    # Keep a nonzero update even for the most exposed row.  This is an
    # interpretable fixed policy, not an outcome-dependent selector.
    alpha = (0.35 + 0.65 * (1.0 - normalized)).clamp(0.35, 1.0)
    median = new_degree.median()
    gate_alpha = (new_degree <= median).to(torch.float32)
    return {
        "degree": degree,
        "new_degree": new_degree,
        "degree_scale": float(scale),
        "degree_normalized": normalized,
        "degree_anchor_alpha": alpha,
        "degree_gate_median": float(median),
        "degree_gate_alpha": gate_alpha,
        "policy": "alpha=0.35+0.65*(1-degree/max_new_degree)",
        "gate_policy": "alpha=1 for degree<=median(new-degree), else alpha=0",
    }


def mix_text(old_text: str, new_text: str, alpha: float) -> str:
    """Make a deterministic bounded text mixture for one anchored row."""
    alpha = float(max(0.0, min(1.0, alpha)))
    if alpha <= 1e-8:
        return str(old_text)[:TEXT_CHARS]
    if alpha >= 1.0 - 1e-8:
        return str(new_text)[:TEXT_CHARS]
    old_chars = max(0, min(TEXT_CHARS, int(round(TEXT_CHARS * (1.0 - alpha)))))
    new_chars = max(0, TEXT_CHARS - old_chars)
    left = str(old_text)[:old_chars].strip()
    right = str(new_text)[:new_chars].strip()
    if left and right:
        return (left + " " + right)[:TEXT_CHARS]
    return (left or right)[:TEXT_CHARS]


def apply_anchor(candidate: dict, c0: dict, data: dict, old_n: int,
                 changed_rows: list[int], arm: str,
                 risk: dict) -> tuple[dict, dict]:
    """Blend candidate X/text with C0 using a fixed structural policy."""
    out = J.T.clone_memory(candidate)
    if arm == "two_prototype":
        alpha_by_slot = {int(slot): 1.0 for slot in changed_rows}
    elif arm == "anchor050":
        alpha_by_slot = {int(slot): 0.50 for slot in changed_rows}
    elif arm == "anchor025":
        alpha_by_slot = {int(slot): 0.25 for slot in changed_rows}
    elif arm == "degree_anchor":
        alpha_by_slot = {
            int(slot): float(risk["degree_anchor_alpha"][int(slot)])
            for slot in changed_rows
        }
    elif arm == "degree_gate":
        alpha_by_slot = {
            int(slot): float(risk["degree_gate_alpha"][int(slot)])
            for slot in changed_rows
        }
    else:
        raise ValueError(f"unknown anchor arm: {arm}")

    mixed_texts = []
    text_slots = []
    for slot in changed_rows:
        index = old_n + int(slot)
        alpha = alpha_by_slot[int(slot)]
        out["x"][index] = (
            (1.0 - alpha) * c0["x"][index].float()
            + alpha * candidate["x"][index].float()
        ).to(out["x"].dtype)
        mixed_texts.append(mix_text(c0["texts"][index], candidate["texts"][index], alpha))
        text_slots.append(index)

    if mixed_texts:
        encoded = data["encoder"].encode(mixed_texts).cpu()
        for index, text, embedding in zip(text_slots, mixed_texts, encoded):
            out["texts"][index] = text
            out["text_emb"][index] = embedding

    audit = {
        "arm": arm,
        "changed_rows": [int(v) for v in changed_rows],
        "alpha_by_slot": {str(k): float(v) for k, v in alpha_by_slot.items()},
        "x_operator": "(1-alpha)*C0-X + alpha*two-prototype-X",
        "text_operator": "bounded old/new character mixture followed by frozen MiniLM re-encode",
        "topology": "copied bitwise from C0",
        "mass_operator": "two-prototype source mass retained; no budget change",
    }
    return out, audit


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    source = Path(args.source_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = V1.load_data(args.dataset, args.minilm, args.device, Path(args.cache_root))
    ids, budgets = R11.D.set_stream(data, args.order)
    stage = int(args.stage)
    if stage <= 0 or stage >= len(ids):
        raise ValueError("anchor smoke stage must be an incremental stage")
    cell = source / args.dataset / args.order
    parent = B.load(cell / f"c0_s{stage - 1}_memory.pt")
    c0 = B.load(cell / f"c0_s{stage}_memory.pt")
    old_n = len(parent["x"])
    q = R11.D.current_assignment(data, ids[stage], c0, old_n, stage, args.device)
    c0_rows = int(len(c0["x"]) - old_n)
    c0_block = {
        "x": c0["x"][old_n:].clone(),
        "text_emb": c0["text_emb"][old_n:].clone(),
        "texts": list(c0["texts"][old_n:]),
        "mass": c0["mass"][old_n:].clone(),
        "depth": c0["depth"][old_n:].clone(),
        "stage": c0["stage"][old_n:].clone(),
    }
    block, build_audit = B.build_two_prototype_block(
        data, ids[stage], q, B.class_rows_from_c0(q, data["y"][ids[stage]]),
        stage, args.device, c0_block
    )
    candidate = B.candidate_from_block(c0, parent, block)
    changed_rows = [int(v) for v in build_audit["changed_rows"]]
    risk = topology_risk(c0, old_n)
    arms = ("two_prototype", "anchor050", "anchor025", "degree_anchor", "degree_gate")
    memories = {"c0": c0}
    arm_audits = {"c0": {"arm": "c0", "topology": "sealed C0"}}
    for arm in arms:
        memory, audit = apply_anchor(candidate, c0, data, old_n, changed_rows, arm, risk)
        if not torch.equal(memory["edge_index"], c0["edge_index"]):
            raise AssertionError("anchor arm changed C0 topology")
        if not torch.equal(memory["x"][:old_n], parent["x"]):
            raise AssertionError("anchor arm changed old X prefix")
        if memory["texts"][:old_n] != parent["texts"]:
            raise AssertionError("anchor arm changed old text prefix")
        memories[arm] = memory
        arm_audits[arm] = audit

    audit = {
        "dataset": args.dataset,
        "order": args.order,
        "stage": stage,
        "source_root": str(source),
        "parent_memory_sha256": J.T.memory_hash(parent),
        "c0_memory_sha256": J.T.memory_hash(c0),
        "old_rows": old_n,
        "new_rows": c0_rows,
        "total_rows": int(len(c0["x"])),
        "budget": int(budgets[stage]),
        "historical_raw_reads": 0,
        "report_labels_used_for_construction": False,
        "same_c0_topology": all(torch.equal(m["edge_index"], c0["edge_index"])
                                 for m in memories.values()),
        "same_old_prefix": all(torch.equal(m["x"][:old_n], parent["x"])
                                for m in memories.values()),
        "build": build_audit,
        "topology_risk": {
            "degree_scale": risk["degree_scale"],
            "policy": risk["policy"],
            "new_degree": [float(v) for v in risk["new_degree"].tolist()],
            "degree_normalized": [float(v) for v in risk["degree_normalized"].tolist()],
            "degree_anchor_alpha": [float(v) for v in risk["degree_anchor_alpha"].tolist()],
            "degree_gate_median": risk["degree_gate_median"],
            "degree_gate_alpha": [float(v) for v in risk["degree_gate_alpha"].tolist()],
        },
        "arms": arm_audits,
    }
    save_json(out / "protocol.json", {
        "name": "tgc-shared-two-prototype-c0-anchor-smoke-v1",
        "dataset": args.dataset,
        "order": args.order,
        "stage": stage,
        "control": "sealed C0 single-mean content and C0 topology",
        "base_candidate": "localized high-dispersion split + nearest-pair merge with shared X/text assignment",
        "arms": list(("c0",) + arms),
        "fixed": ["total rows", "C0 topology", "old synthetic prefix", "student GCN", "batch", "steps"],
        "anchor_policy": "same row-wise alpha applied to X and bounded old/new text mixture",
        "degree_anchor": risk["policy"],
        "text": "top-5 center-ranked snippets, 320 chars, frozen MiniLM re-encode",
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "report_labels_used_for_selection": False,
        "report_seeds": [int(v) for v in args.report_seeds],
    })
    for name, memory in memories.items():
        torch.save(memory, out / f"{name}_memory.pt")
    save_json(out / "build_audit.json", audit)

    reports = {}
    for name, memory in memories.items():
        reports[name] = V1.report_candidate(
            memory, data, stage, [int(v) for v in args.report_seeds], args.device
        )
        reports[name]["margin"] = B.margin_report(
            memory, data, stage, old_n, build_audit["c0_row_classes"],
            [int(v) for v in args.report_seeds], args.device
        )
    delta = {}
    for name in arms:
        delta[name] = {
            "cumulative_f1_pp": float(reports[name]["cumulative_f1_mean"] - reports["c0"]["cumulative_f1_mean"]),
            "current_f1_pp": float(reports[name]["current_f1_mean"] - reports["c0"]["current_f1_mean"]),
            "old_mean_f1_pp": float(reports[name]["old_mean_f1_mean"] - reports["c0"]["old_mean_f1_mean"]),
            "real_new_margin_pp": float(reports[name]["margin"]["real_new_margin_mean"] - reports["c0"]["margin"]["real_new_margin_mean"]),
            "synthetic_margin_pp": float(reports[name]["margin"]["synthetic_margin_mean"] - reports["c0"]["margin"]["synthetic_margin_mean"]),
        }
    save_json(out / "results.json", {
        "protocol": json.loads((out / "protocol.json").read_text(encoding="utf-8")),
        "audit": audit,
        "reports": reports,
        "delta": delta,
    })
    print(json.dumps({
        "event": "complete",
        "dataset": args.dataset,
        "order": args.order,
        "stage": stage,
        "delta": delta,
    }), flush=True)


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
    parser.add_argument("--report-seeds", nargs="+", type=int, default=[150, 151])
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args)
    print("TGC_TWO_PROTOTYPE_ANCHOR_SMOKE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
