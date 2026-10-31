"""Small controlled test for a shared two-prototype content construction.

The experiment deliberately removes the response selector.  It compares the
sealed C0 memory with a candidate that changes only the incoming block's X and
text.  The same deterministic source partition is used to build both
modalities, while the C0 topology, total rows, old prefix, and GCN student
protocol stay fixed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

import run_student_compatible_append as SCA
import run_tgc_joint_student_response as V1
import run_tgc_r11_c0_anchor as R11
import tgc_shared_assignment as J


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def joint_points(x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    return torch.cat((F.normalize(x.float(), dim=1), text_emb.float()), dim=1)


def integer_allocate(weights: torch.Tensor, budget: int,
                     capacities: list[int], minimum: int = 0) -> list[int]:
    """Largest-remainder allocation with per-unit capacities."""
    budget = int(budget)
    if budget < 0 or len(capacities) != len(weights):
        raise ValueError("invalid allocation inputs")
    if budget == 0:
        return [0] * len(capacities)
    caps = torch.tensor([int(v) for v in capacities], dtype=torch.long)
    if (caps < 0).any() or int(caps.sum()) < budget:
        raise ValueError("insufficient prototype capacity")
    n = len(capacities)
    minimum = min(int(minimum), budget // max(1, n))
    alloc = torch.minimum(torch.full((n,), minimum, dtype=torch.long), caps)
    weights = weights.float().clamp_min(1e-8)
    while int(alloc.sum()) < budget:
        left = budget - int(alloc.sum())
        active = alloc < caps
        if not bool(active.any()):
            raise ValueError("allocation exhausted capacities")
        raw = left * weights * active / (weights * active).sum()
        step = torch.minimum(torch.floor(raw).long(), caps - alloc)
        if int(step.sum()) == 0:
            eligible = [i for i in range(n) if bool(active[i])]
            eligible.sort(key=lambda i: (-float(raw[i] - torch.floor(raw[i])), i))
            for i in eligible[:left]:
                alloc[i] += 1
        else:
            alloc += step
    return [int(v) for v in alloc.tolist()]


def class_rows_from_c0(q: torch.Tensor, labels: torch.Tensor) -> list[int]:
    classes = J.row_classes(q, labels, int(q.max()) + 1).cpu().tolist()
    return [int(v) for v in classes]


def _subclusters(points: torch.Tensor, count: int, device: str) -> list[torch.Tensor]:
    count = max(1, min(int(count), len(points)))
    if count == 1:
        return [torch.arange(len(points), dtype=torch.long)]
    assignment = SCA.cluster_assign(
        points, torch.ones(len(points), dtype=torch.float32), count, device=device
    ).cpu()
    result = [torch.where(assignment == i)[0] for i in range(count)]
    result = [ids for ids in result if len(ids)]
    if len(result) < count:
        return [torch.arange(len(points), dtype=torch.long)]
    return result


def build_two_prototype_block(data: dict, ids: torch.Tensor, q: torch.Tensor,
                              c0_rows: list[int], stage: int, device: str,
                              c0_block: dict,
                              content_assignment: torch.Tensor | None = None
                              ) -> tuple[dict, dict]:
    """Make a localized fixed-budget split/merge around C0 source groups.

    A broad class-level recluster changes nearly every row while retaining the
    old C0 topology.  This version changes only three row slots per class:
    one high-dispersion C0 group is split into two shared X/text prototypes,
    and the closest pair of other C0 groups is merged to keep the row budget
    unchanged.  Unaffected rows are copied bitwise from C0.
    """
    source_x = data["x"][ids].float()
    source_text = data["text_emb"][ids].float()
    joint_source = joint_points(source_x, source_text)
    source_texts = [data["texts"][int(i)] for i in ids]
    labels = data["y"][ids].cpu()
    row_classes = class_rows_from_c0(q, labels)
    if content_assignment is not None:
        content_assignment = content_assignment.float().cpu()
        if content_assignment.shape != (len(ids), len(c0_rows)):
            raise ValueError("content assignment shape does not match delta/C0 rows")
    started = time.perf_counter()

    block = {
        "x": c0_block["x"].clone(),
        "text_emb": c0_block["text_emb"].clone(),
        "texts": list(c0_block["texts"]),
        "mass": c0_block["mass"].clone(),
        "depth": c0_block["depth"].clone(),
        "stage": c0_block["stage"].clone(),
    }
    records = []
    for slot, cls in enumerate(row_classes):
        member = torch.where(q == int(slot))[0]
        records.append({
            "class": int(cls),
            "source_positions": member.tolist(),
            "source_original_ids": ids[member].tolist(),
            "mass": int(len(member)),
            "changed": False,
        })

    def materialize(member_positions: torch.Tensor,
                    member_weights: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, str, torch.Tensor]:
        local_points = joint_source[member_positions]
        local_text_emb = source_text[member_positions]
        local_texts = [source_texts[int(i)] for i in member_positions]
        if member_weights is not None:
            weights = member_weights.float().clamp_min(0)
            if float(weights.sum()) <= 0:
                weights = torch.ones_like(weights)
            x_value = (weights[:, None] * source_x[member_positions]).sum(0) / weights.sum()
            chosen_order = torch.argsort(weights, descending=True, stable=True)
            chosen = chosen_order[:5]
            per = max(1, 320 // max(1, len(chosen)))
            text_value = " ".join(local_texts[int(i)][:per] for i in chosen)[:320]
            text_value_emb = data["encoder"].encode([text_value])[0]
            return x_value.cpu(), text_value, text_value_emb.cpu()
        local_members = torch.arange(len(member_positions), dtype=torch.long)
        x_value = SCA.synthesize_x(source_x, joint_source, member_positions, operator="mean")
        text_value, text_value_emb = SCA.encode_cluster_text(
            local_points, local_members, local_texts, data["encoder"],
            "concat", 320, 5, source_text_emb=local_text_emb,
            diversity_lambda=0.5, dual_alpha=0.5,
        )
        return x_value.cpu(), text_value, text_value_emb.cpu()

    class_audit = []
    for cls in sorted(set(int(v) for v in row_classes)):
        slots = [i for i, value in enumerate(row_classes) if int(value) == cls]
        stats = []
        for slot in slots:
            member = torch.where(q == int(slot))[0]
            center = joint_source[member].mean(0)
            dispersion = float((joint_source[member] - center).square().sum()
                               / max(1, len(member)))
            stats.append({"slot": slot, "member": member, "center": center,
                          "dispersion": dispersion})
        audit_row = {"class": cls, "candidate_slots": slots, "changed": False}
        if len(stats) < 3:
            audit_row["reason"] = "fewer_than_three_c0_rows"
            class_audit.append(audit_row)
            continue
        split_pool = [item for item in stats if len(item["member"]) >= 4]
        if not split_pool:
            audit_row["reason"] = "no_splittable_c0_row"
            class_audit.append(audit_row)
            continue
        split = max(split_pool, key=lambda item: (item["dispersion"], -item["slot"]))
        others = [item for item in stats if item["slot"] != split["slot"]]
        pairs = []
        for left in range(len(others)):
            for right in range(left + 1, len(others)):
                distance = float((others[left]["center"] - others[right]["center"]).square().sum())
                pairs.append((distance, others[left], others[right]))
        if not pairs:
            audit_row["reason"] = "no_merge_pair"
            class_audit.append(audit_row)
            continue
        _, merge_left, merge_right = min(pairs, key=lambda row: (row[0], row[1]["slot"], row[2]["slot"]))
        split_assign = SCA.cluster_assign(
            joint_source[split["member"]], torch.ones(len(split["member"])), 2, device=device
        ).cpu()
        split_pieces = [torch.where(split_assign == i)[0] for i in range(2)]
        if any(len(piece) == 0 for piece in split_pieces):
            audit_row["reason"] = "empty_split"
            class_audit.append(audit_row)
            continue
        split_pieces = [split["member"][piece] for piece in split_pieces]
        merged = torch.cat((merge_left["member"], merge_right["member"])).sort().values
        proposal_members = split_pieces + [merged]
        proposal_weights = None
        if content_assignment is not None:
            proposal_weights = [
                content_assignment[member, split["slot"]]
                for member in split_pieces
            ]
            proposal_weights.append(
                content_assignment[merged, merge_left["slot"]]
                + content_assignment[merged, merge_right["slot"]]
            )
        target_slots = [split["slot"], merge_left["slot"], merge_right["slot"]]
        target_centers = torch.stack([
            joint_source[torch.where(q == int(slot))[0]].mean(0) for slot in target_slots
        ])
        proposal_centers = torch.stack([joint_source[m].mean(0) for m in proposal_members])
        cost = torch.cdist(target_centers, proposal_centers).square().numpy()
        slot_order, proposal_order = linear_sum_assignment(cost)
        replacements = []
        for target_index, proposal_index in zip(slot_order, proposal_order):
            slot = target_slots[int(target_index)]
            member = proposal_members[int(proposal_index)]
            member_weights = None if proposal_weights is None else proposal_weights[int(proposal_index)]
            x_value, text_value, text_emb_value = materialize(member, member_weights)
            block["x"][slot] = x_value
            block["text_emb"][slot] = text_emb_value
            block["texts"][slot] = text_value
            block["mass"][slot] = float(len(member))
            records[slot] = {
                "class": cls,
                "source_positions": member.tolist(),
                "source_original_ids": ids[member].tolist(),
                "mass": int(len(member)),
                "changed": True,
                "operation": "split" if len(member) < len(merged) else "merge_or_split",
            }
            replacements.append({"slot": slot, "source_rows": int(len(member)),
                                 "proposal_index": int(proposal_index),
                                 "cost": float(cost[int(target_index), int(proposal_index)])})
        audit_row.update({
            "changed": True,
            "split_slot": int(split["slot"]),
            "merge_slots": [int(merge_left["slot"]), int(merge_right["slot"])],
            "split_sizes": [int(len(v)) for v in split_pieces],
            "merged_size": int(len(merged)),
            "alignment_cost_mean": float(cost[slot_order, proposal_order].mean()),
            "alignment_cost_max": float(cost[slot_order, proposal_order].max()),
            "replacements": replacements,
        })
        class_audit.append(audit_row)

    if len(records) != len(c0_rows) or [int(v) for v in row_classes] != [int(r["class"]) for r in records]:
        raise AssertionError("localized two-prototype row contract failed")
    audit = {
        "operator": "localized_shared_two_prototype_split_merge",
        "c0_row_classes": [int(v) for v in row_classes],
        "classes": class_audit,
        "records": records,
        "changed_rows": [i for i, record in enumerate(records) if record["changed"]],
        "rows": int(len(records)),
        "build_seconds": float(time.perf_counter() - started),
        "text_operator": "top-5 center-ranked snippets, 320 chars, MiniLM re-encode",
        "x_operator": ("P-weighted X over the same final source piece used by text"
                        if content_assignment is not None
                        else "Mean-X over the same final source piece used by text"),
        "topology": "copied from C0; no edge construction or edge count change",
        "alignment": "split/merge proposals matched to their three original C0 slots",
    }
    return block, audit


def candidate_from_block(c0: dict, parent: dict, block: dict) -> dict:
    old_n = len(parent["x"])
    candidate = J.T.clone_memory(c0)
    for key in ("x", "text_emb", "texts", "mass", "depth", "stage"):
        if isinstance(candidate[key], torch.Tensor):
            candidate[key][old_n:] = block[key]
        else:
            candidate[key] = candidate[key][:old_n] + list(block[key])
    if not torch.equal(candidate["edge_index"], c0["edge_index"]):
        raise AssertionError("two-prototype candidate changed C0 topology")
    for key in ("x", "text_emb", "mass", "depth", "stage"):
        assert torch.equal(candidate[key][:old_n], parent[key])
    assert candidate["texts"][:old_n] == parent["texts"]
    return candidate


def margin_report(memory: dict, data: dict, stage: int, old_n: int,
                  row_classes: list[int], seeds: list[int], device: str) -> dict:
    """Measure synthetic and real prompt margins with the canonical student."""
    values = []
    prompt = data["prompt_emb"].to(device)
    syn_adj = V1.U.G.norm_adj(memory["edge_index"], len(memory["x"]), device)
    seen = torch.tensor(data["stage_classes"][stage], dtype=torch.long)
    real_ids = torch.where(
        data["report_dev_mask"] & torch.isin(data["y"], seen)
    )[0]
    for seed in seeds:
        steps = V1.U.protocol_v3.student_steps(len(memory["x"]))
        model = V1.U.G.train_student(
            V1.U.G.memory_to_trainable(memory), int(seed), device, steps,
            batch=V1.U.protocol_v3.STUDENT.batch,
        )
        with torch.no_grad():
            syn_logits = model.logits(memory["x"].to(device), syn_adj, prompt).cpu()
            real_logits = model.logits(
                data["x_device"], data["real_adj"], prompt
            ).cpu()
        new_ids = torch.arange(old_n, len(memory["x"]))
        cls = torch.tensor(row_classes, dtype=torch.long)
        correct = syn_logits[new_ids, cls]
        other = syn_logits[new_ids].clone()
        other[torch.arange(len(new_ids)), cls] = -torch.inf
        real_cls = data["y"][real_ids].long()
        real_correct = real_logits[real_ids, real_cls]
        real_other = real_logits[real_ids].clone()
        real_other[torch.arange(len(real_ids)), real_cls] = -torch.inf
        values.append({
            "seed": int(seed),
            "synthetic_margin_mean": float((correct - other.max(1).values).mean()),
            "real_new_margin_mean": float((real_correct - real_other.max(1).values).mean()),
        })
        del model
    return {
        "per_seed": values,
        "synthetic_margin_mean": float(sum(v["synthetic_margin_mean"] for v in values) / len(values)),
        "real_new_margin_mean": float(sum(v["real_new_margin_mean"] for v in values) / len(values)),
    }


def run(args: argparse.Namespace) -> None:
    source = Path(args.source_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = V1.load_data(args.dataset, args.minilm, args.device, Path(args.cache_root))
    ids, budgets = R11.D.set_stream(data, args.order)
    stage = int(args.stage)
    if stage <= 0 or stage >= len(ids):
        raise ValueError("smoke stage must be an incremental stage")
    cell = source / args.dataset / args.order
    parent = load(cell / f"c0_s{stage - 1}_memory.pt")
    c0 = load(cell / f"c0_s{stage}_memory.pt")
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
    block, build_audit = build_two_prototype_block(
        data, ids[stage], q, class_row_targets(q, data["y"][ids[stage]]),
        stage, args.device, c0_block
    )
    candidate = candidate_from_block(c0, parent, block)
    audit = {
        "dataset": args.dataset,
        "order": args.order,
        "stage": stage,
        "source_root": str(source),
        "parent_memory_sha256": J.T.memory_hash(parent),
        "c0_memory_sha256": J.T.memory_hash(c0),
        "candidate_memory_sha256": J.T.memory_hash(candidate),
        "old_rows": old_n,
        "new_rows": c0_rows,
        "total_rows": int(len(c0["x"])),
        "budget": int(budgets[stage]),
        "historical_raw_reads": 0,
        "report_labels_used_for_construction": False,
        "same_c0_topology": bool(torch.equal(candidate["edge_index"], c0["edge_index"])),
        "same_old_prefix": bool(torch.equal(candidate["x"][:old_n], parent["x"])),
        "build": build_audit,
    }
    save_json(out / "protocol.json", {
        "name": "tgc-shared-two-prototype-content-smoke-v1",
        "dataset": args.dataset,
        "order": args.order,
        "stage": stage,
        "control": "sealed C0 single-mean content and C0 topology",
        "candidate": "localized high-dispersion split + nearest-pair merge with shared X/text assignment",
        "fixed": ["total rows", "C0 topology", "old synthetic prefix", "student GCN", "batch", "steps"],
        "text": "top-5 center-ranked snippets, 320 chars, frozen MiniLM re-encode",
        "student": "original two-layer GCN + trainable text projection + bidirectional InfoNCE",
        "report_labels_used_for_selection": False,
        "report_seeds": [int(v) for v in args.report_seeds],
    })
    torch.save(c0, out / "c0_memory.pt")
    torch.save(candidate, out / "two_prototype_memory.pt")
    save_json(out / "build_audit.json", audit)
    reports = {}
    for name, memory in (("c0", c0), ("two_prototype", candidate)):
        reports[name] = V1.report_candidate(
            memory, data, stage, [int(v) for v in args.report_seeds], args.device
        )
        reports[name]["margin"] = margin_report(
            memory, data, stage, old_n, build_audit["c0_row_classes"],
            [int(v) for v in args.report_seeds], args.device
        )
    save_json(out / "results.json", {
        "protocol": json.loads((out / "protocol.json").read_text(encoding="utf-8")),
        "audit": audit,
        "reports": reports,
        "delta": {
            "cumulative_f1_pp": float(
                reports["two_prototype"]["cumulative_f1_mean"]
                - reports["c0"]["cumulative_f1_mean"]
            ),
            "current_f1_pp": float(
                reports["two_prototype"]["current_f1_mean"]
                - reports["c0"]["current_f1_mean"]
            ),
            "old_mean_f1_pp": float(
                reports["two_prototype"]["old_mean_f1_mean"]
                - reports["c0"]["old_mean_f1_mean"]
            ),
            "real_new_margin_pp": float(
                reports["two_prototype"]["margin"]["real_new_margin_mean"]
                - reports["c0"]["margin"]["real_new_margin_mean"]
            ),
            "synthetic_margin_pp": float(
                reports["two_prototype"]["margin"]["synthetic_margin_mean"]
                - reports["c0"]["margin"]["synthetic_margin_mean"]
            ),
        },
    })
    print(json.dumps({
        "event": "complete",
        "dataset": args.dataset,
        "order": args.order,
        "stage": stage,
        "c0_f1": reports["c0"]["cumulative_f1_mean"],
        "two_prototype_f1": reports["two_prototype"]["cumulative_f1_mean"],
        "delta_pp": reports["two_prototype"]["cumulative_f1_mean"]
        - reports["c0"]["cumulative_f1_mean"],
    }), flush=True)


def class_row_targets(q: torch.Tensor, labels: torch.Tensor) -> list[int]:
    """Return C0 row class labels; kept separate for auditable call sites."""
    return class_rows_from_c0(q, labels.cpu())


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
    print("TGC_TWO_PROTOTYPE_SMOKE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
