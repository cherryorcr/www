"""Message-conditioned synthetic feature projection on a frozen C0 topology.

The proposal, text, budget and C0 edge_index are fixed.  For changed incoming
rows, the synthetic X is projected toward the legal current-delta one/two-hop
message targets while staying inside a data-derived source-cluster trust
region around the two-prototype proposal.  This is a content/topology joint
operator, not a post-hoc payload permutation or an F1 selector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_tgc_role_aware_assignment_20260925 as R
import run_tgc_joint_student_response as V1
import run_tgc_r11_c0_anchor as R11
import run_tgc_two_prototype_role_permutation as Base
import run_tgc_two_prototype_smoke as B


ARMS = ("c0", "identity", "message_x")


def normalized_dense_adjacency(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    edge = edge_index.long().cpu()
    adj = torch.zeros((n, n), dtype=torch.float32)
    if edge.numel():
        adj[edge[0], edge[1]] = 1.0
    adj.fill_diagonal_(1.0)
    degree = adj.sum(1).clamp_min(1.0)
    scale = degree.rsqrt()
    return scale[:, None] * adj * scale[None, :]


def current_delta_adjacency(data: dict, ids: torch.Tensor) -> torch.Tensor:
    """Read only edges whose two endpoints belong to the legal current delta."""
    ids = ids.long().cpu()
    n_global = len(data["x"])
    local = torch.full((n_global,), -1, dtype=torch.long)
    local[ids] = torch.arange(len(ids), dtype=torch.long)
    edge = data["edge_index"].long().cpu()
    keep = (local[edge[0]] >= 0) & (local[edge[1]] >= 0)
    local_edge = torch.stack((local[edge[0, keep]], local[edge[1, keep]]))
    return normalized_dense_adjacency(local_edge, len(ids))


def project_x(c0: dict, structural_block: dict, records: list[dict],
              data: dict, ids: torch.Tensor, old_n: int,
              raw_adj: torch.Tensor) -> tuple[dict, dict]:
    block = {
        key: value.clone() if isinstance(value, torch.Tensor) else list(value)
        for key, value in structural_block.items()
    }
    changed = [i for i, record in enumerate(records) if record.get("changed")]
    if not changed:
        return block, {"changed_rows": 0, "rows": []}

    raw_x = data["x"][ids].float().cpu()
    raw_h1 = raw_adj @ raw_x
    raw_h2 = raw_adj @ raw_h1
    base_x = c0["x"].float().cpu().clone()
    base_x[old_n:] = structural_block["x"].float().cpu()
    n = len(base_x)
    c0_adj = R.normalized_adjacency(c0)
    c0_h1 = c0_adj @ base_x
    c0_a2 = c0_adj @ c0_adj
    c0_h2 = c0_a2 @ base_x
    audit_rows = []

    for slot in changed:
        record = records[slot]
        members = torch.tensor(record["source_positions"], dtype=torch.long)
        if len(members) == 0:
            continue
        proposal = base_x[old_n + slot].clone()
        target_h1 = raw_h1[members].mean(0)
        target_h2 = raw_h2[members].mean(0)
        node = old_n + int(slot)
        phi1 = c0_adj[:, node]
        phi2 = c0_a2[:, node]
        base_h1 = c0_h1 - phi1[:, None] * base_x[node][None, :]
        base_h2 = c0_h2 - phi2[:, None] * base_x[node][None, :]
        # The raw cluster target is a node-level message.  Use the exact C0
        # one/two-hop coefficients at this synthetic node, rather than mixing
        # a vector target with the full graph message matrix.
        phi = torch.tensor([phi1[node], phi2[node]], dtype=torch.float32)
        target = torch.cat((target_h1, target_h2), dim=0)
        base = torch.cat((base_h1[node], base_h2[node]), dim=0)
        residual = target - base
        sensitivity = float(phi.square().sum())
        lam = 1.0 / max(sensitivity, 1e-8)
        unconstrained = (proposal + lam * (phi[:, None] * residual.reshape(2, -1)).sum(0)) / (1.0 + lam * sensitivity)

        source_mean = raw_x[members].mean(0)
        dispersion = float((raw_x[members] - source_mean).square().mean(0).sum().sqrt())
        radius = max(dispersion, float(proposal.norm()) * 0.02, 1e-6)
        delta = unconstrained - proposal
        delta_norm = float(delta.norm())
        scale = min(1.0, radius / max(delta_norm, 1e-12))
        updated = proposal + scale * delta
        block["x"][slot] = updated

        before_h1 = c0_h1[node]
        before_h2 = c0_h2[node]
        after_h1 = base_h1[node] + phi1[node] * updated
        after_h2 = base_h2[node] + phi2[node] * updated
        audit_rows.append({
            "slot": int(slot),
            "source_count": int(len(members)),
            "sensitivity": sensitivity,
            "lambda": lam,
            "dispersion_radius": radius,
            "unconstrained_displacement": delta_norm,
            "applied_displacement": float((updated - proposal).norm()),
            "trust_scale": float(scale),
            "message_error_before": float((before_h1 - target_h1).norm() + (before_h2 - target_h2).norm()),
            "message_error_after": float((after_h1 - target_h1).norm() + (after_h2 - target_h2).norm()),
        })
    # Recompute after all rows are updated: a node's message also depends on
    # neighboring synthetic rows, so the per-row closed-form audit above is
    # only a local prediction.
    final_x = base_x.clone()
    final_x[old_n:] = block["x"].float().cpu()
    final_h1 = c0_adj @ final_x
    final_h2 = c0_a2 @ final_x
    by_slot = {int(row["slot"]): row for row in audit_rows}
    for slot in changed:
        row = by_slot.get(int(slot))
        if row is None:
            continue
        members = torch.tensor(records[slot]["source_positions"], dtype=torch.long)
        target_h1 = raw_h1[members].mean(0)
        target_h2 = raw_h2[members].mean(0)
        node = old_n + int(slot)
        row["message_error_after_global"] = float(
            (final_h1[node] - target_h1).norm() +
            (final_h2[node] - target_h2).norm())
    return block, {"changed_rows": len(audit_rows), "rows": audit_rows}


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
    raw_adj = current_delta_adjacency(data, ids[args.stage])
    message_block, message_audit = project_x(
        c0, structural_block, structural_audit["records"], data,
        ids[args.stage], old_n, raw_adj,
    )
    output = Path(args.output) / f"{args.dataset}_{args.order}_s{args.stage}"
    if (output / "results.json").exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    identity = Base.candidate_from_block(c0, parent, structural_block)
    message_memory = Base.candidate_from_block(c0, parent, message_block)
    for key in ("text_emb", "mass", "depth", "stage"):
        if not torch.equal(message_block[key], structural_block[key]):
            raise AssertionError(f"message-conditioned X changed non-X payload: {key}")
    if message_block["texts"] != structural_block["texts"]:
        raise AssertionError("message-conditioned X changed text payload")
    if not torch.equal(message_memory["edge_index"], c0["edge_index"]):
        raise AssertionError("message-conditioned X changed C0 topology")
    for name, memory in (("c0", c0), ("identity", identity),
                         ("message_x", message_memory)):
        torch.save(memory, output / f"{name}_memory.pt")
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
        "message_x_audit": message_audit,
        "same_c0_topology": True,
        "same_old_prefix": True,
        "text_unchanged_from_identity": True,
    }
    protocol = {
        "name": "tgc-message-conditioned-x-v1",
        "dataset": args.dataset,
        "order": args.order,
        "stage": int(args.stage),
        "arms": list(ARMS),
        "proposal": "frozen two-prototype structural proposal",
        "operator": "closed-form bounded projection to current-delta one/two-hop message targets",
        "topology": "exact C0 edge_index",
        "text": "same frozen two-prototype text; no text optimization",
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
