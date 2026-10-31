"""Controlled C0/k-center content-gap attribution on the small graphs.

The experiment keeps the C0 row order, row budget, topology, student and
training schedule fixed.  It changes only the content of newly appended
rows, so a gain/loss can be attributed to synthetic X, text selection, or
their pairing rather than to a topology or protocol change.

This is a diagnostic experiment.  The anchor arms remain synthetic: an
anchor is used to guide a convex combination or to select local text, but a
raw node is never written as a synthetic row.  ``kcenter_fixed_topology`` is
included only as a reference arm; it deliberately retains real rows and is
not a proposed condensation method.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import run_small_comparison_audited as C
import run_unified_v3_continual as U
import run_student_compatible_append as SCA
import run_p0_random_synthetic as R

G, P = C.G, C.P

ARMS = [
    "c0",
    "anchor_x_c0_text",
    "c0_x_anchor_text",
    "anchor_guided_synthetic",
    "kcenter_fixed_topology",
]
SYNTHETIC_ARMS = set(ARMS) - {"kcenter_fixed_topology"}
ANCHOR_X_ALPHA = 0.5
FUSE_CHARS = int(P.FUSE_CHARS)
FUSE_G = int(P.FUSE_G)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def memory_hash(memory: dict) -> str:
    payload = C.tensor_hash(memory)
    payload += json.dumps(memory.get("texts", []), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    return value


def clone_memory(memory: dict) -> dict:
    return {key: clone_value(value) for key, value in memory.items()}


def stage_groups(dataset: str, count: int, order: str = "canonical") -> list[list[int]]:
    n = 7 if dataset == "cora" else 6
    sizes = [2, 2, n - 4] if count == 3 else [n - 4, 1, 1, 1, 1]
    classes = list(range(n))
    if order != "canonical":
        random.Random(int(order.removeprefix("order"))).shuffle(classes)
    groups, at = [], 0
    for width in sizes:
        groups.append(sorted(classes[at:at + width]))
        at += width
    return P.canonicalize_class_groups(groups)


def set_stream(data: dict, groups: list[list[int]], rate: float):
    data["stage_classes"] = groups
    G.STAGE_CLASSES = groups
    stage_ids = [torch.where(
        data["pool_mask"] & torch.isin(data["y"], torch.tensor(group))
    )[0].sort().values for group in groups]
    budgets = [math.floor(float(rate) * sum(map(len, stage_ids[:i + 1])))
               for i in range(len(stage_ids))]
    if any(b <= (budgets[i - 1] if i else 0)
           for i, b in enumerate(budgets)):
        raise ValueError("registered rate must add rows at every stage")
    return stage_ids, budgets


def source_groups_for_c0(data: dict, ids: torch.Tensor, rows: int,
                         stage: int, device: str) -> list[list[int]]:
    """Recreate C0's source partition in exactly the same row order."""
    ids = torch.sort(ids.long().cpu()).values
    classes = sorted(int(c) for c in data["stage_classes"][stage])
    allocation = U.allocate_rows_by_class(data, ids, classes, int(rows),
                                          "dispersion")
    groups: list[list[int]] = []
    for cls in classes:
        class_ids = torch.sort(ids[data["y"][ids] == int(cls)]).values
        count = min(int(allocation[int(cls)]), len(class_ids))
        if count <= 0:
            continue
        x = data["x"][class_ids].float()
        t = data["text_emb"][class_ids].float()
        joint = torch.cat((F.normalize(x, dim=1), t), dim=1)
        assign = SCA.cluster_assign(joint, torch.ones(len(joint)), count,
                                    device=device)
        for cluster_id in range(int(assign.max().item()) + 1):
            local = torch.where(assign == cluster_id)[0]
            groups.append([int(v) for v in class_ids[local].tolist()])
    return groups


def bounded_anchor_text(data: dict, member_ids: list[int], anchor_local: int,
                        encoder) -> tuple[str, torch.Tensor]:
    """Select a fixed-budget local text set around one source anchor."""
    ids = torch.tensor(member_ids, dtype=torch.long)
    x = data["x"][ids].float()
    t = data["text_emb"][ids].float()
    joint = torch.cat((F.normalize(x, dim=1), t), dim=1)
    anchor = joint[int(anchor_local)].unsqueeze(0)
    distance = torch.cdist(joint, anchor).flatten()
    count = min(FUSE_G, len(ids))
    chosen = torch.argsort(distance, stable=True)[:count]
    per = max(1, FUSE_CHARS // max(1, len(chosen)))
    snippets = [data["texts"][int(ids[int(i)])][:per] for i in chosen]
    fused = " ".join(snippets)[:FUSE_CHARS]
    return fused, encoder.encode([fused])[0]


def anchor_variants(data: dict, block: dict, groups: list[list[int]],
                    encoder) -> tuple[dict, dict]:
    """Create the two anchor ingredients while preserving C0 row order."""
    if len(groups) != len(block["x"]):
        raise AssertionError("C0 source partition row count mismatch")
    anchor_x, anchor_text, anchor_ids = [], [], []
    source_audit = []
    for row, member_ids in enumerate(groups):
        ids = torch.tensor(member_ids, dtype=torch.long)
        x = data["x"][ids].float()
        t = data["text_emb"][ids].float()
        joint = torch.cat((F.normalize(x, dim=1), t), dim=1)
        center = joint.mean(0, keepdim=True)
        # The farthest local anchor preserves a deterministic tail direction;
        # convex interpolation below keeps the final row synthetic.
        distances = torch.cdist(joint, center).flatten()
        local = int(torch.argsort(distances, stable=True)[-1])
        raw_x = x[local]
        mixed_x = ((1.0 - ANCHOR_X_ALPHA) * block["x"][row].float()
                   + ANCHOR_X_ALPHA * raw_x)
        fused, emb = bounded_anchor_text(data, member_ids, local, encoder)
        anchor_x.append(mixed_x)
        anchor_text.append(emb)
        anchor_ids.append(int(ids[local]))
        source_audit.append({
            "row": int(row), "source_ids": member_ids,
            "anchor_source_id": int(ids[local]),
            "anchor_local_index": local,
            "anchor_x_alpha": ANCHOR_X_ALPHA,
            "anchor_distance": float(distances[local]),
            "text_chars": len(fused),
        })
    return {
        "x": torch.stack(anchor_x),
        "text_emb": torch.stack(anchor_text),
        "texts": [bounded_anchor_text(data, members,
                                       next(i for i, v in enumerate(members)
                                            if v == rec["anchor_source_id"]),
                                       encoder)[0]
                   for members, rec in zip(groups, source_audit)],
    }, {"rows": source_audit, "synthetic": True}


def real_kcenter_block(data: dict, ids: torch.Tensor, rows: int, stage: int,
                       device: str) -> tuple[dict, dict]:
    """Build a real-node k-center delta for diagnostic reference only."""
    block, audit = C.real_block(data, ids, int(rows), "k_center", stage, device)
    source_ids = block.get("source_ids")
    block = {key: clone_value(block[key]) for key in
             ("x", "text_emb", "texts", "mass", "depth", "stage")}
    audit = {**audit, "source_ids": (source_ids.tolist()
                                      if isinstance(source_ids, torch.Tensor)
                                      else None)}
    audit = {**audit, "synthetic": False,
             "warning": "real rows; reference only, not a condensation method"}
    return block, audit


def fixed_topology_append(old: dict | None, block: dict, c0: dict) -> dict:
    """Append content while reusing the exact C0 adjacency."""
    if old is None:
        out = {key: clone_value(value) for key, value in block.items()
               if key in ("x", "text_emb", "texts", "mass", "depth", "stage")}
    else:
        out = {
            key: (old[key] + block[key] if key == "texts"
                  else torch.cat((old[key], block[key])))
            for key in ("x", "text_emb", "texts", "mass", "depth", "stage")
        }
    out["edge_index"] = c0["edge_index"].clone()
    if len(out["x"]) != len(c0["x"]):
        raise AssertionError("fixed topology row count mismatch")
    return out


def payload(memory: dict) -> dict:
    tensor_bytes = {key: int(value.numel() * value.element_size())
                    for key, value in memory.items()
                    if isinstance(value, torch.Tensor)}
    text_bytes = sum(len(str(t).encode("utf-8")) for t in memory["texts"])
    return {"tensor_bytes": tensor_bytes, "text_utf8_bytes": text_bytes,
            "logical_bytes": sum(tensor_bytes.values()) + text_bytes,
            "rows": int(len(memory["x"])),
            "edge_entries": int(memory["edge_index"].shape[1])}


def build_variant_block(data: dict, c0_block: dict, groups: list[list[int]],
                        arm: str, encoder, ids: torch.Tensor, stage: int,
                        device: str) -> tuple[dict, dict]:
    if arm == "c0":
        return clone_memory(c0_block), {"synthetic": True, "operator": "C0"}
    if arm == "kcenter_fixed_topology":
        return real_kcenter_block(data, ids, len(c0_block["x"]), stage, device)
    anchors, audit = anchor_variants(data, c0_block, groups, encoder)
    block = clone_memory(c0_block)
    if arm == "anchor_x_c0_text":
        block["x"] = anchors["x"]
        audit["operator"] = "anchor-guided synthetic X + C0 text"
    elif arm == "c0_x_anchor_text":
        block["text_emb"] = anchors["text_emb"]
        block["texts"] = anchors["texts"]
        audit["operator"] = "C0 X + anchor-local text"
    elif arm == "anchor_guided_synthetic":
        block["x"] = anchors["x"]
        block["text_emb"] = anchors["text_emb"]
        block["texts"] = anchors["texts"]
        audit["operator"] = "anchor-guided synthetic X/text pair"
    else:
        raise ValueError(f"unknown attribution arm: {arm}")
    block["anchor_source_ids"] = torch.tensor(
        [row["anchor_source_id"] for row in audit["rows"]], dtype=torch.long)
    return block, audit


def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def fit_one(memory: dict, data: dict, stage: int, seed: int, device: str,
            checkpoint: Path | None = None) -> dict:
    steps = P.student_steps(len(memory["x"]))
    started = time.perf_counter()
    row = C.fit(memory, int(seed), steps, device, data, stage,
                checkpoint=checkpoint)
    row["arm_support_sha256"] = memory_hash(memory)
    row["stage"] = int(stage)
    row["seed"] = int(seed)
    row["fit_seconds_wall"] = float(time.perf_counter() - started)
    return row


def run_dataset(args, dataset: str) -> None:
    out = Path(args.output) / dataset
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completion.json").exists():
        print(json.dumps({"dataset": dataset, "status": "already_complete"}),
              flush=True)
        return
    cache = Path(args.text_cache_root) / f"{dataset}_r005" / "text_emb_pool_only.pt"
    data = R.load_data(dataset, args.minilm, args.device, cache)
    groups = stage_groups(dataset, args.stages, args.order)
    stage_ids, budgets = set_stream(data, groups, args.rate)
    manifest = {
        "name": "content-gap-attribution-v1",
        "dataset": dataset, "order": args.order, "stages": args.stages,
        "rate": args.rate, "stage_classes": groups,
        "budgets": budgets, "arms": ARMS,
        "student_seeds": list(args.seeds),
        "student": {"batch": int(P.STUDENT.batch), "lr": float(P.STUDENT.lr),
                     "steps": "max(50,min(2048,16*rows))"},
        "fixed": ["row budget", "row order", "C0 topology", "student",
                  "batch", "steps", "report seeds", "report-dev"],
        "anchor": {"x_alpha": ANCHOR_X_ALPHA, "text_chars": FUSE_CHARS,
                   "text_g": FUSE_G,
                   "selection": "farthest source from each C0 cluster center",
                   "raw_node_written_as_synthetic": False},
        "scope": "small-graph content attribution; report-dev only",
        "test_labels_evaluated": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    C.save_json(out / "manifest.json", manifest)
    memories = {arm: None for arm in ARMS}
    results = []
    audits = []
    for stage, ids in enumerate(stage_ids):
        budget = budgets[stage]
        keep = 0 if stage == 0 else budgets[stage - 1]
        new_rows = budget - keep
        started = time.perf_counter()
        c0_block = U.build_stage_block(
            data, ids, new_rows, data["encoder"], stage, "joint_g5_c320",
            "concat", args.device, "dispersion", data["stage_classes"][stage])
        source_groups = source_groups_for_c0(data, ids, new_rows, stage,
                                             args.device)
        if len(source_groups) != len(c0_block["x"]):
            raise AssertionError("source group count differs from C0 rows")
        for row, member_ids in enumerate(source_groups):
            expected_x = data["x"][torch.tensor(member_ids)].float().mean(0)
            if not torch.equal(expected_x, c0_block["x"][row]):
                raise AssertionError("C0 source reconstruction mismatch")
            if int(c0_block["mass"][row]) != len(member_ids):
                raise AssertionError("C0 mass/source reconstruction mismatch")
        stage_audits = {"stage": stage, "keep": keep, "new_rows": new_rows,
                        "source_groups": source_groups,
                        "c0_block_build_seconds": time.perf_counter() - started}
        c0_mem = None
        # First create C0; every diagnostic receives the exact C0 topology for
        # this stage and therefore cannot gain from a different edge budget.
        c0_mem = SCA.append_memory(
            memories["c0"], c0_block, keep, topology_mode="frozen_old",
            cross_degree=1, refresh_mode="frozen_old", device=args.device)
        memories["c0"] = c0_mem
        c0_edge = c0_mem["edge_index"].clone()
        for arm in ARMS:
            if arm == "c0":
                memory = clone_memory(c0_mem)
                build_audit = {"operator": "C0", "synthetic": True}
            else:
                block, build_audit = build_variant_block(
                    data, c0_block, source_groups, arm, data["encoder"], ids,
                    stage, args.device)
                memory = fixed_topology_append(
                    memories[arm], block, c0_mem)
                if not torch.equal(memory["edge_index"], c0_edge):
                    raise AssertionError("diagnostic topology differs from C0")
            memories[arm] = memory
            if len(memory["x"]) != budget:
                raise AssertionError(f"{arm} budget mismatch")
            if not torch.equal(memory["edge_index"], c0_edge):
                raise AssertionError(f"{arm} topology mismatch")
            if arm in SYNTHETIC_ARMS and arm != "c0":
                # An anchor arm is only synthetic if it does not copy an exact
                # raw feature vector into a compressed (multi-source) row.
                # A singleton source cluster is already irreducible at this
                # budget and is explicitly exempted from this audit.
                for value, member_ids in zip(memory["x"][keep:], source_groups):
                    if len(member_ids) <= 1:
                        continue
                    raw = data["x"][torch.tensor(member_ids)].float()
                    if bool(torch.isclose(raw, value.unsqueeze(0), atol=0,
                                          rtol=0).all(1).any()):
                        raise AssertionError("anchor arm copied a raw X row")
            torch.save(memory, out / f"{arm}_s{stage}_memory.pt")
            C.save_json(out / f"{arm}_s{stage}_construction.json", {
                "arm": arm, "stage": stage, "keep": keep,
                "new_rows": new_rows, "payload": payload(memory),
                "memory_sha256": memory_hash(memory),
                "topology_sha256": C.tensor_hash({"edge_index": memory["edge_index"]}),
                "build": build_audit,
            })
            stage_audits.setdefault("arms", {})[arm] = {
                "rows": int(len(memory["x"])),
                "memory_sha256": memory_hash(memory),
                "payload": payload(memory),
            }
        audits.append(stage_audits)
        for arm in ARMS:
            memory = memories[arm]
            scores = []
            for seed in args.seeds:
                key = f"{arm}_s{stage}_seed{seed}"
                json_path = out / f"{key}.json"
                checkpoint = out / f"{key}.pt"
                if json_path.exists() and checkpoint.exists():
                    row = json.loads(json_path.read_text())
                    if row["arm_support_sha256"] != memory_hash(memory):
                        raise AssertionError("stale attribution fit")
                else:
                    row = fit_one(memory, data, stage, seed, args.device,
                                  checkpoint=checkpoint)
                    row["arm"] = arm
                    row["dataset"] = dataset
                    row["effective_batch"] = min(int(P.STUDENT.batch),
                                                  len(memory["x"]))
                    # Keep the canonical report metric as JSON-native values.
                    row["metrics"] = json.loads(json.dumps(row["metrics"]))
                    row["checkpoint_sha256"] = C.sha(checkpoint)
                    C.save_json(json_path, row)
                scores.append(row)
            results.append({
                "arm": arm, "stage": stage, "rows": int(len(memory["x"])),
                "steps": int(P.student_steps(len(memory["x"]))),
                "metrics": C.aggregate(scores),
                "per_seed": scores,
            })
            C.save_json(out / "progress.json", {
                "manifest": manifest, "results": results, "audits": audits})
            print(json.dumps({"dataset": dataset, "stage": stage, "arm": arm,
                              "f1": results[-1]["metrics"]["all"]
                              ["cumulative_f1"]["mean"]}), flush=True)
    C.save_json(out / "results.json", {
        "manifest": manifest, "results": results, "audits": audits})
    C.save_json(out / "completion.json", {
        "dataset": dataset, "completed_at": datetime.now(timezone.utc).isoformat(),
        "fits": len(results) * len(args.seeds),
        "expected_fits": len(ARMS) * len(stage_ids) * len(args.seeds),
        "topology_fixed_to_c0": True,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default=(
        "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"))
    ap.add_argument("--text-cache-root", default=(
        "/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--order", default="canonical")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(P.STUDENT_SEEDS))
    args = ap.parse_args()
    torch.set_num_threads(2)
    datasets = [args.dataset] if args.dataset else ["cora", "citeseer"]
    for dataset in datasets:
        run_dataset(args, dataset)
    print("CONTENT_GAP_ATTRIBUTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
