"""Frozen Phase A matrix for Teacher-Guided Continual Revision."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import torch

import c0m_memory_maintenance as M
import run_content_gap_attribution as A
import run_p0_random_synthetic as R
import run_small_comparison_audited as C
import run_student_compatible_append as SCA
import run_unified_v3_continual as U
import tgc_revision_gate as TGC

G, P = C.G, C.P
ARMS = ("c0", "naive_refresh", "c0m", "tgc_r", "tgc_no_preserve", "tgc_shuffled")
MODES = {"tgc_r": "gate", "tgc_no_preserve": "no_preserve", "tgc_shuffled": "shuffled"}
ORDERS = ("canonical", "order19101", "order19102")
FIELDS = ("x", "text_emb", "texts", "mass", "depth", "stage")


def normalize_arms(arms) -> list[str]:
    if not arms or len(arms) != len(set(arms)) or any(a not in ARMS for a in arms):
        raise ValueError("arms must be unique registered names")
    if "c0" not in arms:
        raise ValueError("c0 is required as the incoming-block/topology reference")
    return [a for a in ARMS if a in arms]


def validate_reference(cell: Path, manifest: dict) -> dict:
    reference = json.loads((cell / "manifest.json").read_text())
    for key in ("dataset", "order", "stage_classes", "budgets", "stage_source_ids",
                "source_dataset_sha256", "pool_sha256", "report_mask_sha256"):
        if reference[key] != manifest[key]:
            raise ValueError(f"reference mismatch: {key}")
    for key in ("rate", "student"):
        if reference["protocol"][key] != manifest["protocol"][key]:
            raise ValueError(f"reference protocol mismatch: {key}")
    required = [C.__file__, G.__file__, P.__file__, U.__file__, SCA.__file__]
    base = Path(__file__).resolve().parent.parent
    for filename in required:
        name = str(Path(filename).resolve().relative_to(base)).replace("\\", "/")
        if reference["protocol"]["source_hashes"].get(name) != C.sha(filename):
            raise ValueError(f"reference implementation changed: {name}")
    return reference


def save_json(path: Path, value: dict) -> None:
    C.save_json(path, value)


def memory_hash(memory: dict) -> str:
    return TGC.memory_hash(memory)


def payload(memory: dict) -> dict:
    tensor_bytes = {key: int(value.numel() * value.element_size())
                    for key, value in memory.items()
                    if isinstance(value, torch.Tensor)}
    text_bytes = sum(len(str(t).encode("utf-8")) for t in memory["texts"])
    return {"tensor_bytes": tensor_bytes, "text_utf8_bytes": text_bytes,
            "logical_bytes": sum(tensor_bytes.values()) + text_bytes,
            "rows": int(len(memory["x"])),
            "edge_entries": int(memory["edge_index"].shape[1])}


def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def stable_lock(path: Path, value: dict) -> None:
    normalized = json.loads(json.dumps(value))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != normalized:
            raise ValueError(f"frozen configuration mismatch: {path}")
    else:
        save_json(path, normalized)


def source_hashes() -> dict:
    here = Path(__file__).resolve().parent
    paths = list(here.glob("*.py")) + list(
        (here.parent / "frozen_v1_2026_09_09").glob("*.py"))
    return {str(path.relative_to(here.parent)): C.sha(path)
            for path in sorted(paths)}


def stage_groups(dataset: str, order: str) -> list[list[int]]:
    return A.stage_groups(dataset, 3, order)


def _assert_new_block_equal(c0: dict, memory: dict, new_rows: int) -> None:
    start = len(c0["x"]) - int(new_rows)
    for key in FIELDS:
        if key == "texts":
            equal = c0[key][start:] == memory[key][start:]
        else:
            equal = torch.equal(c0[key][start:], memory[key][start:])
        if not equal:
            raise AssertionError(f"incoming C0 field changed: {key}")


def fit(memory: dict, data: dict, stage: int, seed: int, arm: str,
        out: Path, device: str, steps_override: int | None,
        reference_cell: Path | None = None) -> dict:
    key = f"{arm}_s{stage}_seed{seed}"
    row_path, checkpoint = out / f"{key}.json", out / f"{key}.pt"
    steps = P.student_steps(len(memory["x"]), override=steps_override)
    if row_path.exists() and checkpoint.exists():
        row = json.loads(row_path.read_text())
        if (row["support_sha256"] != C.tensor_hash(memory)
                or int(row["steps"]) != int(steps)
                or C.sha(checkpoint) != row["checkpoint_sha256"]):
            raise AssertionError(f"stale or damaged fit: {key}")
        return row
    if reference_cell is not None and (arm == "c0" or stage == 0):
        source = reference_cell / f"c0_s{stage}_seed{seed}.json"
        source_checkpoint = source.with_suffix(".pt")
        row = json.loads(source.read_text())
        if (row["support_sha256"] != C.tensor_hash(memory)
                or row["support_memory_sha256"] != memory_hash(memory)
                or row["steps"] != steps or row["seed"] != seed
                or row["checkpoint_sha256"] != C.sha(source_checkpoint)):
            raise AssertionError(f"reference fit mismatch: {source}")
        shutil.copy2(source_checkpoint, checkpoint)
        row.update(arm=arm, reused_from=str(source),
                   reused_source_sha256=C.sha(source), executed_here=False)
        save_json(row_path, row)
        return row
    sync(device)
    started = time.perf_counter()
    row = C.fit(memory, seed, steps, device, data, stage,
                checkpoint=checkpoint)
    row.update(arm=arm, stage=int(stage), seed=int(seed),
               support_memory_sha256=memory_hash(memory),
               wall_fit_seconds=float(time.perf_counter() - started),
               effective_batch=min(int(P.STUDENT.batch), len(memory["x"])),
               executed_here=True,
               checkpoint_sha256=C.sha(checkpoint))
    save_json(row_path, row)
    return row


def protocol(args, cfg: TGC.RevisionConfig) -> dict:
    return {
        "name": "tgc-r-phase-a-v2",
        "datasets": [args.dataset] if args.dataset else ["cora", "citeseer"],
        "arms": list(args.arms), "orders": list(args.order),
        "rate": float(args.rate), "student_seeds": list(args.seeds),
        "student": {"batch": int(P.STUDENT.batch), "lr": float(P.STUDENT.lr),
                     "steps_override": args.student_steps},
        "teacher": cfg.dict(), "source_hashes": source_hashes(),
        "primary_metric": "report-dev cumulative Macro-F1",
        "history_access": "previous synthetic memory + current raw delta only",
        "revision": "old synthetic X only; text/mass/depth/stage/old topology immutable",
        "test_labels_evaluated": False, "selector_labels_used": False,
        "promotion": {"mean_final_min_pp": 1.0,
                      "worst_final_min_pp": -2.0,
                      "minimum_cells_ge_minus1": 4,
                      "worst_old_min_pp": -2.0,
                      "mean_new_min_pp_strict": 0.0,
                      "build_cost_max_ratio": 2.0,
                      "logical_memory_max_ratio": 1.5},
    }


def run_cell(data: dict, order: str, args, cfg: TGC.RevisionConfig,
             protocol_manifest: dict) -> None:
    groups = stage_groups(data["dataset"], order)
    out = Path(args.output) / data["dataset"] / order
    out.mkdir(parents=True, exist_ok=True)
    stage_ids, budgets = A.set_stream(data, groups, args.rate)
    manifest = {
        "protocol": protocol_manifest, "dataset": data["dataset"],
        "order": order, "stage_classes": groups, "budgets": budgets,
        "student_seeds": list(args.seeds), "arms": list(args.arms),
        "stage_source_ids": [ids.tolist() for ids in stage_ids],
        "source_dataset_sha256": C.sha(U.DATA_CFG[data["dataset"]]["path"]),
        "pool_sha256": C.tensor_hash({"pool": data["pool_mask"]}),
        "report_mask_sha256": C.tensor_hash({"report": data["report_dev_mask"]}),
    }
    reference_cell = (Path(args.reference_root) / data["dataset"] / order
                      if getattr(args, "reference_root", None) else None)
    if reference_cell is not None:
        validate_reference(reference_cell, manifest)
        manifest["reference_manifest_sha256"] = C.sha(reference_cell / "manifest.json")
        manifest["reference_cell"] = str(reference_cell)
    stable_lock(out / "manifest.json", manifest)
    histories = {arm: [] for arm in args.arms}
    constructions = []
    for stage, ids in enumerate(stage_ids):
        keep = 0 if stage == 0 else int(budgets[stage - 1])
        new_rows = int(budgets[stage] - keep)
        block_path = out / f"c0_s{stage}_block.pt"
        block_audit_path = out / f"c0_s{stage}_block.json"
        if not block_path.exists() and reference_cell is not None:
            shutil.copy2(reference_cell / block_path.name, block_path)
            shutil.copy2(reference_cell / block_audit_path.name, block_audit_path)
        if block_audit_path.exists() and block_path.exists():
            block = torch.load(block_path, map_location="cpu", weights_only=False)
            block_audit = json.loads(block_audit_path.read_text())
            if memory_hash(block) != block_audit["memory_sha256"]:
                raise AssertionError("cached C0 block hash mismatch")
            if block_audit["source_ids"] != ids.tolist():
                raise AssertionError("cached C0 block uses wrong current delta")
        else:
            sync(args.device)
            started = time.perf_counter()
            block = U.build_stage_block(
                data, ids, new_rows, data["encoder"], stage,
                "joint_g5_c320", "concat", args.device, "dispersion",
                groups[stage])
            sync(args.device)
            block_audit = {"seconds": float(time.perf_counter() - started),
                           "memory_sha256": memory_hash(block),
                           "source_ids": ids.tolist()}
            torch.save(block, block_path)
            save_json(block_audit_path, block_audit)
        delta = TGC.raw_delta_block(data, ids, stage) if stage > 0 else None
        # C0 is built first so every arm has an immutable incoming-block reference.
        for arm in args.arms:
            old = histories[arm][-1] if stage else None
            memory_path = out / f"{arm}_s{stage}_memory.pt"
            audit_path = out / f"{arm}_s{stage}_construction.json"
            if arm == "c0" and reference_cell is not None and not memory_path.exists():
                shutil.copy2(reference_cell / memory_path.name, memory_path)
                shutil.copy2(reference_cell / audit_path.name, audit_path)
            if memory_path.exists() and audit_path.exists():
                memory = torch.load(memory_path, map_location="cpu", weights_only=False)
                audit = json.loads(audit_path.read_text())
                if (audit["memory_sha256"] != memory_hash(memory)
                        or audit["block_sha256"] != memory_hash(block)
                        or audit["old_memory_sha256"] != (memory_hash(old) if old else None)):
                    raise AssertionError(f"cached construction mismatch: {arm} s{stage}")
            else:
                sync(args.device)
                started = time.perf_counter()
                build = {"operator": arm}
                if arm == "c0m":
                    memory, maintenance = M.append(
                        old, block, keep, M.MaintenanceConfig(), args.device)
                    M.assert_invariants(old, block, memory, maintenance,
                                        M.MaintenanceConfig())
                    build["maintenance"] = maintenance
                else:
                    working = old
                    if stage > 0 and arm in MODES:
                        certificate = {}
                        working, revision = TGC.revise_old_memory(
                            old, delta, cfg, args.device, MODES[arm], certificate)
                        TGC.assert_revision_invariants(old, working, revision, cfg)
                        certificate_path = out / f"{arm}_s{stage}_certificate.pt"
                        torch.save(certificate, certificate_path)
                        build.update(revision=revision,
                                     certificate_file=certificate_path.name,
                                     certificate_sha256=C.sha(certificate_path))
                    memory = SCA.append_memory(
                        working, block, keep, topology_mode="frozen_old",
                        cross_degree=1,
                        refresh_mode=("bounded_refresh" if arm == "naive_refresh"
                                      else "frozen_old"),
                        refresh_ratio=0.10, refresh_alpha=0.25,
                        device=args.device)
                sync(args.device)
                build["operator_seconds"] = float(time.perf_counter() - started)
                build["block_seconds"] = float(block_audit["seconds"])
                build["build_seconds"] = build["operator_seconds"] + build["block_seconds"]
                audit = {
                    "arm": arm, "stage": stage, "keep": keep,
                    "new_rows": new_rows, "memory_sha256": memory_hash(memory),
                    "tensor_sha256": C.tensor_hash(memory),
                    "old_memory_sha256": memory_hash(old) if old else None,
                    "block_sha256": memory_hash(block), "payload": payload(memory),
                    "build": build,
                }
                torch.save(memory, memory_path)
                save_json(audit_path, audit)
            if len(memory["x"]) != int(budgets[stage]):
                raise AssertionError(f"row budget mismatch: {arm} s{stage}")
            if arm != "c0":
                _assert_new_block_equal(histories["c0"][stage], memory, new_rows)
            if arm in MODES:
                if not torch.equal(memory["edge_index"],
                                   histories["c0"][stage]["edge_index"]):
                    raise AssertionError("TGC-R topology must match C0")
            histories[arm].append(memory)
            constructions.append(audit)
            print(json.dumps({"event": "constructed", "dataset": data["dataset"],
                              "order": order, "stage": stage, "arm": arm}),
                  flush=True)
    results = []
    for stage in range(len(groups)):
        for arm in args.arms:
            memory = histories[arm][stage]
            scores = [fit(memory, data, stage, seed, arm, out, args.device,
                           args.student_steps, reference_cell) for seed in args.seeds]
            results.append({"arm": arm, "stage": stage, "rows": len(memory["x"]),
                            "steps": scores[0]["steps"],
                            "metrics": C.aggregate(scores), "per_seed": scores})
            save_json(out / "progress.json", {"manifest": manifest,
                                               "results": results,
                                               "constructions": constructions})
            print(json.dumps({"event": "evaluated", "dataset": data["dataset"],
                              "order": order, "stage": stage, "arm": arm,
                              "f1": results[-1]["metrics"]["all"][
                                  "cumulative_f1"]["mean"]}), flush=True)
    save_json(out / "results.json", {"manifest": manifest,
                                     "results": results,
                                     "constructions": constructions})
    save_json(out / "completion.json", {
        "status": "complete", "fits": len(results) * len(args.seeds),
        "expected_fits": len(args.arms) * len(groups) * len(args.seeds),
        "executed_fits": sum(r.get("executed_here", True)
                             for e in results for r in e["per_seed"]),
        "reused_fits": sum(not r.get("executed_here", True)
                           for e in results for r in e["per_seed"]),
        "completed_at": datetime.now(timezone.utc).isoformat(),
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
    ap.add_argument("--order", nargs="+", choices=ORDERS, default=list(ORDERS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(P.STUDENT_SEEDS))
    ap.add_argument("--student-steps", type=int, default=None)
    ap.add_argument("--teacher-steps", type=int, default=128)
    ap.add_argument("--reference-root", default=None)
    ap.add_argument("--protected-fraction", type=float, default=0.80)
    ap.add_argument("--revision-fraction", type=float, default=0.10)
    ap.add_argument("--revision-radius", type=float, default=0.10)
    ap.add_argument("--arms", nargs="+", choices=list(ARMS),
                    default=list(ARMS))
    args = ap.parse_args()
    args.arms = normalize_arms(args.arms)
    cfg = TGC.RevisionConfig(
        protected_fraction=args.protected_fraction,
        revision_fraction_of_old=args.revision_fraction,
        revision_radius=args.revision_radius,
        teacher_steps=args.teacher_steps,
    )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("report seeds must be unique")
    torch.set_num_threads(2)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    frozen = protocol(args, cfg)
    stable_lock(root / "protocol.json", frozen)
    datasets = [args.dataset] if args.dataset else ["cora", "citeseer"]
    for dataset in datasets:
        cache = Path(args.text_cache_root) / f"{dataset}_r005" / "text_emb_pool_only.pt"
        data = R.load_data(dataset, args.minilm, args.device, cache)
        for order in args.order:
            run_cell(data, order, args, cfg, frozen)
    print("TGC_R_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
