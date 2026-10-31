"""Independent replay audit for R13 scratch response decisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_tgc_r13_plasticity_gate as B
import tgc_selector_confirmation as Selector

R, T, D = B.R, B.T, B.D


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def check_close(left, right, tolerance=2e-5):
    if abs(float(left) - float(right)) > tolerance:
        raise AssertionError(f"response mismatch: {left} vs {right}")


def audit_cell(data, order, root, device):
    ids, budgets = D.set_stream(data, order)
    out = root / data["dataset"] / order
    manifest = read(out / "manifest.json")
    if manifest["stage_source_ids"] != [x.tolist() for x in ids]:
        raise AssertionError("stage ids changed")
    if manifest["budgets"] != budgets:
        raise AssertionError("budgets changed")
    checked_points = 0
    for stage in range(3):
        decision = read(out / f"s{stage}_decision.json")
        for arm in B.ARMS:
            memory = load(out / f"{arm}_s{stage}_memory.pt")
            B.E.memory_checks(memory, budgets[stage])
            if decision["memories"][arm]["memory_sha256"] != T.memory_hash(memory):
                raise AssertionError("memory decision hash mismatch")
        if stage == 0:
            continue
        old = load(B.SOURCE / data["dataset"] / order / f"c0_s{stage-1}_memory.pt")
        support = T._concat_support(old, T.raw_delta_block(data, ids[stage], stage))
        response = decision["response"]
        if response["historical_raw_reads"] != 0 or response["report_labels_used"]:
            raise AssertionError("illegal response inputs")
        batches = Selector.probe_batches(support["old_rows"], len(support["x"]), stage)
        expected_batches = {k: [x.tolist() for x in v] for k, v in batches.items()}
        if response["probe_batches"] != expected_batches:
            raise AssertionError("probe batches changed")
        records = []
        base = load(out / f"c0_s{stage}_memory.pt")
        candidate = load(out / f"anchor025_s{stage}_memory.pt")
        for saved in response["per_seed"]:
            path = out / saved["checkpoint_file"]
            if R.C.sha(path) != saved["checkpoint_sha256"]:
                raise AssertionError("selector checkpoint hash mismatch")
            trajectories = load(path)
            replay = {}
            for name, memory in (("c0", base), ("candidate", candidate)):
                replay[name] = Selector.scratch_trajectory(
                    memory, support, batches, int(saved["seed"]), device,
                    checkpoints=B.CHECKPOINTS)
                if replay[name]["start_hash"] != trajectories[name]["start_hash"]:
                    raise AssertionError("selector start hash mismatch")
                if replay[name]["batch_sequence"] != trajectories[name]["batch_sequence"]:
                    raise AssertionError("selector batch sequence mismatch")
                for point, ref in zip(trajectories[name]["curve"], replay[name]["curve"]):
                    if point["step"] != ref["step"] or point["prefix_hash"] != ref["prefix_hash"]:
                        raise AssertionError("selector curve structure mismatch")
                    for key in ("old", "new", "mixed"):
                        check_close(point[key], ref[key])
                    checked_points += 1
            records.append({"seed": int(saved["seed"]),
                            "c0": trajectories["c0"]["curve"],
                            "candidate": trajectories["candidate"]["curve"]})
        details = B._gain_records(records)
        for policy in ("selected_curve", "selected_new64", "selected_conservative"):
            expected = B._policy(details, policy)
            if expected != response["policies"][policy]:
                raise AssertionError(f"policy changed: {policy}")
            selected = decision["policies"][policy]["selected_arm"]
            selected_hash = T.memory_hash(load(out / f"{policy}_s{stage}_memory.pt"))
            expected_hash = T.memory_hash(load(out / f"{selected}_s{stage}_memory.pt"))
            if selected_hash != expected_hash:
                raise AssertionError(f"selected memory mismatch: {policy}")
    return {"dataset": data["dataset"], "order": order,
            "replayed_response_points": checked_points}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--text-cache-root", type=Path,
                        default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    cells = []
    for dataset in B.DATASETS:
        data = R.R.load_data(dataset, args.minilm, args.device,
                             args.text_cache_root / f"{dataset}_r005" / "text_emb_pool_only.pt")
        for order in B.ORDERS:
            cells.append(audit_cell(data, order, args.root, args.device))
    result = {"status": "passed", "cells": cells,
              "replayed_response_points": sum(c["replayed_response_points"] for c in cells),
              "report_labels_evaluated": False,
              "historical_raw_reads": 0}
    (args.root / "r13_selector_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
