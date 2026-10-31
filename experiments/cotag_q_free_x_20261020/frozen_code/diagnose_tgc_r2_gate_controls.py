"""Causal gate replay: identical parent and saved teacher, deterministic CPU math."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

import run_tgc_revision_gate as R
import tgc_revision_gate as T


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--capacity", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--text-cache-root", default="/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919")
    args = ap.parse_args()
    torch.set_num_threads(2)
    records = []
    for dataset in ("cora", "citeseer"):
        cache = Path(args.text_cache_root) / f"{dataset}_r005" / "text_emb_pool_only.pt"
        data = R.R.load_data(dataset, args.minilm, "cpu", cache)
        for config in ("baseline", "quota050"):
            orders = R.ORDERS if config == "baseline" else ("canonical", "order19101")
            for order in orders:
                cell = (args.reference / dataset / order if config == "baseline" else
                        args.capacity / f"{dataset}_{config}" / dataset / order)
                manifest = read(cell / "manifest.json")
                ids, _ = R.A.set_stream(data, manifest["stage_classes"], .05)
                cfg = T.RevisionConfig(**manifest["protocol"]["teacher"])
                for stage in (1, 2):
                    build = read(cell / f"tgc_r_s{stage}_construction.json")["build"]
                    cert = torch.load(cell / build["certificate_file"], map_location="cpu", weights_only=False)
                    if R.C.sha(cell / build["certificate_file"]) != build["certificate_sha256"]:
                        raise ValueError("teacher certificate corrupted")
                    old = torch.load(cell / f"tgc_r_s{stage-1}_memory.pt", map_location="cpu", weights_only=False)
                    delta = T.raw_delta_block(data, ids[stage], stage)
                    teacher = R.G.CLIPStudent(old["x"].shape[1])
                    teacher.load_state_dict(cert["state"])
                    teacher.eval()

                    def frozen_fit(support, cfg, device):
                        adjacency = R.G.norm_adj(support["edge_index"], len(support["x"]), "cpu")
                        return teacher, adjacency, []

                    with patch.object(T, "_teacher_fit", side_effect=frozen_fit):
                        gate_memory, gate = T.revise_old_memory(old, delta, cfg, "cpu", "gate")
                        free_memory, free = T.revise_old_memory(old, delta, cfg, "cpu", "no_preserve")
                    T.assert_revision_invariants(old, gate_memory, gate, cfg)
                    T.assert_revision_invariants(old, free_memory, free, cfg)
                    original = build["revision"]
                    record = {"dataset": dataset, "order": order, "config": config, "stage": stage,
                              "teacher_sha256": build["certificate_sha256"],
                              "original_gate_attempts": len(original["traces"]),
                              "original_gate_accepted": len(original["updated_rows"]),
                              "original_preserve_failures": sum(not r.get("preserve_ok", True) for r in original["traces"]),
                              "shared_teacher_gate_accepted": len(gate["updated_rows"]),
                              "shared_teacher_no_preserve_accepted": len(free["updated_rows"]),
                              "shared_teacher_x_bitwise_equal": torch.equal(gate_memory["x"], free_memory["x"]),
                              "shared_teacher_x_max_abs_difference": float((gate_memory["x"] - free_memory["x"]).abs().max())}
                    if config == "baseline":
                        independent = torch.load(cell / f"tgc_no_preserve_s{stage}_memory.pt", map_location="cpu", weights_only=False)
                        primary = torch.load(cell / f"tgc_r_s{stage}_memory.pt", map_location="cpu", weights_only=False)
                        other = read(cell / f"tgc_no_preserve_s{stage}_construction.json")["build"]["revision"]
                        record.update(independent_fits_same_updated_rows=original["updated_rows"] == other["updated_rows"],
                                      independent_fits_x_max_difference=float((primary["x"] - independent["x"]).abs().max()),
                                      independent_fits_teacher_loss_start_difference=original["baseline_loss"] - other["baseline_loss"])
                    records.append(record)
                    print(json.dumps(record), flush=True)
    result = {"records": records, "new_downstream_fits": 0,
              "scope": "gate intervention at fixed saved teacher and parent; no new performance estimate",
              "baseline_identical_replays": sum(r["shared_teacher_x_bitwise_equal"] for r in records if r["config"] == "baseline"),
              "baseline_replay_count": sum(r["config"] == "baseline" for r in records),
              "all_replays_complete": len(records) == 20}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    R.save_json(args.output, result)


if __name__ == "__main__":
    main()
