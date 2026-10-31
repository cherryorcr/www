"""Post-run teacher timing: a separate measurement, never a recovered run timer."""
import argparse
import json
from pathlib import Path
from statistics import mean, median
import time

import torch

import run_tgc_r2b_student_revision as B

R, T = B.R, B.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--reference-root", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--text-cache-root", type=Path,
                    default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = ap.parse_args()
    if args.repeats < 2:
        raise ValueError("at least two timing repetitions")
    torch.set_num_threads(2)
    rows = []
    for dataset in ("cora", "citeseer"):
        data = R.R.load_data(dataset, args.minilm, args.device,
                            args.text_cache_root / f"{dataset}_r005" / "text_emb_pool_only.pt")
        for order in B.ORDERS:
            reference = args.reference_root / dataset / order
            original = B.read(reference / "results.json")
            ids, _ = R.A.set_stream(data, original["manifest"]["stage_classes"], .05)
            for stage in (1, 2):
                old = torch.load(reference / f"c0_s{stage-1}_memory.pt", map_location="cpu", weights_only=False)
                delta = T.raw_delta_block(data, ids[stage], stage)
                baseline = next(c for c in original["constructions"] if c["stage"] == stage and c["arm"] == "c0")
                decision = B.read(args.root / dataset / order / f"s{stage}_decisions.json")
                cfg = T.RevisionConfig(protected_fraction=.8, revision_fraction_of_old=.1,
                                       revision_radius=.1, teacher_steps=128)
                samples = []
                for repeat in range(args.repeats):
                    R.sync(args.device)
                    started = time.perf_counter()
                    _, audit = T.revise_old_memory(old, delta, cfg, args.device, "gate")
                    R.sync(args.device)
                    elapsed = time.perf_counter() - started
                    samples.append({"repeat": repeat, "seconds": elapsed,
                                    "updated_rows_match_original": audit["updated_rows"] == decision["editable_rows"]})
                teacher_seconds = median(s["seconds"] for s in samples)
                c0_seconds = baseline["build"]["build_seconds"]
                for k in B.HORIZONS:
                    candidate = next(a for a in decision["arms"] if a["arm"] == f"student_k{k}")
                    meta_seconds = candidate["audit"]["optimization"]["seconds"]
                    rows.append({"dataset": dataset, "order": order, "stage": stage, "horizon": k,
                                 "historical_c0_build_seconds": c0_seconds,
                                 "remeasured_teacher_seconds_median": teacher_seconds,
                                 "original_meta_optimization_seconds": meta_seconds,
                                 "component_estimate_seconds": c0_seconds + teacher_seconds + meta_seconds,
                                 "component_estimate_over_historical_c0": (c0_seconds + teacher_seconds + meta_seconds) / c0_seconds,
                                 "teacher_timing_samples": samples})
                print(json.dumps({"dataset": dataset, "order": order, "stage": stage,
                                  "teacher_seconds_median": teacher_seconds}), flush=True)
    result = {
        "scope": "post-run component cost estimate, not measured end-to-end latency",
        "rows": rows,
        "aggregate": [{"horizon": k,
                       "mean_component_seconds": mean(r["component_estimate_seconds"] for r in rows if r["horizon"] == k),
                       "mean_ratio_over_historical_c0": mean(r["component_estimate_over_historical_c0"] for r in rows if r["horizon"] == k)}
                      for k in B.HORIZONS],
        "limitations": ["teacher elapsed time not captured by original runner",
                        "teacher time remeasured after formal run, three repetitions per support",
                        "C0 construction time reused from earlier experiment, loads may differ",
                        "does not include cold start, source text encoding, checkpoint IO or audit",
                        "cost replay does not select or change any formal candidate or prediction"],
    }
    R.save_json(args.root / "cost_remeasurement.json", result)
    print(json.dumps(result["aggregate"]))


if __name__ == "__main__":
    main()
