"""Offline R2-B ranking/selection audit using independently seeded students."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import spearmanr
import torch

import run_tgc_revision_gate as R
import tgc_revision_gate as T
from run_tgc_r2_capacity import CONFIGS
from tgc_r2_response import response_curve, select_indices


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_cell(reference, capacity, dataset, order, config):
    return (reference / dataset / order if config in ("c0", "baseline")
            else capacity / f"{dataset}_{config}" / dataset / order)


def canonical_f1(cell, config, stage):
    arm = "c0" if config == "c0" else "tgc_r"
    return next(row["metrics"]["all"]["cumulative_f1"]["mean"]
                for row in read(cell / "results.json")["results"]
                if row["arm"] == arm and row["stage"] == stage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--capacity", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--text-cache-root", default="/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919")
    args = ap.parse_args()
    torch.set_num_threads(2)
    root = args.output / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    configs = ["c0", "baseline"] + list(CONFIGS)
    protocol = {"name": "tgc-r2-response-offline-v1", "dataset": args.dataset,
                "configs": configs, "selection_seeds": [700, 701], "horizons": [32, 64, 128],
                "batch": R.P.STUDENT.batch, "lr": R.P.STUDENT.lr,
                "source_hashes": {Path(p).name: R.C.sha(p) for p in
                                  (__file__, Path(__file__).with_name("tgc_r2_response.py"), R.C.__file__, T.__file__)},
                "reference": str(args.reference), "capacity": str(args.capacity),
                "labels_used_for_selection": False,
                "probe": "common C0 previous synthetic memory + current raw delta; eight fixed batches each old/new",
                "probe_guard_old_loss_tolerance": .02,
                "topk": "C0 plus top k-1 candidates by current teacher relative loss gain; k=2,4",
                "selection_scope": "offline per-stage diagnostic, not a sequential selector rollout",
                "caveat": "candidate histories differ at stage 2; common C0 probe is diagnostic auxiliary memory"}
    R.stable_lock(root / "protocol.json", protocol)
    cache = Path(args.text_cache_root) / f"{args.dataset}_r005" / "text_emb_pool_only.pt"
    data = R.R.load_data(args.dataset, args.minilm, args.device, cache)
    stage_results = []
    for order in ("canonical", "order19101"):
        source = args.reference / args.dataset / order
        manifest = read(source / "manifest.json")
        groups = manifest["stage_classes"]
        ids, _ = R.A.set_stream(data, groups, .05)
        for stage in (1, 2):
            out = root / order / f"s{stage}"
            out.mkdir(parents=True, exist_ok=True)
            previous = torch.load(source / f"c0_s{stage-1}_memory.pt", map_location="cpu", weights_only=False)
            delta = T.raw_delta_block(data, ids[stage], stage)
            support = T._concat_support(previous, delta)
            candidates = []
            prefix_audits = []
            for name in configs:
                cell = source_cell(args.reference, args.capacity, args.dataset, order, name)
                arm = "c0" if name == "c0" else "tgc_r"
                memory_path = cell / f"{arm}_s{stage}_memory.pt"
                memory = torch.load(memory_path, map_location="cpu", weights_only=False)
                construction = read(cell / f"{arm}_s{stage}_construction.json")
                if T.memory_hash(memory) != construction["memory_sha256"]:
                    raise ValueError("candidate artifact hash mismatch")
                if candidates:
                    for key in ("text_emb", "edge_index", "mass", "depth", "stage"):
                        if not torch.equal(memory[key], first_memory[key]):
                            raise ValueError(f"candidate changes non-X content: {name} {key}")
                    if memory["texts"] != first_memory["texts"]:
                        raise ValueError("candidate texts differ")
                else:
                    first_memory = memory
                scores = []
                for seed in (700, 701):
                    target = out / f"{name}_seed{seed}_response.json"
                    if target.exists():
                        record = read(target)
                        if (record["memory_sha256"] != T.memory_hash(memory)
                                or record["probe_sha256"] != T.memory_hash(support)
                                or record["seed"] != seed or record["horizons"] != [32, 64, 128]):
                            raise ValueError("stale response cache")
                    else:
                        record = response_curve(memory, support, seed, args.device)
                        R.save_json(target, record)
                    scores.append(record)
                if name == "c0":
                    direct = response_curve(memory, support, 700, args.device, probes=False)
                    prefix_audits.append({"all_prefixes_exact": [r["state_sha256"] for r in direct["trajectory"]]
                                         == [r["state_sha256"] for r in scores[0]["trajectory"]],
                                         "seed": 700, "verification_seconds": direct["wall_seconds"]})
                revision = construction["build"].get("revision", {})
                gain = ((revision["baseline_loss"] - revision["final_loss"]) / max(abs(revision["baseline_loss"]), 1e-12)
                        if revision else 0.0)
                candidates.append({"name": name, "cell": str(cell), "responses": scores,
                                   "teacher_relative_gain": gain,
                                   "memory_sha256": T.memory_hash(memory),
                                   "construction_sha256": R.C.sha(cell / f"{arm}_s{stage}_construction.json")})
            teacher_rank = sorted(range(1, len(candidates)), key=lambda i: (-candidates[i]["teacher_relative_gain"], i))
            decisions = []
            for k in (2, 4):
                indices = [0] + teacher_rank[:k-1]
                subset = [candidates[i] for i in indices]
                for horizon in (32, 64, 128, None):
                    decision = select_indices(subset, horizon)
                    scores = [m["balanced_loss"] for m in decision["measures"]]
                    permutation = np.random.default_rng(9702 + stage).permutation(k)
                    decisions.append({"k": k, "horizon": horizon or "curve", "indices": indices,
                                      "chosen": {name: indices[decision[name]] for name in ("balanced", "guarded", "own_training_loss")},
                                      "random_chosen": indices[int(np.random.default_rng(9703 + stage).integers(k))],
                                      "shuffled_chosen": indices[int(np.argmin(np.asarray(scores)[permutation]))],
                                      "teacher_chosen": teacher_rank[0],
                                      "measures": decision["measures"]})
            # Persist choices before reading report-dev F1: an explicit selection/report boundary.
            decision_path = out / "label_free_decisions.json"
            R.save_json(decision_path, {"candidates": candidates, "decisions": decisions,
                                       "prefix_audits": prefix_audits, "report_labels_used": False,
                                       "raw_current_ids_sha256": T.tensor_hash(delta["current_delta_ids"])})
            final = [canonical_f1(Path(c["cell"]), c["name"], stage) for c in candidates]
            correlations = []
            for horizon in (32, 64, 128, None):
                all_scores = select_indices(candidates, horizon)["measures"]
                corr = {key: float(spearmanr([-v[key] for v in all_scores], final).statistic)
                        for key in ("balanced_loss", "old_loss", "new_loss", "train_loss")}
                correlations.append({"horizon": horizon or "curve", "spearman": corr})
            outcomes = []
            for decision in decisions:
                oracle = max(final[i] for i in decision["indices"])
                chosen = dict(decision["chosen"], random=decision["random_chosen"],
                              shuffled=decision["shuffled_chosen"], teacher=decision["teacher_chosen"])
                for policy, index in chosen.items():
                    outcomes.append({"k": decision["k"], "horizon": decision["horizon"],
                                     "policy": policy, "chosen": candidates[index]["name"],
                                     "f1": final[index], "delta_vs_c0_pp": final[index] - final[0],
                                     "regret_pp": oracle - final[index], "fallback_to_c0": index == 0})
            result = {"dataset": args.dataset, "order": order, "stage": stage,
                      "correlations": correlations, "outcomes": outcomes,
                      "candidate_names": configs, "canonical_f1": final,
                      "decisions_sha256": R.C.sha(decision_path), "prefix_audits": prefix_audits,
                      "selection_seconds_all_candidates": sum(r["wall_seconds"] for c in candidates for r in c["responses"]),
                      "trajectories": len(candidates) * 2, "new_canonical_fits": 0}
            R.save_json(out / "diagnostic.json", result)
            stage_results.append(result)
            print(json.dumps({"event": "response_diagnostic", "dataset": args.dataset,
                              "order": order, "stage": stage, "correlations": correlations}), flush=True)
    R.save_json(root / "summary.json", {"protocol": protocol, "results": stage_results,
                                        "prefix_audit_passed": all(p["all_prefixes_exact"] for r in stage_results for p in r["prefix_audits"])})
    print("TGC_R2_RESPONSE_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
