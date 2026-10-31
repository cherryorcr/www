"""Audit an already sealed R8 result without rewriting its protocol lock."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code/v2_llm_fusion"
sys.path.insert(0, str(CODE))

import tgc_r6_experiment as E  # noqa: E402
import run_tgc_r8_plasticity as R8  # noqa: E402
from audit_tgc_r2b_student_revision import independent_scores  # noqa: E402


def audit_dataset(dataset: str, args) -> dict:
    spec = R8.protocol()
    E.configure(spec)
    E.D.verify_seal(args.output)
    cache = (Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919")
             / f"{dataset}_r005/text_emb_pool_only.pt")
    data = E.R.R.load_data(dataset, args.minilm, args.device, cache)
    dataset_checked = 0
    dataset_max_error = 0.0
    per_order = {}
    for order in spec["orders"]:
        ids, budgets = E.D.set_stream(data, order)
        out = args.output / dataset / order
        manifest = E.read(out / "manifest.json")
        assert manifest["stage_source_ids"] == [value.tolist() for value in ids]
        assert manifest["budgets"] == budgets
        cache_scores, checked, maximum = {}, 0, 0.0
        for stage in spec["evaluation_stages"]:
            E.D.ARMS = tuple(spec["arms"])
            E.D.verify_decision(out, stage)
            R8.validate(data, out, stage, ids, budgets, args)
            for arm in spec["arms"]:
                memory = E.load(out / f"{arm}_s{stage}_memory.pt")
                for seed in spec["report_seeds"]:
                    row = E.read(out / f"{arm}_s{stage}_seed{seed}.json")
                    checkpoint = out / f"{arm}_s{stage}_seed{seed}.pt"
                    assert row["checkpoint_sha256"] == E.R.C.sha(checkpoint)
                    assert row["support_memory_sha256"] == E.T.memory_hash(memory)
                    assert row["steps"] == E.R.P.student_steps(budgets[stage])
                    assert row["seed"] == seed
                    key = (stage, row["checkpoint_sha256"])
                    if key not in cache_scores:
                        model = E.R.G.CLIPStudent(memory["x"].shape[1]).to(args.device)
                        model.load_state_dict(E.load(checkpoint)["state"])
                        cache_scores[key] = independent_scores(model, data, stage)
                        del model
                    for scope in ("all", "seen"):
                        for metric, value in cache_scores[key][scope].items():
                            if value is not None:
                                maximum = max(maximum,
                                              abs(value - row["metrics"][scope][metric]))
                    checked += 1
        expected = len(spec["evaluation_stages"]) * len(spec["arms"]) * len(spec["report_seeds"])
        assert checked == expected
        assert maximum < 1e-7
        E.save(out / "audit.json", {
            "status": "passed", "fit_slots": checked,
            "unique_reload_scores": len(cache_scores),
            "maximum_metric_error_pp": maximum,
            "test_labels_evaluated": False,
            "protocol_lock_preserved": True,
        })
        dataset_checked += checked
        dataset_max_error = max(dataset_max_error, maximum)
        per_order[order] = {"fit_slots": checked, "maximum_metric_error_pp": maximum}
    return {"dataset": dataset, "fit_slots": dataset_checked,
            "maximum_metric_error_pp": dataset_max_error, "orders": per_order}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cora", "citeseer"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = audit_dataset(args.dataset, args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
