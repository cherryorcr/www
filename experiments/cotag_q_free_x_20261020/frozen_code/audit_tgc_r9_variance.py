"""Independent audit for the P1 variance-decomposition experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import tgc_r6_experiment as E
import run_tgc_r9_variance as R9

R, T = E.R, E.T


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_dataset(data, root, args):
    checked = 0
    maximum = 0.0
    worst = None
    relation_checks = 0
    for order in R9.ORDERS:
        # The loader starts with the canonical class scope.  Each variance
        # cell was constructed/evaluated after its own order was installed;
        # restore that same stream before reloading checkpoints, otherwise
        # non-canonical orders are scored against the wrong prompt groups.
        E.D.set_stream(data, order)
        out = root / data["dataset"] / order
        for stage in R9.STAGES:
            g0 = E.load(out / f"g0_s{stage}_memory.pt")
            c0 = E.load(out / f"c0_s{stage}_memory.pt")
            for arm in R9.ARMS:
                memory = E.load(out / f"{arm}_s{stage}_memory.pt")
                audit = read(out / f"{arm}_s{stage}_build_audit.json")
                assert T.memory_hash(memory) == audit["memory_sha256"]
                assert audit["historical_raw_reads"] == 0
                assert not audit["report_labels_used"]
                if arm == "text_c0":
                    relation = audit["relative_to_g0"]
                    assert relation["h_hash_equal"] and relation["p_max_abs"] == 0
                    assert relation["x_max_abs"] == 0 and relation["edges_equal"]
                    assert not relation["text_equal"]
                if arm == "topology_c0":
                    relation = audit["relative_to_g0"]
                    assert relation["h_hash_equal"] and relation["p_max_abs"] == 0
                    assert relation["x_max_abs"] == 0 and relation["text_equal"]
                    assert not relation["edges_equal"]
                if arm == "p_initial":
                    relation = audit["relative_to_g0"]
                    assert relation["h_hash_equal"] and relation["p_max_abs"] > 0
                for seed in R9.REPORT_SEEDS:
                    row = read(out / f"{arm}_s{stage}_seed{seed}.json")
                    checkpoint = out / f"{arm}_s{stage}_seed{seed}.pt"
                    assert row["seed"] == seed and row["stage"] == stage
                    assert row["support_memory_sha256"] == T.memory_hash(memory)
                    assert row["checkpoint_sha256"] == R.C.sha(checkpoint)
                    saved = E.load(checkpoint)
                    assert R.C.tensor_hash(saved["state"]) == row["end_state_sha256"]
                    model = R.G.CLIPStudent(memory["x"].shape[1]).to(args.device)
                    model.load_state_dict(saved["state"])
                    # Use the exact canonical report entrypoint used by the
                    # fit wrapper.  A separate legacy scorer has a different
                    # class/prompt scope on some cells and is not a valid
                    # reload comparator for this protocol.
                    scores = R.C.evaluate(model, data, stage)
                    for scope in ("all", "seen"):
                        for metric in ("cumulative_f1", "accuracy", "old_f1", "new_f1"):
                            value = scores[scope][metric]
                            if value is None:
                                continue
                            error = abs(float(value)
                                        - float(row["metrics"][scope][metric]))
                            if error > maximum:
                                maximum = error
                                worst = {"order": order, "stage": stage,
                                         "arm": arm, "seed": seed,
                                         "scope": scope, "metric": metric,
                                         "row": row["metrics"][scope][metric],
                                         "reload": value}
                    del model
                    checked += 1
                relation_checks += 1
    if maximum >= 1e-7:
        raise AssertionError(f"reload metric mismatch: {worst}")
    result = {"status": "passed", "dataset": data["dataset"],
              "fit_rows_checked": checked,
              "construction_cells_checked": relation_checks,
              "maximum_reload_metric_error": maximum,
              "worst_reload_metric": worst,
              "test_labels_evaluated": False}
    E.save(root / "r9_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=R9.DATASETS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--text-cache-root", type=Path,
                        default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = parser.parse_args()
    cache = args.text_cache_root / f"{args.dataset}_r005" / "text_emb_pool_only.pt"
    data = R.R.load_data(args.dataset, args.minilm, args.device, cache)
    result = audit_dataset(data, args.root, args)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
