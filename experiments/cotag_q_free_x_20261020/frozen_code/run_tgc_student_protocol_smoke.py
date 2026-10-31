"""Student-only protocol diagnostic on sealed C0 memories.

No memory construction changes.  Compare the canonical scratch student with
warm-start from the previous C0 stage using the same memory, steps and report
seeds.  This is a diagnostic for plasticity, not a new condensation method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_tgc_r15_r11_cross as R15


R, T, D, B = R15.R, R15.T, R15.D, R15.B


def load(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--c0-root", required=True, type=Path)
    ap.add_argument("--dataset", required=True, choices=("cora", "citeseer"))
    ap.add_argument("--order", required=True)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--seed", nargs="+", type=int, default=[150, 151])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    ap.add_argument("--text-cache-root", type=Path,
                    default=Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.text_cache_root / f"{args.dataset}_r005" / "text_emb_pool_only.pt"
    data = R.R.load_data(args.dataset, args.minilm, args.device, cache)
    D.set_stream(data, args.order)
    cell = args.c0_root / args.dataset / args.order
    memory = load(cell / f"c0_s{args.stage}_memory.pt")
    previous = cell / f"c0_s{args.stage - 1}_seed{{seed}}.pt"
    out = args.output / args.dataset / args.order
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seed:
        base_row = json.loads((cell / f"c0_s{args.stage}_seed{seed}.json").read_text())
        previous_state = load(Path(str(previous).format(seed=seed)))["state"]
        checkpoint = out / f"warm_start_s{args.stage}_seed{seed}.pt"
        warm = R.C.fit(memory, int(seed), R.P.student_steps(len(memory["x"])),
                       args.device, data, args.stage, initial=previous_state,
                       checkpoint=checkpoint)
        metrics, base = warm["metrics"]["all"], base_row["metrics"]["all"]
        row = {
            "dataset": args.dataset, "order": args.order, "stage": args.stage,
            "seed": seed, "scratch_final_f1": base["cumulative_f1"],
            "warm_final_f1": metrics["cumulative_f1"],
            "delta_final_f1": metrics["cumulative_f1"] - base["cumulative_f1"],
            "scratch_new_f1": base["new_f1"], "warm_new_f1": metrics["new_f1"],
            "delta_new_f1": metrics["new_f1"] - base["new_f1"],
            "scratch_old_f1": base["old_f1"], "warm_old_f1": metrics["old_f1"],
            "delta_old_f1": metrics["old_f1"] - base["old_f1"],
            "start_state_sha256": warm["start_state_sha256"],
            "previous_state_sha256": R.C.tensor_hash(previous_state),
        }
        save_json(out / f"warm_start_s{args.stage}_seed{seed}.json", {**warm, **row})
        rows.append(row)
        print(json.dumps(row), flush=True)
    save_json(args.output / "manifest.json", {
        "name": "tgc-student-protocol-smoke-v1", "dataset": args.dataset,
        "order": args.order, "stage": args.stage, "seeds": args.seed,
        "memory": "sealed C0; unchanged", "comparison": "scratch vs previous-stage warm-start",
        "report_labels_used_for_construction": False,
    })
    save_json(args.output / "summary.json", {"rows": rows})


if __name__ == "__main__":
    main()
