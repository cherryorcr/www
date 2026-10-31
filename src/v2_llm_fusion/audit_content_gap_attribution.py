"""Independent reload audit for the content-gap attribution experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_content_gap_attribution as A
import run_p0_random_synthetic as R
import run_small_comparison_audited as C
import small_comparison_metrics as M

G, P = C.G, C.P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minilm", default=(
        "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip"))
    ap.add_argument("--text-cache-root", default=(
        "/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919"))
    args = ap.parse_args()
    root = Path(args.root) / args.dataset
    data = R.load_data(args.dataset, args.minilm, args.device,
                       Path(args.text_cache_root) / f"{args.dataset}_r005"
                       / "text_emb_pool_only.pt")
    result = json.loads((root / "results.json").read_text())
    arms = result["manifest"]["arms"]
    checks = []
    failures = []
    for stage in range(len(result["manifest"]["stage_classes"])):
        c0 = torch.load(root / f"c0_s{stage}_memory.pt", map_location="cpu",
                        weights_only=False)
        c0_hash = A.memory_hash(c0)
        for arm in arms:
            memory = torch.load(root / f"{arm}_s{stage}_memory.pt",
                                map_location="cpu", weights_only=False)
            edge_equal = torch.equal(memory["edge_index"], c0["edge_index"])
            rows_equal = len(memory["x"]) == len(c0["x"])
            for seed in result["manifest"]["student_seeds"]:
                row_path = root / f"{arm}_s{stage}_seed{seed}.json"
                ckpt_path = root / f"{arm}_s{stage}_seed{seed}.pt"
                row = json.loads(row_path.read_text())
                saved = torch.load(ckpt_path, map_location="cpu",
                                   weights_only=False)
                model = G.CLIPStudent(data["x"].shape[1]).to(args.device)
                model.load_state_dict(saved["state"])
                model.eval()
                with torch.no_grad():
                    logits = model.logits(
                        data["x_device"], data["real_adj"],
                        data["prompt_emb"].to(args.device)).cpu()
                mask = data["report_dev_mask"]
                metrics = M.score_logits(
                    logits[mask], data["y"][mask].numpy(),
                    data["stage_classes"], stage)
                equal = json.loads(json.dumps(metrics)) == row["metrics"]
                support_equal = row["arm_support_sha256"] == A.memory_hash(memory)
                checks.append({"stage": stage, "arm": arm, "seed": seed,
                               "metrics_equal": equal,
                               "support_equal": support_equal,
                               "edge_equal_to_c0": edge_equal,
                               "rows_equal_to_c0": rows_equal,
                               "c0_hash": c0_hash})
                if not (equal and support_equal and edge_equal and rows_equal):
                    failures.append(checks[-1])
                del model
    out = {"dataset": args.dataset, "checks": len(checks),
           "failures": failures, "failure_count": len(failures),
           "edge_fixed_all": all(v["edge_equal_to_c0"] for v in checks),
           "metrics_reload_all": all(v["metrics_equal"] for v in checks),
           "support_hash_all": all(v["support_equal"] for v in checks),
           "rows_equal_all": all(v["rows_equal_to_c0"] for v in checks)}
    (root / "reload_audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
