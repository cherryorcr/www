"""Explain NC argmax instability without replacing the original fit metrics."""
import argparse
from pathlib import Path

import numpy as np
import torch

import run_topology_causal_ablation as R
from audit_topology_causal_ablation import load, nc_scores, metric_gap
from summarize_topology_causal_ablation import read_json


def main(root, minilm, device):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read_json(root / "manifest.json")
    audit = read_json(root / "independent_audit.json")
    D = R.U.load_unified(manifest["dataset"], manifest["data"], minilm, device,
                         n_stages=5, split_mode="base_incremental")
    x = D["x"].to(device)
    adj = R.U.G.norm_adj(D["edge_index"], len(x), device)
    report = []
    for warning in audit["reload_warnings"]:
        if warning["task"] != "nc":
            continue
        order, arm, stage, seed = (warning[k] for k in ("order", "arm", "stage", "seed"))
        folder = root / f"order_{order}/nc"
        ev = load(folder / "evaluation.pt")
        ckpt = load(folder / f"{arm}_s{stage}_seed{seed}.pt")
        groups = ev["stage_classes"]
        seen = [c for g in groups[:stage + 1] for c in g]
        active = np.isin(ev["y"].numpy(), seen)
        saved = ckpt["logits"]
        stored_metrics = nc_scores(saved, ev["y"].numpy(), groups, stage)
        trials = []
        for repeat in range(3):
            model = R.U.G.CLIPStudent(x.shape[1]).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            with torch.no_grad():
                logits = model.logits(x, adj, ev["prompts"].to(device)).cpu()[ev["ids"]]
            item = dict(repeat=repeat, max_logit_error=float((logits - saved).abs().max()), scopes={})
            for scope in ("all", "seen"):
                allowed = list(range(logits.shape[1])) if scope == "all" else sorted(seen)
                a, b = saved[active][:, allowed], logits[active][:, allowed]
                changed = a.argmax(1) != b.argmax(1)
                selected = a[changed].topk(2, dim=1).values
                margins = selected[:, 0] - selected[:, 1]
                item["scopes"][scope] = dict(changed_predictions=int(changed.sum()),
                                              max_saved_margin=float(margins.max()) if len(margins) else None,
                                              min_saved_margin=float(margins.min()) if len(margins) else None)
            fresh = nc_scores(logits, ev["y"].numpy(), groups, stage)
            item["metrics"] = fresh
            item["max_metric_gap_pp"] = metric_gap(stored_metrics, fresh)
            item["all_f1_delta_pp"] = fresh["all"]["cumulative_f1"] - stored_metrics["all"]["cumulative_f1"]
            trials.append(item)
        report.append(dict(warning=warning, original_metrics=stored_metrics, trials=trials,
                           final_stage_affected=stage == len(groups) - 1))
    result = dict(diagnosis_sha256=R.sha(__file__), records=report,
                  policy="Original metrics and strict audit warning are retained; no tolerance relaxation.")
    R.save_json(root / "nc_reload_roundoff_diagnosis.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--minilm", required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    main(a.root, a.minilm, a.device)
