"""Same-function repeated training; diagnostic only, never replace main scores."""
import argparse
import json
import os
from pathlib import Path

import torch

import run_handoff_student_transfer_20261008 as H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--text-cache", type=Path, required=True)
    p.add_argument("--minilm", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:1")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostic is immutable")
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text())
    data = H.L.load_large("wikics", args.minilm, args.text_cache, args.device, manifest)
    result = dict(name="same-function-same-seed-legacy-repeat", seed=150, steps=2048, batch=16,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        trainer_sha256=H.sha(H.L.U.G.__file__), runner_sha256=H.sha(__file__), rows=[],
        purpose="diagnose numerical reproducibility; do not replace paired table scores")
    for deterministic in (False, True):
        torch.use_deterministic_algorithms(deterministic)
        for arm in ("k_center", "gecc_hard_tag"):
            source = next(r for r in manifest["rows"] if (r["arm"], r["stage"]) == (arm, 0))
            memory = torch.load(source["memory_path"], map_location="cpu", weights_only=False)
            if H.original_hash(memory, source["hash_kind"]) != source["memory_hash"]:
                raise AssertionError("source memory changed")
            runs, states = [], []
            error = None
            try:
                for repeat in range(2):
                    with H.A.preserve_rng():
                        model = H.L.U.G.train_student(memory, 150, args.device, 2048, lr=.001, batch=16)
                    with torch.no_grad():
                        adj = H.L.U.G.norm_adj(data["edge_index"], len(data["x"]), args.device)
                        pred = model.logits(data["x_device"], adj, data["prompt_emb"].to(args.device)).argmax(1).cpu()
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    runs.append(dict(repeat=repeat, model_hash=H.tensor_dict_hash(state),
                        **H.AB.score_prediction(pred, data, 0)))
                    states.append(state)
                    del model
            except RuntimeError as exc:
                error = str(exc)
            value = dict(arm=arm, deterministic=deterministic, runs=runs, error=error,
                source_memory_hash=source["memory_hash"],
                max_parameter_difference=max(float((states[0][k]-states[1][k]).abs().max())
                    for k in states[0]) if len(states) == 2 else None)
            result["rows"].append(value)
            print(json.dumps(value), flush=True)
    torch.use_deterministic_algorithms(False)
    result["status"] = "complete"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    H.V.write_json(args.output, result)


if __name__ == "__main__":
    main()
