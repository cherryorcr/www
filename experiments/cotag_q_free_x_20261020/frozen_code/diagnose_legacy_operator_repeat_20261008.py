"""Localize legacy repeat drift with equivalent dense/CPU training operators."""
import argparse
import json
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
    torch.use_deterministic_algorithms(True)
    manifest = json.loads(args.manifest.read_text())
    source = next(r for r in manifest["rows"] if (r["arm"], r["stage"]) == ("k_center", 0))
    memory = torch.load(source["memory_path"], map_location="cpu", weights_only=False)
    if H.original_hash(memory, source["hash_kind"]) != source["memory_hash"]:
        raise AssertionError("source memory changed")
    out = dict(name="legacy-operator-repeat-diagnostic", source_memory_hash=source["memory_hash"],
        seed=150, steps=2048, batch=16, rows=[], runner_sha256=H.sha(__file__),
        trainer_sha256=H.sha(H.L.U.G.__file__),
        scope="diagnostic only: do not substitute any values into the paired comparison")
    data = H.L.load_large("wikics", args.minilm, args.text_cache, args.device, manifest)
    original_norm, original_model = H.L.U.G.norm_adj, H.L.U.G.CLIPStudent
    for mode, device in (("cuda_dense_training", args.device), ("cpu_sparse_training", "cpu")):
        runs, states, initial_hashes = [], [], []

        def capture_model(*a, **kw):
            model = original_model(*a, **kw)
            initial_hashes.append(H.tensor_dict_hash(model.state_dict()))
            return model

        def adjacency(*a, **kw):
            adj = original_norm(*a, **kw)
            return adj.to_dense() if mode == "cuda_dense_training" else adj

        try:
            H.L.U.G.norm_adj = adjacency
            H.L.U.G.CLIPStudent = capture_model
            for repeat in range(2):
                with H.A.preserve_rng():
                    model = H.L.U.G.train_student(memory, 150, device, 2048, lr=.001, batch=16)
                with torch.no_grad():
                    adj = original_norm(data["edge_index"], len(data["x"]), device)
                    pred = model.logits(data["x"].to(device), adj, data["prompt_emb"].to(device)).argmax(1).cpu()
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                states.append(state)
                runs.append(dict(repeat=repeat, initial_hash=initial_hashes[-1],
                    model_hash=H.tensor_dict_hash(state), **H.AB.score_prediction(pred, data, 0)))
                del model
        finally:
            H.L.U.G.norm_adj, H.L.U.G.CLIPStudent = original_norm, original_model
        row = dict(mode=mode, runs=runs,
            max_parameter_difference=max(float((states[0][k]-states[1][k]).abs().max()) for k in states[0]))
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
    out["status"] = "complete"
    H.V.write_json(args.output, out)


if __name__ == "__main__":
    main()
