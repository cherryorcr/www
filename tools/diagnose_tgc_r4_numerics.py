"""Measure native/functional Adam drift without relaxing any assertion."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "v2_llm_fusion"))
import tgc_student_revision as S
import tgc_revision_gate as T


def canonical(memory, seed, steps, device, foreach=None, dense=False):
    x, text = memory["x"].to(device), memory["text_emb"].to(device)
    adj = S.R.C.training_adjacency(memory, device)
    if dense:
        adj = adj.to_dense()
    torch.manual_seed(seed)
    model = S.R.G.CLIPStudent(x.shape[1], text_dim=text.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.001, foreach=foreach)
    gen = torch.Generator().manual_seed(seed * 7 + 1)
    for _ in range(steps):
        ids = torch.randperm(len(x), generator=gen)[:min(16, len(x))].to(device)
        loss = T._info_nce(model, x, text, adj, ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return {k: v.detach() for k, v in model.named_parameters()}


def functional(memory, seed, steps, device, dense=False):
    x, text = memory["x"].to(device), memory["text_emb"].to(device)
    adj = S.R.C.training_adjacency(memory, device)
    if dense:
        adj = adj.to_dense()
    return {k: v.detach() for k, v in S.unroll(x, adj, text, seed, steps, False).items()}


def compare(a, b, memory, device, dense=False):
    maximum, square, total, mismatched = 0., 0., 0, 0
    for name in a:
        diff = (a[name] - b[name]).abs()
        maximum = max(maximum, float(diff.max()))
        square += float(diff.square().sum())
        total += diff.numel()
        mismatched += int((diff > 2e-5 + 2e-4 * b[name].abs()).sum())
    x, text = memory["x"].to(device), memory["text_emb"].to(device)
    adj = S.R.C.training_adjacency(memory, device)
    if dense:
        adj = adj.to_dense()
    ga, gb = S.graph(a, x, adj), S.graph(b, x, adj)
    ids = torch.arange(min(16, len(x)), device=device)
    return {"max_parameter_abs_difference": maximum, "rms_parameter_difference": (square / total) ** .5,
            "strict_tolerance_mismatch_elements": mismatched, "total_elements": total,
            "min_graph_cosine": float(F.cosine_similarity(ga, gb).min()),
            "probe_nce_abs_difference": float((S.contrastive(a, x, adj, text, ids)
                                               - S.contrastive(b, x, adj, text, ids)).abs())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    m = torch.load(args.memory, map_location="cpu", weights_only=False)
    result = {"memory": str(args.memory), "memory_sha256": T.memory_hash(m), "torch": torch.__version__,
              "device": args.device, "seed": 8100, "steps": 32, "experiments": []}
    for dense in (False, True):
        states = {"native_a": canonical(m, 8100, 32, args.device, dense=dense),
                  "native_b": canonical(m, 8100, 32, args.device, dense=dense),
                  "single_tensor": canonical(m, 8100, 32, args.device, foreach=False, dense=dense),
                  "functional_a": functional(m, 8100, 32, args.device, dense=dense),
                  "functional_b": functional(m, 8100, 32, args.device, dense=dense)}
        for a, b in (("native_a", "native_b"), ("functional_a", "functional_b"),
                     ("native_a", "functional_a"), ("single_tensor", "functional_a")):
            result["experiments"].append({"adjacency": "dense" if dense else "sparse",
                                          "pair": [a, b], **compare(states[a], states[b], m, args.device, dense)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
