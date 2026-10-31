"""Isolate repeated sparse versus dense backward on the exact same legal graph."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/v2_llm_fusion"))

import torch
import run_tgc_r7_stability as V
import tgc_r7_stable as Q


def run(args):
    torch.set_num_threads(2)
    cache = Path("/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919/cora_r005/text_emb_pool_only.pt")
    data = V.R.R.load_data("cora", "/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip", args.device, cache)
    ids, _ = V.D.set_stream(data, "order19341")
    folder = V.SOURCE / "cora/order19341"
    old, c0 = V.E.load(folder / "recursive_s1_memory.pt"), V.E.load(folder / "c0_s2_memory.pt")
    base = V.V.append_parent(old, V.E.load(folder / "c0_s2_block.pt"), args.device)
    q = V.D.current_assignment(data, ids[2], c0, len(old["x"]), 2, args.device)
    delta = V.T.raw_delta_block(data, ids[2], 2)
    support = Q.M.lifted_support(old, delta, base, q)
    a = Q.M.adjacency(support, "cpu")
    sparse, dense = a.to(args.device), a.to_dense().to(args.device)
    torch.testing.assert_close(sparse.to_dense(), dense, atol=0, rtol=0)
    torch.manual_seed(8400)
    model = Q.R.G.CLIPStudent(support["x"].shape[1]).to(args.device)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    gen = torch.Generator().manual_seed(8400*7+1)
    rows = Q.M.sample_rows(len(support["x"]), len(old["x"]), 64, gen, "compact", len(old["x"])/len(base["x"]))
    x, text = support["x"].to(args.device), support["text_emb"].to(args.device)
    result = {}
    with Q.N.deterministic(True):
        for name, adj in (("sparse", sparse), ("dense", dense)):
            records, gradients = [], []
            for i in range(5):
                model.load_state_dict(state)
                model.zero_grad(set_to_none=True)
                output = model.encode_graph(x, adj)
                loss = Q.T._info_nce(model, x, text, adj, rows.to(args.device))
                loss.backward()
                grad = {k: v.grad.detach().clone() for k, v in model.named_parameters()}
                records.append({"forward_hash": Q.N.fingerprint(output), "loss": float(loss.detach()),
                                "gradient_hash": Q.N.fingerprint(grad)})
                gradients.append(grad)
            result[name] = {"repeats": records,
                "same_forward": len({v["forward_hash"] for v in records}) == 1,
                "same_gradients": len({v["gradient_hash"] for v in records}) == 1,
                "max_gradient_abs_difference": max(float((g[k]-gradients[0][k]).abs().max())
                                                   for g in gradients[1:] for k in g)}
    result.update(same_adjacency_values=True, same_model_and_batches=True,
                  scope="one legal Cora s2 condition; same values, five fixed-weight backward repetitions; no report scores")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    V.E.save(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
