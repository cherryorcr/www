"""Zero-shot evaluation of the OFFICIAL pretrained GraphCLIP checkpoint.

Loads the released GPS graph tower + MiniLM text tower from
checkpoints/pretrained_graphclip.pt (config inferred from weights:
in=384, channels=1024, pe_dim=8, layers=12, multihead attention) and runs the
official protocol per node: 2-hop ego subgraph, 32-step random-walk PE,
GPS forward -> 384-d graph embedding vs class-prompt text embeddings.

Evaluated on the SAME deterministic dev split (node_id%10==1) and the same
prompts as our condensation experiments, so the number is directly comparable
to our condensed-student results. No training happens here; this is the
official model as released.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import Data
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.utils import k_hop_subgraph

sys.path.insert(0, "/home/pc/research/GraphCLIP")
from models.gt import GPS  # official code

sys.path.insert(0, str(Path(__file__).parent))
from run_gate1_continual import FrozenMiniLM

PROMPT_SETS = {
    "cora": ["a research paper about case based", "a research paper about genetic algorithms",
             "a research paper about neural networks", "a research paper about probabilistic methods",
             "a research paper about reinforcement learning", "a research paper about rule learning",
             "a research paper about theory"],
    "citeseer": ["a research paper about agents", "a research paper about machine learning",
                 "a research paper about information retrieval", "a research paper about databases",
                 "a research paper about human computer interaction",
                 "a research paper about artificial intelligence"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cora", "citeseer"], required=True)
    parser.add_argument("--ckpt", default="/home/pc/research/GraphCLIP/checkpoints/pretrained_graphclip.pt")
    parser.add_argument("--minilm", default="/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--max-sub", type=int, default=256)
    parser.add_argument("--batch-graphs", type=int, default=48)
    args = parser.parse_args()

    device = args.device
    data = torch.load(f"/home/pc/research/GraphCLIP/processed_data/{args.dataset}.pt",
                      map_location="cpu", weights_only=False)
    x, y, edge_index = data.x.float(), data.y, data.edge_index
    n = len(x)
    dev_nodes = torch.where((torch.arange(n) % 10) == 1)[0]

    # ---- official graph tower ----
    gps = GPS(in_dim=384, channels=1024, out_dim=384, pe_dim=8, num_layers=12,
              attn_type="multihead", attn_kwargs={"dropout": 0.0})
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    graph_sd = {k[len("graph_model."):]: v for k, v in sd.items() if k.startswith("graph_model.")}
    missing, unexpected = gps.load_state_dict(graph_sd, strict=False)
    gps = gps.to(device).eval()
    print(json.dumps({"missing_keys": len(missing), "unexpected_keys": len(unexpected),
                      "missing_sample": missing[:4], "unexpected_sample": unexpected[:4]}), flush=True)

    # ---- text tower (same MiniLM weights extracted from this checkpoint) ----
    encoder = FrozenMiniLM(args.minilm, device)
    prompt_emb = F.normalize(encoder.encode(PROMPT_SETS[args.dataset]), dim=1).to(device)

    pe_tf = AddRandomWalkPE(walk_length=32, attr_name="pe")

    def ego(node: int):
        subset, sub_edges, mapping, _ = k_hop_subgraph(
            int(node), args.hops, edge_index, relabel_nodes=True, num_nodes=n)
        if len(subset) > args.max_sub:  # deterministic fallback to 1 hop
            subset, sub_edges, mapping, _ = k_hop_subgraph(
                int(node), 1, edge_index, relabel_nodes=True, num_nodes=n)
        g = Data(x=x[subset], edge_index=sub_edges, num_nodes=len(subset))
        g = pe_tf(g)
        return g, int(mapping[0])

    preds = []
    with torch.no_grad():
        for start in range(0, len(dev_nodes), args.batch_graphs):
            chunk = dev_nodes[start:start + args.batch_graphs]
            xs, pes, es, batch_vec, centers = [], [], [], [], []
            offset = 0
            for bi, node in enumerate(chunk):
                g, center = ego(int(node))
                xs.append(g.x)
                pes.append(g.pe)
                es.append(g.edge_index + offset)
                batch_vec.append(torch.full((g.num_nodes,), bi, dtype=torch.long))
                centers.append(offset + center)
                offset += g.num_nodes
            bx = torch.cat(xs).to(device)
            bpe = torch.cat(pes).to(device)
            be = torch.cat(es, 1).to(device)
            bb = torch.cat(batch_vec).to(device)
            bc = torch.tensor(centers, device=device)
            g_emb, _ = gps(bx, bpe, be, bb, bc)
            g_emb = F.normalize(g_emb, dim=1)
            preds.append((g_emb @ prompt_emb.T).argmax(1).cpu())
    pred = torch.cat(preds).numpy()
    yy = y[dev_nodes].numpy()
    result = {
        "dataset": args.dataset,
        "model": "official pretrained GraphCLIP (GPS-12L-1024 + MiniLM)",
        "protocol": f"{args.hops}-hop ego subgraph, RW-PE(32), dev split node_id%10==1",
        "n_eval": int(len(dev_nodes)),
        "dev_macro_f1": 100 * f1_score(yy, pred, average="macro", zero_division=0),
        "dev_accuracy": 100 * accuracy_score(yy, pred),
        "test_accessed": False,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output, "w"), indent=1)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
