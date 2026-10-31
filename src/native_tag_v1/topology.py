"""Fixed synthetic links following the TAGSAM add_adj decoder design.

The decoder is fit only to the caller's legally accessible source graph.
It is temporary; the exported graph contains links, not decoder weights.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from native_core import clean_data


class LinkPredictor(nn.Module):
    def __init__(self, dim, hidden=384):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, left, right):
        return (self.mlp(torch.cat((left, right), -1)).sigmoid().squeeze(-1)
                + self.mlp(torch.cat((right, left), -1)).sigmoid().squeeze(-1)) / 2


def fit_link_predictor(source, teacher, seed, device, steps=160, batch_size=256):
    torch.manual_seed(int(seed))
    source = clean_data(source, device)
    teacher.eval()
    with torch.no_grad():
        embeds = teacher.encode_text(source["text_emb"]).detach()
    edges = source["edge_index"]
    edges = edges[:, edges[0] != edges[1]]
    decoder = LinkPredictor(embeds.shape[1]).to(device)
    if edges.shape[1] == 0:
        return None, {"kind": "self_loops_no_source_links", "steps": 0}
    n = len(embeds)
    occupied = torch.zeros(n * n, dtype=torch.bool, device=device)
    occupied[edges[0] * n + edges[1]] = True
    occupied[edges[1] * n + edges[0]] = True
    occupied[torch.arange(n, device=device) * (n + 1)] = True
    negatives = (~occupied).nonzero().flatten()
    if len(negatives) == 0:
        return None, {"kind": "self_loops_no_negative_links", "steps": 0}
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    losses = []
    for step in range(steps):
        pos = edges[:, torch.randint(edges.shape[1], (min(batch_size, edges.shape[1]),), device=device)]
        neg = negatives[torch.randint(len(negatives), (pos.shape[1],), device=device)]
        positive = decoder(embeds[pos[0]], embeds[pos[1]])
        negative = decoder(embeds[neg // n], embeds[neg % n])
        loss = (F.binary_cross_entropy(positive, torch.ones_like(positive))
                + F.binary_cross_entropy(negative, torch.zeros_like(negative))) / 2
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or step == steps - 1:
            losses.append(float(loss.detach()))
    decoder.eval()
    return decoder, {"kind": "tagsam_add_adj_symmetric_mlp", "steps": steps,
                     "source_nonself_directed_edges": edges.shape[1], "loss_first_last": losses,
                     "class_labels_used": False, "decoder_retained": False}


@torch.no_grad()
def apply_link_predictor(initial, teacher, decoder, degree=1.5):
    device = next(teacher.parameters()).device
    result = clean_data(initial, "cpu")
    n = len(result["x"])
    loops = torch.arange(n, device=device).repeat(2, 1)
    if decoder is None or n < 2 or degree <= 0:
        result["edge_index"] = loops.cpu()
        return result
    text = teacher.encode_text(result["text_emb"].to(device))
    pairs = torch.triu_indices(n, n, offset=1, device=device)
    score = decoder(text[pairs[0]], text[pairs[1]])
    count = min(int(n * degree), len(score))
    chosen = pairs[:, torch.topk(score, count).indices]
    result["edge_index"] = torch.cat((chosen, chosen.flip(0), loops), 1).cpu()
    return result
