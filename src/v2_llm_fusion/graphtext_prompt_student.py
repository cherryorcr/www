"""Prompt-only GraphText Student for a common zero-shot evaluation protocol.

The model has no trainable classifier head.  It learns a graph projection and
a shared text projection from synthetic memory, while inference compares the
graph embedding only with frozen MiniLM embeddings of the class prompts.

Unlike the historical row-wise InfoNCE Student, the pair loss treats rows of
the same class as multiple positives.  This avoids turning repeated or very
similar same-class payloads into false negatives.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GraphConv(torch.nn.Module):
    def __init__(self, din: int, dout: int):
        super().__init__()
        self.lin = torch.nn.Linear(int(din), int(dout))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adj, self.lin(x))


class GraphTextPromptStudent(torch.nn.Module):
    """Two-layer GCN with prompt-only zero-shot prediction."""

    def __init__(self, in_dim: int, hidden: int = 384, out: int = 384,
                 text_dim: int = 384):
        super().__init__()
        self.g1 = GraphConv(in_dim, hidden)
        self.g2 = GraphConv(hidden, out)
        self.text_proj = torch.nn.Linear(text_dim, out)
        self.logit_scale = torch.nn.Parameter(
            torch.ones([]) * math.log(1.0 / 0.07))

    def encode_graph(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.g1(x, adj))
        return F.normalize(self.g2(h, adj), dim=1)

    def encode_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_proj(text_emb), dim=1)

    def logits(self, x: torch.Tensor, adj: torch.Tensor,
               prompt_emb: torch.Tensor) -> torch.Tensor:
        graph = self.encode_graph(x, adj)
        prompts = self.encode_text(prompt_emb)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * (graph @ prompts.T)


def normalized_adjacency(edge_index: torch.Tensor, n: int,
                         device: str) -> torch.Tensor:
    """Build the same symmetric-normalized self-loop graph used by the suite."""
    edge_index = edge_index.long().cpu()
    loops = torch.arange(int(n), dtype=torch.long).repeat(2, 1)
    index = torch.cat((edge_index, edge_index.flip(0), loops), dim=1)
    keys = torch.unique(index[0] * int(n) + index[1], sorted=True)
    index = torch.stack((keys // int(n), keys % int(n))).to(device)
    degree = torch.bincount(index[0], minlength=int(n)).float().clamp_min(1.0)
    value = degree[index[0]].rsqrt() * degree[index[1]].rsqrt()
    return torch.sparse_coo_tensor(
        index, value, (int(n), int(n)), device=device).coalesce()


def _masked_logsumexp(values: torch.Tensor, mask: torch.Tensor,
                      dim: int) -> torch.Tensor:
    neg_inf = torch.finfo(values.dtype).min
    return torch.logsumexp(values.masked_fill(~mask, neg_inf), dim=dim)


def _normalized_sample_weights(sample_weights: torch.Tensor | None,
                               n: int, device: torch.device,
                               dtype: torch.dtype) -> torch.Tensor | None:
    if sample_weights is None:
        return None
    weights = sample_weights.to(device=device, dtype=dtype).flatten()
    if len(weights) != int(n):
        raise ValueError("sample_weights must have one value per row")
    if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError("sample_weights must be finite and strictly positive")
    return weights / weights.sum().clamp_min(torch.finfo(dtype).eps)


def multi_positive_pair_loss(graph: torch.Tensor, text: torch.Tensor,
                             labels: torch.Tensor,
                             scale: torch.Tensor,
                             sample_weights: torch.Tensor | None = None,
                             chunk_size: int = 0
                             ) -> torch.Tensor:
    """Symmetric multi-positive graph/text contrastive loss."""
    weights = _normalized_sample_weights(
        sample_weights, len(graph), graph.device, graph.dtype)
    if chunk_size > 0 and len(graph) > chunk_size:
        # Exact full-candidate loss, checkpointed by anchors. No negatives
        # are sampled; checkpointing avoids retaining an n-by-n autograd graph.
        anchor_weights = (torch.full_like(labels, 1.0 / len(labels), dtype=graph.dtype)
                          if weights is None else weights)
        candidate_log_weights = (torch.zeros_like(anchor_weights)
                                 if weights is None else weights.log())

        def block(left, right, local_labels, all_labels, log_weights, temperature):
            logits = temperature * (left @ right.T) + log_weights[None, :]
            mask = local_labels[:, None].eq(all_labels[None, :])
            return torch.logsumexp(logits, dim=1) - _masked_logsumexp(logits, mask, 1)

        terms = []
        for left, right in ((graph, text), (text, graph)):
            for start in range(0, len(graph), chunk_size):
                end = min(len(graph), start + chunk_size)
                values = checkpoint(block, left[start:end], right, labels[start:end],
                                    labels, candidate_log_weights, scale,
                                    use_reentrant=False)
                terms.append((anchor_weights[start:end] * values).sum())
        return 0.5 * torch.stack(terms).sum()
    similarity = scale * (graph @ text.T)
    positive = labels.view(-1, 1).eq(labels.view(1, -1))
    row_logits = similarity if weights is None else similarity + weights.log()[None, :]
    col_logits = similarity if weights is None else similarity + weights.log()[:, None]
    row_loss = -(_masked_logsumexp(row_logits, positive, 1)
                 - torch.logsumexp(row_logits, dim=1))
    col_loss = -(_masked_logsumexp(col_logits, positive, 0)
                 - torch.logsumexp(col_logits, dim=0))
    if weights is None:
        row, col = row_loss.mean(), col_loss.mean()
    else:
        row, col = (weights * row_loss).sum(), (weights * col_loss).sum()
    return 0.5 * (row + col)


def prompt_margin_loss(prompt_logits: torch.Tensor, labels: torch.Tensor,
                       margin: float,
                       sample_weights: torch.Tensor | None = None
                       ) -> torch.Tensor:
    """Multiclass margin against the strongest non-target prompt."""
    positive = prompt_logits.gather(1, labels.view(-1, 1)).squeeze(1)
    target_mask = F.one_hot(labels, num_classes=prompt_logits.shape[1]).bool()
    negative = prompt_logits.masked_fill(target_mask, -torch.inf).max(dim=1).values
    values = F.relu(float(margin) - positive + negative)
    weights = _normalized_sample_weights(
        sample_weights, len(labels), prompt_logits.device, prompt_logits.dtype)
    return values.mean() if weights is None else (weights * values).sum()


def train_prompt_student(
        memory: dict,
        prompt_emb: torch.Tensor,
        seed: int,
        device: str,
        steps: int,
        lr: float = 1e-3,
        batch: int = 24,
        lambda_pair: float = 0.5,
        lambda_prompt: float = 1.0,
        lambda_margin: float = 0.1,
        margin: float = 0.20, model_factory=None) -> GraphTextPromptStudent:
    """Train the fixed protocol; prediction remains prompt-only."""
    torch.manual_seed(int(seed))
    x = memory["x"].float().to(device)
    text = memory["text_emb"].float().to(device)
    labels = memory["labels"].long().to(device)
    prompt_emb = prompt_emb.float().to(device)
    factory = GraphTextPromptStudent if model_factory is None else model_factory
    model = factory(
        int(memory["x"].shape[1]), text_dim=int(text.shape[1])).to(device)
    adj = normalized_adjacency(memory.get("edge_index",
                                           torch.empty((2, 0), dtype=torch.long)),
                               len(x), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    generator = torch.Generator(device="cpu").manual_seed(int(seed) * 7 + 1)
    n = len(x)
    for _ in range(int(steps)):
        rows = torch.randperm(n, generator=generator)[:min(int(batch), n)]
        rows = rows.to(device)
        graph = model.encode_graph(x, adj)[rows]
        text_rows = model.encode_text(text[rows])
        scale = model.logit_scale.exp().clamp(max=100.0)
        pair = multi_positive_pair_loss(graph, text_rows, labels[rows], scale)
        logits = scale * (graph @ model.encode_text(prompt_emb).T)
        prompt = F.cross_entropy(logits, labels[rows])
        margin_loss = prompt_margin_loss(logits, labels[rows], margin)
        loss = (float(lambda_pair) * pair
                + float(lambda_prompt) * prompt
                + float(lambda_margin) * margin_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite prompt Student loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval()


def model_config() -> dict:
    return {
        "name": "GraphTextPromptStudent",
        "graph_encoder": "two-layer symmetric-normalized GCN",
        "text_encoder": "frozen MiniLM embeddings",
        "classifier_head": "none; prompt cosine only",
        "pair_loss": "symmetric multi-positive graph-text InfoNCE",
        "prompt_loss": "cross-entropy over encoded class prompts",
        "margin_loss": "multiclass prompt margin",
        "lambda_pair": 0.5,
        "lambda_prompt": 1.0,
        "lambda_margin": 0.1,
        "margin": 0.20,
        "same_class_rows_are_positive": True,
    }
