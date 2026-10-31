"""Frozen-memory GraphSAGE transfer with the unchanged Prompt training loss.

This is the explicit self-loop mean variant: the neighborhood includes the
node itself, in addition to SAGE's root transform. Edge-free memory therefore
trains both transforms. It is not canonical loop-free GraphSAGE.
"""
import math
import weakref

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

import graphtext_prompt_student as P


class GraphTextSAGEStudent(torch.nn.Module):
    def __init__(self, in_dim, hidden=384, out=384, text_dim=384):
        super().__init__()
        self.g1 = SAGEConv(in_dim, hidden, aggr="mean", normalize=False)
        self.g2 = SAGEConv(hidden, out, aggr="mean", normalize=False)
        self.text_proj = torch.nn.Linear(text_dim, out)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(1 / .07))
        self._adj_ref, self._adj_version, self._binary_adj = None, None, None

    def binary_adjacency(self, adj):
        if (self._adj_ref is None or self._adj_ref() is not adj
                or self._adj_version != adj._version):
            coalesced = adj.coalesce()
            binary = torch.sparse_coo_tensor(coalesced.indices(),
                torch.ones_like(coalesced.values()), adj.shape,
                device=adj.device).coalesce()
            # Sparse message-and-aggregate avoids an E x hidden message tensor.
            self._binary_adj = binary.to_sparse_csr()
            self._adj_ref, self._adj_version = weakref.ref(adj), adj._version
        return self._binary_adj

    def encode_graph(self, x, adj):
        adjacency = self.binary_adjacency(adj)
        return F.normalize(self.g2(F.relu(self.g1(x, adjacency)), adjacency), dim=1)

    def encode_text(self, text_emb):
        return F.normalize(self.text_proj(text_emb), dim=1)

    def logits(self, x, adj, prompt_emb):
        return self.logit_scale.exp().clamp(max=100) * (
            self.encode_graph(x, adj) @ self.encode_text(prompt_emb).T)


def train_sage_student(*args, **kwargs):
    return P.train_prompt_student(*args, **kwargs, model_factory=GraphTextSAGEStudent)


def model_config():
    return dict(P.model_config(), name="GraphTextSAGEStudent",
        graph_encoder="PyG two-layer SAGEConv mean, ReLU, self-loop neighborhood plus separate root map",
        graph_architecture_transfer=True, constructor_unchanged=True,
        topology="same undirected edges and self loops; binary mean aggregation, not GCN symmetric weights",
        caveat="not canonical loop-free SAGE; edge-free training cannot identify separate root/neighbor roles",
        same_text_tower_loss_budget_and_prompt_rule=True)
