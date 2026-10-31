"""SGC(K=2) readout with the unchanged Prompt Student text tower and losses."""
import math
import weakref

import torch
import torch.nn.functional as F

import graphtext_prompt_student as P


class GraphTextSGCStudent(torch.nn.Module):
    """The SGC graph tower is normalize(linear(A_hat^2 X)), without ReLU."""
    def __init__(self, in_dim, hidden=384, out=384, text_dim=384):
        super().__init__()
        self.graph_proj = torch.nn.Linear(in_dim, out)
        self.text_proj = torch.nn.Linear(text_dim, out)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(1 / .07))
        self._cached_key, self._cached_features = None, None
        self._cached_x, self._cached_adj = None, None

    def encode_graph(self, x, adj):
        # Condensed inputs are fixed during downstream training, as in SGC.
        key = (id(x), id(adj), x._version, adj._version)
        if (self._cached_key != key or x.requires_grad or self._cached_x() is not x
                or self._cached_adj() is not adj):
            propagated = torch.sparse.mm(adj, torch.sparse.mm(adj, x))
            if not x.requires_grad:
                self._cached_key, self._cached_features = key, propagated.detach()
                self._cached_x, self._cached_adj = weakref.ref(x), weakref.ref(adj)
        else:
            propagated = self._cached_features
        return F.normalize(self.graph_proj(propagated), dim=1)

    def encode_text(self, text_emb):
        return F.normalize(self.text_proj(text_emb), dim=1)

    def logits(self, x, adj, prompt_emb):
        return self.logit_scale.exp().clamp(max=100) * (
            self.encode_graph(x, adj) @ self.encode_text(prompt_emb).T)


def train_sgc_student(*args, **kwargs):
    return P.train_prompt_student(*args, **kwargs, model_factory=GraphTextSGCStudent)


def model_config():
    return dict(P.model_config(), name="GraphTextSGCStudent",
        graph_encoder="SGC, K=2; A_hat^2 X followed by one trainable linear map; no ReLU",
        graph_architecture_transfer=True, constructor_unchanged=True,
        same_text_tower_loss_budget_and_prompt_rule=True)
