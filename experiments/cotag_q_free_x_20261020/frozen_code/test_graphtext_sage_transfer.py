import torch
import torch.nn.functional as F

import graphtext_prompt_student as P
from graphtext_sage_transfer import GraphTextSAGEStudent, train_sage_student


def test_sage_matches_root_plus_mean_formula_and_invalidates_cache():
    torch.manual_seed(132)
    x = torch.randn(6, 4)
    adj = P.normalized_adjacency(torch.tensor([[0, 1, 3], [1, 2, 4]]), 6, "cpu")
    model = GraphTextSAGEStudent(4, hidden=7, out=8, text_dim=8)
    binary = (adj.to_dense() != 0).float()
    mean = binary / binary.sum(1, keepdim=True)
    hidden = F.relu(model.g1.lin_l(mean @ x) + model.g1.lin_r(x))
    expected = F.normalize(model.g2.lin_l(mean @ hidden) + model.g2.lin_r(hidden), dim=1)
    torch.testing.assert_close(model.encode_graph(x, adj), expected)
    torch.testing.assert_close(model.encode_graph(x, adj), expected)
    empty = P.normalized_adjacency(torch.empty(2, 0, dtype=torch.long), 6, "cpu")
    h = F.relu(model.g1.lin_l(x) + model.g1.lin_r(x))
    expected = F.normalize(model.g2.lin_l(h) + model.g2.lin_r(h), dim=1)
    torch.testing.assert_close(model.encode_graph(x, empty), expected)


def test_edgefree_memory_trains_root_and_neighbor_without_mutation():
    torch.manual_seed(134)
    memory = dict(x=torch.randn(8, 4), text_emb=torch.randn(8, 8),
        labels=torch.tensor([0, 1] * 4), edge_index=torch.empty(2, 0, dtype=torch.long))
    before = {k: v.clone() for k, v in memory.items()}
    prompts = torch.randn(2, 8)
    model = train_sage_student(memory, prompts, 51, "cpu", steps=3)
    for layer in (model.g1, model.g2):
        assert layer.lin_l.weight.grad.abs().sum() > 0
        assert layer.lin_r.weight.grad.abs().sum() > 0
    for k in memory:
        torch.testing.assert_close(before[k], memory[k], rtol=0, atol=0)
    adj = P.normalized_adjacency(memory["edge_index"], 8, "cpu")
    assert model.logits(memory["x"], adj, prompts).shape == (8, 2)
