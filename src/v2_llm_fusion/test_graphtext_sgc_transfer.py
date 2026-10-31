import torch
import torch.nn.functional as F

import graphtext_prompt_student as P
from graphtext_sgc_transfer import GraphTextSGCStudent, train_sgc_student


def test_sgc_matches_two_hop_formula_and_cache_invalidates():
    torch.manual_seed(13)
    x = torch.randn(6, 4)
    adj = P.normalized_adjacency(torch.tensor([[0, 1, 3], [1, 2, 4]]), 6, "cpu")
    model = GraphTextSGCStudent(4, out=8, text_dim=8)
    expected = F.normalize(model.graph_proj(adj.to_dense() @ adj.to_dense() @ x), dim=1)
    torch.testing.assert_close(model.encode_graph(x, adj), expected)
    torch.testing.assert_close(model.encode_graph(x, adj), expected)
    x.add_(.2)
    expected = F.normalize(model.graph_proj(adj.to_dense() @ adj.to_dense() @ x), dim=1)
    torch.testing.assert_close(model.encode_graph(x, adj), expected)
    replacement = P.normalized_adjacency(torch.empty(2, 0, dtype=torch.long), 6, "cpu")
    torch.testing.assert_close(model.encode_graph(x, replacement), F.normalize(model.graph_proj(x), dim=1))


def test_default_factory_preserves_original_student_exactly():
    torch.manual_seed(14)
    memory = dict(x=torch.randn(8, 4), text_emb=torch.randn(8, 8),
        labels=torch.tensor([0, 1] * 4), edge_index=torch.empty(2, 0, dtype=torch.long))
    prompts = torch.randn(2, 8)
    left = P.train_prompt_student(memory, prompts, 51, "cpu", steps=3)
    right = P.train_prompt_student(memory, prompts, 51, "cpu", steps=3, model_factory=P.GraphTextPromptStudent)
    for key, tensor in left.state_dict().items():
        torch.testing.assert_close(tensor, right.state_dict()[key], rtol=0, atol=0)
    sgc = train_sgc_student(memory, prompts, 51, "cpu", steps=3)
    assert isinstance(sgc, GraphTextSGCStudent)
    assert not hasattr(sgc, "g1")
    adj = P.normalized_adjacency(memory["edge_index"], 8, "cpu")
    assert sgc.logits(memory["x"], adj, prompts).shape == (8, 2)
