"""Unit tests for the fixed-X P-only selector."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import student_guided_text_assignment as P


class Encoder:
    def encode(self, texts):
        values = torch.tensor(
            [[len(text), sum(map(ord, text)) % 41, 1.0, 2.0]
             for text in texts], dtype=torch.float32)
        return F.normalize(values, dim=1)


class _Layer(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.lin = nn.Linear(din, dout)


class TinyStudent(nn.Module):
    def __init__(self, in_dim, hidden=8, out=4, text_dim=4):
        super().__init__()
        self.g1 = _Layer(in_dim, hidden)
        self.g2 = _Layer(hidden, out)
        self.text_proj = nn.Linear(text_dim, out)
        self.logit_scale = nn.Parameter(torch.ones([]) * 1.0)


class StudentAPI:
    CLIPStudent = TinyStudent

    @staticmethod
    def norm_adj(edge_index, n, device):
        return torch.eye(n, device=device)

    @staticmethod
    def memory_to_trainable(memory):
        n = len(memory["x"])
        pairs = torch.triu_indices(n, n, offset=1)
        if pairs.shape[1] == 0:
            edges = torch.empty((2, 0), dtype=torch.long)
        else:
            score = (memory["text_emb"][pairs[0]]
                     * memory["text_emb"][pairs[1]]).sum(1)
            count = min(max(1, n // 2), pairs.shape[1])
            chosen = pairs[:, torch.topk(score, count).indices]
            edges = torch.cat((chosen, chosen.flip(0)), dim=1)
        return {"edge_index": edges}


def fixture():
    generator = torch.Generator().manual_seed(7)
    return {
        "x": torch.randn(12, 4, generator=generator),
        "y": torch.tensor([0] * 6 + [1] * 6),
        "ids": torch.arange(100, 112),
        "stage": 1,
        "texts": [f"current source {i}" for i in range(12)],
        "text_emb": F.normalize(torch.randn(12, 4, generator=generator), dim=1),
        "prompt_emb": F.normalize(torch.randn(3, 4, generator=generator), dim=1),
    }


def test_p_only_is_current_delta_only_and_reproducible():
    torch.set_num_threads(2)
    delta = fixture()
    x = torch.randn(4, 4, generator=torch.Generator().manual_seed(8))
    labels = torch.tensor([0, 0, 1, 1])
    cfg = P.PConfig(text_sources=3, text_chars=64,
                    outer_steps=2, unroll_steps=1, student_batch=4)
    first, first_audit = P.build_p_only_block(
        delta, delta["x"], x, labels, None, Encoder(), cfg, StudentAPI,
        "cpu")
    second, second_audit = P.build_p_only_block(
        delta, delta["x"], x, labels, None, Encoder(), cfg, StudentAPI,
        "cpu")
    assert torch.equal(first["x"], x)
    assert torch.equal(first["x"], second["x"])
    assert first["texts"] == second["texts"]
    assert first_audit["text_source_ids"] == second_audit["text_source_ids"]
    assert first_audit["current_delta_only"] is True
    assert first_audit["fixed_x"] is True
    assert all(set(source_ids).issubset(set(delta["ids"].tolist()))
               for source_ids in first_audit["text_source_ids"])
    assert all(len(text) <= 64 for text in first["texts"])


def test_p_only_preserves_existing_parent_prefix():
    delta = fixture()
    parent = {
        "x": torch.randn(2, 4, generator=torch.Generator().manual_seed(9)),
        "text_emb": F.normalize(torch.randn(2, 4,
                                             generator=torch.Generator().manual_seed(10)), dim=1),
        "texts": ["old a", "old b"],
        "mass": torch.ones(2),
        "depth": torch.zeros(2, dtype=torch.long),
        "stage": torch.zeros(2, dtype=torch.long),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    old_x, old_text = parent["x"].clone(), parent["text_emb"].clone()
    block, _ = P.build_p_only_block(
        delta, delta["x"], torch.randn(3, 4), torch.tensor([0, 1, 1]),
        parent, Encoder(), P.PConfig(outer_steps=1, unroll_steps=1),
        StudentAPI, "cpu")
    assert torch.equal(parent["x"], old_x)
    assert torch.equal(parent["text_emb"], old_text)
    assert len(block["x"]) == 3


def test_projection_aware_is_reproducible_and_bounded():
    delta = fixture()
    x = torch.randn(4, 4, generator=torch.Generator().manual_seed(18))
    labels = torch.tensor([0, 0, 1, 1])
    cfg = P.PConfig(text_sources=3, text_chars=64,
                    projection_candidate_pool=4,
                    projection_unroll_steps=1,
                    projection_min_gain=0.0,
                    student_batch=4)
    first, first_audit = P.build_projection_aware_block(
        delta, delta["x"], x, labels, None, Encoder(), cfg, StudentAPI,
        "cpu")
    second, second_audit = P.build_projection_aware_block(
        delta, delta["x"], x, labels, None, Encoder(), cfg, StudentAPI,
        "cpu")
    assert torch.equal(first["x"], x)
    assert torch.equal(first["text_emb"], second["text_emb"])
    assert first["texts"] == second["texts"]
    assert first_audit["text_source_ids"] == second_audit["text_source_ids"]
    assert first_audit["current_delta_only"] is True
    assert first_audit["fixed_x"] is True
    assert first_audit["accepted_rows"] <= len(labels)
    assert all(len(text) <= 64 for text in first["texts"])
    assert all(len(ids) == 3 for ids in first_audit["text_source_ids"])

