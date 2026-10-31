import torch

import teacher_guided_payload_revision as T


def memory(rows=5, dim=4, text_dim=384):
    edge = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    return {
        "x": torch.randn(rows, dim),
        "text_emb": torch.randn(rows, text_dim),
        "texts": [f"t{i}" for i in range(rows)],
        "mass": torch.ones(rows),
        "depth": torch.zeros(rows, dtype=torch.long),
        "stage": torch.zeros(rows, dtype=torch.long),
        "edge_index": edge,
    }


def test_teacher_guided_revision_changes_only_x_and_returns_sparse_assignment():
    torch.manual_seed(12)
    base = memory(5)
    support = memory(8)
    class_id = torch.tensor([-1, -1, 0, 0, 1, 1, 1, 0], dtype=torch.long)
    cfg = T.Config(teacher_steps=1, inner_steps=1, outer_steps=1,
                   batch=4, top_r=2, teacher_seeds=(101,),
                   validation_teacher_seeds=(102,))
    before = T.memory_hash(base)
    result, audit, assignment = T.optimize(
        base, support, class_id, 2, 1, "cpu", cfg)
    assert T.memory_hash(base) == before
    assert not torch.equal(result["x"], base["x"])
    for key in T.IMMUTABLE_FIELDS:
        assert torch.equal(result[key], base[key]) if isinstance(result[key], torch.Tensor) else result[key] == base[key]
    assert len(assignment["indices"]) == 3
    assert assignment["top_r"] == 2
    assert audit["historical_raw_reads"] == 0


def test_assignment_rows_are_class_consistent_at_initialization():
    torch.manual_seed(3)
    base = memory(4)
    support = memory(7)
    class_id = torch.tensor([-1, -1, 0, 0, 1, 1, 0], dtype=torch.long)
    cfg = T.Config(top_r=2)
    assignment = T.build_sparse_assignment(base, support, 2, class_id, cfg, "cpu")
    raw_class = assignment["raw_class"]
    slot_class = assignment["slot_class"]
    for row, cls in enumerate(slot_class.tolist()):
        assert int(cls) in set(int(v) for v in raw_class.tolist())
