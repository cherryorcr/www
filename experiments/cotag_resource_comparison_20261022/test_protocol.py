import pytest
import torch

from protocol_utils import full_support, label_schedule, payload_bytes


def test_full_support_exact_union_sorted():
    assert full_support([torch.tensor([3, 1]), torch.tensor([4])],
                        torch.tensor([0, 1, 0, 1, 1], dtype=torch.bool)).tolist() == [1, 3, 4]


@pytest.mark.parametrize("ids", [[torch.tensor([1, 1, 3])], [torch.tensor([1])]])
def test_reject_duplicate_or_incomplete(ids):
    with pytest.raises(ValueError):
        full_support(ids, torch.tensor([0, 1, 0, 1], dtype=torch.bool))


def test_preserves_incremental_quotas_not_global_reallocation():
    y = torch.tensor([0, 0, 1, 1, 1])
    calls = []

    def allocate(labels, count):
        calls.append((labels.tolist(), count))
        return {int(labels[0]): count}

    out = label_schedule(y, [torch.tensor([0, 1]), torch.tensor([2, 3, 4])],
                         [1, 3], allocate,
                         lambda d: torch.tensor([c for c, n in d.items() for _ in range(n)]))
    assert torch.cat(out).tolist() == [0, 1, 1]
    assert calls == [([0, 0], 1), ([1, 1, 1], 2)]


def test_payload_counts_bytes_not_characters():
    result = payload_bytes(dict(x=torch.zeros((2, 3)), texts=["abc", "\u4e2d"]))
    assert result["total"] == 30
    assert not result["python_object_overhead_included"]
