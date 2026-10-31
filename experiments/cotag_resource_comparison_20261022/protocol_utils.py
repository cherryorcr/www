"""Protocol checks independent of the remote loader."""
import torch


def full_support(stage_ids, pool_mask):
    joined = torch.cat(stage_ids).long().cpu()
    if len(torch.unique(joined)) != len(joined):
        raise ValueError("stage supports overlap")
    full = torch.sort(joined).values
    expected = torch.where(pool_mask.cpu().bool())[0]
    if not torch.equal(full, expected):
        raise ValueError("full support differs from final legal pool")
    return full


def label_schedule(y, stage_ids, budgets, allocator, row_labels):
    if len(stage_ids) != len(budgets):
        raise ValueError("stages and budgets differ")
    previous = 0
    labels = []
    for ids, budget in zip(stage_ids, budgets):
        count = budget - previous
        if count < 1 or count > len(ids):
            raise ValueError("invalid increment budget")
        new = row_labels(allocator(y[ids], count)).cpu().long()
        if len(new) != count or not bool(torch.isin(new, y[ids].cpu()).all()):
            raise ValueError("invalid class allocation")
        labels.append(new)
        previous = budget
    return labels


def payload_bytes(memory):
    tensor_bytes = sum(v.numel() * v.element_size() for v in memory.values()
                       if torch.is_tensor(v))
    text_bytes = sum(len(v.encode("utf-8")) for v in memory["texts"])
    return dict(tensors=tensor_bytes, utf8_text=text_bytes,
                total=tensor_bytes + text_bytes, python_object_overhead_included=False)
