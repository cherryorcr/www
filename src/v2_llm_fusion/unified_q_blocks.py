"""Class-legal storage for an assignment with implicit cross-class zeros.

Only the operations used by the edge-free constructor are implemented. Dense
conversion is deliberately explicit and is intended for small equivalence tests.
"""
from __future__ import annotations

import torch


class ClassBlocks:
    def __init__(self, shape, rows, slots, blocks):
        self.shape = tuple(shape)
        self.rows, self.slots, self.blocks = tuple(rows), tuple(slots), tuple(blocks)
        if not blocks or not (len(rows) == len(slots) == len(blocks)):
            raise ValueError("nonempty, aligned class blocks required")
        self.device = blocks[0].device

    def map(self, fn):
        return ClassBlocks(self.shape, self.rows, self.slots, [fn(b) for b in self.blocks])

    def detach(self):
        return self.map(lambda b: b.detach())

    def cpu(self):
        return ClassBlocks(self.shape, [r.cpu() for r in self.rows],
                           [s.cpu() for s in self.slots], [b.cpu() for b in self.blocks])

    def parameters(self):
        return list(self.blocks)

    def sum(self, dim):
        if dim not in (0, 1):
            raise ValueError("only row/column reductions are supported")
        out = self.blocks[0].new_zeros(self.shape[1 - dim])
        for index, block in zip(self.slots if dim == 0 else self.rows, self.blocks):
            out = out.index_copy(0, index, block.sum(dim))
        return out

    def argmax(self, dim):
        if dim not in (0, 1):
            raise ValueError("only row/column argmax is supported")
        out = torch.empty(self.shape[1 - dim], dtype=torch.long, device=self.device)
        for rows, slots, block in zip(self.rows, self.slots, self.blocks):
            index, sources = (slots, rows) if dim == 0 else (rows, slots)
            out[index] = sources[block.argmax(dim)]
        return out

    def pool(self, values):
        out = values.new_zeros((self.shape[1], values.shape[1]))
        for rows, slots, block in zip(self.rows, self.slots, self.blocks):
            out = out.index_copy(0, slots, block.T @ values.index_select(0, rows))
        return out

    @property
    def T(self):
        return _Transpose(self)

    def anchor(self, other):
        # Keep the original dense mean denominator, including implicit zeros.
        return sum((a - b.detach()).square().sum() for a, b in zip(self.blocks, other.blocks)) / (
            self.shape[0] * self.shape[1])

    def entropy(self):
        value = sum(-(b.clamp_min(1e-12) * b.clamp_min(1e-12).log()).sum() for b in self.blocks)
        missing = self.shape[0] * self.shape[1] - sum(b.numel() for b in self.blocks)
        eps = self.blocks[0].new_tensor(1e-12)
        return (value - missing * eps * eps.log()) / self.shape[0]

    def __getitem__(self, key):
        rows, slot = key
        slot = int(slot)
        for raw_ids, slot_ids, block in zip(self.rows, self.slots, self.blocks):
            local_slot = torch.searchsorted(slot_ids, slot)
            if int(local_slot) < len(slot_ids) and int(slot_ids[local_slot]) == slot:
                requested = torch.as_tensor(rows, dtype=torch.long, device=self.device)
                if torch.equal(raw_ids, requested):
                    return block[:, local_slot]
                local_rows = torch.searchsorted(raw_ids, requested)
                if bool((local_rows >= len(raw_ids)).any()) or not torch.equal(raw_ids[local_rows], requested):
                    raise ValueError("requested a cross-class entry")
                return block[local_rows, local_slot]
        raise IndexError(slot)

    def to_dense(self):
        out = self.blocks[0].new_zeros(self.shape)
        for rows, slots, block in zip(self.rows, self.slots, self.blocks):
            out[rows[:, None], slots[None, :]] = block
        return out


class _Transpose:
    def __init__(self, assignment):
        self.assignment = assignment

    def __matmul__(self, values):
        return self.assignment.pool(values)
