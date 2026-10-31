"""Deterministic row/column-segmented sparse-dense multiplication.

The graph is stored as coalesced COO, but reductions run over sorted segments
instead of sparse CUDA atomics. Backward uses a separately sorted column view.
"""
from __future__ import annotations

import torch


def _blocks(lengths: torch.Tensor, edge_limit: int) -> tuple[tuple[int, int, int, int], ...]:
    if edge_limit <= 0:
        raise ValueError("edge_limit must be positive")
    counts = lengths.tolist()
    result = []
    node_start = edge_start = 0
    while node_start < len(counts):
        node_end = node_start
        edge_end = edge_start
        while node_end < len(counts) and (node_end == node_start or edge_end - edge_start < edge_limit):
            edge_end += counts[node_end]
            node_end += 1
        result.append((node_start, node_end, edge_start, edge_end))
        node_start, edge_start = node_end, edge_end
    return tuple(result)


class _SegmentedSpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, row_cols, row_values, row_lengths,
                col_rows, col_values, col_lengths, row_blocks, col_blocks):
        if x.ndim != 2:
            raise ValueError("sparse-dense multiplication expects a matrix")
        outputs = []
        for node_start, node_end, edge_start, edge_end in row_blocks:
            gathered = x.index_select(0, row_cols[edge_start:edge_end])
            weighted = gathered * row_values[edge_start:edge_end, None]
            outputs.append(torch.segment_reduce(
                weighted, reduce="sum", lengths=row_lengths[node_start:node_end], axis=0))
        ctx.save_for_backward(row_cols, row_values, row_lengths,
                              col_rows, col_values, col_lengths)
        ctx.col_blocks = col_blocks
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        row_cols, row_values, row_lengths, col_rows, col_values, col_lengths = ctx.saved_tensors
        outputs = []
        for node_start, node_end, edge_start, edge_end in ctx.col_blocks:
            gathered = grad_output.index_select(0, col_rows[edge_start:edge_end])
            weighted = gathered * col_values[edge_start:edge_end, None]
            outputs.append(torch.segment_reduce(
                weighted, reduce="sum", lengths=col_lengths[node_start:node_end], axis=0))
        grad_x = torch.cat(outputs, dim=0)
        return grad_x, None, None, None, None, None, None, None, None


class DeterministicAdjacency:
    """Immutable sparse matrix with deterministic segmented ``@`` semantics."""

    def __init__(self, matrix: torch.Tensor, edge_limit: int = 65536):
        if matrix.layout != torch.sparse_coo:
            raise TypeError("matrix must be sparse COO")
        device = matrix.device
        matrix = matrix.detach().cpu().coalesce()
        n, m = matrix.shape
        if n <= 0 or m <= 0:
            raise ValueError("empty adjacency")
        indices = matrix.indices().detach().cpu()
        values = matrix.values().detach().cpu()
        row_key = indices[0] * m + indices[1]
        row_order = torch.argsort(row_key, stable=True)
        col_key = indices[1] * n + indices[0]
        col_order = torch.argsort(col_key, stable=True)

        self.shape = matrix.shape
        self.device = device
        self.dtype = matrix.dtype
        self._row_cols = indices[1, row_order].to(self.device)
        self._row_indices = torch.stack((indices[0, row_order], indices[1, row_order])).to(self.device)
        self._row_values = values[row_order].to(self.device)
        row_lengths_cpu = torch.bincount(indices[0], minlength=n)
        self._row_lengths = row_lengths_cpu.to(self.device)
        self._row_sums = torch.segment_reduce(values[row_order], reduce="sum",
                                              lengths=row_lengths_cpu).to(self.device)
        self._col_rows = indices[0, col_order].to(self.device)
        self._col_values = values[col_order].to(self.device)
        col_lengths_cpu = torch.bincount(indices[1], minlength=m)
        self._col_lengths = col_lengths_cpu.to(self.device)
        self._row_blocks = _blocks(row_lengths_cpu, edge_limit)
        self._col_blocks = _blocks(col_lengths_cpu, edge_limit)
        self._indices = indices.to(self.device)

    def __matmul__(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device:
            raise ValueError("adjacency and dense operand must share a device")
        if x.shape[0] != self.shape[1]:
            raise ValueError("adjacency/dense operand shape mismatch")
        if x.dtype != self.dtype:
            raise ValueError("adjacency/dense operand dtype mismatch")
        return _SegmentedSpMM.apply(
            x, self._row_cols, self._row_values, self._row_lengths,
            self._col_rows, self._col_values, self._col_lengths,
            self._row_blocks, self._col_blocks)

    def to_dense(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self._row_indices, self._row_values, self.shape, device=self.device,
            dtype=self.dtype).to_dense()

    def row_sums(self) -> torch.Tensor:
        return self._row_sums

    def to(self, device: torch.device | str) -> "DeterministicAdjacency":
        device = torch.device(device)
        if device == self.device:
            return self
        sparse = torch.sparse_coo_tensor(
            self._row_indices.to(device), self._row_values.to(device), self.shape,
            device=device, dtype=self.dtype)
        return DeterministicAdjacency(sparse)


def from_sparse(matrix: torch.Tensor, edge_limit: int = 65536) -> DeterministicAdjacency:
    return DeterministicAdjacency(matrix, edge_limit=edge_limit)
