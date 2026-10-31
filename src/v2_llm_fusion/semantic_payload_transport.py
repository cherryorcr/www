"""Exact payload transport used by the frozen Semantic-Recursive runners.

This module deliberately contains no selector, response signal, topology rule,
or Student feedback.  It only makes the already existing permutation of a
structural block explicit and testable.  Keeping this operation separate makes
it possible to audit placement without changing the construction protocol.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import torch


PAYLOAD_KEYS = ("x", "text_emb", "texts", "mass", "depth", "stage")


def clone_block(block: Mapping[str, object]) -> dict[str, object]:
    """Clone all payload fields while preserving the existing field types."""
    result: dict[str, object] = {}
    for key in PAYLOAD_KEYS:
        value = block[key]
        if isinstance(value, torch.Tensor):
            result[key] = value.clone()
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_bijection(mapping: Mapping[int, int], n_rows: int) -> None:
    """Validate the row permutation expected by payload transport."""
    expected = set(range(int(n_rows)))
    sources = {int(source) for source in mapping}
    targets = {int(target) for target in mapping.values()}
    if sources != expected or targets != expected or len(mapping) != int(n_rows):
        raise ValueError("payload mapping must be a complete row bijection")


def apply_payload_permutation(
    structural: Mapping[str, object],
    mapping: Mapping[int, int],
    *,
    validate: bool = True,
) -> dict[str, object]:
    """Move complete payload rows according to a source-to-target mapping.

    The assignment convention matches the existing runners: ``mapping[source]``
    is the target slot.  The output starts as a clone, then every field in a
    payload row is copied together.  No row is synthesized, dropped, or
    duplicated.  The function intentionally does not compute or change edges.
    """
    n_rows = len(structural["x"])
    if validate:
        validate_bijection(mapping, n_rows)
    output = clone_block(structural)
    for source, target in mapping.items():
        source = int(source)
        target = int(target)
        for key in PAYLOAD_KEYS:
            value = structural[key]
            if isinstance(value, torch.Tensor):
                output[key][target] = value[source]
            else:
                output[key][target] = value[source]
    return output


def payload_row_digest(block: Mapping[str, object], index: int) -> str:
    """Return a stable digest for one complete payload row."""
    import hashlib

    digest = hashlib.sha256()
    for key in PAYLOAD_KEYS:
        value = block[key]
        if isinstance(value, torch.Tensor):
            tensor = value[index].detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(key.encode("utf-8"))
            digest.update(repr(value[index]).encode("utf-8"))
    return digest.hexdigest()


def payload_multiset_digest(block: Mapping[str, object]) -> list[str]:
    """Return sorted row digests for a payload conservation audit."""
    return sorted(payload_row_digest(block, i) for i in range(len(block["x"])))
