"""Small, constructor-independent checks for a text-only intervention."""
import torch

NUMERIC_KEYS = ("x", "labels", "mass", "stage", "depth", "edge_index")


def assert_numeric_equal(left, right):
    for key in NUMERIC_KEYS:
        if not torch.equal(left[key], right[key]):
            raise AssertionError(f"Text intervention changed {key}")


def assert_frozen_prefix(memory, parent):
    if parent is None:
        return
    keep = len(parent["x"])
    for key in ("x", "text_emb", "labels", "mass", "stage", "depth"):
        if not torch.equal(memory[key][:keep], parent[key]):
            raise AssertionError(f"Historical prefix changed: {key}")
    if memory["texts"][:keep] != parent["texts"]:
        raise AssertionError("Historical texts changed")


def assert_single_source(single, five, allowed):
    if len(single) != len(five):
        raise AssertionError("Source selection count differs")
    allowed = set(map(int, allowed))
    for one, many in zip(single, five):
        if len(one) != 1 or not many or one[0] != many[0] or one[0] not in allowed:
            raise AssertionError("Single source is not the verified legal nearest source")


def text_statistics(texts):
    lengths = [len(t) for t in texts]
    if not lengths or max(lengths) > 320:
        raise AssertionError("Invalid text length")
    return dict(rows=len(texts), mean_chars=sum(lengths)/len(lengths),
                min_chars=min(lengths), max_chars=max(lengths),
                unique_texts=len(set(texts)), unique_text_ratio=len(set(texts))/len(texts))
