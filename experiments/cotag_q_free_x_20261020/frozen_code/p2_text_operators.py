"""Token-bounded synthetic text targets for controlled P2 experiments."""
from __future__ import annotations

import hashlib
import torch
import torch.nn.functional as F


OPERATORS = ("concat", "diverse_concat", "dual_proto", "mean_pool",
             "weighted_pool", "set_pool", "no_reencode", "no_fuse", "no_text")


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def bounded_prefix(tokenizer, text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    ids = tokenizer.encode(str(text), add_special_tokens=False)[:int(budget)]
    # Decoding a partial wordpiece can alter its next tokenization.
    while ids:
        result = tokenizer.decode(ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
        if token_count(tokenizer, result) <= budget:
            return result
        ids.pop()
    return ""


def bounded_parts(tokenizer, texts: list[str], budget: int) -> list[str]:
    """Water-fill one shared content-token cap across the selected sources."""
    capacities = [token_count(tokenizer, text) for text in texts]
    quotas = [0] * len(texts)
    remaining = int(budget)
    while remaining > 0:
        active = [i for i, cap in enumerate(capacities) if quotas[i] < cap]
        if not active:
            break
        for i in active:
            quotas[i] += 1
            remaining -= 1
            if remaining == 0:
                break
    parts = [bounded_prefix(tokenizer, text, quota)
             for text, quota in zip(texts, quotas)]
    # Both separate encoding and concatenation must satisfy the same cap.
    while (sum(token_count(tokenizer, p) for p in parts) > budget
           or token_count(tokenizer, " ".join(parts)) > budget):
        i = max(range(len(parts)), key=lambda j: (token_count(tokenizer, parts[j]), j))
        parts[i] = bounded_prefix(tokenizer, parts[i], token_count(tokenizer, parts[i]) - 1)
    return parts


def select_members(joint: torch.Tensor, text_emb: torch.Tensor,
                   members: torch.Tensor, count: int, diverse: bool,
                   relevance_weight: float) -> torch.Tensor:
    count = min(int(count), len(members))
    if count < 1:
        raise ValueError("a text cluster must have at least one member")
    dist = torch.cdist(joint[members], joint[members].mean(0, keepdim=True)).flatten()
    order = torch.argsort(dist, stable=True)
    if not diverse:
        return members[order[:count]]
    z = F.normalize(text_emb[members].float(), dim=1)
    center = F.normalize(z.mean(0), dim=0)
    selected = [int(order[0])]
    remaining = [i for i in range(len(members)) if i not in selected]
    lam = float(relevance_weight)
    while remaining and len(selected) < count:
        i = max(remaining, key=lambda j: (
            lam * float(z[j] @ center)
            - (1.0 - lam) * max(float(z[j] @ z[k]) for k in selected), -j))
        selected.append(i)
        remaining.remove(i)
    return members[torch.tensor(selected, dtype=torch.long)]


def construct(joint: torch.Tensor, source_embeddings: torch.Tensor,
              members: torch.Tensor, texts: list[str], encoder,
              operator: str = "concat", budget: int = 64, g: int = 5,
              relevance_weight: float = 0.5, alpha: float = 0.5) -> tuple[str, torch.Tensor, dict]:
    if operator not in OPERATORS or not 1 <= budget <= 240 or g < 1:
        raise ValueError("invalid operator, content-token cap, or member count")
    if not 0 <= relevance_weight <= 1 or not 0 <= alpha <= 1:
        raise ValueError("text mixture weights must be in [0, 1]")
    tokenizer = encoder.tokenizer
    chosen = select_members(joint, source_embeddings, members,
                            1 if operator == "no_fuse" else g,
                            operator in {"diverse_concat", "dual_proto"},
                            relevance_weight)
    selected_texts = [texts[int(i)] for i in chosen]
    if operator == "dual_proto" and len(selected_texts) > 1:
        first_budget = budget // 2
        parts = [bounded_prefix(tokenizer, selected_texts[0], first_budget)]
        parts += bounded_parts(tokenizer, selected_texts[1:], budget - first_budget)
        # Joining tokenized pieces may create an additional boundary token.
        parts = bounded_parts(tokenizer, parts, budget)
    else:
        parts = bounded_parts(tokenizer, selected_texts, budget)
    fused = " ".join(p for p in parts if p)
    if operator == "no_text":
        parts, fused, inputs = [], "", []
        target = torch.zeros_like(source_embeddings[0])
    elif operator in {"concat", "diverse_concat", "no_fuse"}:
        inputs = [fused]
        target = encoder.encode(inputs)[0]
    elif operator == "dual_proto":
        inputs = [parts[0]]
        if len(parts) > 1:
            inputs.append(" ".join(parts[1:]))
        encoded = encoder.encode(inputs)
        target = (alpha * encoded[0] + (1 - alpha) * encoded[1]
                  if len(encoded) > 1 else encoded[0])
        target = F.normalize(target, dim=0)
    else:
        inputs = parts
        encoded = encoder.encode(inputs)
        if operator in {"mean_pool", "no_reencode"}:
            weights = torch.ones(len(encoded)) / len(encoded)
        elif operator == "weighted_pool":
            center = joint[members].mean(0, keepdim=True)
            weights = torch.softmax(-torch.cdist(joint[chosen], center).flatten(), dim=0)
        else:
            # A parameter-free, permutation-invariant content-attention control.
            query = F.normalize(encoded.mean(0), dim=0)
            weights = torch.softmax((F.normalize(encoded, dim=1) @ query) / 0.2, dim=0)
        target = F.normalize((encoded * weights[:, None]).sum(0), dim=0)
    content_tokens = sum(token_count(tokenizer, t) for t in inputs)
    retained_tokens = token_count(tokenizer, fused)
    if content_tokens > budget or retained_tokens > budget:
        raise AssertionError("encoder input exceeded the common content-token cap")
    with_special = sum(len(tokenizer.encode(t, add_special_tokens=True)) for t in inputs)
    audit = {
        "content_token_cap": int(budget), "selected_members": int(len(chosen)),
        "retained_tokens": int(retained_tokens), "encoder_content_tokens": int(content_tokens),
        "encoder_tokens_with_special": int(with_special), "encoder_sequences": len(inputs),
        "empty": not bool(fused), "operator": operator,
        "input_sha256": hashlib.sha256("\0".join(inputs).encode()).hexdigest(),
        "inputs": inputs,
    }
    return fused, target, audit


def aggregate_audit(rows: list[dict]) -> dict:
    n = max(1, len(rows))
    return {
        "rows": len(rows),
        "token_cap_per_row": rows[0]["content_token_cap"] if rows else 0,
        "retained_token_total": sum(r["retained_tokens"] for r in rows),
        "encoder_content_token_total": sum(r["encoder_content_tokens"] for r in rows),
        "encoder_content_token_mean": sum(r["encoder_content_tokens"] for r in rows) / n,
        "encoder_tokens_with_special_total": sum(r["encoder_tokens_with_special"] for r in rows),
        "encoder_sequences": sum(r["encoder_sequences"] for r in rows),
        "empty_rows": sum(r["empty"] for r in rows),
        "budget_violations": sum(r["encoder_content_tokens"] > r["content_token_cap"]
                                 or r["retained_tokens"] > r["content_token_cap"] for r in rows),
    }
