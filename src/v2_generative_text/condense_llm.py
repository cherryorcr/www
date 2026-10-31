"""v2 runner: generative text condensation with Qwen3 summarizer.

Replaces v1's truncation+concatenation with LLM-guided semantic summary.
Otherwise identical structure to frozen_v1 gate1 runner.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from llm_summarizer import ClusterSummarizer


def condense_with_llm(
    x: torch.Tensor,
    text_emb: torch.Tensor,
    texts: list[str],
    k: int,
    text_encoder,
    llm_summarizer: ClusterSummarizer,
    budget_chars: int = 320,
    fuse_g: int = 5,
) -> dict:
    """Static condensation with LLM-generated cluster summaries.

    Args:
        x: Node features [n, d_graph]
        text_emb: Text embeddings [n, d_text]
        texts: Raw text strings
        k: Number of clusters
        text_encoder: Frozen text encoder (for re-encoding summaries)
        llm_summarizer: ClusterSummarizer instance
        budget_chars: Target summary length
        fuse_g: Number of closest texts to include in LLM prompt

    Returns:
        Memory dict with synthetic X, LLM-generated texts, and paired embeddings
    """
    from run_gate1_continual import weighted_cluster  # reuse deterministic clustering

    # Cluster in joint space (same as v1)
    joint = torch.cat((F.normalize(x.float(), dim=1), text_emb), dim=1)
    assign = weighted_cluster(joint, torch.ones(len(x)), k)

    syn_x, fused_texts, mass = [], [], []

    for j in range(k):
        members = torch.where(assign == j)[0]

        # Synthetic X: cluster mean (unchanged from v1)
        syn_x.append(x[members].float().mean(0))
        mass.append(float(len(members)))

        # Sort members by distance to center (for selecting top-g)
        center = joint[members].mean(0, keepdim=True)
        dist = torch.cdist(joint[members], center).flatten()
        order = members[torch.argsort(dist)][:fuse_g]

        # Extract member texts for LLM
        member_texts_for_llm = [texts[int(i)] for i in order]

        # Generate summary via LLM (temperature=0 greedy decode)
        summary = llm_summarizer.summarize_cluster(
            member_texts_for_llm,
            cluster_size=len(members),
            budget_chars=budget_chars,
            max_input_texts=fuse_g,
        )
        fused_texts.append(summary)

    # Re-encode generated summaries (self-encoding: pairing is constructive)
    fused_emb = text_encoder.encode(fused_texts)

    return {
        "x": torch.stack(syn_x),
        "text_emb": fused_emb,
        "texts": fused_texts,
        "mass": torch.tensor(mass),
        "depth": torch.zeros(k, dtype=torch.long),  # all leaf nodes
        "stage": torch.full((k,), 0, dtype=torch.long),  # caller will overwrite
    }
