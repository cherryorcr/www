"""LLM-based text summarization for cluster fusion (v2).

Replaces v1's truncation+concatenation with Qwen3-guided semantic summary.
Temperature=0 greedy decoding ensures determinism.
"""
from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ClusterSummarizer:
    """Qwen3-32B-AWQ summarizer for text-anchored graph condensation."""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, torch_dtype=torch.float16, trust_remote_code=True
        ).eval()
        self.device = device

    @torch.no_grad()
    def summarize_cluster(
        self,
        member_texts: list[str],
        cluster_size: int,
        budget_chars: int = 320,
        max_input_texts: int = 7,
    ) -> str:
        """Generate a deterministic semantic summary for a cluster of texts.

        Args:
            member_texts: Texts from cluster members (already sorted by distance to center)
            cluster_size: Total cluster size (for context)
            budget_chars: Target output length
            max_input_texts: Maximum number of input texts to include in prompt

        Returns:
            A synthetic text that covers common themes and preserves discriminative detail
        """
        # Take top-k closest to center (already sorted by caller)
        selected = member_texts[:max_input_texts]

        # Build prompt
        texts_block = "\n".join(f"{i+1}. {t[:200]}" for i, t in enumerate(selected))  # truncate each to 200 chars for prompt
        prompt = f"""You are summarizing research paper abstracts for graph condensation. Given {len(selected)} texts from a cluster of {cluster_size} papers, write ONE concise paragraph (~{budget_chars} characters) that:
1. Covers the common research theme across all texts
2. Preserves technical terms and distinguishing details
3. Reads as a coherent standalone abstract (no meta-commentary like "these papers discuss...")

INPUT TEXTS:
{texts_block}

OUTPUT (one paragraph, {budget_chars} chars):"""

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)

        # Greedy decode (temperature=0 equivalent via do_sample=False)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=budget_chars // 3,  # rough char-to-token ratio ~3:1
            do_sample=False,  # greedy = deterministic
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Extract generated text
        generated = self.tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Post-process: take first paragraph, truncate to budget
        summary = generated.strip().split("\n\n")[0]  # first paragraph
        if len(summary) > budget_chars:
            summary = summary[:budget_chars].rsplit(" ", 1)[0]  # truncate at word boundary

        return summary
