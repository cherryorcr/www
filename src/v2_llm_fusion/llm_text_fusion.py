"""LLM-driven text fusion for GraphCLIP constructive condensation (v2).

This module implements LLM-based text summarization to replace the simple
concatenation in v1. The goal is to generate semantically coherent, information-rich
fused abstracts that preserve key technical terms and topical coverage.

Supported LLM backends:
- OpenAI (GPT-4o, GPT-4-turbo)
- Anthropic Claude (Claude-3.5-Sonnet, Claude-3-Opus)
- Local/Open-source (Llama-3.1-70B via vLLM or Ollama)
"""
from __future__ import annotations

import os
import time
from typing import Literal

# LLM client imports (install as needed)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import requests
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


LLMBackend = Literal["gpt4o", "gpt4-turbo", "claude-sonnet", "claude-opus", "llama70b", "custom-openai"]


class LLMTextFusion:
    """LLM-driven text fusion for cluster member abstracts."""

    def __init__(self, backend: LLMBackend = "gpt4o",
                 openai_api_key: str | None = None,
                 anthropic_api_key: str | None = None,
                 vllm_endpoint: str | None = None,
                 custom_base_url: str | None = None,
                 custom_model: str | None = None):
        """Initialize LLM fusion client.

        Args:
            backend: Which LLM to use
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            anthropic_api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            vllm_endpoint: vLLM server endpoint for Llama (e.g., http://localhost:8000/v1)
            custom_base_url: Custom OpenAI-compatible API base URL
            custom_model: Model name for custom endpoint
        """
        self.backend = backend

        if backend in ["gpt4o", "gpt4-turbo"]:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed. Run: pip install openai")
            self.client = openai.OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-4o" if backend == "gpt4o" else "gpt-4-turbo-2024-04-09"

        elif backend in ["claude-sonnet", "claude-opus"]:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.model = "claude-3-5-sonnet-20241022" if backend == "claude-sonnet" else "claude-3-opus-20240229"

        elif backend == "llama70b":
            if not VLLM_AVAILABLE:
                raise ImportError("requests package not installed. Run: pip install requests")
            self.vllm_endpoint = vllm_endpoint or os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
            self.model = "meta-llama/Llama-3.1-70B-Instruct"

        elif backend == "custom-openai":
            # Custom OpenAI-compatible endpoint (e.g., third-party API proxy)
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed. Run: pip install openai")
            base_url = custom_base_url or os.getenv("CUSTOM_API_BASE_URL")
            api_key = openai_api_key or os.getenv("CUSTOM_API_KEY")
            if not base_url:
                raise ValueError("custom_base_url is required for custom-openai backend")
            self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
            self.model = custom_model or os.getenv("CUSTOM_MODEL", "gpt-4o")
            print(f"Using custom endpoint: {base_url}, model: {self.model}")

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def fuse_texts(self, texts: list[str], cluster_id: int,
                   target_words: int = 200,
                   domain: str = "machine learning research") -> str:
        """Generate a fused abstract from cluster member texts.

        Args:
            texts: List of member abstracts (already selected top-g by distance)
            cluster_id: Cluster identifier (for prompt context)
            target_words: Target length of fused abstract
            domain: Domain context for the fusion

        Returns:
            Fused abstract string
        """
        # Build prompt
        prompt = self._build_prompt(texts, cluster_id, target_words, domain)

        # Call LLM
        if self.backend in ["gpt4o", "gpt4-turbo", "custom-openai"]:
            return self._call_openai(prompt, max_tokens=int(target_words * 1.5))
        elif self.backend in ["claude-sonnet", "claude-opus"]:
            return self._call_anthropic(prompt, max_tokens=int(target_words * 1.5))
        elif self.backend == "llama70b":
            return self._call_vllm(prompt, max_tokens=int(target_words * 1.5))

    def _build_prompt(self, texts: list[str], cluster_id: int,
                      target_words: int, domain: str) -> str:
        """Construct the fusion prompt."""
        # Format texts with numbering
        formatted_texts = "\n\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))

        prompt = f"""You are a research assistant tasked with creating a coherent, information-rich abstract by fusing multiple related research paper abstracts.

**Domain**: {domain}

**Task**: Below are {len(texts)} abstracts from Cluster {cluster_id}. These papers share similar topics and methods. Generate a single {target_words}-word fused abstract that:

1. **Preserves key technical terms** (algorithms, architectures, datasets, metrics)
2. **Maintains semantic coherence** (reads like a natural abstract, not a list)
3. **Covers main topics** from all input abstracts
4. **Avoids redundancy** (consolidate overlapping concepts)
5. **Uses precise language** (no vague phrases like "various methods")

**Input Abstracts**:

{formatted_texts}

**Output**: Generate ONLY the fused abstract ({target_words} words), no preamble or meta-commentary.

Fused Abstract:"""

        return prompt

    def _call_openai(self, prompt: str, max_tokens: int = 300) -> str:
        """Call OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,  # Lower temperature for consistency
                top_p=0.95
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            # Fallback to concatenation
            return self._fallback_concat(prompt)

    def _call_anthropic(self, prompt: str, max_tokens: int = 300) -> str:
        """Call Anthropic Claude API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return self._fallback_concat(prompt)

    def _call_vllm(self, prompt: str, max_tokens: int = 300) -> str:
        """Call vLLM server (Llama-3.1-70B)."""
        try:
            response = requests.post(
                f"{self.vllm_endpoint}/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "top_p": 0.95
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except Exception as e:
            print(f"vLLM API error: {e}")
            return self._fallback_concat(prompt)

    def _fallback_concat(self, prompt: str) -> str:
        """Fallback to simple concatenation if LLM fails."""
        # Extract texts from prompt (crude parsing)
        lines = prompt.split("\n")
        texts = []
        for line in lines:
            if line.strip() and line[0].isdigit() and ". " in line:
                texts.append(line.split(". ", 1)[1])

        if not texts:
            return "[LLM fusion failed, no fallback available]"

        # Simple truncated concat
        concat = " ".join(texts)
        return concat[:800]  # Truncate to reasonable length


def batch_fuse_texts(texts_per_cluster: list[list[str]],
                     backend: LLMBackend = "gpt4o",
                     target_words: int = 200,
                     domain: str = "machine learning research",
                     **llm_kwargs) -> list[str]:
    """Batch process text fusion for multiple clusters.

    Args:
        texts_per_cluster: List of text lists, one per cluster
        backend: LLM backend to use
        target_words: Target abstract length
        domain: Domain context
        **llm_kwargs: Additional args for LLMTextFusion init

    Returns:
        List of fused abstracts, one per cluster
    """
    fusioner = LLMTextFusion(backend=backend, **llm_kwargs)
    fused = []

    for i, texts in enumerate(texts_per_cluster):
        print(f"Fusing cluster {i+1}/{len(texts_per_cluster)}...")
        fused_abstract = fusioner.fuse_texts(texts, cluster_id=i,
                                              target_words=target_words,
                                              domain=domain)
        fused.append(fused_abstract)

        # Rate limiting: sleep 0.5s between calls
        if i < len(texts_per_cluster) - 1:
            time.sleep(0.5)

    return fused


# ============================= Example Usage ================================

if __name__ == "__main__":
    # Example: fuse 3 sample abstracts
    sample_texts = [
        "We propose a novel graph neural network for node classification. "
        "Our method uses attention mechanisms to aggregate neighbor features. "
        "Experiments on Cora show 2% improvement over GCN.",

        "This paper introduces an attention-based GNN architecture. "
        "We apply self-attention to learn importance weights for neighbors. "
        "Results demonstrate state-of-the-art performance on citation networks.",

        "Graph attention networks (GAT) leverage masked self-attention layers. "
        "Each node computes attention coefficients for its neighbors. "
        "GAT achieves competitive results on transductive node classification."
    ]

    # Initialize fusion (defaults to GPT-4o)
    fusioner = LLMTextFusion(backend="gpt4o")

    # Fuse texts
    fused = fusioner.fuse_texts(sample_texts, cluster_id=0, target_words=150)

    print("=== Original Texts ===")
    for i, text in enumerate(sample_texts):
        print(f"{i+1}. {text}\n")

    print("=== Fused Abstract ===")
    print(fused)
