"""LLM re-ranking baseline: an LLM scores every candidate 0-10 in one request.

This is the comparison Jev classification has to beat to justify itself: a
general-purpose chat model given the same candidates. Works with any
OpenAI-compatible chat completions API (OpenAI, Vercel AI Gateway, OpenRouter,
Ollama).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

PROMPT = """You rank passages for a search system. Score how well each passage answers or \\
directly supports the query, from 0 (unrelated) to 10 (directly answers it or is the exact \\
evidence needed). Judge each passage on its own content.

Query: {query}

Passages:
{passages}

Reply with JSON only: {{"scores": [{{"id": 1, "score": 0}}, ...]}} with one entry for every passage id."""

# USD per million tokens (input, output); estimates for cost reporting only
PRICES = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40), "gpt-4o-mini": (0.15, 0.60),
          "gpt-4.1": (2.00, 8.00), "claude-haiku-4.5": (1.00, 5.00)}

_DEFAULTS = {"gateway": ("https://ai-gateway.vercel.sh/v1", "AI_GATEWAY_API_KEY"),
             "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
             "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")}


@dataclass
class LLMReranker:
    model: str = "openai/gpt-4.1-mini"
    provider: str = "gateway"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    max_passage_chars: int = 1500
    concurrency: int = 8
    timeout_s: float = 90.0
    retries: int = 3

    def __post_init__(self) -> None:
        d_url, d_env = _DEFAULTS.get(self.provider, _DEFAULTS["openai"])
        self.url = (self.base_url or d_url).rstrip("/") + "/chat/completions"
        self.key_env = self.api_key_env or d_env
        self._sem: Optional[asyncio.Semaphore] = None
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "failures": 0}

    @property
    def spec(self) -> str:
        return f"llm-rerank:{self.model}"

    def cost_usd(self) -> Optional[float]:
        short = self.model.split("/")[-1]
        if short not in PRICES:
            return None
        pin, pout = PRICES[short]
        return round((self.usage["input_tokens"] * pin + self.usage["output_tokens"] * pout) / 1e6, 5)

    def prompt(self, query: str, texts: Sequence[str]) -> str:
        blocks = "\n\n".join(f"[{i + 1}] {t[:self.max_passage_chars]}" for i, t in enumerate(texts))
        return PROMPT.format(query=query, passages=blocks)

    async def score(self, client: httpx.AsyncClient, query: str, texts: Sequence[str]) -> Tuple[List[Optional[float]], Dict[str, Any]]:
        """Scores in input order (None where the model skipped an id), plus usage for this call."""
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.concurrency)
        key = os.environ.get(self.key_env, "")
        if not key:
            raise RuntimeError(f"{self.key_env} is not set (LLM re-ranker)")
        body = {"model": self.model, "temperature": 0, "max_tokens": 16 * len(texts) + 64,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": self.prompt(query, texts)}]}
        last: Exception = RuntimeError("no attempt made")
        async with self._sem:
            for attempt in range(self.retries):
                try:
                    r = await client.post(self.url, json=body, headers={"Authorization": f"Bearer {key}"},
                                          timeout=self.timeout_s)
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
                    r.raise_for_status()
                    data = r.json()
                    usage = data.get("usage") or {}
                    self.usage["requests"] += 1
                    self.usage["input_tokens"] += int(usage.get("prompt_tokens", 0))
                    self.usage["output_tokens"] += int(usage.get("completion_tokens", 0))
                    return parse_scores(data["choices"][0]["message"]["content"] or "", len(texts)), usage
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    last = exc
                    await asyncio.sleep(1.5 * (attempt + 1))
        self.usage["failures"] += 1
        raise RuntimeError(f"LLM re-ranker failed: {last}")


def parse_scores(content: str, n: int) -> List[Optional[float]]:
    """Read ``{"scores": [{"id", "score"}]}``; tolerate code fences and stray text."""
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        return [None] * n
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [None] * n
    out: List[Optional[float]] = [None] * n
    for item in data.get("scores", []) if isinstance(data, dict) else []:
        try:
            i = int(item["id"]) - 1
            s = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= i < n:
            out[i] = max(0.0, min(10.0, s))
    return out


__all__ = ["LLMReranker", "parse_scores", "PRICES"]
