"""Embedding providers (FR-I1).

A provider is ``"<name>:<model>"``:

* ``openai:text-embedding-3-small``  - OpenAI (``OPENAI_API_KEY``) or any OpenAI-compatible
  endpoint via ``base_url`` / ``api_key_env``
* ``gateway:openai/text-embedding-3-small`` - Vercel AI Gateway (``AI_GATEWAY_API_KEY``)
* ``ollama:nomic-embed-text``        - local Ollama (``OLLAMA_HOST``, default http://localhost:11434)
* ``local:BAAI/bge-small-en-v1.5``   - sentence-transformers, in process (``...[local]``)
* ``hash:256``                       - deterministic hashing embedder for tests and offline demos;
  no semantic quality, never use in production
* a Python callable ``list[str] -> list[list[float]]`` passed to :func:`make_embedder`
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import random
import re
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Union

import httpx

from .. import __version__

EmbedFn = Callable[[List[str]], Union[List[List[float]], Awaitable[List[List[float]]]]]


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    provider: str = "base"
    model: str = ""
    metric: str = "cosine"
    batch_size: int = 128
    dim: Optional[int] = None
    tokens_used: int = 0

    @property
    def spec(self) -> str:
        return f"{self.provider}:{self.model}"

    async def _embed(self, texts: List[str], kind: str) -> List[List[float]]:
        raise NotImplementedError

    async def _batched(self, texts: Sequence[str], kind: str) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[i:i + self.batch_size]]
            vecs = await self._embed(batch, kind)
            if len(vecs) != len(batch):
                raise EmbeddingError(f"{self.spec} returned {len(vecs)} vectors for {len(batch)} inputs")
            out.extend(vecs)
        if out:
            d = len(out[0])
            if self.dim is None:
                self.dim = d
            elif d != self.dim:
                raise EmbeddingError(f"{self.spec} returned dimension {d}, expected {self.dim}")
        return out

    async def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return await self._batched(texts, "document")

    async def embed_query(self, text: str) -> List[float]:
        return (await self._batched([text], "query"))[0]

    async def aclose(self) -> None:
        return None


class _HTTPEmbedder(Embedder):
    """Holds one httpx.AsyncClient per event loop, so sync wrappers that call asyncio.run
    repeatedly never reuse a client bound to a closed loop."""

    _timeout: float = 60.0
    _transport: Any = None

    def _http(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        cur = getattr(self, "_client_loop", None)
        if cur is not loop or getattr(self, "_client", None) is None:
            self._client = httpx.AsyncClient(timeout=self._timeout, transport=self._transport,
                                             headers={"User-Agent": f"jev-retrieval/{__version__}"})
            self._client_loop = loop
        return self._client

    async def aclose(self) -> None:
        c = getattr(self, "_client", None)
        if c is not None and getattr(self, "_client_loop", None) is asyncio.get_running_loop():
            await c.aclose()
        self._client = None


class OpenAICompatible(_HTTPEmbedder):
    def __init__(self, model: str, *, provider: str = "openai", base_url: Optional[str] = None,
                 api_key_env: Optional[str] = None, batch_size: int = 128, dimensions: Optional[int] = None,
                 timeout_s: float = 60.0, max_retries: int = 5, transport: Any = None) -> None:
        self.provider = provider
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.max_retries = max_retries
        defaults = {"openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
                    "gateway": ("https://ai-gateway.vercel.sh/v1", "AI_GATEWAY_API_KEY")}
        d_url, d_env = defaults.get(provider, defaults["openai"])
        self.base_url = (base_url or d_url).rstrip("/")
        self.api_key_env = api_key_env or d_env
        self._timeout = timeout_s
        self._transport = transport

    async def _embed(self, texts: List[str], kind: str) -> List[List[float]]:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise EmbeddingError(f"{self.api_key_env} is not set (embedder {self.spec})")
        body: dict = {"model": self.model, "input": texts}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        for attempt in range(self.max_retries + 1):
            try:
                r = await self._http().post(f"{self.base_url}/embeddings", json=body,
                                            headers={"Authorization": f"Bearer {key}"})
            except httpx.HTTPError as exc:
                if attempt >= self.max_retries:
                    raise EmbeddingError(f"{self.spec}: {exc}") from exc
                await asyncio.sleep(min(8, 0.5 * 2 ** attempt))
                continue
            if r.status_code == 200:
                data = r.json()
                self.tokens_used += int((data.get("usage") or {}).get("total_tokens", 0) or 0)
                rows = sorted(data["data"], key=lambda x: x.get("index", 0))
                return [row["embedding"] for row in rows]
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                await asyncio.sleep(min(8, 0.5 * 2 ** attempt) + random.random() * 0.2)
                continue
            raise EmbeddingError(f"{self.spec}: HTTP {r.status_code}: {r.text[:200]}")
        raise EmbeddingError(f"{self.spec}: retries exhausted")


class Ollama(_HTTPEmbedder):
    def __init__(self, model: str, *, base_url: Optional[str] = None, batch_size: int = 64) -> None:
        self.provider = "ollama"
        self.model = model
        self.batch_size = batch_size
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self._timeout = 120.0

    async def _embed(self, texts: List[str], kind: str) -> List[List[float]]:
        r = await self._http().post(f"{self.base_url}/api/embed", json={"model": self.model, "input": texts})
        if r.status_code != 200:
            raise EmbeddingError(f"{self.spec}: HTTP {r.status_code}: {r.text[:200]}")
        return r.json()["embeddings"]


class SentenceTransformers(Embedder):
    def __init__(self, model: str, *, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError('local embeddings need: pip install "jev-rag-retrieval[local]"') from exc
        self.provider = "local"
        self.model = model
        self.batch_size = batch_size
        self._m = SentenceTransformer(model)

    async def _embed(self, texts: List[str], kind: str) -> List[List[float]]:
        vecs = await asyncio.to_thread(self._m.encode, texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class HashEmbedder(Embedder):
    """Feature-hashed bag of words. Deterministic, offline, no semantic quality: tests and demos only."""

    _WORD = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 256) -> None:
        self.provider = "hash"
        self.model = str(dim)
        self.dim = dim

    async def _embed(self, texts: List[str], kind: str) -> List[List[float]]:
        out = []
        for t in texts:
            v = [0.0] * int(self.model)
            for w in self._WORD.findall(t.lower()):
                h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")
                v[h % len(v)] += 1.0 if (h >> 63) & 1 else -1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class CallableEmbedder(Embedder):
    def __init__(self, fn: EmbedFn, *, name: str = "callable", metric: str = "cosine", batch_size: int = 128) -> None:
        self.provider = "callable"
        self.model = name
        self.metric = metric
        self.batch_size = batch_size
        self._fn = fn

    async def _embed(self, texts: List[str], kind: str) -> List[List[float]]:
        res = self._fn(texts)
        if asyncio.iscoroutine(res) or isinstance(res, Awaitable):
            res = await res  # type: ignore[misc]
        return [list(map(float, v)) for v in res]  # type: ignore[union-attr]


def make_embedder(spec: Union[str, EmbedFn, Embedder], **options: Any) -> Embedder:
    if isinstance(spec, Embedder):
        return spec
    if callable(spec):
        return CallableEmbedder(spec, **{k: v for k, v in options.items() if k in ("name", "metric", "batch_size")})
    if ":" not in spec:
        raise ValueError(f"embedder spec must be '<provider>:<model>', got {spec!r}")
    provider, model = spec.split(":", 1)
    provider = provider.lower()
    if provider in ("openai", "gateway"):
        keep = ("base_url", "api_key_env", "batch_size", "dimensions", "timeout_s", "transport")
        return OpenAICompatible(model, provider=provider, **{k: v for k, v in options.items() if k in keep})
    if provider == "ollama":
        return Ollama(model, **{k: v for k, v in options.items() if k in ("base_url", "batch_size")})
    if provider == "local":
        return SentenceTransformers(model, **{k: v for k, v in options.items() if k == "batch_size"})
    if provider == "hash":
        return HashEmbedder(int(model or 256))
    raise ValueError(f"unknown embedding provider {provider!r}: use openai, gateway, ollama, local or hash, "
                     "or pass a callable")
