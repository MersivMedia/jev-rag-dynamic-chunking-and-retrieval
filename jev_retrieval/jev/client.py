"""Async Jev client for TypeSafe's System One API.

Three backends speak the same wire format (``POST {base}/v1/systemone``):

* ``typesafe``   - TypeSafe direct (``TYPESAFE_API_KEY``, model ``jev-1.13.0``)
* ``vercel``     - Vercel AI Gateway (``AI_GATEWAY_API_KEY``, model ``typesafe-ai/jev``)
* ``openrouter`` - OpenRouter System One API (``OPENROUTER_API_KEY``, model ``typesafe/jev-1.13``)

``backend: auto`` picks the first backend whose key is set, in that order.

The client adds what batch RAG work needs on top of the raw API: a shared rate
limiter (requests and tokens per second), bounded concurrency, retries with
backoff that honour ``retry-after``, an on-disk answer cache, and usage
accounting. Any failure raises :class:`JevError`; callers decide how to fall back.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import httpx

from .. import __version__
from .cache import AnswerCache, cache_key
from .questions import Answer, Question, parse_answer, questions_to_wire

BACKENDS: Dict[str, Dict[str, str]] = {
    "typesafe": {
        "base_url": "https://api.typesafe.ai",
        "api_key_env": "TYPESAFE_API_KEY",
        "model": "jev-1.13.0",
    },
    "vercel": {
        "base_url": "https://ai-gateway.vercel.sh/typesafe",
        "api_key_env": "AI_GATEWAY_API_KEY",
        "model": "typesafe-ai/jev",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "typesafe/jev-1.13",
    },
}
AUTO_ORDER = ("typesafe", "vercel", "openrouter")
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}

# Published limits for jev-1.13.0 (https://docs.typesafe.ai/models).
STATE_BUDGET_TOKENS = 32_000  # state + longest single question
REQUEST_BUDGET_TOKENS = 64_000
PRICE_PER_MILLION_INPUT = 0.042


class JevError(RuntimeError):
    """Any failure to obtain answers. Callers treat it as 'no decision'."""

    def __init__(self, message: str, *, status: Optional[int] = None, kind: str = "error"):
        super().__init__(message)
        self.status = status
        self.kind = kind


def estimate_tokens(obj: Any) -> int:
    """Conservative token estimate for budget packing (about 3 characters per token)."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return len(text) // 3 + 8


@dataclass
class JevResponse:
    model: str
    answers: Dict[str, Answer]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False


@dataclass
class JevConfig:
    backend: str = "auto"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    cache_dir: Optional[str] = ".jev-retrieval/cache"
    max_rps: float = 30.0  # published limit is 40; keep headroom
    max_tokens_per_s: float = 80_000.0  # published limit is 100k
    max_concurrency: int = 16
    timeout_s: float = 30.0
    max_retries: int = 5
    zero_data_retention: bool = False
    price_per_million: float = PRICE_PER_MILLION_INPUT

    def resolve(self) -> Dict[str, str]:
        backend = self.backend
        if backend == "auto":
            for name in AUTO_ORDER:
                env = BACKENDS[name]["api_key_env"]
                if os.environ.get(env):
                    backend = name
                    break
            else:
                raise JevError(
                    "no Jev key found: set TYPESAFE_API_KEY, AI_GATEWAY_API_KEY or OPENROUTER_API_KEY",
                    kind="config",
                )
        if backend not in BACKENDS:
            raise JevError(f"unknown Jev backend {backend!r}", kind="config")
        preset = BACKENDS[backend]
        key_env = self.api_key_env or preset["api_key_env"]
        return {
            "backend": backend,
            "base_url": (self.base_url or preset["base_url"]).rstrip("/"),
            "model": self.model or preset["model"],
            "api_key": os.environ.get(key_env, ""),
            "api_key_env": key_env,
        }


@dataclass
class Usage:
    requests: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0
    retries: int = 0
    latency_ms: List[float] = field(default_factory=list)

    def add(self, resp: JevResponse) -> None:
        if resp.cached:
            self.cached += 1
            return
        self.requests += 1
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.latency_ms.append(resp.latency_ms)

    def cost(self, price_per_million: float = PRICE_PER_MILLION_INPUT) -> float:
        return self.input_tokens * price_per_million / 1_000_000

    def to_dict(self, price_per_million: float = PRICE_PER_MILLION_INPUT) -> Dict[str, Any]:
        lat = sorted(self.latency_ms)
        return {
            "requests": self.requests,
            "cached": self.cached,
            "errors": self.errors,
            "retries": self.retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost(price_per_million), 6),
            "latency_ms_p50": round(lat[len(lat) // 2], 1) if lat else None,
            "latency_ms_p90": round(lat[int(len(lat) * 0.9)], 1) if lat else None,
        }


class _Limiter:
    """Token buckets for requests/s and tokens/s, shared by every task on one loop."""

    def __init__(self, rps: float, tps: float) -> None:
        self.rps = max(0.1, rps)
        self.tps = max(1.0, tps)
        self._req = self.rps
        self._tok = self.tps
        self._t = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        dt = now - self._t
        self._t = now
        self._req = min(self.rps, self._req + dt * self.rps)
        self._tok = min(self.tps, self._tok + dt * self.tps)

    async def acquire(self, tokens: int) -> None:
        tokens = min(tokens, int(self.tps))
        while True:
            async with self._lock:
                self._refill()
                if self._req >= 1 and self._tok >= tokens:
                    self._req -= 1
                    self._tok -= tokens
                    return
                wait = max((1 - self._req) / self.rps, (tokens - self._tok) / self.tps, 0.005)
            await asyncio.sleep(wait)

    def penalise(self, seconds: float) -> None:
        """After a 429, drain the request bucket so every task backs off together."""
        self._req = min(self._req, -seconds * self.rps)


class JevClient:
    """Use as ``async with JevClient(config) as jev: await jev.ask(state, questions)``."""

    def __init__(self, config: Optional[JevConfig] = None, *, transport: Optional[httpx.AsyncBaseTransport] = None,
                 cache: Optional[AnswerCache] = None) -> None:
        self.config = config or JevConfig()
        self._transport = transport
        self._resolved: Optional[Dict[str, str]] = None
        self.cache = cache if cache is not None else AnswerCache(self.config.cache_dir)
        self.usage = Usage()
        self._http: Optional[httpx.AsyncClient] = None
        self._limiter: Optional[_Limiter] = None
        self._sem: Optional[asyncio.Semaphore] = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def resolved(self) -> Dict[str, str]:
        if self._resolved is None:
            self._resolved = self.config.resolve()
        return self._resolved

    @property
    def model(self) -> str:
        return self.resolved["model"]

    def available(self) -> bool:
        try:
            return bool(self.resolved["api_key"])
        except JevError:
            return False

    async def __aenter__(self) -> "JevClient":
        self._http = httpx.AsyncClient(
            transport=self._transport,
            headers={"User-Agent": f"jev-retrieval/{__version__} (+https://github.com/MersivMedia/jev-rag-retrieval)"},
            timeout=self.config.timeout_s,
        )
        self._limiter = _Limiter(self.config.max_rps, self.config.max_tokens_per_s)
        self._sem = asyncio.Semaphore(self.config.max_concurrency)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # -- API -----------------------------------------------------------------

    def build_payload(self, state: Any, questions: Mapping[str, Question]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "state": state,
            "questions": questions_to_wire(questions),
        }
        if self.resolved["backend"] == "vercel" and self.config.zero_data_retention:
            payload["providerOptions"] = {"gateway": {"zeroDataRetention": True}}
        return payload

    async def ask(self, state: Any, questions: Mapping[str, Question]) -> JevResponse:
        payload = self.build_payload(state, questions)
        key = cache_key(payload["model"], payload["state"], payload["questions"])
        hit = self.cache.get(key)
        if hit is not None:
            resp = JevResponse(model=hit["model"], cached=True,
                               answers={q: parse_answer(a) for q, a in hit["answers"].items()})
            self.usage.add(resp)
            return resp
        if not self.resolved["api_key"]:
            raise JevError(f"{self.resolved['api_key_env']} is not set", kind="config")
        if self._http is None or self._limiter is None or self._sem is None:
            raise JevError("JevClient must be used inside 'async with'", kind="config")

        est = estimate_tokens(payload["state"]) + estimate_tokens(payload["questions"])
        if est > REQUEST_BUDGET_TOKENS:
            raise JevError(f"request estimated at {est} tokens exceeds the {REQUEST_BUDGET_TOKENS} budget",
                           kind="too_large")
        url = f"{self.resolved['base_url']}/v1/systemone"
        headers = {"Authorization": f"Bearer {self.resolved['api_key']}", "Content-Type": "application/json"}

        async with self._sem:
            attempt = 0
            r: Optional[httpx.Response] = None
            t0 = time.monotonic()
            while True:
                await self._limiter.acquire(est)
                t0 = time.monotonic()
                r = None
                try:
                    r = await self._http.post(url, json=payload, headers=headers)
                except httpx.TimeoutException as exc:
                    err: Optional[JevError] = JevError(f"timeout: {exc}", kind="timeout")
                    status = None
                except httpx.HTTPError as exc:
                    err = JevError(f"transport error: {exc}", kind="transport")
                    status = None
                else:
                    status = r.status_code
                    err = None if status == 200 else JevError(_error_message(r), status=status,
                                                               kind=_error_kind(r))
                if err is None:
                    break
                retryable = status is None or status in RETRYABLE_STATUS
                if not retryable or attempt >= self.config.max_retries:
                    self.usage.errors += 1
                    raise err
                attempt += 1
                self.usage.retries += 1
                delay = (_retry_after(r) if r is not None else None) or \
                    min(8.0, 0.25 * (2 ** attempt)) + random.uniform(0, 0.1)
                if status == 429:
                    self._limiter.penalise(delay)
                await asyncio.sleep(delay)

        assert r is not None
        latency = (time.monotonic() - t0) * 1000.0
        try:
            body = r.json()
            answers = {qid: parse_answer(a) for qid, a in (body.get("answers") or {}).items()}
        except Exception as exc:
            self.usage.errors += 1
            raise JevError(f"unparseable response: {exc}", kind="parse") from exc
        missing = set(payload["questions"]) - set(answers)
        if missing:
            self.usage.errors += 1
            raise JevError(f"response missing answers for {sorted(missing)[:5]}", kind="parse")
        usage = body.get("usage") or {}
        resp = JevResponse(
            model=str(body.get("model") or self.model),
            answers=answers,
            input_tokens=int(usage.get("input_tokens", usage.get("inputTokens", 0)) or 0),
            output_tokens=int(usage.get("output_tokens", usage.get("outputTokens", 0)) or 0),
            latency_ms=latency,
        )
        self.cache.put(key, {"model": resp.model, "answers": body.get("answers")})
        self.usage.add(resp)
        return resp

    async def ask_many(self, jobs: List[Tuple[Any, Mapping[str, Question]]]) -> List[Any]:
        """Run many requests concurrently. Each result is a JevResponse or a JevError."""
        async def one(state: Any, qs: Mapping[str, Question]) -> Any:
            try:
                return await self.ask(state, qs)
            except JevError as exc:
                return exc
        return list(await asyncio.gather(*(one(s, q) for s, q in jobs)))


def _retry_after(resp: Any) -> Optional[float]:
    try:
        value = resp.headers.get("retry-after")
        return min(30.0, float(value)) if value is not None else None
    except (ValueError, AttributeError):
        return None


def _error_message(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error")
        if isinstance(err, dict):
            return f"HTTP {resp.status_code}: {err.get('message') or err}"
        if err:
            return f"HTTP {resp.status_code}: {err}"
    except Exception:
        pass
    return f"HTTP {resp.status_code}: {resp.text[:200]}"


def _error_kind(resp: httpx.Response) -> str:
    try:
        err = resp.json().get("error")
        if isinstance(err, dict) and err.get("type"):
            return str(err["type"])
    except Exception:
        pass
    return {401: "auth", 403: "forbidden", 422: "invalid_request", 429: "rate_limited"}.get(
        resp.status_code, "http")
