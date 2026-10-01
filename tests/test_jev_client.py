import asyncio

import httpx
import pytest

from jev_retrieval.jev import Choice, JevClient, JevConfig, JevError, Noul, Score, SpecError, question_from_dict
from jev_retrieval.jev.questions import parse_answer


def run(c):
    return asyncio.run(c)


def test_question_specs_validate():
    assert Noul("x?").to_wire() == {"type": "noul", "instructions": "x?"}
    with pytest.raises(SpecError):
        Choice("x?", {"only": None}).to_wire()
    with pytest.raises(SpecError):
        Choice("x?", {str(i): None for i in range(256)}).to_wire()
    with pytest.raises(SpecError):
        Score("x?", ["a"]).to_wire()
    with pytest.raises(SpecError):
        Score("x?", [str(i) for i in range(11)]).to_wire()
    with pytest.raises(SpecError):
        Noul("x?", {"yes": "y"}).to_wire()
    assert question_from_dict({"type": "choice", "instructions": "q", "criteria": {"a": 1, "b": 2}}).to_wire()["type"] == "choice"
    with pytest.raises(SpecError):
        question_from_dict({"type": "choice", "instructions": "q", "criteria": ["a", "b"]})


def test_parse_answer_dialects():
    assert parse_answer({"type": "noul", "noul": 0.8}).value == 0.8
    assert parse_answer({"type": "boolean", "probability": 0.3}).value == 0.3  # Vercel native
    a = parse_answer({"type": "choice", "probabilities": {"x": 0.7, "y": 0.3}})
    assert a.choice == "x" and 0 < a.confidence < 1
    s = parse_answer({"type": "score", "probabilities": {"0": 0.5, "1": 0.5}})
    assert s.score == pytest.approx(0.5)


def test_auto_backend_order(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    assert JevConfig().resolve()["backend"] == "openrouter"
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "v")
    assert JevConfig().resolve()["backend"] == "vercel"
    monkeypatch.setenv("TYPESAFE_API_KEY", "t")
    r = JevConfig().resolve()
    assert r["backend"] == "typesafe" and r["model"] == "jev-1.13.0"


def test_no_key_is_a_clear_error():
    with pytest.raises(JevError) as e:
        JevConfig().resolve()
    assert "TYPESAFE_API_KEY" in str(e.value)


def test_ask_retries_then_caches(fake_jev, tmp_path):
    fj, transport = fake_jev
    fj.fail_status, fj.fail_times = 503, 2
    cfg = JevConfig(cache_dir=str(tmp_path / "cache"), max_retries=3)

    async def go():
        async with JevClient(cfg, transport=transport) as c:
            r1 = await c.ask("some text", {"q": Noul("Is this text filler with no usable information?")})
            r2 = await c.ask("some text", {"q": Noul("Is this text filler with no usable information?")})
            return c, r1, r2

    c, r1, r2 = run(go())
    assert not r1.cached and r2.cached
    assert c.usage.retries == 2 and c.usage.requests == 1 and c.usage.cached == 1
    assert len(fj.calls) == 3  # two 503s + one success; the second ask never hit the network
    assert fj.calls[-1]["model"] == "jev-1.13.0"


def test_non_retryable_raises(fake_jev, tmp_path):
    fj, transport = fake_jev
    fj.fail_status, fj.fail_times = 422, 1

    async def go():
        async with JevClient(JevConfig(cache_dir=None), transport=transport) as c:
            await c.ask("x", {"q": Noul("?")})

    with pytest.raises(JevError) as e:
        run(go())
    assert e.value.status == 422


def test_missing_answer_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    t = httpx.MockTransport(lambda r: httpx.Response(200, json={"model": "m", "answers": {}, "usage": {}}))

    async def go():
        async with JevClient(JevConfig(cache_dir=None), transport=t) as c:
            await c.ask("x", {"q": Noul("?")})

    with pytest.raises(JevError, match="missing answers"):
        run(go())


def test_ask_many_returns_errors_in_place(fake_jev):
    fj, transport = fake_jev
    fj.fail_status, fj.fail_times = 401, 1

    async def go():
        async with JevClient(JevConfig(cache_dir=None, max_retries=0), transport=transport) as c:
            return await c.ask_many([("a", {"q": Noul("?")}), ("b", {"q": Noul("?")})])

    res = run(go())
    assert sum(isinstance(r, JevError) for r in res) == 1
    assert sum(not isinstance(r, JevError) for r in res) == 1


def test_oversize_request_refused(fake_jev):
    _, transport = fake_jev

    async def go():
        async with JevClient(JevConfig(cache_dir=None), transport=transport) as c:
            await c.ask("x" * 400_000, {"q": Noul("?")})

    with pytest.raises(JevError, match="budget"):
        run(go())


def test_clients_sharing_a_key_share_one_rate_budget(monkeypatch):
    """Jev limits are per key: two clients opened at once must not double the request rate."""
    import asyncio
    import time

    import httpx

    from jev_retrieval.jev.client import JevClient, JevConfig
    from jev_retrieval.jev.questions import Noul

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    stamps = []

    def handler(request: httpx.Request) -> httpx.Response:
        stamps.append(time.monotonic())
        return httpx.Response(200, json={"model": "jev", "answers": {"q": {"type": "noul", "noul": 0.5}},
                                         "usage": {"input_tokens": 10, "output_tokens": 1}})

    transport = httpx.MockTransport(handler)
    cfg = JevConfig(backend="typesafe", cache_dir=None, max_rps=10, max_concurrency=16)

    async def worker(i: int) -> None:
        async with JevClient(cfg, transport=transport) as jev:
            for j in range(10):
                await jev.ask({"i": i, "j": j}, {"q": Noul("x?")})

    async def go() -> float:
        t0 = time.monotonic()
        await asyncio.gather(worker(1), worker(2))
        return time.monotonic() - t0

    elapsed = asyncio.run(go())
    assert len(stamps) == 20
    # 20 requests at 10/s with a 10-request burst: at least ~1 s when shared, ~0 s if each client had its own budget
    assert elapsed >= 0.9, elapsed
