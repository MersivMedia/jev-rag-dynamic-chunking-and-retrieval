"""Shared fixtures: a deterministic fake Jev served through httpx.MockTransport.

The fake answers from keywords so tests can assert pipeline behaviour without
the network. It speaks the real wire format (POST /v1/systemone, typed answers).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

TOPICS = {
    "auth": ("token", "sso", "sign-on", "session", "login"),
    "billing": ("invoice", "refund", "payment", "plan", "billing", "fee"),
    "api": ("api", "webhook", "endpoint", "rate limit", "requests per minute"),
}


def _topic(text: str) -> Optional[str]:
    t = text.lower()
    best, n = None, 0
    for name, words in TOPICS.items():
        k = sum(t.count(w) for w in words)
        if k > n:
            best, n = name, k
    return best


def _text_of(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x)


class FakeJev:
    """Callable transport handler. ``fail`` = set of question-id prefixes to 500 on."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.fail_status: Optional[int] = None
        self.fail_times = 0
        self.overrides: Dict[str, Callable[[Any, Any], Any]] = {}

    def answer(self, qid: str, q: Dict[str, Any], state: Any) -> Dict[str, Any]:
        instr = q["instructions"]
        itext = _text_of(instr)
        stext = _text_of(state)
        for prefix, fn in self.overrides.items():
            if qid.startswith(prefix):
                return fn(q, state)
        if q["type"] == "choice":
            opts = list(q["criteria"])
            src = state["query"] if isinstance(state, dict) and "query" in state else _text_of(state.get("text", state)
                                                                                                    if isinstance(state, dict) else state)
            top = _topic(src)
            pick = top if top in opts else ("any" if "any" in opts else "other")
            probs = {o: (0.94 if o == pick else 0.06 / (len(opts) - 1)) for o in opts}
            return {"type": "choice", "choice": pick, "probabilities": probs, "confidence": 0.9}
        if q["type"] == "score":
            return {"type": "score", "score": 1.0, "probabilities": {"0": 0.5, "1": 0.5}, "confidence": 0.1}
        # nouls
        if isinstance(instr, dict) and "previous" in instr:  # boundary questions (inline style)
            same = _topic(instr["previous"]) == _topic(instr["sentence"]) and _topic(instr["sentence"]) is not None
            refers = bool(re.match(r"\s*(this|it|they|these|such|that)\b", instr["sentence"], re.I))
            if qid.startswith("r"):
                return {"type": "noul", "noul": 0.9 if refers else (0.6 if same else 0.05)}
            return {"type": "noul", "noul": 0.92 if (same or refers) else 0.04}
        low = itext.lower()
        passage = state.get("passage", "") if isinstance(state, dict) else ""
        query = state.get("query", "") if isinstance(state, dict) else ""
        text = state.get("text", "") if isinstance(state, dict) else stext
        if "instructions addressed to an ai" in low:
            src = passage or text
            return {"type": "noul", "noul": 0.97 if "ignore all previous instructions" in src.lower() else 0.02}
        if "boilerplate" in low:
            return {"type": "noul", "noul": 0.95 if ("cookies" in text.lower() or "|" in text and "privacy" in
                                                    text.lower()) else 0.03}
        if "filler" in low:
            return {"type": "noul", "noul": 0.9 if len(text.split()) < 4 else 0.05}
        if "understood on its own" in low:
            return {"type": "noul", "noul": 0.3 if re.match(r"\s*(this|it|they)\b", text, re.I) else 0.85}
        if "address the subject" in low:
            return {"type": "noul", "noul": 0.95 if _topic(passage) and _topic(passage) == _topic(query) else 0.04}
        if "usable in a direct answer" in low:
            return {"type": "noul", "noul": 0.9 if _topic(passage) and _topic(passage) == _topic(query) else 0.03}
        if "conflict with a factual premise" in low:
            return {"type": "noul", "noul": 0.93 if "30 days" in query and "14 days" in passage else 0.05}
        if "contain the information needed" in low:
            ps = state.get("passages", []) if isinstance(state, dict) else []
            return {"type": "noul", "noul": 0.95 if any(_topic(p) == _topic(query) for p in ps) and ps else 0.05}
        if "reference documents could answer" in low:
            return {"type": "noul", "noul": 0.9 if _topic(query) else 0.2}
        return {"type": "noul", "noul": 0.5}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append(body)
        if self.fail_times > 0 and self.fail_status:
            self.fail_times -= 1
            return httpx.Response(self.fail_status, json={"error": {"type": "overloaded", "message": "busy"}},
                                  headers={"retry-after": "0"})
        answers = {qid: self.answer(qid, q, body["state"]) for qid, q in body["questions"].items()}
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": answers,
                                         "usage": {"input_tokens": 100 + len(json.dumps(body)) // 4,
                                                   "output_tokens": 10 * len(answers)}})


@pytest.fixture
def fake_jev(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    for k in ("AI_GATEWAY_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    fj = FakeJev()
    return fj, httpx.MockTransport(fj)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k in ("TYPESAFE_API_KEY", "AI_GATEWAY_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
