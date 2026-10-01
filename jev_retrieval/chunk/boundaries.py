"""Jev boundary questions (FR-C2).

For every candidate gap inside a section, two Nouls go into one packed request.
The state is the surrounding window of text (context only); each question
carries the sentence pair it asks about:

* ``continues``:   does ``sentence`` continue the specific point that ``previous`` is making?
* ``refers_back``: does ``sentence`` depend on ``previous`` to be understood?

Why the pair is inline and not referenced by position: Jev answered
``sentences[i]``-style questions against the wrong sentences (mean absolute
error 0.63 on a labelled 8-gap probe, vs 0.10 for named keys and 0.07 for inline
pairs; see docs/RESULTS.md). Counting and indexing are a known weak spot
(https://docs.typesafe.ai/model-jaggedness/jev-1.13). ``style: keyed`` uses
named keys instead, about 25% fewer tokens.

Gaps that code already decides (section boundaries, right after a heading) are
never asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from ..jev import JevClient, JevError, Noul, estimate_tokens
from ..jev.client import STATE_BUDGET_TOKENS
from ..types import Unit
from .segmenter import Gap

CONTINUES = "In the document, does `sentence` continue the specific point that `previous` is making?"
REFERS_BACK = ("Does `sentence` depend on `previous` to be understood, for example by referring back to it "
               "with words like this, it, these or such?")
CONTINUES_KEYED = "Does `{i}` continue the specific point that `{p}` is making?"
REFERS_BACK_KEYED = ("Does `{i}` depend on `{p}` to be understood, for example by referring back to it with "
                     "words like this, it, these or such?")

MAX_UNIT_CHARS = 1200  # a unit's text is truncated to this in state and questions


@dataclass
class BoundaryConfig:
    style: str = "inline"  # inline | keyed
    window_units: int = 24
    overlap_units: int = 4
    w_continues: float = 0.6
    w_refers_back: float = 0.4
    paragraph_discount: float = 0.7  # multiply cost at a paragraph break
    protect_refers_back: float = 0.5


@dataclass
class BoundaryResult:
    gaps: List[Gap]
    answers: Dict[int, Dict[str, float]] = field(default_factory=dict)  # gap index -> {continues, refers_back}
    requests: int = 0


def candidate_gaps(units: Sequence[Unit]) -> List[int]:
    """Gaps worth asking Jev about: inside a section and not directly after a heading."""
    return [i for i in range(1, len(units))
            if units[i].section == units[i - 1].section and units[i - 1].kind != "heading"]


def _windows(n: int, size: int, overlap: int) -> List[Tuple[int, int]]:
    if n <= size:
        return [(0, n)]
    step = max(1, size - overlap)
    out = []
    s = 0
    while True:
        e = min(n, s + size)
        out.append((s, e))
        if e == n:
            break
        s += step
    return out


def _key(i: int) -> str:
    return f"s{i:03d}"


Plan = Tuple[Any, Dict[str, Noul], Dict[str, int]]


def plan_requests(text: str, units: Sequence[Unit], cfg: BoundaryConfig) -> List[Plan]:
    """Build (state, questions, qid -> gap) per window. Pure: can be dry-run and costed."""
    if cfg.style not in ("inline", "keyed"):
        raise ValueError(f"boundary style must be 'inline' or 'keyed', got {cfg.style!r}")
    gaps = candidate_gaps(units)
    if not gaps:
        return []
    texts = [text[u.start:u.end][:MAX_UNIT_CHARS] for u in units]
    size = max(4, cfg.window_units)
    per_q = 2 if cfg.style == "inline" else 0

    def window_tokens(s: int, e: int) -> int:
        body = estimate_tokens(texts[s:e])
        return body + (e - s) * 60 + per_q * sum(estimate_tokens(t) for t in texts[s:e])

    while size > 4 and max(window_tokens(s, e) for s, e in _windows(len(units), size, cfg.overlap_units)) \
            > STATE_BUDGET_TOKENS // 2:
        size //= 2
    windows = _windows(len(units), size, min(cfg.overlap_units, size // 2))
    owner: Dict[int, int] = {}
    for g in gaps:
        best_w, best_margin = 0, -1
        for wi, (s, e) in enumerate(windows):
            if s <= g - 1 and g < e:
                margin = min(g - 1 - s, e - 1 - g)
                if margin > best_margin:
                    best_w, best_margin = wi, margin
        owner[g] = best_w
    plans: List[Plan] = []
    for wi, (s, e) in enumerate(windows):
        mine = [g for g in gaps if owner[g] == wi]
        if not mine:
            continue
        questions: Dict[str, Noul] = {}
        qmap: Dict[str, int] = {}
        if cfg.style == "keyed":
            state: Any = {_key(i): texts[i] for i in range(s, e)}
            for g in mine:
                questions[f"c{g}"] = Noul(CONTINUES_KEYED.format(i=_key(g), p=_key(g - 1)))
                questions[f"r{g}"] = Noul(REFERS_BACK_KEYED.format(i=_key(g), p=_key(g - 1)))
        else:
            state = " ".join(texts[s:e])
            for g in mine:
                pair = {"previous": texts[g - 1], "sentence": texts[g]}
                questions[f"c{g}"] = Noul({**pair, "question": CONTINUES})
                questions[f"r{g}"] = Noul({**pair, "question": REFERS_BACK})
        for g in mine:
            qmap[f"c{g}"] = g
            qmap[f"r{g}"] = g
        plans.append((state, questions, qmap))
    return plans


def gap_cost(p_continues: float, p_refers: float, paragraph_break: bool, cfg: BoundaryConfig) -> float:
    c = cfg.w_continues * p_continues + cfg.w_refers_back * p_refers
    if paragraph_break:
        c *= cfg.paragraph_discount
    return max(0.0, min(1.0, c))


async def jev_boundaries(jev: JevClient, text: str, units: Sequence[Unit], cfg: BoundaryConfig) -> BoundaryResult:
    """Ask Jev about every candidate gap. Raises JevError if any window fails."""
    plans = plan_requests(text, units, cfg)
    results = await jev.ask_many([(s, q) for s, q, _ in plans])
    answers: Dict[int, Dict[str, float]] = {}
    errors = []
    for (_state, _qs, qmap), res in zip(plans, results):
        if isinstance(res, JevError):
            errors.append(str(res))
            continue
        for qid, g in qmap.items():
            key = "continues" if qid.startswith("c") else "refers_back"
            answers.setdefault(g, {})[key] = float(res.answers[qid].value)
    if errors:
        raise JevError(f"{len(errors)} of {len(plans)} boundary requests failed: {errors[0]}", kind="partial")
    gaps: List[Gap] = []
    for i in range(1, len(units)):
        prev, cur = units[i - 1], units[i]
        if prev.kind == "heading":
            gaps.append(Gap(i, 1.0, protected=True))
            continue
        if i not in answers:
            gaps.append(Gap(i, 0.0))  # section boundary: the segmenter never crosses it anyway
            continue
        a = answers[i]
        pc, pr = a.get("continues", 0.5), a.get("refers_back", 0.5)
        gaps.append(Gap(i, gap_cost(pc, pr, cur.first_in_block, cfg), protected=pr >= cfg.protect_refers_back))
    return BoundaryResult(gaps=gaps, answers=answers, requests=len(plans))
