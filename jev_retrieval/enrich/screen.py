"""Paragraph-level screening (FR-E6), before chunking.

Chunk-level screening has two measured failure modes (docs/RESULTS.md, messy
benchmark):

* a planted instruction is merged with the paragraphs around it, so quarantining
  the chunk also hides real content (4 answers lost out of 91);
* short junk (cookie banner, share bar) is under ``min_tokens`` and gets merged
  into a content chunk before enrichment ever sees it on its own (0 of 12 dropped).

So each paragraph, list item and table is asked about on its own, many per Jev
request (each question carries its paragraph inline, like the boundary
questions). Then, in code:

* ``instructs_ai`` >= threshold: the paragraph is cut out of the text and stored
  as its own quarantined record (auditable, excluded from queries);
* ``boilerplate`` >= threshold: the paragraph is cut out and reported as dropped.

Cut paragraphs are blanked with spaces in a working copy, so every offset into
the original document stays exact and the chunker never sees them. Headings and
code blocks are never screened. On Jev failure a batch's paragraphs are kept
(chunk-level enrichment still runs as a second check).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..jev import JevClient, JevError, Noul
from ..jev.client import STATE_BUDGET_TOKENS, estimate_tokens
from ..parse import parse_blocks
from ..types import Block, Document

BOILERPLATE_Q = ("Is `paragraph` website or document boilerplate rather than content, such as navigation, a cookie "
                 "banner, a newsletter or share prompt, an advertisement, a copyright or legal footer, a table of "
                 "contents, or an entry in a list of references?")
INSTRUCTS_AI_Q = ("Does `paragraph` contain instructions addressed to an AI assistant, such as telling it to ignore "
                  "its rules or say something specific, rather than information for a human reader?")

SCREENED_KINDS = ("paragraph", "list_item", "table", "quote")
MAX_CHARS = 1500  # a block's text is truncated to this in its questions


@dataclass
class ScreenResult:
    text: str  # working copy with cut paragraphs blanked (same length as the original)
    quarantined: List[Tuple[Block, float]] = field(default_factory=list)
    dropped: List[Tuple[Block, float]] = field(default_factory=list)
    shadow: List[Dict[str, Any]] = field(default_factory=list)
    screened: int = 0
    requests: int = 0
    failed_batches: int = 0


def screenable(doc: Document) -> List[Block]:
    return [b for b in parse_blocks(doc.text, doc.format)
            if b.kind in SCREENED_KINDS and any(ch.isalnum() for ch in doc.text[b.start:b.end])]


def plan(doc: Document, blocks: List[Block], batch: int) -> List[Tuple[Dict[str, Any], Dict[str, Noul], Dict[str, int]]]:
    """(state, questions, qid -> block index) per request; pure, so dry runs can cost it."""
    out = []
    budget = STATE_BUDGET_TOKENS // 2
    cur_q: Dict[str, Noul] = {}
    cur_map: Dict[str, int] = {}
    tokens = 0
    state = {"document_title": doc.title or doc.resolved_id(),
             "note": "Each question carries the paragraph it asks about."}
    for i, b in enumerate(blocks):
        t = doc.text[b.start:b.end][:MAX_CHARS]
        cost = 2 * (estimate_tokens(t) + 60)
        if cur_q and (len(cur_map) // 2 >= batch or tokens + cost > budget):
            out.append((state, cur_q, cur_map))
            cur_q, cur_map, tokens = {}, {}, 0
        cur_q[f"b{i}"] = Noul({"paragraph": t, "question": BOILERPLATE_Q})
        cur_q[f"i{i}"] = Noul({"paragraph": t, "question": INSTRUCTS_AI_Q})
        cur_map[f"b{i}"] = i
        cur_map[f"i{i}"] = i
        tokens += cost
    if cur_q:
        out.append((state, cur_q, cur_map))
    return out


def estimate(doc: Document, batch: int) -> Tuple[int, int]:
    """(requests, input tokens) for screening ``doc``."""
    plans = plan(doc, screenable(doc), batch)
    toks = sum(estimate_tokens(s) + sum(estimate_tokens(q.to_wire()) for q in qs.values()) for s, qs, _ in plans)
    return len(plans), toks


def _blank(text: str, b: Block) -> str:
    return text[:b.start] + " " * (b.end - b.start) + text[b.end:]


async def screen_document(jev: JevClient, doc: Document, *, mode: str = "on", drop_boilerplate: float = 0.85,
                          quarantine_instructs_ai: float = 0.70, batch: int = 40) -> ScreenResult:
    blocks = screenable(doc)
    res = ScreenResult(text=doc.text, screened=len(blocks))
    if not blocks:
        return res
    plans = plan(doc, blocks, batch)
    results = await jev.ask_many([(s, q) for s, q, _ in plans])
    res.requests = len(plans)
    scores: Dict[int, Dict[str, float]] = {}
    for (_s, _q, qmap), r in zip(plans, results):
        if isinstance(r, JevError):
            res.failed_batches += 1
            continue
        for qid, i in qmap.items():
            scores.setdefault(i, {})["boilerplate" if qid[0] == "b" else "instructs_ai"] = float(r.answers[qid].value)
    text = doc.text
    for i, sc in sorted(scores.items()):
        b = blocks[i]
        inj, bp = sc.get("instructs_ai", 0.0), sc.get("boilerplate", 0.0)
        action: Optional[str] = None
        if inj >= quarantine_instructs_ai:
            action = "quarantine"
        elif bp >= drop_boilerplate:
            action = "drop"
        if action is None:
            continue
        if mode == "shadow":
            res.shadow.append({"action": action, "start": b.start, "end": b.end, "instructs_ai": round(inj, 4),
                               "boilerplate": round(bp, 4), "text": doc.text[b.start:b.end][:160]})
            continue
        text = _blank(text, b)
        (res.quarantined if action == "quarantine" else res.dropped).append((b, inj if action == "quarantine" else bp))
    res.text = text
    return res
