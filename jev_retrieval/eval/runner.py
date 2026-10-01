"""Record candidates once, score systems offline.

``record()`` ingests an :class:`EvalSet` into a collection, then for every query:

1. recalls the top ``candidates`` chunks by vector similarity (within the
   query's ``scope`` document when set),
2. asks Jev the passage questions about every candidate (one request each),
3. optionally asks an LLM re-ranker to score every candidate (one request),
4. asks Jev the gate question on three candidate sets: the vector top 8, the
   passages threshold-mode classification includes at default thresholds, and
   the rank-mode top 5,
5. writes one JSON line with all of that plus which labelled units each
   candidate contains.

Recordings are append-only and resumable. ``systems.evaluate()`` then turns a
recording into metrics for any system and threshold without new API calls.
The one place this differs from the live pipeline: the gate is recorded for
the default thresholds only, so tuned thresholds report gate results for the
default passage set (the tuning report says so).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from ..jev.client import JevClient, JevError
from ..pipeline import Pipeline
from ..retrieve.stages import GATE_QUESTION, PASSAGE_QUESTIONS, ClassifyConfig, evidence_order, route_passage
from ..stores.filters import parse
from . import EvalQuery, EvalSet, units_for
from .llm_rerank import LLMReranker

# The two passage sets the gate is recorded on. Pinned explicitly so recordings
# keep their meaning when library defaults change.
DEFAULT_CLASSIFY = ClassifyConfig(select="threshold", max_passages=8)  # gate.jev_default
RANK_CLASSIFY = ClassifyConfig(select="rank", max_passages=5)         # gate.jev_rank


@dataclass
class RecordOptions:
    candidates: int = 30
    gate_top: int = 8
    concurrency: int = 4
    llm: Optional[LLMReranker] = None
    jev: bool = True
    max_passage_chars: int = 6000
    gate_passage_chars: int = 3000


def _row_path(out_dir: Path, collection: str) -> Path:
    return out_dir / f"record_{collection}.jsonl"


def _done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" not in row:
            out.add(row["id"])
    return out


async def _gate(jev: JevClient, query: str, texts: List[str], chars: int) -> Optional[float]:
    if not texts:
        return 0.0
    resp = await jev.ask({"query": query, "passages": [t[:chars] for t in texts]}, {"answerable": GATE_QUESTION})
    return round(float(resp.answers["answerable"].value), 4)


async def record_query(rag: Pipeline, jev: Optional[JevClient], http: httpx.AsyncClient, ds: EvalSet,
                       q: EvalQuery, collection: str, opt: RecordOptions) -> Dict[str, Any]:
    t0 = time.monotonic()
    vec = await rag.embedder.embed_query(q.text)
    t_embed = time.monotonic()
    where: Dict[str, Any] = {"ne": {"quarantined": True}}
    if q.scope:
        where = {"and": [where, {"eq": {"doc_id": q.scope}}]}
    hits = await asyncio.to_thread(rag.store.query, collection, vec, parse(where), opt.candidates)
    t_recall = time.monotonic()
    cands = [{"id": h.id, "doc_id": str(h.metadata.get("doc_id", "")), "vs": round(float(h.score), 5),
              "tokens": int(h.metadata.get("tokens") or max(1, len(h.text) // 4)),
              "units": units_for(q, str(h.metadata.get("doc_id", "")), h.text, ds.level)} for h in hits]
    texts = [h.text for h in hits]
    row: Dict[str, Any] = {"id": q.id, "answerable": q.answerable, "scope": q.scope, "type": q.meta.get("type"),
                           "gate_top": opt.gate_top, "cands": cands, "ms": {"embed": round((t_embed - t0) * 1000, 1),
                                                  "recall": round((t_recall - t_embed) * 1000, 1)}}

    async def jev_part() -> None:
        if jev is None or not hits:
            return
        t = time.monotonic()
        jobs = [({"query": q.text, "passage": tx[:opt.max_passage_chars]}, PASSAGE_QUESTIONS) for tx in texts]
        results = await jev.ask_many(jobs)
        row["ms"]["classify"] = round((time.monotonic() - t) * 1000, 1)
        errs = 0
        for c, r in zip(cands, results):
            if isinstance(r, JevError):
                errs += 1
                c["jev"] = None
            else:
                c["jev"] = {k: round(float(a.value), 4) for k, a in r.answers.items()}
        row["jev_errors"] = errs
        # gate on (a) vector top-N, (b) what default classification would include
        t = time.monotonic()
        inc = [i for i, c in enumerate(cands) if c.get("jev") and route_passage(c["jev"], DEFAULT_CLASSIFY) == "include"]
        inc.sort(key=lambda i: evidence_order(cands[i]))
        inc = inc[:DEFAULT_CLASSIFY.max_passages]
        con = [i for i, c in enumerate(cands) if c.get("jev") and route_passage(c["jev"], DEFAULT_CLASSIFY) == "conflict"]
        rank_inc = [i for i, c in enumerate(cands) if not c.get("jev") or route_passage(c["jev"], RANK_CLASSIFY) == "include"]
        rank_inc.sort(key=lambda i: evidence_order(cands[i]))
        rank_con = [i for i, c in enumerate(cands) if c.get("jev") and route_passage(c["jev"], RANK_CLASSIFY) == "conflict"]
        rank_set = rank_inc[:RANK_CLASSIFY.max_passages] + rank_con
        gates = await asyncio.gather(
            _gate(jev, q.text, texts[:opt.gate_top], opt.gate_passage_chars),
            _gate(jev, q.text, [texts[i] for i in inc + con], opt.gate_passage_chars),
            _gate(jev, q.text, [texts[i] for i in rank_set], opt.gate_passage_chars),
            return_exceptions=True)
        row["gate"] = {"vector_top": None if isinstance(gates[0], BaseException) else gates[0],
                       "jev_default": None if isinstance(gates[1], BaseException) else gates[1],
                       "jev_rank": None if isinstance(gates[2], BaseException) else gates[2],
                       "jev_default_ids": [cands[i]["id"] for i in inc + con],
                       "jev_rank_ids": [cands[i]["id"] for i in rank_set]}
        row["ms"]["gate"] = round((time.monotonic() - t) * 1000, 1)

    async def llm_part() -> None:
        if opt.llm is None or not hits:
            return
        t = time.monotonic()
        try:
            scores, usage = await opt.llm.score(http, q.text, texts)
            for c, s in zip(cands, scores):
                c["llm"] = s
            row["llm_usage"] = {"in": int(usage.get("prompt_tokens", 0)), "out": int(usage.get("completion_tokens", 0))}
        except RuntimeError as exc:
            row["llm_error"] = str(exc)[:300]
        row["ms"]["llm"] = round((time.monotonic() - t) * 1000, 1)

    await asyncio.gather(jev_part(), llm_part())
    row["ms"]["total"] = round((time.monotonic() - t0) * 1000, 1)
    return row


async def record(rag: Pipeline, ds: EvalSet, collection: str, out_dir: Path, opt: RecordOptions,
                 *, ingest: bool = True, limit: Optional[int] = None,
                 progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Ingest (unless the collection already has every document) and record every query."""
    say = progress or (lambda s: None)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {"dataset": ds.summary(), "collection": collection,
                            "candidates": opt.candidates, "llm": opt.llm.spec if opt.llm else None}
    if ingest:
        rep = await rag.aingest(ds.docs, collection, concurrency=4)
        meta["ingest"] = {"chunks": sum(d.chunks for d in rep.docs), "dropped": sum(d.dropped for d in rep.docs),
                          "quarantined": sum(d.quarantined for d in rep.docs), "failed": rep.count("failed"),
                          "unchanged": rep.count("skipped_unchanged"), "jev": rep.jev,
                          "embedding_tokens": rep.embedding_tokens, "seconds": round(rep.seconds, 1)}
        say(f"ingested {len(ds.docs)} docs into {collection}: {meta['ingest']['chunks']} chunks "
            f"({meta['ingest']['unchanged']} unchanged)")
    path = _row_path(out_dir, collection)
    done = _done_ids(path)
    todo = [q for q in ds.queries if q.id not in done]
    if limit is not None:
        todo = todo[:limit]
    say(f"recording {len(todo)} queries ({len(done)} already done)")
    jev_client = JevClient(rag.jev_config, transport=rag.jev_transport) if opt.jev else None
    if jev_client is not None and not jev_client.available():
        jev_client = None
    sem = asyncio.Semaphore(opt.concurrency)
    lock = asyncio.Lock()
    n = {"done": 0}
    t0 = time.monotonic()

    async def one(jev: Optional[JevClient], http: httpx.AsyncClient, q: EvalQuery) -> None:
        async with sem:
            try:
                row = await record_query(rag, jev, http, ds, q, collection, opt)
            except Exception as exc:  # recorded and retried on the next run
                row = {"id": q.id, "error": f"{type(exc).__name__}: {exc}"[:400]}
        async with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            n["done"] += 1
            if n["done"] % 25 == 0 or n["done"] == len(todo):
                say(f"  {n['done']}/{len(todo)} queries")

    async with httpx.AsyncClient() as http:
        if jev_client is not None:
            async with jev_client as jev:
                await asyncio.gather(*(one(jev, http, q) for q in todo))
                meta["jev_usage"] = jev.usage.to_dict(rag.jev_config.price_per_million)
        else:
            await asyncio.gather(*(one(None, http, q) for q in todo))
    meta["record_seconds"] = round(time.monotonic() - t0, 1)
    if opt.llm is not None:
        meta["llm_usage"] = dict(opt.llm.usage, cost_usd=opt.llm.cost_usd())
    with (out_dir / f"record_{collection}.meta.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
    return meta


def load_recording(out_dir: Path, collection: str) -> List[Dict[str, Any]]:
    """Latest successful row per query id."""
    rows: Dict[str, Dict[str, Any]] = {}
    path = _row_path(out_dir, collection)
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" not in r:
            rows[r["id"]] = r
    return list(rows.values())


__all__ = ["RecordOptions", "record", "record_query", "load_recording"]
