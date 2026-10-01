"""The Pipeline: ingest documents into a vector store, retrieve with Jev.

Sync methods (``ingest``, ``retrieve``, ``answer``) wrap async ones
(``aingest``, ``aretrieve``, ``aanswer``) for use outside an event loop.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from . import __version__
from .chunk import ChunkConfig, chunk_document, estimate_jev_requests
from .config import Config
from .embed import Embedder, make_embedder
from .enrich import EnrichConfig, enrich_chunks
from .enrich import screen as para_screen
from .jev import JevClient, JevConfig
from .jev.client import estimate_tokens
from .parse import load_file, iter_paths
from .retrieve import Answer, AnswerConfig, RetrievalResult, RetrieveConfig, generate, retrieve
from .stores import BASE_FILTER_FIELDS, VectorStore, make_store
from .tokens import TokenCounter, default_counter
from .types import Chunk, Document, Record


class ManifestMismatch(RuntimeError):
    """The collection was built with a different embedder / dimension / metric."""


@dataclass
class DocReport:
    doc_id: str
    status: str  # ingested | skipped_unchanged | failed | dry_run
    chunks: int = 0
    dropped: int = 0
    quarantined: int = 0
    stale_deleted: int = 0
    chunker: str = ""
    fallback: Optional[str] = None
    error: Optional[str] = None
    dropped_detail: List[Dict[str, Any]] = field(default_factory=list)
    screened_paragraphs: int = 0


@dataclass
class IngestReport:
    collection: str
    docs: List[DocReport] = field(default_factory=list)
    jev: Dict[str, Any] = field(default_factory=dict)
    embedding_tokens: int = 0
    seconds: float = 0.0
    dry_run: bool = False
    estimate: Dict[str, Any] = field(default_factory=dict)

    def count(self, status: str) -> int:
        return sum(1 for d in self.docs if d.status == status)

    def summary(self) -> str:
        chunks = sum(d.chunks for d in self.docs)
        dropped = sum(d.dropped for d in self.docs)
        quar = sum(d.quarantined for d in self.docs)
        fb = [d for d in self.docs if d.fallback]
        lines = [f"collection {self.collection}: {len(self.docs)} docs "
                 f"({self.count('ingested')} ingested, {self.count('skipped_unchanged')} unchanged, "
                 f"{self.count('failed')} failed{', dry run' if self.dry_run else ''})",
                 f"chunks {chunks} stored, {dropped} dropped, {quar} quarantined"]
        if self.jev:
            lines.append(f"Jev {self.jev.get('requests', 0)} requests ({self.jev.get('cached', 0)} cached), "
                         f"{self.jev.get('input_tokens', 0):,} tokens, ${self.jev.get('cost_usd', 0):.5f}")
        if self.embedding_tokens:
            lines.append(f"embedding {self.embedding_tokens:,} tokens")
        if self.estimate:
            e = self.estimate
            lines.append(f"estimate: {e['jev_requests']} Jev requests, ~{e['jev_tokens']:,} tokens, "
                         f"~${e['jev_cost_usd']:.5f} of Jev (chunking + enrichment)")
        if fb:
            lines.append(f"fallbacks: {len(fb)} docs used structural chunking ({(fb[0].fallback or '')[:120]})")
        failed = [d for d in self.docs if d.status == "failed"]
        for d in failed[:5]:
            lines.append(f"FAILED {d.doc_id}: {d.error}")
        lines.append(f"time {self.seconds:.1f}s")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"collection": self.collection, "dry_run": self.dry_run, "seconds": round(self.seconds, 2),
                "jev": self.jev, "embedding_tokens": self.embedding_tokens, "estimate": self.estimate,
                "docs": [d.__dict__ for d in self.docs]}


_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")


def _tidy(c: Chunk) -> Chunk:
    """Collapse the blank runs left where screened paragraphs were cut out of a chunk."""
    body = _BLANK_LINES.sub("\n\n", c.text).strip()
    if body == c.text:
        return c
    prefix = c.embed_text[: len(c.embed_text) - len(c.text)] if c.embed_text.endswith(c.text) else ""
    return dataclasses.replace(c, text=body, embed_text=prefix + body if prefix else body)


def _run(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("called a sync Pipeline method inside an event loop; use the a* async methods instead")


class Pipeline:
    def __init__(self, *, store: Union[VectorStore, Mapping[str, Any], None] = None,
                 embedder: Union[str, Embedder, Any] = None, jev: Optional[JevConfig] = None,
                 chunking: Optional[ChunkConfig] = None, enrich: Optional[EnrichConfig] = None,
                 retrieve: Optional[RetrieveConfig] = None, answer: Optional[AnswerConfig] = None,
                 token_counter: Optional[TokenCounter] = None, trace_dir: Optional[str] = ".jev-retrieval/traces",
                 embedder_options: Optional[Mapping[str, Any]] = None, jev_transport: Any = None) -> None:
        if embedder is None:
            raise ValueError("Pipeline needs an embedder, e.g. embedder='openai:text-embedding-3-small' "
                             "(or 'hash:256' for offline tests)")
        self.store = store if isinstance(store, VectorStore) else make_store(store or {"kind": "memory"})
        self.embedder = make_embedder(embedder, **dict(embedder_options or {}))
        self.jev_config = jev or JevConfig()
        self.chunking = chunking or ChunkConfig()
        self.enrich_cfg = enrich or EnrichConfig()
        self.retrieve_cfg = retrieve or RetrieveConfig()
        self.answer_cfg = answer or AnswerConfig()
        self.count = token_counter or default_counter()
        self.trace_dir = trace_dir
        self.jev_transport = jev_transport

    @classmethod
    def from_config(cls, path_or_config: Union[str, Config, None] = None, **overrides: Any) -> "Pipeline":
        cfg = path_or_config if isinstance(path_or_config, Config) else Config.load(path_or_config)
        kw: Dict[str, Any] = dict(store=cfg.store, embedder=cfg.embedder_spec, jev=cfg.jev, chunking=cfg.chunking,
                                  enrich=cfg.enrich, retrieve=cfg.retrieve, answer=cfg.answer,
                                  embedder_options=cfg.embedder_options)
        kw.update(overrides)
        return cls(**kw)

    # -- helpers -------------------------------------------------------------------

    def _jev(self) -> Optional[JevClient]:
        client = JevClient(self.jev_config, transport=self.jev_transport)
        return client if client.available() else None

    def _filter_fields(self) -> Dict[str, str]:
        f = dict(BASE_FILTER_FIELDS)
        for t in self.enrich_cfg.taxonomy.fields:
            f[f"tag_{t.name}"] = "keyword"
        return f

    def _manifest(self, dim: int) -> Dict[str, Any]:
        return {"embedder": self.embedder.spec, "dim": dim, "metric": self.embedder.metric,
                "chunker": self.chunking.method, "taxonomy_version": self.enrich_cfg.taxonomy.version,
                "pipeline_version": __version__, "created_at": dt.datetime.now(dt.timezone.utc).isoformat()}

    def _check_manifest(self, collection: str, dim: Optional[int], *, create: bool) -> None:
        m = self.store.get_manifest(collection)
        if m is None:
            if not create:
                raise ManifestMismatch(f"collection {collection!r} has no jev-retrieval manifest: ingest into it first, "
                                       "or query it in retrieve-only mode")
            assert dim is not None
            self.store.ensure_collection(collection, dim, self.embedder.metric, self._filter_fields())
            self.store.put_manifest(collection, self._manifest(dim))
            return
        if m.get("embedder") != self.embedder.spec:
            raise ManifestMismatch(f"collection {collection!r} was built with embedder {m.get('embedder')!r}; "
                                   f"this pipeline uses {self.embedder.spec!r}. Mixing embedding models silently "
                                   "ruins retrieval: use the same one, or re-embed into a new collection")
        if dim is not None and int(m.get("dim", dim)) != dim:
            raise ManifestMismatch(f"collection {collection!r} has dimension {m.get('dim')}, embedder gives {dim}")
        if dim is not None:
            self.store.ensure_collection(collection, int(m["dim"]), m.get("metric", "cosine"), self._filter_fields())

    def _record(self, c: Chunk, vector: List[float], doc: Document, jev_model: str, now: str) -> Record:
        meta: Dict[str, Any] = {
            "doc_id": c.doc_id, "source_uri": c.source_uri, "title": c.title,
            "section_path": " > ".join(c.section_path), "chunk_index": c.chunk_index,
            "char_start": c.char_start, "char_end": c.char_end, "tokens": c.tokens, "content_hash": c.content_hash,
            "doc_hash": _doc_hash(doc), "quarantined": c.quarantined, "chunker": c.chunker, "jev_model": jev_model,
            "pipeline_version": __version__, "ingested_at": now,
        }
        for k, v in c.answers.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
        for k, v in (doc.metadata or {}).items():
            if isinstance(v, (str, int, float, bool)) and k not in meta:
                meta[f"m_{k}"] = v
        return Record(c.id, vector, c.text, meta)

    def _write_trace(self, collection: str, doc_id: str, payload: Mapping[str, Any]) -> None:
        if not self.trace_dir:
            return
        d = Path(os.path.expanduser(self.trace_dir)) / "ingest" / collection
        d.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in doc_id)[:150]
        (d / f"{safe}.json").write_text(json.dumps(payload, indent=2, default=str))

    # -- ingest --------------------------------------------------------------------

    def _documents(self, sources: Iterable[Union[str, Document]]) -> List[Document]:
        docs: List[Document] = []
        paths: List[str] = []
        for s in sources:
            if isinstance(s, Document):
                docs.append(s)
            else:
                paths.append(str(s))
        for f, root in iter_paths(paths):
            docs.append(load_file(f, root=root))
        return docs

    async def aingest(self, sources: Iterable[Union[str, Document]], collection: str, *, dry_run: bool = False,
                      limit: Optional[int] = None, force: bool = False, concurrency: int = 4) -> IngestReport:
        t0 = time.monotonic()
        docs = self._documents(sources)
        if limit is not None:
            docs = docs[:limit]
        report = IngestReport(collection=collection, dry_run=dry_run)
        if dry_run:
            reqs = toks = 0
            uses_jev_chunking = self.chunking.method == "jev" and self.chunking.mode != "off"
            uses_enrich = self.enrich_cfg.mode != "off"
            uses_screen = self.enrich_cfg.screen_paragraphs != "off" and uses_enrich
            for d in docs:
                if uses_screen:
                    r, t = para_screen.estimate(d, self.enrich_cfg.screen_batch)
                    reqs += r
                    toks += t
                n_chunks = max(1, len(d.text) // max(1, self.chunking.target_tokens * 4))
                if uses_jev_chunking:
                    r, t = estimate_jev_requests(d, self.chunking, self.count)
                    reqs += r
                    toks += t
                if uses_enrich:
                    reqs += n_chunks
                    toks += n_chunks * (self.chunking.target_tokens + 400)
                report.docs.append(DocReport(d.resolved_id(), "dry_run"))
            report.estimate = {"jev_requests": reqs, "jev_tokens": toks,
                               "jev_cost_usd": round(toks * self.jev_config.price_per_million / 1e6, 6)}
            report.seconds = time.monotonic() - t0
            return report
        jev = self._jev()
        try:
            if jev is not None:
                await jev.__aenter__()
            sem = asyncio.Semaphore(max(1, concurrency))
            manifest_lock = asyncio.Lock()
            ready = {"ok": False}

            async def one(doc: Document) -> DocReport:
                async with sem:
                    return await self._ingest_one(doc, collection, jev, force, manifest_lock, ready)

            report.docs = list(await asyncio.gather(*(one(d) for d in docs)))
        finally:
            if jev is not None:
                report.jev = jev.usage.to_dict(self.jev_config.price_per_million)
                await jev.__aexit__(None, None, None)
        report.embedding_tokens = self.embedder.tokens_used
        report.seconds = time.monotonic() - t0
        return report

    async def _ingest_one(self, doc: Document, collection: str, jev: Optional[JevClient], force: bool,
                          manifest_lock: asyncio.Lock, ready: Dict[str, bool]) -> DocReport:
        did = doc.resolved_id()
        rep = DocReport(did, "ingested")
        try:
            # skip unchanged (FR-I5)
            if not force and self.store.get_manifest(collection) is not None:
                existing = await asyncio.to_thread(self.store.list_ids, collection, {"eq": {"doc_id": did}}, 100_000)
                if existing:
                    sample = await asyncio.to_thread(self.store.get, collection, existing[:1])
                    if sample and sample[0].metadata.get("doc_hash") == _doc_hash(doc):
                        rep.status = "skipped_unchanged"
                        rep.chunks = len(existing)
                        return rep
            sres = None
            work = doc
            ec = self.enrich_cfg
            if jev is not None and ec.screen_paragraphs != "off" and ec.mode != "off":
                sres = await para_screen.screen_document(
                    jev, doc, mode="shadow" if ec.mode == "shadow" else ec.screen_paragraphs,
                    drop_boilerplate=ec.drop_boilerplate, quarantine_instructs_ai=ec.quarantine_instructs_ai,
                    batch=ec.screen_batch)
                rep.screened_paragraphs = sres.screened
                if sres.text != doc.text:
                    work = dataclasses.replace(doc, text=sres.text)
            chunks, ctrace = await chunk_document(work, self.chunking, jev=jev, embedder=self.embedder,
                                                  count=self.count)
            if work is not doc:
                chunks = [_tidy(c) for c in chunks]
            screened_out = []
            if sres is not None:
                for b, p in sres.dropped:
                    rep.dropped_detail.append({"chunk_index": None, "reason": "boilerplate", "level": "paragraph",
                                               "score": round(p, 4), "text": doc.text[b.start:b.end][:160]})
                for k, (b, p) in enumerate(sres.quarantined):
                    body = doc.text[b.start:b.end]
                    screened_out.append(Chunk(
                        text=body, doc_id=did, chunk_index=len(chunks) + k, char_start=b.start, char_end=b.end,
                        section_path=list(b.section_path), chunker="screen", tokens=self.count(body),
                        title=doc.title, source_uri=doc.source_uri, embed_text=body,
                        answers={"q_instructs_ai": round(p, 4)}, quarantined=True))
            rep.chunker = ctrace.method_used
            rep.fallback = ctrace.fallback_reason if ctrace.method_requested == "jev" and \
                ctrace.method_used != "jev" and self.chunking.mode != "off" else None
            # exact duplicates within the document (FR-E5)
            seen, unique = set(), []
            for c in chunks:
                if c.content_hash in seen:
                    rep.dropped_detail.append({"chunk_index": c.chunk_index, "reason": "duplicate"})
                    continue
                seen.add(c.content_hash)
                unique.append(c)
            er = await enrich_chunks(jev, unique, self.enrich_cfg)
            n_para_dropped = len(sres.dropped) if sres is not None else 0
            rep.dropped = len(er.dropped) + (len(chunks) - len(unique)) + n_para_dropped
            rep.quarantined = len(er.quarantined) + len(screened_out)
            rep.dropped_detail += [{"chunk_index": c.chunk_index, "reason": c.dropped_reason, "level": "chunk",
                                    "text": c.text[:160]} for c in er.dropped]
            stored = er.kept + er.quarantined + screened_out
            stored.sort(key=lambda c: c.chunk_index)
            vectors = await self.embedder.embed_documents([c.embed_text or c.text for c in stored]) if stored else []
            async with manifest_lock:
                if not ready["ok"]:
                    dim = len(vectors[0]) if vectors else (self.embedder.dim or 0)
                    if not dim:
                        dim = len(await self.embedder.embed_query("dimension probe"))
                    await asyncio.to_thread(self._check_manifest, collection, dim, create=True)
                    ready["ok"] = True
            now = dt.datetime.now(dt.timezone.utc).isoformat()
            model = jev.model if jev is not None else ""
            records = [self._record(c, v, doc, model, now) for c, v in zip(stored, vectors)]
            batch = self.store.capabilities().max_batch
            for i in range(0, len(records), batch):
                await asyncio.to_thread(self.store.upsert, collection, records[i:i + batch])
            # stale delete (FR-I2): this doc's records that are no longer present
            keep = {r.id for r in records}
            existing = await asyncio.to_thread(self.store.list_ids, collection, {"eq": {"doc_id": did}}, 100_000)
            stale = [i for i in existing if i not in keep]
            if stale:
                await asyncio.to_thread(self.store.delete, collection, stale)
            rep.stale_deleted = len(stale)
            rep.chunks = len(records)
            self._write_trace(collection, did, {"doc_id": did, "chunking": ctrace.to_dict(),
                                                "dropped": rep.dropped_detail,
                                                "quarantined": [c.id for c in er.quarantined + screened_out],
                                                "enrich_failed": er.failed, "shadow_actions": er.shadow_actions,
                                                "screen": None if sres is None else {
                                                    "paragraphs": sres.screened, "requests": sres.requests,
                                                    "failed_batches": sres.failed_batches,
                                                    "quarantined": [doc.text[b.start:b.end][:160]
                                                                    for b, _ in sres.quarantined],
                                                    "shadow": sres.shadow}})
        except Exception as exc:  # one bad document never aborts the batch
            if isinstance(exc, ManifestMismatch):
                raise
            rep.status = "failed"
            rep.error = f"{type(exc).__name__}: {exc}"
        return rep

    def ingest(self, sources: Iterable[Union[str, Document]], collection: str, **kw: Any) -> IngestReport:
        return _run(self.aingest(sources, collection, **kw))

    # -- chunk only (for inspect) ---------------------------------------------------------

    async def achunk(self, doc: Union[str, Document], method: Optional[str] = None) -> Any:
        d = doc if isinstance(doc, Document) else load_file(str(doc))
        cfg = self.chunking if method is None else ChunkConfig(**{**self.chunking.__dict__, "method": method})
        jev = self._jev()
        if jev is None:
            return await chunk_document(d, cfg, jev=None, embedder=self.embedder, count=self.count)
        async with jev:
            chunks, trace = await chunk_document(d, cfg, jev=jev, embedder=self.embedder, count=self.count)
            trace.jev_usage = jev.usage.to_dict()  # type: ignore[attr-defined]
            return chunks, trace

    # -- retrieve --------------------------------------------------------------------

    async def aretrieve(self, query: str, collection: str, *, where: Optional[Mapping[str, Any]] = None,
                        retrieve_only: bool = False, text_field: Optional[str] = None) -> RetrievalResult:
        if not retrieve_only:
            await asyncio.to_thread(self._check_manifest, collection, None, create=False)
        jev = self._jev()
        if jev is None:
            res = await retrieve(query=query, jev=None, store=self.store, collection=collection,
                                 embedder=self.embedder, taxonomy=self.enrich_cfg.taxonomy, cfg=self.retrieve_cfg,
                                 where=where, route_enabled=not retrieve_only)
            res.degraded = True
            res.trace["jev"] = "no Jev key: vector ranking only"
            return res
        async with jev:
            res = await retrieve(query=query, jev=jev, store=self.store, collection=collection,
                                 embedder=self.embedder, taxonomy=self.enrich_cfg.taxonomy, cfg=self.retrieve_cfg,
                                 where=where, route_enabled=not retrieve_only)
            res.trace["jev"] = jev.usage.to_dict(self.jev_config.price_per_million)
            res.trace["jev_model"] = jev.model
        return res

    def retrieve(self, query: str, collection: str, **kw: Any) -> RetrievalResult:
        return _run(self.aretrieve(query, collection, **kw))

    async def aanswer(self, query: str, collection: str, **kw: Any) -> Answer:
        res = await self.aretrieve(query, collection, **kw)
        return await generate(res, self.answer_cfg)

    def answer(self, query: str, collection: str, **kw: Any) -> Answer:
        return _run(self.aanswer(query, collection, **kw))

    # -- maintenance -------------------------------------------------------------------

    def delete_document(self, collection: str, doc_id: str) -> int:
        ids = self.store.list_ids(collection, {"eq": {"doc_id": doc_id}})
        if ids:
            self.store.delete(collection, ids=ids)
        return len(ids)

    def close(self) -> None:
        try:
            _run(self.embedder.aclose())
        except RuntimeError:
            pass
        self.store.close()


def _doc_hash(doc: Document) -> str:
    from .types import normalise_text, sha256
    return sha256(normalise_text(doc.text) + "\x1f" + doc.title)[:32]


__all__ = ["Pipeline", "IngestReport", "DocReport", "ManifestMismatch", "estimate_tokens"]
