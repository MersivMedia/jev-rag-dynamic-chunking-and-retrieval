"""Chunk a document: parse -> units -> gap costs -> DP segmenter -> chunks.

Methods (FR-C7):

* ``jev``                - Jev boundary questions set the gap costs (default)
* ``structural``         - code-only costs from headings and paragraph breaks
* ``fixed``              - windows of ``target_tokens`` with ``overlap_tokens``, sentence-aligned
* ``semantic-embedding`` - cut where adjacent sentence embeddings diverge

If Jev fails, ``jev`` falls back to ``structural`` and records it (NFR-2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..jev import JevClient, JevError
from ..parse import build_units, parse_blocks
from ..tokens import TokenCounter, default_counter
from ..types import Chunk, Document, Unit
from .boundaries import BoundaryConfig, jev_boundaries, plan_requests
from .segmenter import Gap, SegmenterConfig, segment, structural_costs

METHODS = ("jev", "structural", "fixed", "semantic-embedding")


@dataclass
class ChunkConfig:
    # structural is the default: in every benchmark so far (docs/RESULTS.md) Jev chunking tied or lost to
    # structural once paragraph screening ran, at about twice the ingest cost. "jev" stays available.
    method: str = "structural"
    mode: str = "on"  # on | shadow | off  (applies to the Jev method)
    min_tokens: int = 64
    target_tokens: int = 350
    max_tokens: int = 800
    overlap_tokens: int = 0
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)

    def segmenter(self) -> SegmenterConfig:
        return SegmenterConfig(self.min_tokens, self.target_tokens, self.max_tokens)


@dataclass
class ChunkTrace:
    method_requested: str
    method_used: str
    units: int = 0
    jev_requests: int = 0
    fallback_reason: Optional[str] = None
    gaps: List[Dict[str, Any]] = field(default_factory=list)  # per-gap scores, for `inspect`
    shadow_cuts: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [], {})}


def _fixed(units: Sequence[Unit], cfg: ChunkConfig) -> List[Tuple[int, int]]:
    """Sentence-aligned windows of about target_tokens; never crosses a section (headings are hard)."""
    out: List[Tuple[int, int]] = []
    i, n = 0, len(units)
    while i < n:
        j, tok = i, 0
        while j < n and (tok + units[j].tokens <= cfg.target_tokens or j == i) \
                and (j == i or units[j].section == units[i].section):
            tok += units[j].tokens
            j += 1
        out.append((i, j))
        if j >= n:
            break
        if not cfg.overlap_tokens or units[j].section != units[j - 1].section:
            i = j
            continue
        back, k = 0, j
        while k - 1 > i and back + units[k - 1].tokens <= cfg.overlap_tokens:
            k -= 1
            back += units[k].tokens
        i = k
    return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _with_overlap(ranges: List[Tuple[int, int]], units: Sequence[Unit], overlap: int) -> List[Tuple[int, int]]:
    if overlap <= 0:
        return ranges
    out = []
    for k, (s, e) in enumerate(ranges):
        if k and units[s].section == units[s - 1].section:
            back, ns = 0, s
            while ns - 1 >= ranges[k - 1][0] and back + units[ns - 1].tokens <= overlap:
                ns -= 1
                back += units[ns].tokens
            s = ns
        out.append((s, e))
    return out


def _merge_small(ranges: List[Tuple[int, int]], units: Sequence[Unit], gaps: Dict[int, Gap],
                 cfg: ChunkConfig) -> List[Tuple[int, int]]:
    """FR-C4: merge a chunk under min_tokens into its more continuous neighbour in the same section."""
    def tok(r: Tuple[int, int]) -> int:
        return sum(u.tokens for u in units[r[0]:r[1]])

    rs = list(ranges)
    changed = True
    while changed:
        changed = False
        for k, r in enumerate(rs):
            if tok(r) >= cfg.min_tokens:
                continue
            left = k > 0 and units[rs[k - 1][0]].section == units[r[0]].section
            right = k + 1 < len(rs) and units[rs[k + 1][0]].section == units[r[0]].section
            opts = []
            if left and tok(rs[k - 1]) + tok(r) <= cfg.max_tokens:
                g = gaps.get(r[0])
                opts.append(((g.cost if g else 0.5), k - 1))
            if right and tok(rs[k + 1]) + tok(r) <= cfg.max_tokens:
                g = gaps.get(rs[k + 1][0])
                opts.append(((g.cost if g else 0.5), k + 1))
            if not opts:
                continue
            _, other = max(opts)  # higher cut cost = more continuous = better merge partner
            a, b = sorted((k, other))
            rs[a:b + 1] = [(rs[a][0], rs[b][1])]
            changed = True
            break
    return rs


def _to_chunks(doc: Document, text: str, units: Sequence[Unit], ranges: List[Tuple[int, int]],
               blocks_paths: List[List[str]], chunker: str, count: TokenCounter) -> List[Chunk]:
    did = doc.resolved_id()
    chunks = []
    for idx, (s, e) in enumerate(ranges):
        start, end = units[s].start, units[e - 1].end
        body = text[start:end]
        if not any(ch.isalnum() for ch in body):
            continue  # only markup (bullets, pipes, rules): nothing to retrieve
        idx = len(chunks)
        # a chunk that opens with stacked headings belongs to the deepest one: use the first content unit
        first_content = next((u for u in units[s:e] if u.kind != "heading"), units[e - 1])
        path = blocks_paths[first_content.block]
        parts = list(path)
        if doc.title and (not parts or parts[0].strip().lower() != doc.title.strip().lower()):
            parts.insert(0, doc.title)
        prefix = " > ".join(p for p in parts if p)
        chunks.append(Chunk(
            text=body, doc_id=did, chunk_index=idx, char_start=start, char_end=end, section_path=path,
            chunker=chunker, tokens=count(body), title=doc.title, source_uri=doc.source_uri,
            embed_text=f"{prefix}\n\n{body}" if prefix else body,
        ))
    return chunks


async def chunk_document(doc: Document, cfg: ChunkConfig, *, jev: Optional[JevClient] = None, embedder: Any = None,
                         count: Optional[TokenCounter] = None) -> Tuple[List[Chunk], ChunkTrace]:
    if cfg.method not in METHODS:
        raise ValueError(f"unknown chunking method {cfg.method!r}; choose from {METHODS}")
    count = count or default_counter()
    text = doc.text
    blocks = parse_blocks(text, doc.format)
    units = build_units(text, blocks, cfg.max_tokens, count)
    trace = ChunkTrace(method_requested=cfg.method, method_used=cfg.method, units=len(units))
    if not units:
        return [], trace
    paths = [b.section_path for b in blocks]
    seg = cfg.segmenter()

    if cfg.method == "fixed":
        ranges = _fixed(units, cfg)
        return _to_chunks(doc, text, units, ranges, paths, "fixed", count), trace

    gaps: List[Gap]
    used = cfg.method
    if cfg.method == "semantic-embedding":
        if embedder is None:
            raise ValueError("semantic-embedding chunking needs an embedder")
        vecs = await embedder.embed_documents([text[u.start:u.end] for u in units])
        gaps = []
        for i in range(1, len(units)):
            if units[i - 1].kind == "heading":
                gaps.append(Gap(i, 1.0, protected=True))
            else:
                sim = _cosine(vecs[i - 1], vecs[i])
                gaps.append(Gap(i, max(0.0, min(1.0, sim)) * (0.7 if units[i].first_in_block else 1.0)))
    elif cfg.method == "jev" and cfg.mode in ("on", "shadow") and jev is not None:
        try:
            res = await jev_boundaries(jev, text, units, cfg.boundary)
            trace.jev_requests = res.requests
            jev_gaps = res.gaps
            trace.gaps = [{"gap": g.index, "cost": round(g.cost, 3), "protected": g.protected,
                           **{k: round(v, 3) for k, v in res.answers.get(g.index, {}).items()}} for g in jev_gaps]
        except JevError as exc:
            jev_gaps = None
            trace.fallback_reason = str(exc)
        if jev_gaps is None:
            gaps, used = structural_costs(units), "structural"
        elif cfg.mode == "shadow":
            gaps, used = structural_costs(units), "structural"
            gmap = {g.index: g for g in jev_gaps}
            trace.shadow_cuts = [s for s, _ in _merge_small(segment(units, jev_gaps, seg), units, gmap, cfg)][1:]
        else:
            gaps = jev_gaps
    else:
        if cfg.method == "jev":
            trace.fallback_reason = "jev mode off" if cfg.mode == "off" else "no Jev client"
        gaps, used = structural_costs(units), "structural"
    trace.method_used = used
    gmap = {g.index: g for g in gaps}
    ranges = _merge_small(segment(units, gaps, seg), units, gmap, cfg)
    ranges = _with_overlap(ranges, units, cfg.overlap_tokens)
    return _to_chunks(doc, text, units, ranges, paths, used, count), trace


def estimate_jev_requests(doc: Document, cfg: ChunkConfig, count: Optional[TokenCounter] = None) -> Tuple[int, int]:
    """(requests, estimated input tokens) Jev chunking would use for ``doc``. No network."""
    from ..jev import estimate_tokens
    count = count or default_counter()
    blocks = parse_blocks(doc.text, doc.format)
    units = build_units(doc.text, blocks, cfg.max_tokens, count)
    plans = plan_requests(doc.text, units, cfg.boundary)
    return len(plans), sum(estimate_tokens(s) + estimate_tokens({k: q.to_wire() for k, q in qs.items()})
                           for s, qs, _ in plans)
