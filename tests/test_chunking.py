import asyncio

import pytest

from jev_retrieval.chunk import ChunkConfig, chunk_document
from jev_retrieval.chunk.boundaries import BoundaryConfig, candidate_gaps, plan_requests
from jev_retrieval.chunk.segmenter import Gap, SegmenterConfig, segment
from jev_retrieval.jev import JevClient, JevConfig
from jev_retrieval.parse import build_units, html_to_markdown, parse_blocks, split_sentences
from jev_retrieval.tokens import estimate_tokens
from jev_retrieval.types import Document

DOC = """# Guide

## Tokens

Refresh tokens expire after 14 days. They can be renewed once. Access tokens last one hour.

Invoices are issued monthly. A late payment fee of 2% applies to every invoice.

| Token | Lifetime |
|---|---|
| refresh | 14 days |

## Office

Our office is in Oakland.
"""


def test_sentences_keep_offsets_and_abbreviations():
    s = "Dr. Smith paid $3.50 on Jan. 4 at 3 p.m. It worked, e.g. on v1.2.3! Did it? Yes."
    spans = split_sentences(s)
    got = [s[a:b] for a, b in spans]
    assert got == ["Dr. Smith paid $3.50 on Jan. 4 at 3 p.m. It worked, e.g. on v1.2.3!", "Did it?", "Yes."] or \
        got == ["Dr. Smith paid $3.50 on Jan. 4 at 3 p.m.", "It worked, e.g. on v1.2.3!", "Did it?", "Yes."]
    for a, b in spans:
        assert s[a:b].strip() == s[a:b]


def test_blocks_and_section_paths():
    blocks = parse_blocks(DOC)
    kinds = [b.kind for b in blocks]
    assert kinds.count("heading") == 3 and "table" in kinds
    table = next(b for b in blocks if b.kind == "table")
    assert table.section_path == ["Guide", "Tokens"]
    office = [b for b in blocks if b.kind == "paragraph"][-1]
    assert office.section_path == ["Guide", "Office"]


def test_html_loader_strips_chrome():
    title, md = html_to_markdown("<html><head><title>T</title><script>x()</script></head><body><nav>Home</nav>"
                                 "<h2>Sub</h2><p>Body text.</p><table><tr><th>a</th><th>b</th></tr>"
                                 "<tr><td>1</td><td>2</td></tr></table><footer>(c)</footer></body></html>")
    assert title == "T"
    assert "## Sub" in md and "Body text." in md and "| a | b |" in md and "|---|---|" in md
    assert "Home" not in md and "x()" not in md and "(c)" not in md


def test_segmenter_respects_limits_and_sections():
    units = build_units(DOC, parse_blocks(DOC), 60, estimate_tokens)
    gaps = [Gap(i, 0.5) for i in range(1, len(units))]
    ranges = segment(units, gaps, SegmenterConfig(min_tokens=1, target_tokens=15, max_tokens=60))
    for s, e in ranges:
        assert len({units[i].section for i in range(s, e) if units[i].kind != "heading"}) <= 1
        assert sum(u.tokens for u in units[s:e]) + (e - s) <= 60 + 60  # joins counted loosely
    covered = [i for s, e in ranges for i in range(s, e)]
    assert covered == list(range(len(units)))  # full, ordered, no overlap


def test_segmenter_cuts_at_cheapest_gap():
    units = build_units("A one. A two. B three. B four.", parse_blocks("A one. A two. B three. B four.", "text"),
                        100, estimate_tokens)
    gaps = [Gap(1, 0.9), Gap(2, 0.05), Gap(3, 0.9)]
    ranges = segment(units, gaps, SegmenterConfig(min_tokens=1, target_tokens=6, max_tokens=100))
    assert ranges == [(0, 2), (2, 4)]


def test_protected_gap_is_not_cut():
    text = "Alpha one. This matters. Beta two. Beta three."
    units = build_units(text, parse_blocks(text, "text"), 100, estimate_tokens)
    gaps = [Gap(1, 0.0, protected=True), Gap(2, 0.3), Gap(3, 0.9)]
    ranges = segment(units, gaps, SegmenterConfig(min_tokens=1, target_tokens=5, max_tokens=100))
    assert (0, 1) not in ranges


def test_plan_skips_code_decided_gaps_and_uses_inline_pairs():
    units = build_units(DOC, parse_blocks(DOC), 800, estimate_tokens)
    gaps = candidate_gaps(units)
    for g in gaps:
        assert units[g - 1].kind != "heading" and units[g].section == units[g - 1].section
    plans = plan_requests(DOC, units, BoundaryConfig())
    qs = plans[0][1]
    assert set(k[0] for k in qs) == {"c", "r"}
    one = next(iter(qs.values())).to_wire()["instructions"]
    assert set(one) == {"previous", "sentence", "question"}
    keyed = plan_requests(DOC, units, BoundaryConfig(style="keyed"))
    assert isinstance(keyed[0][0], dict) and all(k.startswith("s") for k in keyed[0][0])


def test_structural_and_fixed_never_cross_headings():
    doc = Document(text=DOC, doc_id="g", title="Guide")
    for method in ("structural", "fixed"):
        chunks, tr = asyncio.run(chunk_document(doc, ChunkConfig(method=method, min_tokens=1, target_tokens=20,
                                                                 max_tokens=80), count=estimate_tokens))
        assert tr.method_used == method
        for c in chunks:
            body = c.text.split("\n")
            headings_mid = [ln for ln in body[1:] if ln.startswith("## ") and ln != body[0]]
            content_before = any(ln.strip() and not ln.startswith("#") for ln in body[:body.index(headings_mid[0])]) \
                if headings_mid else False
            assert not content_before, f"{method} chunk crosses a heading: {c.text!r}"


def test_jev_chunking_cuts_at_topic_change(fake_jev):
    fj, transport = fake_jev
    doc = Document(text=DOC, doc_id="g", title="Guide")

    async def go():
        async with JevClient(JevConfig(cache_dir=None), transport=transport) as jev:
            return await chunk_document(doc, ChunkConfig(method="jev", min_tokens=1, target_tokens=30, max_tokens=120), jev=jev,
                                        count=estimate_tokens)

    chunks, tr = asyncio.run(go())
    assert tr.method_used == "jev" and tr.jev_requests == 1
    tokens_chunk = next(c for c in chunks if "Refresh tokens" in c.text)
    assert "Invoices" not in tokens_chunk.text  # the cut lands at the topic change
    assert "They can be renewed once" in tokens_chunk.text  # "They ..." is never orphaned
    assert all(c.id and c.embed_text.startswith("Guide >") for c in chunks)


def test_jev_failure_falls_back_to_structural(fake_jev):
    fj, transport = fake_jev
    fj.fail_status, fj.fail_times = 401, 99
    doc = Document(text=DOC, doc_id="g")

    async def go():
        async with JevClient(JevConfig(cache_dir=None, max_retries=0), transport=transport) as jev:
            return await chunk_document(doc, ChunkConfig(method="jev"), jev=jev, count=estimate_tokens)

    chunks, tr = asyncio.run(go())
    assert tr.method_used == "structural" and "401" in (tr.fallback_reason or "")
    assert chunks and all(c.chunker == "structural" for c in chunks)


def test_ids_are_deterministic():
    doc = Document(text=DOC, doc_id="g")
    a, _ = asyncio.run(chunk_document(doc, ChunkConfig(method="structural"), count=estimate_tokens))
    b, _ = asyncio.run(chunk_document(doc, ChunkConfig(method="structural"), count=estimate_tokens))
    assert [c.id for c in a] == [c.id for c in b]


def test_oversize_sentence_is_split():
    text = "word " * 500 + "end."
    units = build_units(text, parse_blocks(text, "text"), 50, estimate_tokens)
    assert len(units) > 5 and max(u.tokens for u in units) <= 50


def test_unknown_method():
    with pytest.raises(ValueError):
        asyncio.run(chunk_document(Document(text="x"), ChunkConfig(method="nope")))


def test_default_chunker_is_structural_and_makes_no_jev_calls(fake_jev):
    """Structural is the default (docs/RESULTS.md: Jev chunking tied or lost on every benchmark)."""
    fj, transport = fake_jev
    assert ChunkConfig().method == "structural"
    doc = Document(text=DOC, doc_id="g", title="Guide")

    async def go():
        async with JevClient(JevConfig(cache_dir=None), transport=transport) as jev:
            return await chunk_document(doc, ChunkConfig(), jev=jev, count=estimate_tokens)

    chunks, tr = asyncio.run(go())
    assert chunks and tr.method_used == "structural" and tr.jev_requests == 0 and not tr.fallback_reason
