import asyncio

from jevrag.chunk import ChunkConfig, chunk_document
from jevrag.parse import html_to_markdown
from jevrag.tokens import estimate_tokens
from jevrag.types import Document

PAGE = """<html><title>T</title><body>
<p>Intro paragraph with words.</p>
<ul><li></li><li> </li><li><a href="#"></a></li></ul>
<table><tr><th>Name</th><th>Year</th></tr><tr><td></td><td></td></tr><tr><td>Bebop</td><td>1940s</td></tr></table>
<ul><li>Real item</li><li></li></ul>
</body></html>"""


def test_html_drops_empty_list_items_and_rows_but_keeps_tables():
    _, md = html_to_markdown(PAGE)
    assert "|---|---|" in md and "| Bebop | 1940s |" in md and "- Real item" in md
    assert "\n-\n" not in md and not md.rstrip().endswith("-")
    assert "|  |  |" not in md


def test_markup_only_chunks_are_not_emitted():
    doc = Document(text="Real sentence here.\n\n-\n\n---\n\n| |\n\nAnother real sentence.", doc_id="d")
    chunks, _ = asyncio.run(chunk_document(doc, ChunkConfig(method="fixed", min_tokens=1, target_tokens=2,
                                                            max_tokens=50), count=estimate_tokens))
    assert chunks and all(any(c.isalnum() for c in ch.text) for ch in chunks)
    assert [ch.chunk_index for ch in chunks] == list(range(len(chunks)))
