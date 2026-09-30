"""File loaders. Each returns a :class:`~jevrag.types.Document` in Markdown form.

* ``.md`` / ``.markdown`` / ``.txt`` / ``.rst``: read as is
* ``.html`` / ``.htm``: converted with the stdlib HTML parser (scripts, styles,
  nav, header, footer and aside removed)
* ``.pdf``: text layer via PyMuPDF (``pip install ...[pdf]``); scanned pages need OCR first
* ``.docx``: paragraphs and heading styles via python-docx (``...[docx]``)
"""

from __future__ import annotations

import html
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from ..types import Document

SUPPORTED = {".md", ".markdown", ".txt", ".rst", ".html", ".htm", ".pdf", ".docx"}


class _HTMLToMarkdown(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "aside", "noscript", "svg", "form", "button"}
    BLOCK = {"p", "div", "section", "article", "main", "br", "tr", "table", "ul", "ol", "blockquote", "dd", "dt"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: List[str] = []
        self.skip = 0
        self.pre = 0
        self.title = ""
        self._in_title = False
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None
        self._rows_in_table = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        if tag == "title":
            self._in_title = True
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag == "pre":
            self.pre += 1
            self.out.append("\n\n```\n")
        elif tag == "table":
            self._rows_in_table = 0
            self.out.append("\n\n")
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
        elif tag == "blockquote":
            self.out.append("\n\n> ")
        elif tag in self.BLOCK:
            self.out.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag == "title":
            self._in_title = False
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n\n")
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
            self.out.append("\n```\n\n")
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()).replace("|", "\\|"))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.out.append("| " + " | ".join(self._row) + " |\n")
                if self._rows_in_table == 0:
                    self.out.append("|" + "---|" * len(self._row) + "\n")
                self._rows_in_table += 1
            self._row = None
        elif tag == "table":
            self.out.append("\n")
        elif tag in self.BLOCK:
            self.out.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self.skip:
            return
        if self._cell is not None:
            self._cell.append(data)
            return
        if self.pre:
            self.out.append(data)
        else:
            self.out.append(re.sub(r"\s+", " ", data))


def html_to_markdown(source: str) -> tuple:
    p = _HTMLToMarkdown()
    p.feed(source)
    p.close()
    text = "".join(p.out)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return html.unescape(p.title.strip()), text.strip()


def _first_heading(text: str) -> str:
    for line in text.split("\n"):
        m = re.match(r"^#{1,2}\s+(.*)", line)
        if m:
            return m.group(1).strip()
    return ""


def load_file(path: str, doc_id: Optional[str] = None, root: Optional[str] = None) -> Document:
    p = Path(path)
    ext = p.suffix.lower()
    rel = os.path.relpath(p, root) if root else str(p)
    did = doc_id or rel.replace(os.sep, "/")
    uri = p.resolve().as_uri()
    if ext in (".md", ".markdown", ".rst"):
        text = p.read_text(encoding="utf-8", errors="replace")
        return Document(text=text, doc_id=did, title=_first_heading(text) or p.stem, source_uri=uri, format="markdown")
    if ext == ".txt":
        text = p.read_text(encoding="utf-8", errors="replace")
        return Document(text=text, doc_id=did, title=p.stem, source_uri=uri, format="text")
    if ext in (".html", ".htm"):
        title, text = html_to_markdown(p.read_text(encoding="utf-8", errors="replace"))
        return Document(text=text, doc_id=did, title=title or _first_heading(text) or p.stem, source_uri=uri,
                        format="markdown")
    if ext == ".pdf":
        try:
            import pymupdf  # type: ignore
        except ImportError as exc:
            raise ImportError('PDF support needs the pdf extra: pip install "jev-rag-dynamic-chunking-and-retrieval[pdf]"') from exc
        with pymupdf.open(str(p)) as doc:
            pages = [page.get_text("text") for page in doc]
            title = (doc.metadata or {}).get("title") or p.stem
        text = "\n\n".join(pg.strip() for pg in pages if pg.strip())
        return Document(text=text, doc_id=did, title=title, source_uri=uri, format="text")
    if ext == ".docx":
        try:
            import docx  # type: ignore
        except ImportError as exc:
            raise ImportError('DOCX support needs the docx extra: pip install "jev-rag-dynamic-chunking-and-retrieval[docx]"') from exc
        d = docx.Document(str(p))
        parts: List[str] = []
        for para in d.paragraphs:
            t = para.text.strip()
            if not t:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            m = re.match(r"Heading (\d)", style)
            if m:
                parts.append("#" * min(6, int(m.group(1))) + " " + t)
            elif style.lower().startswith("list"):
                parts.append("- " + t)
            else:
                parts.append(t)
        for table in d.tables:
            rows = [[c.text.strip().replace("|", "\\|") for c in row.cells] for row in table.rows]
            if rows:
                parts.append("\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
                                       + ["| " + " | ".join(r) + " |" for r in rows[1:]]))
        text = "\n\n".join(parts)
        return Document(text=text, doc_id=did, title=d.core_properties.title or _first_heading(text) or p.stem,
                        source_uri=uri, format="markdown")
    raise ValueError(f"unsupported file type {ext!r} ({path}); supported: {sorted(SUPPORTED)}")


def iter_paths(paths: Iterable[str]) -> Iterator[tuple]:
    """Yield (file, root) for files and for supported files under directories, sorted."""
    for raw in paths:
        p = Path(os.path.expanduser(str(raw)))
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED and not any(
                        part.startswith(".") for part in f.relative_to(p).parts):
                    yield str(f), str(p)
        elif p.is_file():
            yield str(p), str(p.parent)
        else:
            raise FileNotFoundError(str(p))


def load_paths(paths: Iterable[str]) -> List[Document]:
    return [load_file(f, root=root) for f, root in iter_paths(paths)]
