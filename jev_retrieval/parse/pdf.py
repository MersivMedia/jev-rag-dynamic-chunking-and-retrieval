"""PDF to Markdown with PyMuPDF.

Plain ``page.get_text()`` returns every page as one run of hard-wrapped lines:
words hyphenated across lines, running headers and page numbers mixed into the
prose, and no headings. That hurts both chunking and embedding, so this loader:

* reads text blocks (PyMuPDF groups lines into paragraph-like blocks) and joins
  each block's lines into one paragraph, with de-hyphenation;
* drops page numbers and running headers/footers: blocks in the top or bottom
  8% of the page whose text repeats (digits masked) on at least 40% of pages of
  a 3+ page document;
* marks headings: short blocks that are all bold, or set noticeably larger
  than the body font, become ``#``/``##``/``###`` by relative size.

Scanned PDFs have no text layer; they come out empty (OCR them first).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Tuple

_WS = re.compile(r"[ \t\u00a0]+")
_DIGITS = re.compile(r"\d+")
_CAPTION = re.compile(r"^(figure|fig\.|table|algorithm|listing)\s*\d", re.I)
_WATERMARK = re.compile(r"^arXiv:\d{4}\.\d{4,5}(v\d+)?\b")
_SOFT_HYPHEN = re.compile(r"(?<=[a-z])- (?=[a-z])")


Block = Tuple[str, float, bool, int, bool]  # text, font size, bold, chars, in page margin


def _block_text(block: Dict[str, Any], page_h: float) -> Block:
    lines: List[str] = []
    sizes: Counter = Counter()
    bold_chars = chars = 0
    for line in block.get("lines", []):
        text = "".join(s["text"] for s in line["spans"]).strip()
        if not text:
            continue
        if lines and lines[-1].endswith("-") and len(lines[-1]) > 1 and lines[-1][-2].islower() and text[:1].islower():
            lines[-1] = lines[-1][:-1] + text  # word hyphenated across a line break
        else:
            lines.append(text)
        for s in line["spans"]:
            n = len(s["text"].strip())
            sizes[round(s["size"], 1)] += n
            chars += n
            if s["flags"] & 16 or "bold" in s.get("font", "").lower():
                bold_chars += n
    text = _WS.sub(" ", " ".join(lines)).strip()
    text = _SOFT_HYPHEN.sub("", text) if "- " in text and _looks_wrapped(block) else text
    size = sizes.most_common(1)[0][0] if sizes else 0.0
    y0, y1 = block["bbox"][1], block["bbox"][3]
    margin = page_h > 0 and (y1 <= page_h * 0.08 or y0 >= page_h * 0.92)
    return text, size, chars > 0 and bold_chars / chars > 0.9, chars, bool(margin)


def _looks_wrapped(block: Dict[str, Any]) -> bool:
    return len(block.get("lines", [])) > 1


def pdf_to_markdown(path: str) -> Tuple[str, str]:
    """Return (title, markdown) for a PDF file."""
    import pymupdf  # type: ignore

    pages: List[List[Block]] = []
    with pymupdf.open(path) as doc:
        meta_title = ((doc.metadata or {}).get("title") or "").strip()
        for page in doc:
            d = page.get_text("dict", flags=pymupdf.TEXT_DEHYPHENATE)
            h = float(page.rect.height)
            pages.append([_block_text(b, h) for b in d["blocks"] if b.get("type") == 0])

    body = Counter()
    for pg in pages:
        for _text, size, _bold, chars, _m in pg:
            body[size] += chars
    body_size = body.most_common(1)[0][0] if body else 10.0

    repeated: set = set()
    if len(pages) >= 3:
        seen = Counter()
        for pg in pages:
            for key in {_DIGITS.sub("#", b[0].lower()) for b in pg if b[4] and len(b[0]) < 120}:
                seen[key] += 1
        repeated = {k for k, n in seen.items() if n >= max(2, 0.4 * len(pages))}

    heading_sizes = sorted({b[1] for pg in pages for b in pg if b[1] >= body_size * 1.15}, reverse=True)

    def level(size: float) -> int:
        if size in heading_sizes:
            return min(3, 1 + heading_sizes.index(size))
        return 3 if size < body_size * 1.15 else 2

    out: List[str] = []
    title = meta_title
    for pg in pages:
        for text, size, bold, _chars, margin in pg:
            if not text or (margin and text.isdigit()) or _WATERMARK.match(text) or \
                    (margin and _DIGITS.sub("#", text.lower()) in repeated):
                continue
            words = len(text.split())
            is_heading = (words <= 12 and not text.endswith((".", ",", ";", ":")) and not _CAPTION.match(text)
                          and (bold or size >= body_size * 1.15) and any(c.isalpha() for c in text))
            if is_heading:
                if not title and size >= body_size * 1.3:
                    title = text
                out.append("#" * level(size) + " " + text)
            else:
                out.append(text)
    return title, "\n\n".join(out) + "\n"
