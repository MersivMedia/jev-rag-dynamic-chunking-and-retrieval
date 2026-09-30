"""Split a document's text into structural blocks with offsets.

Markdown is the internal format: plain text is treated as Markdown without
markup, and HTML/PDF/DOCX loaders convert to Markdown first. Structure is read
in code and never asked of Jev: headings, code fences, tables, lists, quotes.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from ..types import Block

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^(```|~~~)")
_LIST = re.compile(r"^\s{0,6}(?:[-*+•]|\d{1,3}[.)])\s+")
_TABLE = re.compile(r"^\s*\|.*\|\s*$")
_QUOTE = re.compile(r"^\s{0,3}>")
_SETEXT = re.compile(r"^(=+|-+)\s*$")


def _lines(text: str) -> List[Tuple[int, int, str]]:
    out = []
    pos = 0
    for line in text.split("\n"):
        out.append((pos, pos + len(line), line))
        pos += len(line) + 1
    return out


def parse_blocks(text: str, fmt: str = "markdown") -> List[Block]:
    lines = _lines(text)
    blocks: List[Block] = []
    path: List[str] = []
    levels: List[int] = []
    markdown = fmt != "text"
    i = 0
    n = len(lines)

    def push(kind: str, start: int, end: int, level: int = 0) -> None:
        # trim surrounding whitespace, keep offsets exact
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            blocks.append(Block(kind=kind, start=start, end=end, section_path=list(path), level=level))

    while i < n:
        s, e, line = lines[i]
        if not line.strip():
            i += 1
            continue
        if markdown:
            m = _HEADING.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                while levels and levels[-1] >= level:
                    levels.pop()
                    path.pop()
                levels.append(level)
                path.append(title)
                push("heading", s, e, level)
                i += 1
                continue
            if _FENCE.match(line.strip()):
                fence = line.strip()[:3]
                j = i + 1
                while j < n and not lines[j][2].strip().startswith(fence):
                    j += 1
                end = lines[min(j, n - 1)][1]
                push("code", s, end)
                i = j + 1
                continue
            if _TABLE.match(line):
                j = i
                while j + 1 < n and _TABLE.match(lines[j + 1][2]):
                    j += 1
                push("table", s, lines[j][1])
                i = j + 1
                continue
            if _QUOTE.match(line):
                j = i
                while j + 1 < n and lines[j + 1][2].strip() and _QUOTE.match(lines[j + 1][2]):
                    j += 1
                push("quote", s, lines[j][1])
                i = j + 1
                continue
            if _LIST.match(line):
                j = i
                # continuation lines: indented, non-empty, not a new item
                while j + 1 < n and lines[j + 1][2].strip() and not _LIST.match(lines[j + 1][2]) \
                        and lines[j + 1][2].startswith((" ", "\t")):
                    j += 1
                push("list_item", s, lines[j][1])
                i = j + 1
                continue
        # paragraph: consecutive non-empty lines that don't start another block
        j = i
        while j + 1 < n:
            nxt = lines[j + 1][2]
            if not nxt.strip():
                break
            if markdown and (_HEADING.match(nxt) or _FENCE.match(nxt.strip()) or _TABLE.match(nxt)
                             or _LIST.match(nxt) or _QUOTE.match(nxt)):
                break
            if markdown and _SETEXT.match(nxt) and j == i:
                break
            j += 1
        if markdown and j + 1 < n and _SETEXT.match(lines[j + 1][2]) and j == i:
            level = 1 if lines[j + 1][2].strip().startswith("=") else 2
            title = line.strip()
            while levels and levels[-1] >= level:
                levels.pop()
                path.pop()
            levels.append(level)
            path.append(title)
            push("heading", s, e, level)
            i = j + 2
            continue
        push("paragraph", s, lines[j][1])
        i = j + 1
    return blocks
