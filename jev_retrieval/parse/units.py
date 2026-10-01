"""Turn blocks into units: the pieces the chunker places cuts between.

* prose blocks (paragraph, list item, quote) -> one unit per sentence
* tables and code -> one atomic unit, split by lines only if over ``max_tokens``
* headings -> a unit glued to what follows it; every heading starts a new section
* any single unit over ``max_tokens`` is split at whitespace
"""

from __future__ import annotations

from typing import List

from ..tokens import TokenCounter
from ..types import Block, Unit
from .sentences import split_sentences


def _split_long(text: str, start: int, end: int, max_tokens: int, count: TokenCounter) -> List[tuple]:
    """Split [start, end) at whitespace into spans of at most max_tokens."""
    spans = []
    cur = start
    while cur < end:
        # grow a window word by word
        j = cur
        last_ok = None
        while j < end:
            k = j
            while k < end and not text[k].isspace():
                k += 1
            if count(text[cur:k]) > max_tokens and last_ok is not None:
                break
            last_ok = k
            while k < end and text[k].isspace():
                k += 1
            j = k
        stop = last_ok if last_ok is not None else end
        spans.append((cur, stop))
        cur = stop
        while cur < end and text[cur].isspace():
            cur += 1
    return spans


def build_units(text: str, blocks: List[Block], max_tokens: int, count: TokenCounter) -> List[Unit]:
    units: List[Unit] = []
    section = -1
    prev_path: object = None
    for bi, b in enumerate(blocks):
        if b.kind == "heading" or b.section_path != prev_path:
            section += 1
            prev_path = b.section_path
        if b.kind == "heading":
            units.append(Unit(b.start, b.end, count(text[b.start:b.end]), "heading", bi, True, section))
            continue
        if b.kind in ("table", "code"):
            tokens = count(text[b.start:b.end])
            if tokens <= max_tokens:
                units.append(Unit(b.start, b.end, tokens, b.kind, bi, True, section))
                continue
            # split by lines, keeping a table's header row with every piece is left to the reader;
            # we keep line boundaries so rows are never broken
            line_start = b.start
            first = True
            pieces = []
            for line in text[b.start:b.end].split("\n"):
                line_end = line_start + len(line)
                pieces.append((line_start, line_end))
                line_start = line_end + 1
            cur_s, cur_e, cur_t = pieces[0][0], pieces[0][1], count(text[pieces[0][0]:pieces[0][1]])
            for s, e in pieces[1:]:
                t = count(text[s:e])
                if cur_t + t > max_tokens:
                    units.append(Unit(cur_s, cur_e, cur_t, b.kind, bi, first, section))
                    first = False
                    cur_s, cur_e, cur_t = s, e, t
                else:
                    cur_e, cur_t = e, cur_t + t
            units.append(Unit(cur_s, cur_e, cur_t, b.kind, bi, first, section))
            continue
        first = True
        for s, e in split_sentences(text[b.start:b.end], base=b.start):
            t = count(text[s:e])
            if t <= max_tokens:
                units.append(Unit(s, e, t, "sentence", bi, first, section))
                first = False
                continue
            for ps, pe in _split_long(text, s, e, max_tokens, count):
                units.append(Unit(ps, pe, count(text[ps:pe]), "sentence", bi, first, section))
                first = False
    return units
