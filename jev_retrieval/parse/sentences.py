"""Sentence segmentation that keeps character offsets.

Handles common abbreviations (e.g., i.e., U.S., Dr.), decimals, version
numbers, URLs, e-mail addresses, ellipses and initials. It never splits inside
a token, only at whitespace after terminal punctuation followed by something
that looks like a sentence start.
"""

from __future__ import annotations

import re
from typing import List, Tuple

ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "cf", "al", "approx", "est", "fig", "figs", "no", "nos", "vol", "vols",
    "pp", "p", "ed", "eds", "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "ft", "inc", "ltd",
    "co", "corp", "dept", "univ", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun", "u.s", "u.k", "u.n", "e.u", "a.m", "p.m",
    "ph.d", "b.sc", "m.sc", "ref", "sec", "ch", "art", "min", "max", "avg", "resp", "incl", "excl",
}

# Candidate boundary: terminal punctuation, optional closing quotes/brackets, whitespace.
_BOUNDARY = re.compile(r"([.!?…]+)([\"'”’)\]]*)(\s+)")
_START_OK = re.compile(r"[\"'“‘(\[]*[A-Z0-9À-ÖØ-Þ#*`>•\-–—]")


def _prev_word(text: str, end: int) -> str:
    i = end
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    return text[i:end]


def split_sentences(text: str, base: int = 0) -> List[Tuple[int, int]]:
    """Return (start, end) spans of sentences in ``text``, offset by ``base``."""
    spans: List[Tuple[int, int]] = []
    start = 0
    n = len(text)
    # skip leading whitespace
    while start < n and text[start].isspace():
        start += 1
    for m in _BOUNDARY.finditer(text):
        punct_start = m.start(1)
        end = m.end(2)
        nxt = m.end(3)
        if nxt >= n:
            break
        if punct_start < start:
            continue
        punct = m.group(1)
        word = _prev_word(text, punct_start).lower().strip("\"'“‘([")
        if punct == ".":
            if word in ABBREVIATIONS or word.rstrip(".") in ABBREVIATIONS:
                continue
            if len(word) == 1 and word.isalpha():  # initials: "J. Smith"
                continue
            if "." in word and all(len(p) <= 2 for p in word.split(".") if p):  # "U.S", "a.m"
                continue
        if not _START_OK.match(text, nxt):
            continue
        spans.append((base + start, base + end))
        start = nxt
    tail_end = n
    while tail_end > start and text[tail_end - 1].isspace():
        tail_end -= 1
    if tail_end > start:
        spans.append((base + start, base + tail_end))
    return spans
