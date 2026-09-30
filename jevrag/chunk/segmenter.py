"""Cut placement in code (FR-C3/C4).

Given the units of a document and a cut cost at every gap, choose cuts by
dynamic programming:

    total = sum(cut_weight * cost[gap] for each cut)
          + sum(size_weight * ((tokens - target) / target) ** 2 for each chunk)
          + penalties (under min_tokens, cutting a protected gap)

subject to: no chunk crosses a section, no chunk exceeds ``max_tokens``
(units are pre-split so a single unit always fits). Output is deterministic
for a given set of costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ..types import Unit

UNDER_MIN_PENALTY = 4.0
PROTECTED_PENALTY = 6.0


@dataclass
class Gap:
    """The gap before unit ``index`` (between index-1 and index)."""

    index: int
    cost: float  # 0 = ideal place to cut, 1 = worst
    protected: bool = False  # e.g. refers_back >= 0.5, or right after a heading


@dataclass
class SegmenterConfig:
    min_tokens: int = 64
    target_tokens: int = 350
    max_tokens: int = 800
    cut_weight: float = 1.0
    size_weight: float = 0.6


def _sections(units: Sequence[Unit]) -> List[Tuple[int, int]]:
    """[start, end) unit ranges of each section, merging heading-only sections forward."""
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, len(units) + 1):
        if i == len(units) or units[i].section != units[i - 1].section:
            ranges.append((start, i))
            start = i
    merged: List[Tuple[int, int]] = []
    carry: Optional[int] = None
    for s, e in ranges:
        if carry is not None:
            s = carry
            carry = None
        if all(u.kind == "heading" for u in units[s:e]) and (s, e) != ranges[-1]:
            carry = s
            continue
        merged.append((s, e))
    if carry is not None:
        merged.append((carry, len(units)))
    return merged


def segment(units: Sequence[Unit], costs: Sequence[Gap], cfg: SegmenterConfig) -> List[Tuple[int, int]]:
    """Return [start, end) unit ranges, one per chunk, in order."""
    if not units:
        return []
    gap = {g.index: g for g in costs}
    out: List[Tuple[int, int]] = []
    target = max(1, cfg.target_tokens)
    for s, e in _sections(units):
        n = e - s
        prefix = [0]
        for u in units[s:e]:
            prefix.append(prefix[-1] + u.tokens + 1)  # +1 for the joining whitespace
        section_tokens = prefix[-1]
        inf = float("inf")
        best = [inf] * (n + 1)
        back = [0] * (n + 1)
        best[0] = 0.0
        for j in range(1, n + 1):
            i = j - 1
            while i >= 0:
                tok = prefix[j] - prefix[i]
                if tok > cfg.max_tokens and j - i > 1:
                    break
                if best[i] < inf:
                    c = best[i]
                    if i > 0:
                        g = gap.get(s + i)
                        gc = g.cost if g else 0.5
                        c += cfg.cut_weight * gc
                        if g and g.protected:
                            c += PROTECTED_PENALTY
                    c += cfg.size_weight * ((tok - target) / target) ** 2
                    if tok < cfg.min_tokens and section_tokens >= cfg.min_tokens:
                        c += UNDER_MIN_PENALTY
                    if c < best[j]:
                        best[j] = c
                        back[j] = i
                i -= 1
        cuts = []
        j = n
        while j > 0:
            cuts.append((s + back[j], s + j))
            j = back[j]
        out.extend(reversed(cuts))
    return out


def structural_costs(units: Sequence[Unit]) -> List[Gap]:
    """Code-only cut costs: cheap at block boundaries, expensive mid-paragraph."""
    gaps = []
    for i in range(1, len(units)):
        prev, cur = units[i - 1], units[i]
        if prev.kind == "heading":
            gaps.append(Gap(i, 1.0, protected=True))
        elif cur.first_in_block:
            gaps.append(Gap(i, 0.2))
        else:
            gaps.append(Gap(i, 0.85))
    return gaps
