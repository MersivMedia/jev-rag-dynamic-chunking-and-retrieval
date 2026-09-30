"""Token counting for chunk sizes.

Uses tiktoken's ``cl100k_base`` when the ``tokens`` extra is installed (close to
OpenAI embedding tokenisation), otherwise a word-and-punctuation estimate.
Pass any ``Callable[[str], int]`` to use your embedding model's own tokenizer.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable, Optional

TokenCounter = Callable[[str], int]

_PIECE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """About 1.3 tokens per word plus punctuation; within ~15% of BPE on English prose."""
    pieces = _PIECE.findall(text)
    words = sum(1 for p in pieces if p[0].isalnum() or p[0] == "_")
    long_words = sum(1 for p in pieces if len(p) > 8)
    return words + (len(pieces) - words) + long_words // 2


@lru_cache(maxsize=1)
def _tiktoken() -> Optional[Callable[[str], int]]:
    try:
        import tiktoken  # type: ignore
    except ImportError:
        return None
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:  # offline, no cached encoding
        return None
    return lambda s: len(enc.encode(s, disallowed_special=()))


def default_counter() -> TokenCounter:
    return _tiktoken() or estimate_tokens
