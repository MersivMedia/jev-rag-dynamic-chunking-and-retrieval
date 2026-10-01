"""Minimal ``.env`` loader (stdlib only).

Supported syntax, one assignment per line::

    # comment
    KEY=value
    export KEY=value
    KEY="value with spaces"      # double quotes: \\n \\t \\" \\\\ escapes
    KEY='literal value'          # single quotes: taken as is
    KEY=value # trailing comment (unquoted values only; needs a space before #)
    KEY=                         # empty: ignored, so blank template lines never mask real values

No variable expansion (``$OTHER`` stays literal) and no multi-line values.
Variables already set in the environment win unless ``override=True``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Union

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


def _unquote(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        return raw[1:-1]
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), "\\" + m.group(1)), raw[1:-1])
    if raw[:1] in ("'", '"'):
        # quoted value followed by a comment: KEY="a b"  # note
        q = raw[0]
        end = raw.find(q, 1)
        while q == '"' and end > 0 and raw[end - 1] == "\\":
            end = raw.find(q, end + 1)
        if end > 0:
            return _unquote(raw[:end + 1])
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def parse_env_file(path: Union[str, Path]) -> Dict[str, str]:
    """Parse a .env file into a dict. Raises ``ValueError`` naming the first bad line."""
    out: Dict[str, str] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"{path}:{n}: expected KEY=value")
        out[m.group(1)] = _unquote(m.group(2))
    return out


def load_env_file(path: Union[str, Path] = ".env", *, override: bool = False) -> List[str]:
    """Set variables from ``path`` into ``os.environ``; return the names that were set.

    Empty values are skipped. A missing file is not an error (returns ``[]``).
    """
    p = Path(path)
    if not p.is_file():
        return []
    applied = []
    for key, value in parse_env_file(p).items():
        if value == "":
            continue
        if override or not os.environ.get(key):
            os.environ[key] = value
            applied.append(key)
    return applied
