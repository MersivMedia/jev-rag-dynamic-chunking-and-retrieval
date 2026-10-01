"""Retrieval result objects (FR-R7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Passage:
    id: str
    text: str
    doc_id: str
    source_uri: str
    title: str
    section_path: List[str]
    chunk_index: int
    vector_score: float
    scores: Dict[str, float] = field(default_factory=dict)  # Jev probabilities
    metadata: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None  # for dropped passages
    expanded_from: Optional[str] = None  # set on neighbour chunks

    @property
    def citation(self) -> str:
        title = self.title or self.doc_id
        path = list(self.section_path)
        if path and path[0].strip().lower() == title.strip().lower():
            path = path[1:]
        where = " > ".join(path)
        src = self.source_uri or self.doc_id
        return f"{title}{' > ' + where if where else ''} ({src})"

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text, "doc_id": self.doc_id, "source_uri": self.source_uri,
                "title": self.title, "section_path": self.section_path, "chunk_index": self.chunk_index,
                "vector_score": round(self.vector_score, 4), "scores": self.scores, "reason": self.reason,
                "expanded_from": self.expanded_from}


@dataclass
class RetrievalResult:
    query: str
    passages: List[Passage] = field(default_factory=list)
    conflicts: List[Passage] = field(default_factory=list)
    dropped: List[Passage] = field(default_factory=list)
    abstain: bool = False
    reason: Optional[str] = None
    gate_p: Optional[float] = None
    answerable_p: Optional[float] = None
    filter_used: Optional[Dict[str, Any]] = None
    degraded: bool = False
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_prompt(self, *, max_chars_per_passage: int = 4000) -> str:
        """Evidence and conflicts as separate, numbered, cited blocks for the LLM."""
        lines: List[str] = []
        n = 0
        if self.passages:
            lines.append("<evidence>")
            for p in self.passages:
                n += 1
                lines.append(f'<passage id="{n}" source="{_attr(p.citation)}">')
                lines.append(p.text[:max_chars_per_passage])
                lines.append("</passage>")
            lines.append("</evidence>")
        if self.conflicts:
            lines.append("<conflicting_evidence note=\"These passages contradict an assumption in the question.\">")
            for p in self.conflicts:
                n += 1
                lines.append(f'<passage id="{n}" source="{_attr(p.citation)}">')
                lines.append(p.text[:max_chars_per_passage])
                lines.append("</passage>")
            lines.append("</conflicting_evidence>")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query, "abstain": self.abstain, "reason": self.reason, "gate_p": self.gate_p,
                "answerable_p": self.answerable_p, "filter_used": self.filter_used, "degraded": self.degraded,
                "passages": [p.to_dict() for p in self.passages],
                "conflicts": [p.to_dict() for p in self.conflicts],
                "dropped": [p.to_dict() for p in self.dropped], "trace": self.trace}


def _attr(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
