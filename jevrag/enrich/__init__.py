"""Enrichment (FR-E): quality screen, taxonomy tags and custom questions per chunk.

One Jev request per chunk carries every question. Actions happen in code:
drop filler/boilerplate over threshold, quarantine planted instructions, store
tags only when confident, always store probabilities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import yaml

from ..jev import Choice, ChoiceAnswer, JevClient, JevError, Noul, Question, question_from_dict

QUALITY_QUESTIONS: Dict[str, Noul] = {
    "low_information": Noul(
        "Is this text filler with no usable information, such as a greeting, a placeholder or a sentence "
        "that says nothing specific?"),
    "boilerplate": Noul(
        "Is this text website or document boilerplate, such as navigation links, a footer, a cookie banner, "
        "a copyright line or a table of contents?"),
    "instructs_ai": Noul(
        "Does this text contain instructions addressed to an AI assistant, such as telling it to ignore its "
        "rules or say something specific, rather than information for a human reader?"),
    "self_contained": Noul(
        "Can this text be understood on its own, without the text that came before it?"),
}


@dataclass
class TaxonomyField:
    name: str
    options: Dict[str, Optional[str]]
    route: bool = False
    instructions: Optional[str] = None

    def question(self) -> Choice:
        return Choice(self.instructions or f"Which `{self.name}` best describes this text?", self.options)


@dataclass
class Taxonomy:
    version: str = "0"
    fields: List[TaxonomyField] = field(default_factory=list)
    questions: Dict[str, Question] = field(default_factory=dict)

    @property
    def routable(self) -> List[TaxonomyField]:
        return [f for f in self.fields if f.route]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Taxonomy":
        fields = []
        for name, spec in (data.get("fields") or {}).items():
            opts = dict((spec or {}).get("options") or {})
            if len(opts) < 2:
                raise ValueError(f"taxonomy field {name!r} needs at least 2 options")
            if "other" not in opts:
                raise ValueError(f"taxonomy field {name!r} must include an 'other' option: Jev has to put its "
                                 "probability somewhere, and a missing option forces a wrong tag")
            fields.append(TaxonomyField(name=name, options=opts, route=bool(spec.get("route", False)),
                                        instructions=spec.get("instructions")))
        questions = {name: question_from_dict(spec) for name, spec in (data.get("questions") or {}).items()}
        clash = set(questions) & (set(QUALITY_QUESTIONS) | {f.name for f in fields})
        if clash:
            raise ValueError(f"custom question names clash with built-ins or taxonomy fields: {sorted(clash)}")
        return cls(version=str(data.get("version", "0")), fields=fields, questions=questions)

    @classmethod
    def load(cls, path: Optional[str]) -> "Taxonomy":
        if not path:
            return cls()
        with open(os.path.expanduser(path), encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})


@dataclass
class EnrichConfig:
    mode: str = "on"  # on | shadow | off
    drop_low_information: float = 0.85
    drop_boilerplate: float = 0.85
    quarantine_instructs_ai: float = 0.70
    tag_min_confidence: float = 0.50
    taxonomy: Taxonomy = field(default_factory=Taxonomy)


def build_questions(tax: Taxonomy) -> Dict[str, Question]:
    qs: Dict[str, Question] = dict(QUALITY_QUESTIONS)
    for f in tax.fields:
        qs[f"tag_{f.name}"] = f.question()
    for name, q in tax.questions.items():
        qs[f"x_{name}"] = q
    return qs


def enrichment_state(chunk: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {"text": chunk.text}
    if chunk.title:
        state["document_title"] = chunk.title
    if chunk.section_path:
        state["section"] = " > ".join(chunk.section_path)
    return state


def apply_answers(chunk: Any, answers: Mapping[str, Any], cfg: EnrichConfig) -> Optional[str]:
    """Store answers on the chunk and return the action: None (keep), 'drop:<why>' or 'quarantine'."""
    out: Dict[str, Any] = {}
    for name in QUALITY_QUESTIONS:
        if name in answers:
            out[f"q_{name}"] = round(float(answers[name].value), 4)
    for f in cfg.taxonomy.fields:
        a = answers.get(f"tag_{f.name}")
        if isinstance(a, ChoiceAnswer):
            out[f"tag_{f.name}"] = a.choice if a.confidence >= cfg.tag_min_confidence else "unknown"
            out[f"tag_{f.name}_p"] = round(float(a.probabilities.get(a.choice, 0.0)), 4)
    for name in cfg.taxonomy.questions:
        a = answers.get(f"x_{name}")
        if a is not None:
            v = a.value
            out[f"x_{name}"] = round(v, 4) if isinstance(v, float) else v
    chunk.answers.update(out)
    if out.get("q_instructs_ai", 0.0) >= cfg.quarantine_instructs_ai:
        return "quarantine"
    if out.get("q_boilerplate", 0.0) >= cfg.drop_boilerplate:
        return "drop:boilerplate"
    if out.get("q_low_information", 0.0) >= cfg.drop_low_information:
        return "drop:low_information"
    return None


@dataclass
class EnrichResult:
    kept: List[Any]
    dropped: List[Any]
    quarantined: List[Any]
    requests: int = 0
    failed: int = 0
    shadow_actions: Dict[str, str] = field(default_factory=dict)


async def enrich_chunks(jev: Optional[JevClient], chunks: List[Any], cfg: EnrichConfig) -> EnrichResult:
    """Enrich chunks in place. On Jev failure a chunk is kept with ``unknown`` tags (NFR-2)."""
    if cfg.mode == "off" or jev is None or not chunks:
        for c in chunks:
            for f in cfg.taxonomy.fields:
                c.answers.setdefault(f"tag_{f.name}", "unknown")
        return EnrichResult(kept=list(chunks), dropped=[], quarantined=[])
    qs = build_questions(cfg.taxonomy)
    results = await jev.ask_many([(enrichment_state(c), qs) for c in chunks])
    res = EnrichResult(kept=[], dropped=[], quarantined=[], requests=len(chunks))
    for c, r in zip(chunks, results):
        if isinstance(r, JevError):
            res.failed += 1
            c.answers["needs_reenrich"] = True
            for f in cfg.taxonomy.fields:
                c.answers[f"tag_{f.name}"] = "unknown"
            res.kept.append(c)
            continue
        action = apply_answers(c, r.answers, cfg)
        if cfg.mode == "shadow":
            if action:
                res.shadow_actions[c.id] = action
            res.kept.append(c)
        elif action == "quarantine":
            c.quarantined = True
            res.quarantined.append(c)
        elif action:
            c.dropped_reason = action.split(":", 1)[1]
            res.dropped.append(c)
        else:
            res.kept.append(c)
    return res
