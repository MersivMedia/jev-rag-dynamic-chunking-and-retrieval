"""Typed Jev questions and answers.

Three primitives, mirroring TypeSafe's System One API
(https://docs.typesafe.ai/api):

* ``Noul``   - is this statement true?      -> probability in [0, 1]
* ``Choice`` - which of these options?      -> option, per-option probabilities, confidence
* ``Score``  - where on this ordered scale? -> weighted score, per-level probabilities, confidence

Limits are validated client-side so a malformed spec fails when it is built,
not as a 422 halfway through an ingest: Choice 2..255 options, Score 2..10 levels.
Adapted from Jermes (MIT, Mersiv Media).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

Instructions = Union[str, Dict[str, Any], List[Any]]


class SpecError(ValueError):
    """A question spec violates the Jev API contract."""


@dataclass(frozen=True)
class Noul:
    instructions: Instructions
    criteria: Optional[Mapping[str, Any]] = None  # {"true": ..., "false": ...}

    def to_wire(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            bad = set(self.criteria) - {"true", "false"}
            if bad:
                raise SpecError(f"Noul criteria keys must be 'true'/'false', got {sorted(bad)}")
            out["criteria"] = dict(self.criteria)
        return out


@dataclass(frozen=True)
class Choice:
    instructions: Instructions
    criteria: Mapping[str, Any]  # option -> description (or None)

    def to_wire(self) -> Dict[str, Any]:
        n = len(self.criteria)
        if n < 2:
            raise SpecError("Choice needs at least 2 options")
        if n > MAX_CHOICE_OPTIONS:
            raise SpecError(f"Choice supports at most {MAX_CHOICE_OPTIONS} options, got {n}")
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.criteria)}


@dataclass(frozen=True)
class Score:
    instructions: Instructions
    criteria: List[Any]  # ordered low -> high

    def to_wire(self) -> Dict[str, Any]:
        n = len(self.criteria)
        if not MIN_SCORE_LEVELS <= n <= MAX_SCORE_LEVELS:
            raise SpecError(f"Score needs {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} levels, got {n}")
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.criteria)}


Question = Union[Noul, Choice, Score]


def questions_to_wire(questions: Mapping[str, Question]) -> Dict[str, Dict[str, Any]]:
    if not questions:
        raise SpecError("at least one question is required")
    return {qid: q.to_wire() for qid, q in questions.items()}


def question_from_dict(spec: Mapping[str, Any]) -> Question:
    """Build a question from YAML-style config: {type, instructions, criteria?}."""
    kind = str(spec.get("type", "noul")).lower()
    instructions = spec.get("instructions")
    if not instructions:
        raise SpecError(f"question is missing 'instructions': {dict(spec)!r}")
    criteria = spec.get("criteria")
    if kind == "noul":
        return Noul(instructions, criteria)
    if kind == "choice":
        if not isinstance(criteria, Mapping):
            raise SpecError("a choice question needs 'criteria' as a map of option -> description")
        return Choice(instructions, dict(criteria))
    if kind == "score":
        if not isinstance(criteria, list):
            raise SpecError("a score question needs 'criteria' as an ordered list of levels")
        return Score(instructions, list(criteria))
    raise SpecError(f"unknown question type {kind!r} (expected noul, choice or score)")


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


def _distribution_confidence(probabilities: Mapping[str, float]) -> float:
    """Fallback confidence when a backend omits it: ``(n * p_max - 1) / (n - 1)``.

    1.0 when all mass is on one option, 0.0 for a uniform split. Only used when
    the response carries no ``confidence`` field.
    """
    values = [float(v) for v in probabilities.values()]
    n = len(values)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (n * max(values) - 1.0) / (n - 1.0)))


@dataclass(frozen=True)
class NoulAnswer:
    noul: float
    type: str = "noul"

    @property
    def probability(self) -> float:
        return self.noul

    @property
    def value(self) -> float:
        return self.noul


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: Dict[str, float]
    confidence: float
    type: str = "choice"

    def top(self, k: int) -> List[tuple]:
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:k]

    @property
    def value(self) -> str:
        return self.choice


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    probabilities: Dict[str, float]
    confidence: float
    legend: Dict[str, Any] = field(default_factory=dict)
    type: str = "score"

    @property
    def value(self) -> float:
        return self.score


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


def parse_answer(raw: Mapping[str, Any]) -> Answer:
    """Parse one answer. Accepts TypeSafe's ``noul`` and Vercel's ``boolean`` dialects."""
    kind = raw.get("type")
    if kind in ("noul", "boolean"):
        value = raw.get("noul", raw.get("probability"))
        if value is None:
            raise ValueError(f"noul answer missing probability: {raw!r}")
        return NoulAnswer(noul=float(value))
    if kind == "choice":
        probs = {str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()}
        choice = raw.get("choice")
        if choice is None and probs:
            choice = max(probs.items(), key=lambda kv: kv[1])[0]
        conf = raw.get("confidence")
        return ChoiceAnswer(
            choice=str(choice),
            probabilities=probs,
            confidence=float(conf) if conf is not None else _distribution_confidence(probs),
        )
    if kind == "score":
        probs = {str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()}
        score = raw.get("score")
        if score is None:
            score = sum(int(k) * v for k, v in probs.items())
        conf = raw.get("confidence")
        return ScoreAnswer(
            score=float(score),
            probabilities=probs,
            confidence=float(conf) if conf is not None else _distribution_confidence(probs),
            legend=dict(raw.get("legend") or {}),
        )
    raise ValueError(f"unknown answer type {kind!r}")


def answer_to_dict(answer: Answer) -> Dict[str, Any]:
    if isinstance(answer, NoulAnswer):
        return {"type": "noul", "noul": answer.noul}
    if isinstance(answer, ChoiceAnswer):
        return {
            "type": "choice",
            "choice": answer.choice,
            "probabilities": answer.probabilities,
            "confidence": answer.confidence,
        }
    return {
        "type": "score",
        "score": answer.score,
        "probabilities": answer.probabilities,
        "confidence": answer.confidence,
        "legend": answer.legend,
    }
