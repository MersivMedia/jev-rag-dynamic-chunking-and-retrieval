"""Evaluation harness (FR-V1): measure retrieval on labelled data.

Loads a dataset, ingests its corpus into a collection, runs every query through
one or more retrieval *systems*, and scores the results against relevance
labels. Datasets:

* ``beir:<dir>``   BEIR layout (``corpus.jsonl``, ``queries.jsonl``, ``qrels/<split>.tsv``)
* ``qasper:<file>`` QASPER JSON (papers with questions, evidence paragraphs, unanswerable flags)
* ``jsonl:<file>``  one object per line: ``{"query", "relevant": [doc ids], "evidence": [quotes],
  "answerable": bool}``, plus a ``corpus`` file alongside (see :func:`load_jsonl`)

Relevance is judged at document level for BEIR (a returned passage counts if
its ``doc_id`` is labelled relevant) and at evidence level for QASPER and JSONL
(a passage counts if it contains a labelled evidence paragraph or quote).

Systems:

* ``vector``  top-k by embedding similarity, no Jev
* ``jev``     the full pipeline: recall, Jev classification, gate
* ``llm-rerank:<model>`` an LLM scores every candidate 0-10 (OpenAI-compatible
  chat API), a baseline for Jev classification

The runner records every query's candidate list once (vector order, Jev scores,
LLM scores, which labelled units each candidate contains, and the gate score),
then systems are applied offline in :mod:`.systems`. Every system therefore
ranks the *same* candidates, and thresholds can be tuned on a dev split
without new API calls.

Metrics are computed on what each system would hand an LLM, so ``recall@10``
for ``jev`` is measured on its included passages (at most ``max_passages``).
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..types import Document

__all__ = ["EvalQuery", "EvalSet", "load_dataset", "load_beir", "load_qasper", "load_jsonl",
           "evidence_match", "units_for", "ideal_units", "score_ranking", "score_query", "summarise"]


@dataclass
class EvalQuery:
    id: str
    text: str
    relevant: Dict[str, int] = field(default_factory=dict)  # doc_id -> grade (BEIR)
    evidence: List[str] = field(default_factory=list)         # evidence paragraphs/quotes
    answerable: bool = True
    scope: Optional[str] = None  # restrict search to this doc_id (QASPER: one paper per question)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSet:
    name: str
    docs: List[Document]
    queries: List[EvalQuery]
    level: str  # "doc" or "evidence"

    def summary(self) -> Dict[str, Any]:
        words = sum(len(d.text.split()) for d in self.docs)
        return {"name": self.name, "docs": len(self.docs), "words": words, "queries": len(self.queries),
                "answerable": sum(q.answerable for q in self.queries), "level": self.level}


# -- loaders ---------------------------------------------------------------------

def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_beir(path: str, *, split: str = "test", max_queries: Optional[int] = None,
              distractors: Optional[int] = None, seed: int = 0) -> EvalSet:
    """BEIR directory. ``max_queries`` samples queries; ``distractors`` keeps every
    relevant document plus that many random others (None keeps the full corpus)."""
    d = Path(path)
    qrels: Dict[str, Dict[str, int]] = {}
    with (d / "qrels" / f"{split}.tsv").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if i == 0 and not parts[-1].lstrip("-").isdigit():
                continue  # header
            qid, did, grade = parts[0], parts[1], int(parts[2])
            if grade > 0:
                qrels.setdefault(qid, {})[did] = grade
    texts = {str(q["_id"]): q["text"] for q in _read_jsonl(d / "queries.jsonl")}
    qids = sorted(q for q in qrels if q in texts)
    rng = random.Random(seed)
    if max_queries is not None and len(qids) > max_queries:
        qids = sorted(rng.sample(qids, max_queries))
    queries = [EvalQuery(id=q, text=texts[q], relevant=qrels[q]) for q in qids]
    needed = {did for q in queries for did in q.relevant}
    corpus = {str(c["_id"]): c for c in _read_jsonl(d / "corpus.jsonl")}
    keep = set(corpus) if distractors is None else needed | set(
        rng.sample(sorted(set(corpus) - needed), min(distractors, len(corpus) - len(needed))))
    docs = [Document(text=c.get("text", ""), doc_id=did, title=c.get("title", "") or "", format="text")
            for did, c in corpus.items() if did in keep and c.get("text")]
    return EvalSet(name=f"beir:{d.name}", docs=docs, queries=queries, level="doc")


def _qasper_markdown(paper: Dict[str, Any]) -> str:
    out = [f"# {paper.get('title', '')}", "", "## Abstract", "", paper.get("abstract", ""), ""]
    for sec in paper.get("full_text", []):
        name = (sec.get("section_name") or "").strip()
        if name:
            out += [f"## {name}", ""]
        for para in sec.get("paragraphs", []):
            para = (para or "").strip()
            if para:
                out += [para, ""]
    return "\n".join(out)


_FLOAT_REF = re.compile(r"^(FLOAT SELECTED|TABLE|FIGURE)", re.I)


def load_qasper(path: str, *, max_papers: Optional[int] = None, seed: int = 0) -> EvalSet:
    """QASPER JSON. Each question is searched within its own paper (``scope``).

    The first annotator's answer is used. A question is answerable unless that
    annotator marked it unanswerable; evidence that points at a figure or table
    (``FLOAT SELECTED: ...``) is skipped because the text of floats isn't in
    ``full_text``. Answerable questions left with no text evidence are dropped.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pids = sorted(data)
    if max_papers is not None and len(pids) > max_papers:
        pids = sorted(random.Random(seed).sample(pids, max_papers))
    docs, queries = [], []
    for pid in pids:
        p = data[pid]
        docs.append(Document(text=_qasper_markdown(p), doc_id=pid, title=p.get("title", ""), format="markdown"))
        for qa in p.get("qas", []):
            a = qa["answers"][0]["answer"]
            unanswerable = bool(a.get("unanswerable"))
            ev = [e.strip() for e in a.get("evidence", []) if e and e.strip() and not _FLOAT_REF.match(e.strip())]
            if not unanswerable and not ev:
                continue
            queries.append(EvalQuery(id=qa["question_id"], text=qa["question"], evidence=ev,
                                     answerable=not unanswerable, scope=pid,
                                     meta={"type": "unanswerable" if unanswerable else
                                           "yes_no" if a.get("yes_no") is not None else
                                           "extractive" if a.get("extractive_spans") else "free_form"}))
    return EvalSet(name="qasper", docs=docs, queries=queries, level="evidence")


def load_jsonl(path: str, *, corpus: Optional[str] = None, split: Optional[str] = None) -> EvalSet:
    """Your own data. ``path`` holds queries; ``corpus`` (default: ``corpus.jsonl``
    next to it) holds ``{"id", "text", "title"?}`` objects. A query object has
    ``query`` (or ``text``), optional ``id``, and either ``relevant`` (doc ids,
    document-level scoring) or ``evidence`` (quotes, evidence-level scoring),
    plus ``answerable`` (default true), ``scope`` (search only this doc id) and
    ``split`` (``split=`` keeps only matching queries)."""
    qp = Path(path)
    cp = Path(corpus) if corpus else qp.with_name("corpus.jsonl")
    docs = [Document(text=c["text"], doc_id=str(c["id"]), title=c.get("title", ""),
                     format=c.get("format", "markdown")) for c in _read_jsonl(cp)]
    queries, level = [], "doc"
    for i, q in enumerate(_read_jsonl(qp)):
        if split is not None and q.get("split", split) != split:
            continue
        ev = list(q.get("evidence") or [])
        if ev:
            level = "evidence"
        rel = q.get("relevant") or []
        queries.append(EvalQuery(id=str(q.get("id", i)), text=q.get("query") or q["text"],
                                 relevant={str(r): 1 for r in rel} if isinstance(rel, list) else
                                 {str(k): int(v) for k, v in rel.items()},
                                 evidence=ev, answerable=bool(q.get("answerable", True)), scope=q.get("scope"),
                                 meta={k: v for k, v in q.items() if k in ("split", "type")}))
    return EvalSet(name=f"jsonl:{qp.stem}", docs=docs, queries=queries, level=level)


def load_dataset(spec: str, **kw: Any) -> EvalSet:
    kind, _, path = spec.partition(":")
    if kind == "beir":
        return load_beir(path, **kw)
    if kind == "qasper":
        return load_qasper(path, **{k: v for k, v in kw.items() if k in ("max_papers", "seed")})
    if kind == "jsonl":
        return load_jsonl(path, **{k: v for k, v in kw.items() if k in ("corpus", "split")})
    raise ValueError(f"unknown dataset {spec!r}: use beir:<dir>, qasper:<file> or jsonl:<file>")


# -- matching --------------------------------------------------------------------

_WS = re.compile(r"\s+")
_NON = re.compile(r"[^0-9a-z ]+")


def _norm(s: str) -> str:
    return _WS.sub(" ", _NON.sub(" ", s.lower())).strip()


def evidence_match(evidence: str, passage: str, *, min_overlap: float = 0.6) -> bool:
    """Does ``passage`` contain this evidence?

    Exact normalised containment, or (for evidence split across a chunk
    boundary) at least ``min_overlap`` of the evidence's word 5-grams.
    """
    e, p = _norm(evidence), _norm(passage)
    if not e:
        return False
    if e in p:
        return True
    ew = e.split()
    if len(ew) < 5:
        return False
    grams = {" ".join(ew[i:i + 5]) for i in range(len(ew) - 4)}
    pw = p.split()
    pgrams = {" ".join(pw[i:i + 5]) for i in range(len(pw) - 4)}
    return len(grams & pgrams) / len(grams) >= min_overlap


# -- scoring ---------------------------------------------------------------------

def units_for(q: EvalQuery, doc_id: str, text: str, level: str) -> Dict[str, int]:
    """Which labelled units a passage contains, with their gains.

    Document level: ``{doc_id: grade}`` when the document is labelled relevant.
    Evidence level: ``{"e<j>": 1}`` for every evidence item the passage contains.
    """
    if level == "doc":
        g = q.relevant.get(str(doc_id), 0)
        return {str(doc_id): g} if g > 0 else {}
    return {f"e{j}": 1 for j, ev in enumerate(q.evidence) if evidence_match(ev, text)}


def ideal_units(q: EvalQuery, level: str) -> Dict[str, int]:
    if level == "doc":
        return {d: g for d, g in q.relevant.items() if g > 0}
    return {f"e{j}": 1 for j in range(len(q.evidence))}


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def score_ranking(ranked_units: Sequence[Mapping[str, int]], ideal: Mapping[str, int], *, k: int = 10) -> Dict[str, Any]:
    """Metrics for one ranked list, given the units each item contains.

    A unit counts once, at the first item that contains it, so two chunks of
    one relevant document don't double-count. ``recall`` is the share of units
    found in the top ``k``; ``ndcg`` uses graded gains against the ideal order.
    """
    seen: set = set()
    gains: List[float] = []
    first: Optional[int] = None
    for i, units in enumerate(list(ranked_units)[:k]):
        new = {u: g for u, g in units.items() if u not in seen and g > 0}
        if new:
            seen |= set(new)
            first = i if first is None else first
        gains.append(float(sum(new.values())))
    total = len(ideal)
    idcg = _dcg(sorted((float(g) for g in ideal.values()), reverse=True)[:k])
    return {"recall": len(seen) / total if total else 0.0, "any_hit": bool(seen), "hit_at_1": first == 0,
            "mrr": 1.0 / (first + 1) if first is not None else 0.0,
            "ndcg": min(1.0, _dcg(gains) / idcg) if idcg else 0.0, "returned": min(k, len(ranked_units))}


def score_query(q: EvalQuery, passages: Sequence[Dict[str, Any]], level: str, *, k: int = 10) -> Dict[str, Any]:
    """Convenience wrapper: metrics for ``{"doc_id", "text"}`` passages."""
    if not q.answerable:
        return {"answerable": False}
    ranked = [units_for(q, str(p.get("doc_id", "")), p.get("text", ""), level) for p in passages]
    return {"answerable": True, **score_ranking(ranked, ideal_units(q, level), k=k)}


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def summarise(rows: Sequence[Dict[str, Any]], *, k: int = 10) -> Dict[str, Any]:
    """Mean metrics over per-query score rows (``{"answerable", "recall", ..., "abstain"?, "tokens"?}``)."""
    ans = [r for r in rows if r.get("answerable")]
    una = [r for r in rows if not r.get("answerable")]
    out: Dict[str, Any] = {
        "queries": len(rows), "answerable": len(ans), "unanswerable": len(una),
        f"recall@{k}": _mean([r["recall"] for r in ans]),
        f"ndcg@{k}": _mean([r["ndcg"] for r in ans]),
        "mrr": _mean([r["mrr"] for r in ans]),
        "hit@1": _mean([float(r["hit_at_1"]) for r in ans]),
        "any_hit": _mean([float(r["any_hit"]) for r in ans]),
        "passages_mean": _mean([float(r.get("returned", 0)) for r in rows]),
        "context_tokens_mean": _mean([float(r.get("tokens", 0)) for r in rows]),
    }
    if any("abstain" in r for r in rows):
        out["false_abstain"] = _mean([float(bool(r.get("abstain"))) for r in ans])
        if una:
            out["true_abstain"] = _mean([float(bool(r.get("abstain"))) for r in una])
    return out
