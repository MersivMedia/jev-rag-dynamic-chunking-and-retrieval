"""Turn a recording into metrics for each retrieval system, with no API calls.

Systems (``name`` -> what an LLM would receive):

* ``vector@N``          the vector top N
* ``jev``               Jev classification with the given ``ClassifyConfig``
                        (library default: rank mode, top 5): candidates routed
                        ``include``, sorted by evidence score then vector score,
                        capped at ``max_passages``, plus conflicts
* ``jev-rerank@N``      every candidate sorted by Jev's evidence score, top N
                        (classification used as a re-ranker, no dropping)
* ``llm-rerank@N``      every candidate sorted by the LLM's score, top N

Gate metrics come from the gate scores in the recording: ``vector@N`` uses the
gate asked about the vector top N (only when N equals the recorded
``gate_top``); ``jev`` uses the gate recorded for threshold mode (top 8) or
rank mode (top 5), whichever ``select`` asks for.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import asdict, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..retrieve.stages import ClassifyConfig, evidence_order, route_passage
from . import score_ranking, summarise


def _num(x: Any, default: float) -> float:
    return float(x) if x is not None else default


def _ideal(row: Mapping[str, Any], qideal: Mapping[str, Mapping[str, int]]) -> Mapping[str, int]:
    return qideal.get(row["id"], {})


def select(row: Mapping[str, Any], system: str, cfg: ClassifyConfig) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """Passages a system hands the LLM, and the gate score that applies (if any)."""
    cands: List[Dict[str, Any]] = row["cands"]
    name, _, n = system.partition("@")
    top = int(n) if n else None
    gate = row.get("gate") or {}
    if name == "vector":
        k = top or 8
        return cands[:k], gate.get("vector_top") if k == row.get("gate_top", 8) else None
    if name == "jev":
        scored = [c for c in cands if c.get("jev")]
        unscored = [c for c in cands if not c.get("jev")]  # kept unscored, as the live pipeline does
        inc = [c for c in scored if route_passage(c["jev"], cfg) == "include"] + unscored
        con = [c for c in scored if route_passage(c["jev"], cfg) == "conflict"]
        inc.sort(key=evidence_order)
        return inc[:cfg.max_passages] + con, (gate.get("jev_default") if cfg.select == "threshold" else gate.get("jev_rank"))
    if name == "jev-rerank":
        ranked = sorted(cands, key=evidence_order)
        return ranked[:top or 8], None
    if name == "llm-rerank":
        ranked = sorted(cands, key=lambda c: (-_num(c.get("llm"), -1.0), -c["vs"]))
        return ranked[:top or 8], None
    raise ValueError(f"unknown system {system!r}")


def evaluate(rows: Sequence[Mapping[str, Any]], systems: Sequence[str], *, qideal: Mapping[str, Mapping[str, int]],
             cfg: Optional[ClassifyConfig] = None, gate_min: float = 0.35, k: int = 10) -> Dict[str, Dict[str, Any]]:
    """Metrics per system. ``qideal`` maps query id -> labelled units (see ``ideal_units``)."""
    cfg = cfg or ClassifyConfig()
    out: Dict[str, Dict[str, Any]] = {}
    for system in systems:
        if system.startswith("llm-rerank") and not any(c.get("llm") is not None for r in rows for c in r["cands"]):
            continue
        if system.startswith("jev") and not any(c.get("jev") for r in rows for c in r["cands"]):
            continue
        per: List[Dict[str, Any]] = []
        for row in rows:
            chosen, gate_p = select(row, system, cfg)
            s: Dict[str, Any] = {"answerable": row["answerable"], "tokens": sum(c["tokens"] for c in chosen)}
            if row["answerable"]:
                s.update(score_ranking([c["units"] for c in chosen], _ideal(row, qideal), k=k))
            else:
                s["returned"] = len(chosen)
            if gate_p is not None:
                s["abstain"] = gate_p < gate_min
            per.append(s)
        out[system] = summarise(per, k=k)
    return out


def bootstrap_delta(rows: Sequence[Mapping[str, Any]], a: str, b: str, *, qideal: Mapping[str, Mapping[str, int]],
                    metric: str = "recall", cfg: Optional[ClassifyConfig] = None, k: int = 10,
                    n: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap of mean(metric[a] - metric[b]) over answerable queries: point estimate and 95% interval."""
    cfg = cfg or ClassifyConfig()
    diffs: List[float] = []
    for row in rows:
        if not row["answerable"]:
            continue
        sa = score_ranking([c["units"] for c in select(row, a, cfg)[0]], _ideal(row, qideal), k=k)
        sb = score_ranking([c["units"] for c in select(row, b, cfg)[0]], _ideal(row, qideal), k=k)
        diffs.append(float(sa[metric]) - float(sb[metric]))
    if not diffs:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return {"delta": round(sum(diffs) / len(diffs), 4), "lo": round(means[int(0.025 * n)], 4),
            "hi": round(means[int(0.975 * n)], 4), "n": len(diffs)}


def tune(rows: Sequence[Mapping[str, Any]], *, qideal: Mapping[str, Mapping[str, int]], k: int = 10,
         recall_floor: Optional[float] = None, max_recall_drop: float = 0.02,
         base: Optional[ClassifyConfig] = None) -> Dict[str, Any]:
    """Grid-search classification thresholds on (dev) rows.

    Objective: the fewest context tokens subject to recall@k no more than
    ``max_recall_drop`` below the vector top-``max_passages`` baseline (or
    ``recall_floor`` when given). Ties go to higher nDCG.
    """
    base = base or ClassifyConfig(select="threshold", max_passages=8)  # the grid below tunes threshold mode
    vec = evaluate(rows, [f"vector@{base.max_passages}"], qideal=qideal, k=k)[f"vector@{base.max_passages}"]
    floor = recall_floor if recall_floor is not None else (vec[f"recall@{k}"] or 0.0) - max_recall_drop
    grid = {"min_relevant": [0.2, 0.35, 0.5, 0.65], "min_evidence": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "max_passages": [3, 5, 8, 10]}
    best: Optional[Tuple[Tuple[float, float], ClassifyConfig, Dict[str, Any]]] = None
    tried = []
    for mr, me, mp in itertools.product(grid["min_relevant"], grid["min_evidence"], grid["max_passages"]):
        cfg = replace(base, min_relevant=mr, min_evidence=me, max_passages=mp)
        m = evaluate(rows, ["jev"], qideal=qideal, cfg=cfg, k=k)["jev"]
        ok = (m[f"recall@{k}"] or 0.0) >= floor
        tried.append({"min_relevant": mr, "min_evidence": me, "max_passages": mp, "ok": ok,
                      f"recall@{k}": m[f"recall@{k}"], "tokens": m["context_tokens_mean"], f"ndcg@{k}": m[f"ndcg@{k}"]})
        if not ok:
            continue
        key = (m["context_tokens_mean"] or 0.0, -(m[f"ndcg@{k}"] or 0.0))
        if best is None or key < best[0]:
            best = (key, cfg, m)
    return {"floor": round(floor, 4), "vector_baseline": vec,
            "best": None if best is None else {"config": {f: getattr(best[1], f) for f in grid}, "metrics": best[2]},
            "tried": tried}


def tune_gate(rows: Sequence[Mapping[str, Any]], *, max_false_abstain: float = 0.05) -> Dict[str, Any]:
    """The gate threshold that maximises correct abstentions with false abstentions at or under the cap."""
    ans = [r["gate"]["jev_default"] for r in rows if r["answerable"] and (r.get("gate") or {}).get("jev_default") is not None]
    una = [r["gate"]["jev_default"] for r in rows if not r["answerable"] and (r.get("gate") or {}).get("jev_default") is not None]
    best = None
    curve = []
    for t in [x / 100 for x in range(5, 96, 5)]:
        fa = sum(p < t for p in ans) / len(ans) if ans else 0.0
        ta = sum(p < t for p in una) / len(una) if una else 0.0
        curve.append({"threshold": t, "false_abstain": round(fa, 4), "true_abstain": round(ta, 4)})
        if fa <= max_false_abstain and (best is None or ta > best["true_abstain"] or
                                        (ta == best["true_abstain"] and fa < best["false_abstain"])):
            best = curve[-1]
    return {"best": best, "curve": curve, "answerable": len(ans), "unanswerable": len(una)}


def config_dict(cfg: ClassifyConfig) -> Dict[str, Any]:
    return asdict(cfg)


__all__ = ["select", "evaluate", "bootstrap_delta", "tune", "tune_gate", "config_dict"]
