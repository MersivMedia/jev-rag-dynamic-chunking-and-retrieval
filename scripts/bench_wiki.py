"""Larger-scale benchmark: long Wikipedia articles, many generated questions.

Stages (each resumable; outputs go to .jevrag/bench/ by default):

    python scripts/bench_wiki.py build    # fetch articles, generate questions with verbatim evidence
    python scripts/bench_wiki.py ingest   # ingest into one collection per chunker config
    python scripts/bench_wiki.py query    # run every question through vector-only and full Jev retrieval
    python scripts/bench_wiki.py report   # metrics -> report.json + report.md

Design
------
* Ingested corpus: long articles (thousands of words each).
* Answerable questions: an LLM reads one paragraph sampled across the article (start, middle,
  end) and writes a self-contained question plus a VERBATIM evidence span. Spans are checked
  against the source; questions whose span isn't verbatim are discarded. A retrieval "hit" is
  mechanical: a returned passage contains the evidence span (fuzzy match >= 0.8 of its length).
* Unanswerable questions: generated the same way from held-out articles on related topics that
  are NOT ingested, so a correct system should abstain. Non-abstentions are judged by an LLM
  (do the returned passages actually answer it?) to separate gate misses from overlap.
* Configs: jev chunking + enrichment (the default), structural, fixed (both without enrichment).
  Retrieval modes: `vector` (top 8 by similarity, no Jev) and `jev` (route + classify + gate).

Keys: a Jev key and AI_GATEWAY_API_KEY (embeddings + question generation) from the environment
or ./.env. Needs a store; default is pgvector at $DATABASE_URL, `--store qdrant-local` for embedded.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import functools
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevrag import Document, Pipeline  # noqa: E402
from jevrag.chunk import ChunkConfig  # noqa: E402
from jevrag.enrich import EnrichConfig  # noqa: E402
from jevrag.envfile import load_env_file  # noqa: E402
from jevrag.jev import JevConfig  # noqa: E402
from jevrag.retrieve import ClassifyConfig, GateConfig, RetrieveConfig, RouteConfig  # noqa: E402
from jevrag.tokens import estimate_tokens  # noqa: E402

INGEST = ["Apollo 11", "Python (programming language)", "French Revolution", "Mount Everest", "Transistor",
          "Great Depression", "Photosynthesis", "CRISPR gene editing", "Tardigrade", "Byzantine Empire"]
HOLDOUT = ["Apollo 12", "Ruby (programming language)", "Russian Revolution", "K2", "Vacuum tube",
           "Chemosynthesis"]
UA = "jevrag-bench/1.0 (https://github.com/MersivMedia/jev-rag-retrieval)"
GATEWAY = "https://ai-gateway.vercel.sh/v1/chat/completions"
QGEN_MODEL = "openai/gpt-4.1-mini"
EMBEDDER = "gateway:openai/text-embedding-3-small"
CONFIGS = {
    "jev": dict(chunking=ChunkConfig(method="jev"), enrich=EnrichConfig(mode="on")),
    "structural": dict(chunking=ChunkConfig(method="structural"), enrich=EnrichConfig(mode="off")),
    "fixed": dict(chunking=ChunkConfig(method="fixed"), enrich=EnrichConfig(mode="off")),
    # controls: same screening + enrichment as "jev", different chunker; separates the chunker's effect
    "structural_screen": dict(chunking=ChunkConfig(method="structural"), enrich=EnrichConfig(mode="on")),
    "fixed_screen": dict(chunking=ChunkConfig(method="fixed"), enrich=EnrichConfig(mode="on")),
}
TOP_N = 8  # passages handed to the LLM in vector mode; equals classify.max_passages


# ---------------------------------------------------------------------------------------------
# helpers

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")).strip().lower()


@functools.lru_cache(maxsize=200_000)
def _norm_cached(s: str) -> str:
    return norm(s)


def evidence_in(evidence: str, text: str, ratio: float = 0.8) -> bool:
    """True if ``text`` contains the evidence span (one contiguous match >= ratio of its length)."""
    e, t = _norm_cached(evidence), _norm_cached(text)
    if not e:
        return False
    if e in t:
        return True
    # cheap filter before the quadratic matcher: a match covering `ratio` of the span must share
    # most of its words, so skip texts that don't
    ew = set(e.split())
    if len(ew & set(t.split())) < ratio * 0.8 * len(ew):
        return False
    m = difflib.SequenceMatcher(None, e, t, autojunk=False).find_longest_match(0, len(e), 0, len(t))
    return m.size >= ratio * len(e)


def pct(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)


async def chat_json(client: httpx.AsyncClient, prompt: str) -> Dict[str, Any]:
    key = os.environ["AI_GATEWAY_API_KEY"]
    for attempt in range(4):
        r = await client.post(GATEWAY, headers={"Authorization": f"Bearer {key}"}, json={
            "model": QGEN_MODEL, "temperature": 0.2, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]}, timeout=60)
        if r.status_code in (429, 500, 502, 503, 529):
            await asyncio.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])
    raise RuntimeError(f"question model kept failing: {r.status_code}")


def to_markdown(extract: str, title: str) -> str:
    """Wikipedia plain-text extract -> Markdown (== H == headings become #...)."""
    out = [f"# {title}", ""]
    for line in extract.splitlines():
        m = re.match(r"^(=+)\s*(.+?)\s*=+\s*$", line)
        if m:
            out += ["", "#" * min(6, len(m.group(1))) + " " + m.group(2), ""]
        elif line.strip():
            out += [line.strip(), ""]
    return "\n".join(out)


def paragraphs(md: str) -> List[str]:
    return [p for p in md.split("\n\n") if not p.startswith("#") and len(p.split()) >= 60]


SKIP_SECTIONS = re.compile(r"^#+ (See also|References|Notes|Further reading|External links|Bibliography|"
                           r"Sources|Citations|Footnotes)\b", re.M)


# ---------------------------------------------------------------------------------------------
# build

async def fetch(client: httpx.AsyncClient, title: str) -> Dict[str, Any]:
    r = await client.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query", "prop": "extracts|info", "explaintext": 1, "inprop": "url", "redirects": 1,
        "titles": title, "format": "json", "formatversion": 2}, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    p = r.json()["query"]["pages"][0]
    md = to_markdown(p["extract"], p["title"])
    cut = SKIP_SECTIONS.search(md)
    if cut:
        md = md[:cut.start()].rstrip() + "\n"
    return {"title": p["title"], "url": p["fullurl"], "revid": p.get("lastrevid"), "markdown": md,
            "words": len(md.split())}


QPROMPT = """You write test questions for a search system. Below is one paragraph from the Wikipedia
article "{title}".

Write ONE factual question that this paragraph answers. Rules:
- The question must stand alone: name the subject explicitly. Never say "this paragraph",
  "the text", "the article" or "according to".
- Ask about a specific fact (a number, date, name, cause or outcome), not a summary.
- Do not copy a whole sentence from the paragraph into the question.
Then copy the SHORTEST exact span from the paragraph (8 to 40 words, character-for-character)
that contains the answer.

Return JSON: {{"question": "...", "answer": "short answer", "evidence": "exact span"}}

Paragraph:
{para}"""


async def make_questions(client: httpx.AsyncClient, art: Dict[str, Any], n: int, answerable: bool,
                         rng: random.Random) -> List[Dict[str, Any]]:
    paras = paragraphs(art["markdown"])
    if not paras:
        return []
    # stratify across the article so late sections are tested, not just the lead
    idx = sorted({min(len(paras) - 1, int((i + rng.random()) * len(paras) / n)) for i in range(n)})
    sem = asyncio.Semaphore(4)

    async def one(i: int) -> Optional[Dict[str, Any]]:
        async with sem:
            for _ in range(2):
                q = await chat_json(client, QPROMPT.format(title=art["title"], para=paras[i]))
                ev = str(q.get("evidence", ""))
                if ev and norm(ev) in norm(paras[i]) and len(ev.split()) >= 5:
                    return {"question": q["question"].strip(), "answer": str(q.get("answer", "")),
                            "evidence": ev, "doc": art["title"], "para_index": i, "n_paras": len(paras),
                            "answerable": answerable}
        return None

    return [q for q in await asyncio.gather(*(one(i) for i in idx)) if q]


async def cmd_build(a: argparse.Namespace) -> None:
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    async with httpx.AsyncClient() as client:
        arts = {}
        for t in INGEST + HOLDOUT:
            arts[t] = await fetch(client, t)
            print(f"fetched {arts[t]['title']:<34} {arts[t]['words']:>6} words")
        qs: List[Dict[str, Any]] = []
        for t in INGEST:
            qs += await make_questions(client, arts[t], a.per_doc, True, rng)
        for t in HOLDOUT:
            qs += await make_questions(client, arts[t], a.per_holdout, False, rng)
    for i, q in enumerate(qs):
        q["id"] = f"q{i:03d}"
    data = {"ingest": [arts[t] for t in INGEST], "holdout": [{k: v for k, v in arts[t].items() if k != "markdown"}
                                                             for t in HOLDOUT],
            "questions": qs, "question_model": QGEN_MODEL, "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    (out / "dataset.json").write_text(json.dumps(data, indent=1))
    na = sum(q["answerable"] for q in qs)
    print(f"questions: {na} answerable, {len(qs) - na} unanswerable -> {out / 'dataset.json'}")


# ---------------------------------------------------------------------------------------------
# pipelines

def store_spec(a: argparse.Namespace, cfg: str) -> Dict[str, Any]:
    if a.store == "qdrant-local":
        return {"kind": "qdrant", "path": str(Path(a.dir) / f"qdrant_{cfg}")}
    return {"kind": "pgvector", "dsn_env": "DATABASE_URL"}


def pipeline(a: argparse.Namespace, cfg: str, cache: bool = True) -> Pipeline:
    return Pipeline(store=store_spec(a, cfg), embedder=EMBEDDER,
                    # one cache per config: a shared cache would reuse answers for chunks that two
                    # chunkers cut identically and understate the later configs' cost and latency
                    jev=JevConfig(cache_dir=None if cache is False else str(Path(a.dir) / f"cache_{cfg}")),
                    retrieve=RetrieveConfig(top_k=30, route=RouteConfig(mode="off"),
                                            classify=ClassifyConfig(max_passages=TOP_N), gate=GateConfig()),
                    trace_dir=str(Path(a.dir) / "traces" / cfg), **CONFIGS[cfg])


def docs(data: Dict[str, Any]) -> List[Document]:
    return [Document(text=d["markdown"], doc_id=d["url"], title=d["title"], source_uri=d["url"])
            for d in data["ingest"]]


async def cmd_ingest(a: argparse.Namespace) -> None:
    data = json.loads((Path(a.dir) / "dataset.json").read_text())
    results = {}
    for cfg in a.configs:
        rag = pipeline(a, cfg)
        est = await rag.aingest(docs(data), f"bench_{cfg}", dry_run=True)
        print(f"[{cfg}] dry run estimate: {est.estimate}")
        t0 = time.monotonic()
        rep = await rag.aingest(docs(data), f"bench_{cfg}", concurrency=a.concurrency)
        wall = time.monotonic() - t0
        print(rep.summary())
        results[cfg] = {"report": rep.to_dict(), "seconds": round(wall, 1), "estimate": est.estimate}
        rag.close()
    path = Path(a.dir) / "ingest.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(results)
    path.write_text(json.dumps(prev, indent=1, default=str))


# ---------------------------------------------------------------------------------------------
# query

def passage_dict(p: Any) -> Dict[str, Any]:
    return {"id": p.id, "doc_id": p.doc_id, "chunk_index": p.chunk_index, "vector_score": round(p.vector_score, 4),
            "scores": {k: round(v, 3) for k, v in (p.scores or {}).items()}, "reason": p.reason, "text": p.text}


async def cmd_query(a: argparse.Namespace) -> None:
    data = json.loads((Path(a.dir) / "dataset.json").read_text())
    qs = data["questions"][: a.limit] if a.limit else data["questions"]
    for cfg in a.configs:
        for mode in a.modes:
            await run_queries(a, qs, cfg, mode)


async def run_queries(a: argparse.Namespace, qs: List[Dict[str, Any]], cfg: str, mode: str,
                      collection: Optional[str] = None) -> None:
    collection = collection or f"bench_{cfg}"
    path = Path(a.dir) / f"runs_{cfg}_{mode}.jsonl"
    done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
    todo = [q for q in qs if q["id"] not in done]
    if not todo:
        print(f"[{cfg}/{mode}] all {len(qs)} done")
        return
    rag = pipeline(a, cfg)
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    n = {"done": 0}

    async def one(q: Dict[str, Any]) -> None:
        async with sem:
            t0 = time.monotonic()
            try:
                if mode == "vector":
                    res = await vector_only(rag, q["question"], collection)
                else:
                    res = await rag.aretrieve(q["question"], collection)
                row = {"id": q["id"], "ms": round((time.monotonic() - t0) * 1000, 1),
                       "abstain": res.abstain, "reason": res.reason, "gate_p": res.gate_p,
                       "answerable_p": res.answerable_p, "degraded": res.degraded,
                       "passages": [passage_dict(p) for p in res.passages],
                       "conflicts": [passage_dict(p) for p in res.conflicts],
                       "dropped": [passage_dict(p) for p in res.dropped],
                       "jev": res.trace.get("jev") if isinstance(res.trace.get("jev"), dict) else None,
                       "stages": {k: v for k, v in res.trace.items() if k not in ("jev", "jev_model")}}
            except Exception as exc:
                row = {"id": q["id"], "error": f"{type(exc).__name__}: {exc}"}
            async with lock:
                with path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                n["done"] += 1
                if n["done"] % 25 == 0:
                    print(f"[{cfg}/{mode}] {n['done']}/{len(todo)}", flush=True)

    await asyncio.gather(*(one(q) for q in todo))
    print(f"[{cfg}/{mode}] finished {len(todo)} queries", flush=True)
    rag.close()


async def vector_only(rag: Pipeline, query: str, collection: str) -> Any:
    """Plain dense retrieval: top TOP_N by similarity, no Jev at all."""
    from jevrag.retrieve.result import Passage, RetrievalResult
    vec = await rag.embedder.embed_query(query)
    hits = await asyncio.to_thread(rag.store.query, collection, vec, {"ne": {"quarantined": True}}, TOP_N)
    res = RetrievalResult(query=query)
    for h in hits:
        m = h.metadata
        res.passages.append(Passage(id=h.id, text=h.text, doc_id=m.get("doc_id", ""),
                                    source_uri=m.get("source_uri", ""), title=m.get("title", ""),
                                    section_path=[], chunk_index=int(m.get("chunk_index", 0)),
                                    vector_score=h.score))
    return res


async def cmd_latency(a: argparse.Namespace) -> None:
    """One query at a time, cache off: per-query latency without our own rate limiter queueing."""
    data = json.loads((Path(a.dir) / "dataset.json").read_text())
    rng = random.Random(11)
    qs = rng.sample(data["questions"], min(a.latency_n, len(data["questions"])))
    out: Dict[str, Any] = {}
    for cfg in a.configs:
        rag = pipeline(a, cfg, cache=False)
        rows = []
        for q in qs:
            t0 = time.monotonic()
            res = await rag.aretrieve(q["question"], f"bench_{cfg}")
            st = res.trace
            rows.append({"id": q["id"], "ms": round((time.monotonic() - t0) * 1000, 1),
                         "stages_ms": {**{k: v for k, v in st.items() if k.endswith("_ms")},
                                       **{k: v["ms"] for k, v in st.items() if isinstance(v, dict) and "ms" in v}},
                         "jev": res.trace.get("jev")})
        rag.close()
        ms = [r["ms"] for r in rows]
        out[cfg] = {"n": len(rows), "p50": pct(ms, 0.5), "p90": pct(ms, 0.9), "max": max(ms),
                    "cached": sum((r["jev"] or {}).get("cached", 0) for r in rows), "rows": rows}
        print(f"[{cfg}] sequential, uncached: p50 {out[cfg]['p50']} ms  p90 {out[cfg]['p90']} ms  "
              f"max {out[cfg]['max']} ms  (cached answers: {out[cfg]['cached']})", flush=True)
    (Path(a.dir) / "latency.json").write_text(json.dumps(out, indent=1, default=str))


# ---------------------------------------------------------------------------------------------
# report

JUDGE = """Question: {q}

Passages:
{p}

Does any passage state the answer to the question? Reply with JSON {{"answers": true or false,
"why": "one short sentence"}}. Answer true only if the passage actually states the specific fact asked."""


async def judge_all(items: List[Dict[str, Any]], cache_path: Path) -> Dict[str, Dict[str, Any]]:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [it for it in items if it["key"] not in cache]
    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient() as client:
        async def one(it: Dict[str, Any]) -> None:
            async with sem:
                ptxt = "\n\n".join(f"[{i + 1}] {t[:2500]}" for i, t in enumerate(it["texts"])) or "(none)"
                cache[it["key"]] = await chat_json(client, JUDGE.format(q=it["question"], p=ptxt))
        await asyncio.gather(*(one(it) for it in todo))
    cache_path.write_text(json.dumps(cache, indent=1))
    return cache


def load_runs(d: Path, cfg: str, mode: str) -> Dict[str, Dict[str, Any]]:
    p = d / f"runs_{cfg}_{mode}.jsonl"
    rows: Dict[str, Dict[str, Any]] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            rows[r["id"]] = r  # last write wins on reruns
    return rows


async def cmd_report(a: argparse.Namespace) -> None:
    d = Path(a.dir)
    data = json.loads((d / "dataset.json").read_text())
    ingest = json.loads((d / "ingest.json").read_text()) if (d / "ingest.json").exists() else {}
    qs = {q["id"]: q for q in data["questions"]}
    rep: Dict[str, Any] = {"dataset": {
        "articles": [{"title": x["title"], "words": x["words"], "revid": x["revid"]} for x in data["ingest"]],
        "words_total": sum(x["words"] for x in data["ingest"]),
        "holdout": [x["title"] for x in data["holdout"]],
        "answerable": sum(q["answerable"] for q in qs.values()),
        "unanswerable": sum(not q["answerable"] for q in qs.values()),
        "question_model": data["question_model"], "built": data["built"]}, "ingest": {}, "retrieval": {}}

    for cfg, ing in ingest.items():
        r = ing["report"]
        chunks = [x["chunks"] for x in r["docs"]]
        rep["ingest"][cfg] = {"seconds": ing["seconds"], "chunks": sum(chunks), "jev": r.get("jev"),
                              "dropped": sum(x.get("dropped", 0) for x in r["docs"]),
                              "quarantined": sum(x.get("quarantined", 0) for x in r["docs"]),
                              "fallbacks": [x["doc_id"] for x in r["docs"] if x.get("fallback")],
                              "failed": [x["doc_id"] for x in r["docs"] if x["status"] == "failed"]}

    judge_items = []
    for cfg in a.configs:
        rag = pipeline(a, cfg)
        # evidence coverage: is each answerable evidence span present in some stored chunk at all?
        coll = f"bench_{cfg}"
        ids = await asyncio.to_thread(rag.store.list_ids, coll, None, 1_000_000)
        stored = [r.text for r in await asyncio.to_thread(rag.store.get, coll, ids)]
        rag.close()
        stored_tokens = [estimate_tokens(t) for t in stored]
        covered = {qid: any(evidence_in(q["evidence"], t) for t in stored)
                   for qid, q in qs.items() if q["answerable"]}
        rep["ingest"].setdefault(cfg, {})["chunk_tokens"] = {
            "n": len(stored), "mean": round(statistics.mean(stored_tokens), 1) if stored else None,
            "p10": pct(stored_tokens, 0.1), "p50": pct(stored_tokens, 0.5), "p90": pct(stored_tokens, 0.9),
            "max": max(stored_tokens) if stored else None}
        rep["ingest"][cfg]["evidence_coverage"] = round(sum(covered.values()) / max(1, len(covered)), 3)
        rep["ingest"][cfg]["evidence_not_stored"] = [qid for qid, ok in covered.items() if not ok]

        for mode in a.modes:
            runs = load_runs(d, cfg, mode)
            if not runs:
                continue
            ans = [r for qid, r in runs.items() if qs[qid]["answerable"] and "error" not in r]
            un = [r for qid, r in runs.items() if not qs[qid]["answerable"] and "error" not in r]
            errors = [qid for qid, r in runs.items() if "error" in r]

            def ctx(r: Dict[str, Any]) -> List[str]:
                return [p["text"] for p in r["passages"] + r["conflicts"]]

            def hit(r: Dict[str, Any]) -> bool:
                return (not r["abstain"]) and any(evidence_in(qs[r["id"]]["evidence"], t) for t in ctx(r))

            def rank(r: Dict[str, Any]) -> Optional[int]:
                for i, p in enumerate(r["passages"]):
                    if evidence_in(qs[r["id"]]["evidence"], p["text"]):
                        return i + 1
                return None

            ranks = [rank(r) for r in ans]
            ctx_tokens = [sum(estimate_tokens(t) for t in ctx(r)) for r in ans + un if not r["abstain"]]
            m: Dict[str, Any] = {
                "queries": len(runs), "errors": errors,
                "answerable_n": len(ans), "unanswerable_n": len(un),
                "hit_rate": round(sum(hit(r) for r in ans) / max(1, len(ans)), 3),
                "hit_at_1": round(sum(1 for x in ranks if x == 1) / max(1, len(ans)), 3),
                "mrr": round(sum(1 / x for x in ranks if x) / max(1, len(ans)), 3),
                "false_abstain": round(sum(r["abstain"] for r in ans) / max(1, len(ans)), 3),
                "true_abstain": round(sum(r["abstain"] for r in un) / max(1, len(un)), 3),
                "context_tokens_mean": round(statistics.mean(ctx_tokens), 1) if ctx_tokens else 0,
                "passages_mean": round(statistics.mean(len(r["passages"]) + len(r["conflicts"]) for r in ans + un), 2),
                "latency_ms": {"p50": pct([r["ms"] for r in runs.values() if "ms" in r], 0.5),
                               "p90": pct([r["ms"] for r in runs.values() if "ms" in r], 0.9)},
                "degraded": sum(1 for r in runs.values() if r.get("degraded")),
            }
            if mode == "jev":
                costs = [r["jev"]["cost_usd"] for r in runs.values() if r.get("jev") and "cost_usd" in r["jev"]]
                reqs = [r["jev"]["requests"] for r in runs.values() if r.get("jev")]
                m["jev_cost_per_query_usd"] = round(statistics.mean(costs), 6) if costs else None
                m["jev_requests_per_query"] = round(statistics.mean(reqs), 1) if reqs else None
                # where did evidence go? compare with the vector run on the same collection
                vec = load_runs(d, cfg, "vector")
                lost = [r["id"] for r in ans if not hit(r) and vec.get(r["id"]) and "error" not in vec[r["id"]]
                        and any(evidence_in(qs[r["id"]]["evidence"], t) for t in ctx(vec[r["id"]]))]
                m["lost_vs_vector"] = len(lost)
                m["lost_vs_vector_by"] = {
                    "gate_abstained": sum(1 for x in lost if runs[x]["abstain"]),
                    "classified_out": sum(1 for x in lost if not runs[x]["abstain"] and any(
                        evidence_in(qs[x]["evidence"], p["text"]) for p in runs[x]["dropped"])),
                }
                m["lost_vs_vector_by"]["not_in_top_k"] = m["lost_vs_vector"] - sum(m["lost_vs_vector_by"].values())
                gained = [r["id"] for r in ans if hit(r) and vec.get(r["id"]) and "error" not in vec[r["id"]]
                          and not any(evidence_in(qs[r["id"]]["evidence"], t) for t in ctx(vec[r["id"]]))]
                m["gained_vs_vector"] = len(gained)
                for r in un:
                    if not r["abstain"]:
                        judge_items.append({"key": f"{cfg}|{r['id']}", "question": qs[r["id"]]["question"],
                                            "texts": ctx(r)})
            rep["retrieval"][f"{cfg}/{mode}"] = m

    if judge_items:
        verdicts = await judge_all(judge_items, d / "judge_cache.json")
        for cfg in a.configs:
            key = f"{cfg}/jev"
            if key not in rep["retrieval"]:
                continue
            mine = [it for it in judge_items if it["key"].startswith(cfg + "|")]
            answered_ok = sum(1 for it in mine if verdicts.get(it["key"], {}).get("answers"))
            rep["retrieval"][key]["unanswerable_not_abstained"] = len(mine)
            rep["retrieval"][key]["of_which_corpus_actually_answers"] = answered_ok
            rep["retrieval"][key]["examples_not_abstained"] = [
                {"q": it["question"], "judge": verdicts.get(it["key"])} for it in mine[:6]]

    if (d / "latency.json").exists():
        lat = json.loads((d / "latency.json").read_text())
        rep["latency_sequential_uncached"] = {k: {kk: v[kk] for kk in ("n", "p50", "p90", "max", "cached")}
                                              for k, v in lat.items()}
    (d / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str)[:6000])


# ---------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "ingest", "query", "latency", "report"])
    ap.add_argument("--dir", default=".jevrag/bench")
    ap.add_argument("--store", default="pgvector", choices=["pgvector", "qdrant-local"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--modes", nargs="+", default=["vector", "jev"], choices=["vector", "jev"])
    ap.add_argument("--per-doc", type=int, default=12)
    ap.add_argument("--per-holdout", type=int, default=7)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    ap.add_argument("--latency-n", type=int, default=25)
    a = ap.parse_args()
    load_env_file(".env")
    asyncio.run({"build": cmd_build, "ingest": cmd_ingest, "query": cmd_query, "latency": cmd_latency,
                 "report": cmd_report}[a.stage](a))


if __name__ == "__main__":
    main()
