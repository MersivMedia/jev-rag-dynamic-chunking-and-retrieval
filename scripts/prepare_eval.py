"""Prepare eval samples as jsonl datasets (corpus.jsonl + queries.jsonl with a split field).

SciFact: full corpus; test = BEIR test qrels (300), dev = 100 sampled from train qrels.
FiQA:    test = 300 sampled from test qrels, dev = 100 from dev qrels; corpus = every
         labelled doc for those queries + 10,000 random distractors.
QASPER:  100 papers sampled from the test file; questions from 30 papers -> dev,
         70 papers -> test. Every question is scoped to its own paper.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "jev-rag-retrieval"))
from jev_retrieval.eval import load_beir, load_qasper  # noqa: E402

RAW = Path.home() / "jev-rag-retrieval/.jev-retrieval/eval_data"
OUT = RAW / "prepared"


def write(name, corpus, queries):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "corpus.jsonl").write_text("\n".join(json.dumps(c) for c in corpus) + "\n")
    (d / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    words = sum(len(c["text"].split()) for c in corpus)
    splits = {s: sum(q["split"] == s for q in queries) for s in ("dev", "test")}
    una = sum(not q.get("answerable", True) for q in queries)
    print(f"{name:<8} docs={len(corpus):>6} words={words:>9} queries={splits} unanswerable={una}")


def beir(name, dev_split, n_test, n_dev, distractors, seed):
    test = load_beir(str(RAW / name), split="test", max_queries=n_test, distractors=0, seed=seed)
    dev = load_beir(str(RAW / name), split=dev_split, max_queries=n_dev, distractors=0, seed=seed + 1)
    test_ids = {q.id for q in test.queries}
    dev.queries = [q for q in dev.queries if q.id not in test_ids]
    needed = {d for q in test.queries + dev.queries for d in q.relevant}
    full = {}
    with open(RAW / name / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c.get("text"):
                full[str(c["_id"])] = c
    missing = needed - set(full)
    if missing:  # FiQA ships a few labelled posts with empty text: unretrievable, so drop those labels
        for q in test.queries + dev.queries:
            q.relevant = {d: g for d, g in q.relevant.items() if d not in missing}
        before = len(test.queries) + len(dev.queries)
        test.queries = [q for q in test.queries if q.relevant]
        dev.queries = [q for q in dev.queries if q.relevant]
        print(f"{name}: dropped {len(missing)} labels pointing at empty docs {sorted(missing)}; "
              f"{before - len(test.queries) - len(dev.queries)} queries lost all labels")
        needed -= missing
    keep = set(full) if distractors is None else needed | set(
        random.Random(seed).sample(sorted(set(full) - needed), distractors))
    corpus = [{"id": i, "title": full[i].get("title", ""), "text": full[i]["text"], "format": "text"} for i in sorted(keep)]
    queries = [{"id": q.id, "query": q.text, "relevant": q.relevant, "split": s}
               for s, qs in (("test", test.queries), ("dev", dev.queries)) for q in qs]
    write(name, corpus, queries)


beir("scifact", "train", 300, 100, None, 11)
beir("fiqa", "dev", 300, 100, 10_000, 12)

qa = load_qasper(str(RAW / "qasper/qasper-test-v0.3.json"), max_papers=100, seed=13)
pids = sorted(str(d.doc_id) for d in qa.docs)
dev_papers = set(random.Random(14).sample(pids, 30))
corpus = [{"id": d.doc_id, "title": d.title, "text": d.text, "format": "markdown"} for d in qa.docs]
queries = [{"id": q.id, "query": q.text, "evidence": q.evidence, "answerable": q.answerable, "scope": q.scope,
            "type": q.meta.get("type"), "split": "dev" if q.scope in dev_papers else "test"} for q in qa.queries]
write("qasper", corpus, queries)
