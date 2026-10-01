"""``jev-retrieval`` command line: init, check, ingest, query, inspect, delete, eval."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

from . import __version__
from .config import DEFAULT_FILE, STORE_TEMPLATES, Config, render_template


def _pipeline(args: argparse.Namespace) -> Any:
    from .pipeline import Pipeline
    if not Path(args.config).exists():
        raise FileNotFoundError(f"{args.config} not found: run `jev-retrieval init` first, or pass -c path/to/jev-retrieval.yaml")
    cfg = Config.load(args.config)
    if not cfg.embedder_spec:
        raise ValueError(f"{args.config} has no embedder.model")
    return Pipeline.from_config(cfg), cfg


def _emit(args: argparse.Namespace, obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


# -- commands -----------------------------------------------------------------------

def _template(name: str) -> str:
    from importlib.resources import files
    return files("jev_retrieval").joinpath("templates", name).read_text(encoding="utf-8")


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.config)
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    if args.full:
        path.write_text(_template("jev-retrieval.example.yaml"))
        print(f"wrote {path} (every setting, with defaults; pick one store block)")
    else:
        path.write_text(render_template(args.store, args.embedder))
        print(f"wrote {path} (store: {args.store}, embedder: {args.embedder})")
    env = Path(args.env_file or ".env")
    if env.exists():
        print(f"kept existing {env}")
    else:
        env.write_text(_template("env.example"))
        try:
            env.chmod(0o600)
        except OSError:
            pass
        print(f"wrote {env} (blank; fill in a Jev key and your embedding key, keep it out of git)")
    print("next: `jev-retrieval check`")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    rag, cfg = _pipeline(args)
    ok = True

    async def run() -> None:
        nonlocal ok
        from .jev import JevClient, JevError, Noul
        c = JevClient(cfg.jev)
        header = False
        try:
            r = c.resolved
            print(f"jev       backend={r['backend']} model={r['model']} key={r['api_key_env']}"
                  f"{'' if r['api_key'] else ' (NOT SET)'}")
            header = True
            async with c:
                resp = await c.ask("Refresh tokens expire after 14 days.",
                                   {"t": Noul("Does this text state a time limit?")})
            print(f"          ok: answered {resp.answers['t'].value:.2f} in {resp.latency_ms:.0f} ms"
                  f"{' (cached)' if resp.cached else ''}")
        except JevError as exc:
            ok = False
            print(f"{'' if header else 'jev':<10}FAILED: {exc}")
        try:
            v = await rag.embedder.embed_query("hello")
            print(f"embedder  {rag.embedder.spec}: ok, dimension {len(v)}")
        except Exception as exc:
            ok = False
            print(f"embedder  {rag.embedder.spec}: FAILED: {exc}")
        await rag.embedder.aclose()

    asyncio.run(run())
    try:
        from .types import Record
        name = "jev_retrieval_check"
        rag.store.ensure_collection(name, 3, "cosine", {"doc_id": "keyword"})
        rag.store.upsert(name, [Record("00000000-0000-0000-0000-000000000001", [1.0, 0.0, 0.0], "check",
                                       {"doc_id": "check"})])
        hits = rag.store.query(name, [1.0, 0.0, 0.0], None, 1)
        try:
            rag.store.drop_collection(name)
        except NotImplementedError:
            rag.store.delete(name, ids=["00000000-0000-0000-0000-000000000001"])
        print(f"store     {rag.store.kind}: ok, round trip score {hits[0].score:.2f}" if hits else
              f"store     {rag.store.kind}: FAILED (no hit)")
        ok = ok and bool(hits)
    except Exception as exc:
        ok = False
        print(f"store     {rag.store.kind}: FAILED: {exc}")
    return 0 if ok else 1


def cmd_ingest(args: argparse.Namespace) -> int:
    rag, _ = _pipeline(args)
    try:
        rep = rag.ingest(args.paths, collection=args.collection, dry_run=args.dry_run, limit=args.limit,
                         force=args.force)
    finally:
        rag.close()
    if args.json:
        _emit(args, rep.to_dict())
    else:
        print(rep.summary())
        if args.verbose:
            for d in rep.docs:
                print(f"  {d.status:18} {d.chunks:4} chunks  {d.dropped:3} dropped  {d.quarantined:2} quar  "
                      f"{d.chunker or '-':10}  {d.doc_id}")
                for x in d.dropped_detail:
                    print(f"      dropped [{x['reason']}] {x.get('text', '')[:100]!r}")
    return 1 if rep.count("failed") else 0


def _where(arg: Optional[str]) -> Optional[dict]:
    return json.loads(arg) if arg else None


def cmd_query(args: argparse.Namespace) -> int:
    rag, _ = _pipeline(args)
    if args.top_k:
        rag.retrieve_cfg.top_k = args.top_k
    if args.no_route:
        rag.retrieve_cfg.route.mode = "off"
    try:
        if args.answer:
            ans = rag.answer(args.query, collection=args.collection, where=_where(args.where),
                             retrieve_only=args.retrieve_only)
            res = ans.retrieval
        else:
            ans = None
            res = rag.retrieve(args.query, collection=args.collection, where=_where(args.where),
                               retrieve_only=args.retrieve_only)
    finally:
        rag.close()
    if args.json:
        out = res.to_dict()
        if ans is not None:
            out["answer"] = {"text": ans.text, "abstained": ans.abstained, "citations": ans.citations}
        _emit(args, out)
        return 0
    print(f"query: {res.query}")
    print(f"filter: {json.dumps(res.filter_used) if res.filter_used else 'none'}   "
          f"gate: {res.gate_p if res.gate_p is not None else '-'}   abstain: {res.abstain}"
          f"{'   DEGRADED' if res.degraded else ''}")
    if res.abstain:
        print(f"  -> abstain: {res.reason}")
    for label, group in (("passage", res.passages), ("conflict", res.conflicts)):
        for i, p in enumerate(group, 1):
            ev = p.scores.get("contains_answer_evidence")
            print(f"\n[{label} {i}] evidence={ev if ev is not None else '-'} vector={p.vector_score:.3f}  "
                  f"{p.citation}")
            print("  " + p.text[:args.chars].replace("\n", "\n  "))
    if res.dropped and args.verbose:
        print("\ndropped:")
        for p in res.dropped:
            print(f"  [{p.reason}] {p.scores}  {p.text[:90]!r}")
    jt = res.trace.get("jev")
    if isinstance(jt, dict):
        print(f"\nJev: {jt['requests']} requests, {jt['input_tokens']} tokens, ${jt['cost_usd']:.5f}; "
              f"total {res.trace.get('total_ms')} ms")
    if ans is not None:
        print("\nanswer:\n" + ans.text)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    rag, _ = _pipeline(args)
    if args.collection:
        where = {"eq": {"quarantined": True}} if args.quarantined else None
        ids = rag.store.list_ids(args.collection, where, limit=args.limit)
        recs = rag.store.get(args.collection, ids)
        recs.sort(key=lambda r: (str(r.metadata.get("doc_id")), int(r.metadata.get("chunk_index", 0))))
        for r in recs:
            m = r.metadata
            print(f"{m.get('doc_id')} #{m.get('chunk_index')}  {m.get('tokens')}t  chunker={m.get('chunker')}  "
                  f"quarantined={m.get('quarantined')}  "
                  + "  ".join(f"{k}={v}" for k, v in sorted(m.items()) if k.startswith(("q_", "tag_"))))
            print("  " + r.text[:args.chars].replace("\n", "\n  "))
        print(f"\n{len(recs)} records")
        return 0
    if not args.path:
        print("give a file to chunk, or --collection to list stored records", file=sys.stderr)
        return 2
    methods = [None] + [m.strip() for m in (args.compare or "").split(",") if m.strip()]
    for method in methods:
        chunks, trace = asyncio.run(rag.achunk(args.path, method=method))
        print(f"== {trace.method_used} (requested {trace.method_requested}), {len(chunks)} chunks"
              f"{', fallback: ' + trace.fallback_reason if trace.fallback_reason else ''}")
        for c in chunks:
            print(f"-- #{c.chunk_index} {c.tokens}t  {' > '.join(c.section_path)}")
            print("   " + c.text[:args.chars].replace("\n", "\n   "))
        if args.verbose and trace.gaps:
            print("gap scores (cost: 0 = natural cut, 1 = keep together):")
            for g in trace.gaps:
                print(f"   {g}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    rag, _ = _pipeline(args)
    n = rag.delete_document(args.collection, args.doc)
    print(f"deleted {n} records for doc_id {args.doc!r} from {args.collection}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Record candidates for a labelled dataset, then report every system."""
    import asyncio
    import json as _json

    from .eval import ideal_units, load_dataset
    from .eval.llm_rerank import LLMReranker
    from .eval.runner import RecordOptions, load_recording, record
    from .eval.systems import bootstrap_delta, evaluate

    kw = {}
    if args.split:
        kw["split"] = args.split
    if args.max_queries is not None:
        kw["max_queries"] = args.max_queries
    if args.distractors is not None:
        kw["distractors"] = args.distractors
    if args.max_papers is not None:
        kw["max_papers"] = args.max_papers
    ds = load_dataset(args.dataset, **kw)
    out_dir = Path(args.out)
    collection = args.collection
    print(f"dataset   {_json.dumps(ds.summary())}")
    if not args.report_only:
        rag, _cfg = _pipeline(args)
        llm = LLMReranker(model=args.llm_rerank) if args.llm_rerank else None
        opt = RecordOptions(candidates=args.candidates, concurrency=args.concurrency, llm=llm, jev=not args.no_jev)
        if args.dry_run:
            est = rag.ingest(ds.docs, collection, dry_run=True).estimate
            n = len(ds.queries) if args.limit is None else min(args.limit, len(ds.queries))
            per_q = args.candidates + 3
            print(f"ingest    {_json.dumps(est)}")
            print(f"queries   {n} x ({args.candidates} classify + 3 gate) = {n * per_q} Jev requests"
                  + (f", {n} {args.llm_rerank} requests" if llm else ""))
            return 0
        try:
            meta = asyncio.run(record(rag, ds, collection, out_dir, opt, ingest=not args.no_ingest,
                                      limit=args.limit, progress=print))
        finally:
            rag.close()
        for key in ("ingest", "jev_usage", "llm_usage"):
            if meta.get(key):
                print(f"{key:<10}{_json.dumps(meta[key])}")
    rows = load_recording(out_dir, collection)
    qideal = {q.id: ideal_units(q, ds.level) for q in ds.queries}
    rows = [r for r in rows if r["id"] in qideal]
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    rep = evaluate(rows, systems, qideal=qideal, k=args.k)
    k = args.k
    cols = [f"recall@{k}", f"ndcg@{k}", "mrr", "hit@1", "passages_mean", "context_tokens_mean", "false_abstain", "true_abstain"]
    print(f"\n{len(rows)} recorded queries ({sum(r['answerable'] for r in rows)} answerable)")
    print(f"{'system':<16}" + "".join(f"{c.replace('_mean', ''):>15}" for c in cols))
    for name, m in rep.items():
        print(f"{name:<16}" + "".join(f"{'' if m.get(c) is None else m.get(c):>15}" for c in cols))
    if args.compare:
        a, b = args.compare.split(",")
        for metric in ("recall", "ndcg"):
            d = bootstrap_delta(rows, a, b, qideal=qideal, metric=metric, k=k)
            print(f"{a} - {b} {metric}@{k}: {d['delta']:+.4f} (95% CI {d['lo']:+.4f} to {d['hi']:+.4f}, n={d['n']})")
    if args.json_out:
        Path(args.json_out).write_text(_json.dumps({"dataset": ds.summary(), "systems": rep}, indent=1))
    return 0


# -- parser ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jev-retrieval", description="Jev-steered ingestion and retrieval")
    p.add_argument("--version", action="version", version=f"jev-retrieval {__version__}")
    p.add_argument("-c", "--config", default=DEFAULT_FILE, help=f"config file (default {DEFAULT_FILE})")
    p.add_argument("--env-file", default=None, help="load environment variables from this file (default ./.env)")
    p.add_argument("--no-env-file", action="store_true", help="don't load a .env file")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="write a starter jev-retrieval.yaml")
    s.add_argument("--store", default="memory", choices=sorted(STORE_TEMPLATES))
    s.add_argument("--embedder", default="openai:text-embedding-3-small")
    s.add_argument("--full", action="store_true", help="write every setting with comments (same as jev-retrieval.example.yaml)")
    s.add_argument("--force", action="store_true", help="overwrite an existing config (never overwrites .env)")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("check", help="one live Jev call, one embedding, one store round trip")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("ingest", help="parse, chunk, enrich, embed and store files or folders")
    s.add_argument("paths", nargs="+")
    s.add_argument("--collection", required=True)
    s.add_argument("--dry-run", action="store_true", help="estimate Jev requests and cost; no Jev or store writes")
    s.add_argument("--limit", type=int, help="only the first N documents")
    s.add_argument("--force", action="store_true", help="re-process unchanged documents")
    s.add_argument("--json", action="store_true")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("query", help="retrieve (and optionally answer) a question")
    s.add_argument("query")
    s.add_argument("--collection", required=True)
    s.add_argument("--where", help='JSON filter, e.g. \'{"eq": {"tag_product": "billing"}}\'')
    s.add_argument("--top-k", type=int)
    s.add_argument("--no-route", action="store_true")
    s.add_argument("--retrieve-only", action="store_true",
                   help="query a collection jev-retrieval didn't build (no manifest check, no routing)")
    s.add_argument("--answer", action="store_true", help="have the configured LLM answer from the passages")
    s.add_argument("--chars", type=int, default=400)
    s.add_argument("--json", action="store_true")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_query)

    s = sub.add_parser("inspect", help="show how a file would be chunked, or list stored records")
    s.add_argument("path", nargs="?")
    s.add_argument("--compare", help="also chunk with these methods, e.g. structural,fixed")
    s.add_argument("--collection")
    s.add_argument("--quarantined", action="store_true")
    s.add_argument("--limit", type=int, default=200)
    s.add_argument("--chars", type=int, default=300)
    s.add_argument("-v", "--verbose", action="store_true", help="print the score at every candidate cut")
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("delete", help="delete one document's records")
    s.add_argument("--collection", required=True)
    s.add_argument("--doc", required=True, help="doc_id")
    s.set_defaults(fn=cmd_delete)

    s = sub.add_parser("eval", help="measure retrieval systems on a labelled dataset")
    s.add_argument("dataset", help="beir:<dir>, qasper:<file> or jsonl:<file>")
    s.add_argument("--collection", required=True, help="collection to ingest the corpus into")
    s.add_argument("--out", default=".jev-retrieval/eval", help="where recordings go (default .jev-retrieval/eval)")
    s.add_argument("--split", help="BEIR qrels split or jsonl split field (default test for BEIR)")
    s.add_argument("--max-queries", type=int, help="sample this many queries (BEIR)")
    s.add_argument("--distractors", type=int, help="BEIR: keep relevant docs plus this many random others")
    s.add_argument("--max-papers", type=int, help="QASPER: sample this many papers")
    s.add_argument("--candidates", type=int, default=30, help="vector candidates per query (default 30)")
    s.add_argument("--llm-rerank", metavar="MODEL", help="also record an LLM re-ranking baseline, e.g. openai/gpt-4.1-mini")
    s.add_argument("--no-jev", action="store_true", help="record vector (and LLM) only")
    s.add_argument("--no-ingest", action="store_true", help="the collection is already ingested")
    s.add_argument("--limit", type=int, help="record only N more queries (try 1 before a batch)")
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--dry-run", action="store_true", help="estimate requests and cost; no calls")
    s.add_argument("--report-only", action="store_true", help="score the existing recording; no API calls")
    s.add_argument("--systems", default="vector@8,vector@10,jev,jev-rerank@8,llm-rerank@8")
    s.add_argument("--compare", help="paired bootstrap of two systems, e.g. jev,vector@10")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--json-out", help="write the report as JSON")
    s.set_defaults(fn=cmd_eval)
    return p


def _load_env(args: argparse.Namespace) -> None:
    """Load ./.env (or --env-file) before any command; shell variables win."""
    if args.no_env_file or args.cmd == "init":
        return
    from .envfile import load_env_file
    path = Path(args.env_file or ".env")
    if args.env_file and not path.is_file():
        raise FileNotFoundError(f"--env-file {path} not found")
    if not path.is_file():
        return
    if os.name == "posix" and path.stat().st_mode & 0o077:
        print(f"warning: {path} is readable by other users; run `chmod 600 {path}`", file=sys.stderr)
    applied = load_env_file(path)
    if args.cmd == "check":
        print(f"env       {len(applied)} variables from {path}" + (f": {', '.join(applied)}" if applied else ""))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _load_env(args)
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # user-facing: one line, no traceback unless asked
        if os.environ.get("JEV_RETRIEVAL_DEBUG"):
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
