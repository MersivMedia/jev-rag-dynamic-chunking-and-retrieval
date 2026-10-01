"""M2 report: tune on dev, report on test, go/no-go per PRD bar.

Reads recordings written by ``jev-retrieval eval`` (one collection per dataset,
dev and test splits recorded into the same file) and prints / writes:

* test-split metrics for every system at default thresholds,
* classification thresholds tuned on dev (fewest tokens with recall@10 within
  2 points of vector@10), then applied unchanged to test,
* paired bootstrap intervals for jev vs vector@10 and jev vs llm-rerank,
* gate threshold tuned on QASPER dev (false abstain <= 5%), applied to test,
* the PRD M2 bars, evaluated.

Usage: python scripts/eval_m2.py [--out .jev-retrieval/eval] [--json docs/m2.json]
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_retrieval.eval import ideal_units, load_jsonl  # noqa: E402
from jev_retrieval.eval.runner import load_recording  # noqa: E402
from jev_retrieval.eval.systems import bootstrap_delta, evaluate, tune, tune_gate  # noqa: E402
from jev_retrieval.retrieve import ClassifyConfig  # noqa: E402

DATA = Path(".jev-retrieval/eval_data/prepared")
THRESHOLD = ClassifyConfig(select="threshold", max_passages=8)  # the pre-1.0 default, as measured in M2
SYSTEMS = ["vector@8", "vector@10", "jev", "jev-rerank@8", "llm-rerank@8"]
K = 10


def split_rows(name: str, out: Path, collection: str):
    ds = load_jsonl(str(DATA / name / "queries.jsonl"))
    split = {q.id: q.meta.get("split") for q in ds.queries}
    qideal = {q.id: ideal_units(q, ds.level) for q in ds.queries}
    rows = load_recording(out, collection)
    dev = [r for r in rows if split.get(r["id"]) == "dev"]
    test = [r for r in rows if split.get(r["id"]) == "test"]
    return dev, test, qideal, ds


def fmt(m, keys):
    return "  ".join(f"{k.replace('_mean', '')}={m.get(k)}" for k in keys if m.get(k) is not None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".jev-retrieval/eval")
    ap.add_argument("--json")
    ap.add_argument("--datasets", default="scifact,fiqa,qasper")
    ap.add_argument("--extra", default="qasper:eval_qasper_jevchunk",
                    help="name:collection pairs scored on test with default thresholds (chunker comparison)")
    a = ap.parse_args()
    out = Path(a.out)
    report = {}
    keys = [f"recall@{K}", f"ndcg@{K}", "hit@1", "passages_mean", "context_tokens_mean", "false_abstain", "true_abstain"]
    for name in a.datasets.split(","):
        dev, test, qideal, ds = split_rows(name, out, f"eval_{name}")
        if not test:
            print(f"\n## {name}: no test rows recorded")
            continue
        r = {"dev_rows": len(dev), "test_rows": len(test), "level": ds.level}
        print(f"\n## {name}: dev {len(dev)}, test {len(test)} ({ds.level} level)")
        base = evaluate(test, SYSTEMS, qideal=qideal, cfg=THRESHOLD, k=K)
        r["test_default"] = base
        for s, m in base.items():
            print(f"  {s:<14} {fmt(m, keys)}")
        t = tune(dev, qideal=qideal, k=K, base=THRESHOLD) if dev else {"best": None}
        r["tuned_on_dev"] = {k: v for k, v in t.items() if k != "tried"}
        if t.get("best"):
            cfg = replace(THRESHOLD, **t["best"]["config"])
            tuned = evaluate(test, ["jev"], qideal=qideal, cfg=cfg, k=K)["jev"]
            r["test_tuned"] = {"config": t["best"]["config"], "metrics": tuned}
            print(f"  jev (tuned on dev: {t['best']['config']}) {fmt(tuned, keys)}")
        else:
            cfg = THRESHOLD
            print("  tuning: no threshold set met the recall floor on dev")
        # preregistered rank mode (docs/m2_preregistration.json), unchanged
        rank_cfg = ClassifyConfig(select="rank", max_passages=5)
        rk = evaluate(test, ["jev"], qideal=qideal, cfg=rank_cfg, k=K)["jev"]
        r["test_rank_prereg"] = {"config": {"select": "rank", "max_passages": 5}, "metrics": rk}
        print(f"  jev rank@5 (preregistered) {fmt(rk, keys)}")
        for other in ("vector@10", "llm-rerank@8"):
            if other in base:
                for metric in ("recall", "ndcg"):
                    d = bootstrap_delta(test, "jev", other, qideal=qideal, metric=metric, cfg=rank_cfg, k=K)
                    r.setdefault("bootstrap_rank", {})[f"rank@5-vs-{other}:{metric}"] = d
                    print(f"  jev rank@5 - {other} {metric}@{K}: {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] n={d['n']}")
        deltas = {}
        for other in ("vector@10", "llm-rerank@8", "vector@8"):
            if other not in base:
                continue
            for metric in ("recall", "ndcg"):
                d = bootstrap_delta(test, "jev", other, qideal=qideal, metric=metric, cfg=cfg, k=K)
                deltas[f"jev-vs-{other}:{metric}"] = d
                print(f"  jev{' (tuned)' if t.get('best') else ''} - {other} {metric}@{K}: {d['delta']:+.4f} "
                      f"[{d['lo']:+.4f}, {d['hi']:+.4f}] n={d['n']}")
        if "llm-rerank@8" in base:
            for metric in ("ndcg",):
                d = bootstrap_delta(test, "jev-rerank@8", "llm-rerank@8", qideal=qideal, metric=metric, k=K)
                deltas[f"jev-rerank@8-vs-llm-rerank@8:{metric}"] = d
                print(f"  jev-rerank@8 - llm-rerank@8 {metric}@{K}: {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
        r["bootstrap"] = deltas
        if any(not row["answerable"] for row in dev + test):
            g = tune_gate(dev)
            r["gate_tuned_on_dev"] = g["best"]
            if g["best"]:
                th = g["best"]["threshold"]
                tg = evaluate(test, ["jev"], qideal=qideal, cfg=THRESHOLD, gate_min=th, k=K)["jev"]
                r["gate_test"] = {"threshold": th, "false_abstain": tg.get("false_abstain"), "true_abstain": tg.get("true_abstain")}
                print(f"  gate tuned on dev: threshold {th} -> test false_abstain {tg.get('false_abstain')}, "
                      f"true_abstain {tg.get('true_abstain')} (default 0.35: {base['jev'].get('false_abstain')}, "
                      f"{base['jev'].get('true_abstain')})")
        if any(not row["answerable"] for row in test):
            rank_cfg = ClassifyConfig(select="rank", max_passages=5)
            have = [row for row in test if "jev_rank" in (row.get("gate") or {})]
            if len(have) == len(test):
                gr = evaluate(test, ["jev"], qideal=qideal, cfg=rank_cfg, k=K)["jev"]
                r["gate_rank_test_default"] = {"false_abstain": gr.get("false_abstain"), "true_abstain": gr.get("true_abstain")}
                print(f"  gate on rank@5 passages (0.35): false_abstain {gr.get('false_abstain')}, true_abstain {gr.get('true_abstain')}")
            else:
                print(f"  gate on rank@5 passages: recorded for {len(have)}/{len(test)} rows; run scripts/backfill_rank_gate.py")
        # cost per query from recorded usage
        jev_req = sum(len([c for c in row["cands"] if c.get("jev")]) + 2 for row in test)
        llm_in = sum((row.get("llm_usage") or {}).get("in", 0) for row in test)
        llm_out = sum((row.get("llm_usage") or {}).get("out", 0) for row in test)
        r["llm_cost_per_query_usd"] = round((llm_in * 0.40 + llm_out * 1.60) / 1e6 / max(1, len(test)), 6)
        r["jev_requests_per_query"] = round(jev_req / max(1, len(test)), 1)
        ms = sorted(row["ms"].get("classify", 0) + row["ms"].get("gate", 0) for row in test if "classify" in row["ms"])
        lms = sorted(row["ms"].get("llm", 0) for row in test if "llm" in row["ms"])
        r["latency_ms"] = {"jev_classify_gate_p50": ms[len(ms) // 2] if ms else None,
                           "llm_rerank_p50": lms[len(lms) // 2] if lms else None}
        print(f"  per query: Jev {r['jev_requests_per_query']} requests; LLM rerank ${r['llm_cost_per_query_usd']}; "
              f"latency p50 Jev classify+gate {r['latency_ms']['jev_classify_gate_p50']} ms, "
              f"LLM rerank {r['latency_ms']['llm_rerank_p50']} ms")
        report[name] = r

    # chunker comparison (QASPER only: the BEIR documents are single short abstracts/posts)
    for pair in [p for p in a.extra.split(",") if p]:
        name, coll = pair.split(":")
        if not (out / f"record_{coll}.jsonl").exists():
            continue
        dev, test, qideal, _ = split_rows(name, out, coll)
        m = evaluate(test, ["vector@8", "vector@10", "jev"], qideal=qideal, cfg=THRESHOLD, k=K)
        report.setdefault("chunker", {})[coll] = m
        print(f"\n## chunker: {coll} (test {len(test)})")
        for s, v in m.items():
            print(f"  {s:<14} {fmt(v, keys)}")

    # PRD M2 bars
    print("\n## PRD M2 bars")
    bars = {}
    for name, r in report.items():
        if name == "chunker":
            continue
        v, j = r["test_default"]["vector@10"], r["test_default"]["jev"]
        cut = 1 - (j["context_tokens_mean"] or 0) / (v["context_tokens_mean"] or 1)
        drop = (v[f"recall@{K}"] or 0) - (j[f"recall@{K}"] or 0)
        ok = cut >= 0.40 and drop <= 0.02
        bars[f"classification:{name}:default"] = {"token_cut": round(cut, 3), "recall_drop": round(drop, 4), "pass": ok}
        print(f"  classification on {name} (defaults): tokens -{cut:.0%}, recall@10 drop {drop * 100:+.1f} pts -> "
              f"{'PASS' if ok else 'FAIL'}")
        rk = r["test_rank_prereg"]["metrics"]
        cut_r = 1 - (rk["context_tokens_mean"] or 0) / (v["context_tokens_mean"] or 1)
        drop_r = (v[f"recall@{K}"] or 0) - (rk[f"recall@{K}"] or 0)
        ok_r = cut_r >= 0.40 and drop_r <= 0.02
        bars[f"classification:{name}:rank@5"] = {"token_cut": round(cut_r, 3), "recall_drop": round(drop_r, 4), "pass": ok_r}
        print(f"  classification on {name} (rank@5, preregistered): tokens -{cut_r:.0%}, recall@10 drop "
              f"{drop_r * 100:+.1f} pts -> {'PASS' if ok_r else 'FAIL'}")
        if r.get("test_tuned"):
            jt = r["test_tuned"]["metrics"]
            cut_t = 1 - (jt["context_tokens_mean"] or 0) / (v["context_tokens_mean"] or 1)
            drop_t = (v[f"recall@{K}"] or 0) - (jt[f"recall@{K}"] or 0)
            ok_t = cut_t >= 0.40 and drop_t <= 0.02
            bars[f"classification:{name}:tuned"] = {"token_cut": round(cut_t, 3), "recall_drop": round(drop_t, 4), "pass": ok_t}
            print(f"  classification on {name} (tuned on dev): tokens -{cut_t:.0%}, recall@10 drop {drop_t * 100:+.1f} pts -> "
                  f"{'PASS' if ok_t else 'FAIL'}")
    if "chunker" in report and "qasper" in report:
        s = report["qasper"]["test_default"]
        jc = next(iter(report["chunker"].values()))
        for sysname in ("vector@10", "jev"):
            d = (jc[sysname][f"recall@{K}"] or 0) - (s[sysname][f"recall@{K}"] or 0)
            bars[f"chunking:qasper:{sysname}"] = {"jev_minus_structural_recall@10": round(d, 4), "win": d > 0}
            print(f"  Jev chunking vs structural on QASPER ({sysname}): recall@10 {d * 100:+.1f} pts")
        print("  Jev chunking can be measured on 1 of the 3 sets only (SciFact and FiQA documents are single "
              "short abstracts/posts, nothing to cut), so the 2-of-3 bar cannot be met: structural stays default.")
    report["bars"] = bars
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
