# Known issues

Current limits of the v1.0 development build. Each has a workaround or a milestone.

## Quality

- **Classification can cost recall.** The default (`select: rank`, top 5) lost 4.9 points of recall@10 against vector top 10 on FiQA, where answers spread over several posts; it was within 2 points on SciFact and QASPER ([Results](RESULTS.md#public-datasets-m2-scifact-fiqa-qasper)). Workaround: raise `max_passages` (8 recovered vector recall on all three sets), or run classification in `shadow` mode. `select: threshold` sends less context but lost 4 to 22 points.
- **The gate misfires on statements.** It asks whether passages "contain the information needed to answer" the query; on SciFact, whose queries are claims, it wrongly refused 26% of answerable ones. On real unanswerable questions (QASPER) it caught 29 to 41%, far fewer than on synthetic ones. Workaround: `gate.mode: shadow` for claim-style or search-box queries, and tune `answer_min` on your own data with `jev-retrieval eval`.
- **In-house benchmarks overstated the gains.** Jev classification, the gate and paragraph screening clearly helped on both ([Results](RESULTS.md)), but their questions were machine-written and both looked much better than the public sets did. Default thresholds are starting points chosen on small probes. Workaround: run stages in `shadow` mode and compare on your data.
- **Jev chunking hasn't earned its place.** It tied or lost to `structural` on every benchmark, including QASPER, at about twice the ingest cost, so `structural` is the default. `method: jev` is kept for long text without structure; test it with `jev-retrieval inspect <file> --compare jev` before using it.
- **Thresholds are per model version.** A new Jev version can shift probabilities. The default model IDs differ by backend (`jev-1.13.0` on TypeSafe, `typesafe-ai/jev` on Vercel, `typesafe/jev-1.13` on OpenRouter); pin `jev.model` if you need stability.
- **English first.** TypeSafe documents lower accuracy outside English.

## Pipeline

- **One classification request per candidate.** With `top_k: 30`, a query makes about 32 Jev requests, so one key at the published 40 requests per second handles about 1.25 queries per second. Lower `top_k` for more throughput. Packed multi-passage classification is planned (M4).
- **Token counts are estimates** unless the `tokens` extra (tiktoken) is installed or you pass your embedder's tokenizer as `token_counter`.
- **Duplicates are removed within a document only.** Cross-document near-duplicate removal is planned.
- **`semantic-embedding` chunking embeds every sentence**, which costs one embedding per sentence.
- **No `reenrich` or `reembed` commands yet.** Changing the taxonomy or embedder means ingesting into a new collection (or `--force` into the same one, for taxonomy changes).
- **Very long passages are truncated in some Jev questions.** Classification sees the first 6,000 characters of a passage, the gate the first 3,000 per passage, and boundary questions the first 1,200 per sentence or table. Enrichment sends the whole chunk. With the default `max_tokens: 800`, chunks stay under these caps.

## Stores

- **Pinecone is experimental and untested against a live index** (deferred to M4). The adapter follows the `pinecone` 10.x SDK signatures and has not been run through the conformance suite. Filtered listing and deletes use a filtered query and are capped at 10,000 ids per call.
- **Chroma** can't express existence tests natively, and its `$ne` / `$nin` also match records without the field. jev-retrieval queries a superset and filters in Python, over-fetching for queries. Very selective negative filters can return fewer than `top_k`.
- **LangChain bridge** filters in Python after over-fetching (`overfetch`, default 4 times `top_k`), and relies on the wrapped store honouring caller-supplied ids for replace and stale-delete. Use `JevRetrievalEmbeddings` so stored vectors come from jev_retrieval's `embed_text`.
- **Qdrant embedded mode** ignores payload indexes (a Qdrant limitation); use a Qdrant server for large collections.
- **`--retrieve-only`** expects jev-retrieval's field layout (text in the store's `text` field). A configurable text field is planned.

## Security

- **Injection screening is one layer.** It caught all 6 planted instructions in the messy benchmark (and kept them out of every query's context), but it hasn't been measured on a real attack set. Keep treating retrieved text as data in your LLM's system prompt (jev-retrieval's `answer()` does).

## Measured weaknesses (messy-document benchmark)

- **Paragraph screening drops reference lists.** Bibliographies and citation lists score as boilerplate and are cut (about 1,000 paragraphs in the messy benchmark, almost all references). Good for most RAG, wrong if users ask about citations: use `enrich.screen_paragraphs: shadow` to review first, or `off`.
- **Occasional false-positive quarantine.** One harmless Wikipedia maintenance tag ("Use dmy dates from January 2026") scored 0.74 for `instructs_ai` and was quarantined. Quarantined paragraphs are stored, not deleted, so they can be audited with `jev-retrieval inspect`.
- **Injection screening misses what it isn't shown.** Every planted injection so far was its own paragraph. An instruction woven into a sentence of real content shares that paragraph's fate: the paragraph is quarantined whole. This hasn't been measured.
- **Fixed in 1.0.0.dev0 (paragraph screening):** chunk-level quarantine hid 4 answers that shared a chunk with an injection, and short boilerplate under `min_tokens` was merged into content chunks (0 of 12 dropped). Both are measured fixed in [Results](RESULTS.md#paragraph-level-screening-rerun-of-the-messy-benchmark).
