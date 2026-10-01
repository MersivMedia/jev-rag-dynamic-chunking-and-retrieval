# Known issues

Current limits of the v1.0 development build. Each has a workaround or a milestone.

## Quality

- **Retrieval quality is unbenchmarked.** Nothing yet shows Jev chunking or classification beating cheaper baselines on real datasets (see [Results](RESULTS.md)). Default thresholds are starting points chosen on small probes. Workaround: run with stages in `shadow` mode and compare. Planned: M2.
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
- **Chroma** can't express existence tests natively, and its `$ne` / `$nin` also match records without the field. jevrag queries a superset and filters in Python, over-fetching for queries. Very selective negative filters can return fewer than `top_k`.
- **LangChain bridge** filters in Python after over-fetching (`overfetch`, default 4 times `top_k`), and relies on the wrapped store honouring caller-supplied ids for replace and stale-delete. Use `JevragEmbeddings` so stored vectors come from jevrag's `embed_text`.
- **Qdrant embedded mode** ignores payload indexes (a Qdrant limitation); use a Qdrant server for large collections.
- **`--retrieve-only`** expects jevrag's field layout (text in the store's `text` field). A configurable text field is planned.

## Security

- **Injection screening is one layer.** It caught the planted instruction in every test so far, but it hasn't been measured on a real attack set. Keep treating retrieved text as data in your LLM's system prompt (jevrag's `answer()` does).

## Measured weaknesses (messy-document benchmark)

- **Quarantine is chunk-level.** A planted injection is caught, but the whole chunk around it is quarantined: in testing, 3 of 6 such chunks also held real answers. Workaround: `enrich.quarantine_instructs_ai: 1.01` turns off ingest quarantine; Jev classification still blocked every injection at query time. Fix planned: screen at paragraph level or cut a boundary around instruction-like text.
- **Short junk paragraphs are merged, not dropped.** Boilerplate shorter than `chunking.min_tokens` (64) is merged into a neighbouring content chunk before enrichment sees it, so it isn't dropped. Lowering `min_tokens` helps at the cost of more small chunks.
- **Reference lists from raw HTML survive.** Wikipedia "References" sections are kept and can be retrieved; the gate once passed a citation-list chunk (gate 0.88). Strip reference sections before ingest if you can.
