| FR-C2 | **Boundary questions.** Within each section, for each adjacent sentence pair, ask two Nouls in one packed request per window. The state is the window's text; each question carries its own pair inline as `{previous, sentence, question}`. `continues`: "In the document, does `sentence` continue the specific point that `previous` is making?" `refers_back`: "Does `sentence` depend on `previous` to be understood, for example by referring back to it with words like this, it, these or such?" Questions must not point at sentences by list position: in a labelled probe, `sentences[i]`-style references gave a mean absolute error of 0.63 against 0.07 for inline pairs and 0.10 for named keys (RESULTS.md). Windows are sized to the state budget and overlap by 4 units; each gap is asked in the window where it sits furthest from an edge. Wording is tuned further in M2 [7] |
# jev-rag-dynamic-chunking-and-retrieval: Product Requirements Document

Name: **jev-rag-dynamic-chunking-and-retrieval**: repository `MersivMedia/jev-rag-dynamic-chunking-and-retrieval` and PyPI package `jev-rag-dynamic-chunking-and-retrieval`. Short forms: Python import `jevrag`, CLI command `jevrag`. "jevrag" below means the tool.

Source material: "Jev chunking and ingestion for RAG" sample design and code [1].

## 1. Summary

jevrag is an open-source Python library and CLI that puts Jev, TypeSafe AI's System One decision model, into the three places a RAG pipeline makes judgment calls it usually makes badly:

- **Where to cut documents into chunks.** Jev judges whether each sentence continues the point of the one before it. Code then places cuts where continuity is lowest, inside hard size limits.
- **What goes into the vector database.** Jev screens each chunk for filler, boilerplate and planted instructions, and tags it against a taxonomy you define. Every tag carries a probability.
- **What reaches the answering LLM.** After a normal vector search, Jev classifies each retrieved passage (relevant, usable evidence, contradicts the question, injection), and a final check decides whether the kept passages can answer the question at all. If they can't, the pipeline says so without calling the LLM.

It works with any popular vector database through one small adapter interface. The first release ships native adapters for pgvector, Qdrant and Chroma plus a bridge to any LangChain vector store, with a Pinecone adapter included as experimental until it is tested against a live index; Weaviate, Milvus, MongoDB Atlas, Elasticsearch, OpenSearch, Redis, LanceDB, Azure AI Search, turbopuffer and a LlamaIndex bridge follow in M4 (Section 11).

Jev never generates text. It answers typed questions (yes/no probability, pick-one-of-N, or a score on a scale), and plain code with visible thresholds decides what to do [2]. That split is the design principle of the whole tool.

**What is proven and what is not.** Jev's ability to classify RAG passages and re-rank shortlists is shown in TypeSafe's own cookbooks (re-ranking raised top-1 accuracy from 5% to 18% on 40 legal queries) [5][6]. Those are vendor-run examples. Nobody has yet shown that Jev-placed chunk boundaries beat fixed-size or embedding-based chunking on retrieval quality. Milestone M2 (Section 11) measures exactly that before any feature becomes a default, and the README makes no performance claim until M2 results exist.

## 2. Review of the sample design

The sample [1] has the right shape: code in control, Jev for narrow judgments, a database abstraction, thresholds, and an answerability gate. As code it would not run, and several design choices would hurt quality or cost. Each finding below maps to a requirement later in this PRD.

### 2.1 API mismatches

| Sample code | Problem | Requirement |
|---|---|---|
| `jev.evaluate(context=..., policy=...)`, `jev.evaluate_multi(...)` | These methods do not exist. Jev's API is `POST /v1/systemone` with a `state` and a map of named, typed questions (`noul`, `choice`, `score`); the Python SDK call is `client.system_one(state=..., questions=...)` [2][10] | FR-J1 |
| `"score": "Rate from 0.0 to 1.0 how relevant..."` | Not a valid question. A Score needs an ordered list of 2 to 10 described levels [2]. A yes/no probability is a Noul | FR-R3 |
| Tagging policy mixes a YES/NO string, a category question and a `choices` list in one dict | Each question must be its own typed entry with its own criteria [2] | FR-E2 |
| `gate_check.choice == "NO"` on a question phrased "(YES/NO)" | A Noul returns a probability (`noul`), not a `choice` field [2] | FR-R5 |
| `judgment.choice == "YES_BREAK"` | Uses the top option only and throws away the probability, even though the prose says to threshold at 0.75 | FR-C3 |

### 2.2 Design problems

| Area | Problem | Requirement |
|---|---|---|
| Chunking cost and speed | One Jev request per sentence, in sequence. A 2,000-sentence document is 2,000 round trips. Jev evaluates many questions against one state in parallel in a single request, and TypeSafe's structure-recovery cookbook asks one question per adjacent line pair, all in one request [7][8]. Batching 13 questions into one request was 12.2x cheaper and 10.0x faster with no change in answers [9] | FR-C2 |
| Chunking question wording | "Does adding the next snippet break the semantic or conceptual unit?" is the broad kind of question TypeSafe found unreliable: asking "same paragraph?" merged lists into run-on blocks, while the narrower "does this line pick up mid-sentence?" did not (17 blocks versus 12 on the same memo) [7] | FR-C2 |
| Growing `current_block` state | The whole chunk-so-far is resent each time, so token cost grows with the square of chunk length, and Jev's accuracy drops as state fills with detail unrelated to the question [4] | FR-C2 |
| Sentence splitting | `split(". ")` breaks on "e.g. ", "U.S. ", decimals and version numbers, drops the periods, and ignores headings, lists, tables and code | FR-P2 |
| Size limit | `len(current_chunk) > 2000` counts characters, not the embedding model's tokens; there is no minimum size, so one-sentence chunks survive | FR-C4 |
| Record IDs | `generate_uuid()` gives a new ID on every run, so re-ingesting a document duplicates it and old versions are never removed | FR-I2 |
| Embeddings | One embedding call per chunk; no batching, retries or check that the vector size matches the collection | FR-I4 |
| Category list | Fixed to Technical/Legal/Financial/General with no "other". Probability has to land somewhere, so off-list text gets forced into a wrong tag [4] | FR-E2 |
| Injection screening | Promised in the prose, absent from the code | FR-E1 |
| Query routing | The router's single top category becomes a hard `WHERE` filter. One wrong route returns zero relevant results | FR-R1 |
| Re-ranking | 20 sequential requests, one question each; TypeSafe's own passage cookbook asks four questions per passage in one request and routes in code [5] | FR-R3 |
| Fixed thresholds | 0.70 for relevance, 0.75 for chunk integrity, with no way to tune them against labelled data | FR-V2 |
| Citations | The source list is bare domains (`https://www.youtube.com`, `https://medium.com`), so no claim in the prose can be checked | Section 9 |

### 2.3 Adapter bugs

| Adapter | Problem |
|---|---|
| Pinecone | Installs `pinecone-client`, which is deprecated; the package is now `pinecone` (10.0.0) [12]. Stores full text in metadata without checking Pinecone's per-record metadata limit [14] |
| Chroma | Queries return distances (lower is better) while the other adapters return similarities (higher is better), so one threshold can't work across stores. Collections are created without an explicit distance metric, so the metric depends on whichever embedding function Chroma attaches (its base default is L2) [16] |
| MongoDB Atlas | `insert_many` is not an upsert: re-running ingestion raises duplicate-key errors. `$vectorSearch` filters only work on fields indexed as `filter` type in the vector index, and the adapter never creates that index [13] |
| Qdrant | Calls `client.search(...)`, which the current client (1.19.1) no longer has; the replacement is `query_points` [11]. Bootstraps a 1536-dimension collection when the first batch is empty |
| All | No delete, no get-by-ID, no collection setup, no batching limits, no filter translation beyond exact match |

## 3. Goals and non-goals

### 3.1 Goals

1. **One pipeline, any vector database.** Switching stores is a config change; chunking, enrichment and retrieval code never changes.
2. **Better context for the LLM.** Fewer irrelevant passages, no planted instructions, and conflicts marked as conflicts.
3. **Honest abstention.** When the corpus can't answer, say so before paying for generation.
4. **Measured, not asserted.** Every Jev stage has a code-only fallback, and a built-in evaluation harness compares them on your own data.
5. **Cheap and fast enough to leave on.** Jev costs $0.042 per million input tokens and output is free [3]; the pipeline must keep per-query Jev cost around a tenth of a cent (Section 8).
6. **Predictable reruns.** The same document and config produce the same chunks and record IDs; re-ingesting only pays for what changed.

### 3.2 Non-goals

- **Not a vector database or an LLM framework.** jevrag writes to and reads from existing stores and returns passages; generation is an optional thin helper.
- **No text generation by Jev.** Summaries, contextual headers and query rewrites, where offered, come from code or an LLM, never from Jev [4].
- **No images, audio or scanned pages.** Jev is text-only [3]. OCR and transcription happen upstream.
- **No fine-tuning.** Jev can't be fine-tuned; all customisation is in state and question wording [3].
- **English first.** Other languages work less accurately [3]; they are supported but not tuned.
- **No TypeScript package in v1.0.**

## 4. Users and use cases

| User | Need |
|---|---|
| Developer adding RAG to an app | `pip install`, point at a folder and a database, get good retrieval without writing a chunker |
| Team with an existing vector store | Keep the store and its data, add Jev re-ranking and gating at query time only |
| Agent builder (Hermes, LangChain, LlamaIndex) | A retriever tool or MCP server the agent can call |
| Evaluator | Compare chunkers and re-rankers on their own labelled questions before switching anything on |

## 5. What Jev is, in brief

Jev takes a `state` (text, a JSON object or an array) and a map of named questions, evaluates every question against the same state in parallel, and returns one typed answer per question [2]:

| Question type | Asks | Returns |
|---|---|---|
| Noul | Is this statement true? | A probability from 0 to 1 |
| Choice | Which of these options? (up to 255) | The top option, a probability per option, a confidence |
| Score | Where on this ordered scale? (2 to 10 levels) | A probability-weighted score, a probability per level, a confidence |

Operating envelope for `jev-1.13.0` as of this writing [3]:

| Parameter | Value |
|---|---|
| Price | $0.042 per million input tokens; output not charged |
| Rate limits | 100,000 tokens per second; 40 requests per second (adjusting during early access) |
| Context | 64k tokens per request; 32k for state plus the longest single question |
| Input | Text only |

Known weak spots, from TypeSafe's jaggedness page [4], each with the rule this design follows:

| Weak spot | Rule in jevrag |
|---|---|
| Counting and arithmetic | All counting, lengths and maths in code |
| Accuracy falls as state fills with unrelated detail | Send only the text a question needs; one passage per request by default |
| Adversarial content in state can move answers | Jev is one layer of injection defence, never the only one |
| Instructions that contradict criteria | Criteria extend the instruction; no inverted Nouls |
| Noul and Choice outputs aren't interchangeable | Thresholds are calibrated per question and never carried across types |
| Generation | Never asked |

Jev is served directly by TypeSafe, through Vercel AI Gateway, and through OpenRouter's System One API, all with the same request format. Jermes has run the first two in production since September 2026 [15].

## 6. Architecture

```
documents
  -> 1 parse      (code)  blocks, headings, sentences, offsets
  -> 2 chunk      (Jev)   continuity per sentence pair -> cuts placed in code
  -> 3 enrich     (Jev)   quality flags + taxonomy tags per chunk
  -> 4 embed      (API)   batched, any embedding provider
  -> 5 store      (any DB adapter)  deterministic IDs, upsert, stale delete

query
  -> 6 route      (Jev)   taxonomy filter, only when confident
  -> 7 recall     (DB)    dense or hybrid top-K
  -> 8 classify   (Jev)   4 questions per passage -> include / conflict / drop
  -> 9 expand     (DB)    optional neighbouring chunks
  -> 10 gate      (Jev)   can these passages answer the query?
  -> result       passages + conflicts + trace (+ optional LLM answer)
```

Every Jev stage has a code-only fallback and a mode: `off`, `shadow` (compute and log, don't act), or `on`. Every decision is written to a trace with its probabilities, thresholds and outcome.

### 6.1 Repository layout

```
jev-rag-dynamic-chunking-and-retrieval/
  jevrag/         Python package
    jev/          client, backends, batching, cache, rate limiter
    parse/        loaders (txt, md, html, pdf, docx), block + sentence segmentation
    chunk/        boundary questions, DP segmenter, fallback chunkers
    enrich/       quality questions, taxonomy tagger
    embed/        providers (OpenAI, Cohere, Voyage, Gemini, Mistral, Ollama, sentence-transformers)
    stores/       base interface, filter DSL, one module per database, bridges
    retrieve/     router, recall, classifier, expander, gate, prompt builder
    eval/         datasets, metrics, baselines, calibrate
    cli.py  server.py  mcp.py
  docs/           RESULTS.md KNOWN_ISSUES.md DATABASES.md TUNING.md
  tests/          unit, adapter conformance (docker-compose), recorded Jev fixtures
```

## 7. Functional requirements

### 7.1 Jev client (FR-J)

| ID | Requirement |
|---|---|
| FR-J1 | Speak the real System One API (`state` + typed `questions`) for three backends: TypeSafe (`TYPESAFE_API_KEY`), Vercel AI Gateway (`AI_GATEWAY_API_KEY`) and OpenRouter (`OPENROUTER_API_KEY`), auto-detected from the environment [15] |
| FR-J2 | Pin a versioned model ID (`jev-1.13.0`) by default, not `jev-latest`, because thresholds are tuned per version and aliases move [3]. Record the answering model's ID on every stored decision |
| FR-J3 | Pack requests to the budget: 32k tokens for state plus the longest question, 64k in total [3]. Split oversize work across requests automatically |
| FR-J4 | Shared async rate limiter for requests per second and tokens per second, retries with backoff on 429 and 529, honouring `retry-after` [2] |
| FR-J5 | Content-addressed answer cache (hash of model ID, state and questions) on disk, so re-running ingestion or evaluation only pays for changed inputs |
| FR-J6 | Validate question specs client-side (Choice 2 to 255 options, Score 2 to 10 levels) so malformed questions fail at build time, not as a 422 mid-run |
| FR-J7 | Optional zero-data-retention flag where the backend supports it |

### 7.2 Parsing (FR-P)

| ID | Requirement |
|---|---|
| FR-P1 | Loaders for plain text, Markdown, HTML, PDF (text layer, optional `pymupdf` extra) and DOCX. Each produces blocks with type (heading, paragraph, list item, table, code, quote), heading path and character offsets into the source |
| FR-P2 | Sentence segmentation that handles abbreviations, decimals, version numbers, URLs and list items, keeps punctuation, and keeps offsets |
| FR-P3 | Structural facts are read in code, never asked of Jev: headings are hard boundaries, tables and code blocks are atomic units (split by rows or lines only when they exceed the maximum size) |
| FR-P4 | Stable `doc_id` from the source URI (or user-supplied), plus a content hash of the normalised text |

### 7.3 Chunking (FR-C)

| ID | Requirement |
|---|---|
| FR-C1 | Chunk sizes in the embedding model's tokens: `min_tokens` (default 64), `target_tokens` (default 350), `max_tokens` (default 800, never above the embedder's limit) |
| FR-C2 | **Boundary questions.** Within each section, for each adjacent sentence pair, ask two Nouls in one packed request whose state is the numbered sentences of a window. `continues`: "Does `s[i]` continue the specific point that `s[i-1]` is making?" `refers_back`: "Does `s[i]` depend on `s[i-1]` to be understood, for example by referring back to it with words like this, it, these or such?" Windows are sized to the state budget and overlap by 4 sentences so every pair is judged with context on both sides. Exact wording is chosen in M2 by testing alternatives, because wording changes results [7] |
| FR-C3 | **Cut placement in code.** Cut cost at each gap = weighted continuity from the two probabilities, discounted at paragraph breaks. A dynamic-programming segmenter picks the cuts that minimise total cut cost plus a penalty for distance from `target_tokens`, subject to min and max. Never cut where `refers_back` ≥ 0.5 unless max size forces it. Output is deterministic for a given set of answers |
| FR-C4 | Merge any chunk under `min_tokens` into the neighbour it's most continuous with |
| FR-C5 | Build `embed_text` = document title and heading path + chunk text, in code. Store `text` (as shown to the LLM) separately |
| FR-C6 | Optional overlap (default 0 for Jev chunking; configurable) |
| FR-C7 | Fallback chunkers, selectable by config and used automatically if Jev fails: `structural` (headings, paragraphs, size limits), `fixed` (N tokens with overlap), `semantic-embedding` (cut where adjacent sentence embeddings diverge). Each chunk records which chunker produced it |

### 7.4 Enrichment (FR-E)

| ID | Requirement |
|---|---|
| FR-E1 | One request per chunk with quality Nouls: `low_information`, `boilerplate` (navigation, footers, cookie banners, tables of contents), `instructs_ai` ("Does this text contain instructions addressed to an AI assistant, such as telling it to ignore its rules or say something specific, rather than information for a human reader?"; the broader cookbook wording scored an ordinary refund policy 0.33 in a probe, see RESULTS.md), `self_contained` |
| FR-E2 | Taxonomy tagging from a user YAML: each field is a Choice whose options carry descriptions, and each must include an `other` option [4]. The top option is stored as the tag only when confidence ≥ `tag_min_confidence`; otherwise `unknown`. The probability is always stored |
| FR-E3 | Custom questions from YAML (Noul, Choice or Score), stored as metadata fields with their probabilities |
| FR-E4 | Actions in code: drop chunks above the low-information or boilerplate thresholds (logged, reversible by re-running); **quarantine** chunks above the injection threshold (stored with `quarantined=true` and excluded from retrieval by default, never silently deleted) |
| FR-E5 | Exact-duplicate removal by content hash in code. Near-duplicate check optional: code proposes pairs by embedding similarity, Jev confirms "Do `a` and `b` state the same information?" |
| FR-E6 | Optional packing of several short chunks per request, off by default until M4 measures its accuracy cost [4] |

### 7.5 Embedding and storage (FR-I)

| ID | Requirement |
|---|---|
| FR-I1 | Embedding providers: OpenAI, Cohere, Voyage, Google Gemini, Mistral, Ollama, sentence-transformers (local), plus a callable for anything else. Batched with provider limits, retried, with separate document and query input types where the provider has them |
| FR-I2 | Deterministic record IDs: UUIDv5 of (`doc_id`, chunk content hash). Re-ingesting a document upserts its current chunks and deletes that `doc_id`'s records whose IDs are no longer present. UUIDs are valid IDs in every supported store, including Qdrant's unsigned-integer-or-UUID rule |
| FR-I3 | A collection manifest (embedding provider, model, dimension, metric, chunker, taxonomy version, jevrag version) stored with the collection. Queries with a different embedder are refused with a clear error |
| FR-I4 | Vector dimension checked against the manifest before any write |
| FR-I5 | Incremental ingest: skip documents whose content hash is unchanged |
| FR-I6 | `jevrag delete --doc`, `jevrag reenrich` (re-run enrichment without re-chunking or re-embedding) and `jevrag reembed` (new embedder into a new collection) |

Record schema (logical; each adapter maps it to native fields):

| Field | Type | Notes |
|---|---|---|
| `id` | UUID string | FR-I2 |
| `vector` | float list | plus `sparse` where hybrid is on |
| `text` | string | shown to the LLM |
| `embed_text` | string | optional; not stored by default |
| `doc_id`, `source_uri`, `title` | string | |
| `section_path` | string | "Guide > Auth > Tokens" |
| `chunk_index`, `char_start`, `char_end` | integer | for neighbour expansion and citations |
| `content_hash` | string | |
| `tags.<field>`, `tags.<field>_p` | string, float | taxonomy |
| `q.<name>` | float or string | quality and custom answers |
| `quarantined` | bool | |
| `chunker`, `jev_model`, `pipeline_version`, `ingested_at` | string | provenance |

### 7.6 Store adapters (FR-S)

| ID | Requirement |
|---|---|
| FR-S1 | One interface: `ensure_collection(name, dim, metric, filter_fields)`, `upsert(records)`, `delete(ids=None, where=None)`, `get(ids)`, `query(vector, where, top_k, sparse=None)`, `capabilities()`. About 200 lines per adapter |
| FR-S2 | Portable filter language (`eq`, `ne`, `in`, `nin`, `gt/gte/lt/lte`, `exists`, `and`, `or`, `not`) translated per store, with a clear error when a store can't express a filter |
| FR-S3 | Scores normalised to similarity in [0, 1], higher is better, for every metric; the raw score is kept alongside |
| FR-S4 | Capability flags: hybrid search, array metadata, namespaces, server-side filtering on unindexed fields, maximum batch size, maximum metadata size. The pipeline adapts: for example, flattening list tags where arrays aren't supported, or truncating stored text to the store's metadata limit with a warning |
| FR-S5 | `ensure_collection` creates what each store needs for filtering to work, such as Qdrant payload indexes and MongoDB `filter`-type vector index fields [13] |
| FR-S6 | A conformance test suite every adapter must pass: round trip, upsert idempotency, stale delete, every filter operator, score normalisation, empty-batch handling. Local stores run in docker-compose in CI; hosted-only stores (Pinecone, Atlas) run nightly behind secrets |

Adapter plan:

| Milestone | Stores |
|---|---|
| First release (M1 to M3) | pgvector, Qdrant, Chroma, LangChain bridge; Pinecone ships as experimental (code written, not live-tested) |
| M4 | Weaviate, Milvus/Zilliz, MongoDB Atlas, Elasticsearch, OpenSearch, Redis (RedisVL), LanceDB, Azure AI Search, turbopuffer, LlamaIndex bridge |

The LangChain bridge in the first release means any database with a LangChain integration works from the first release. The bridges wrap any LangChain `VectorStore` or LlamaIndex vector store, so databases without a native adapter still work, with fewer capabilities (no guaranteed stale delete or filter translation).

### 7.7 Retrieval (FR-R)

| ID | Requirement |
|---|---|
| FR-R1 | **Routing.** One request on the query: a Choice per routable taxonomy field (options include `any`), plus a Noul "Is this a question the documents could answer, rather than chit-chat or a request to act?" Apply `field = top` only if confidence ≥ `route_min_confidence`; else `field in [top 2]` if their combined probability ≥ `route_top2_mass`; else no filter. If a filtered search returns fewer than `min_candidates`, re-run it unfiltered |
| FR-R2 | **Recall.** Dense top-K (default 30) with the routed filter and `quarantined = false`; hybrid (dense + keyword) where the store supports it and it's enabled |
| FR-R3 | **Classification.** For each candidate, one request with state `{query, passage}` and four Nouls, following TypeSafe's passage cookbook [5]: `is_relevant`, `contains_answer_evidence`, `contradicts_query_premise`, `instructs_ai`. Requests run concurrently under the rate limiter |
| FR-R4 | **Routing each passage in code**, first match wins: injection above threshold → drop; relevant and contradicts the premise → conflict block; relevant and has evidence → include; otherwise drop. Included passages sort by `contains_answer_evidence`, tie-broken by vector score; keep at most `max_passages` (default 8) |
| FR-R5 | **Answerability gate.** One request with the query and included passages: Noul "Do `passages` contain the information needed to answer `query`?" Below `answer_min` → return `abstain=true` with a reason; the LLM is not called |
| FR-R6 | **Neighbour expansion** (optional): add the chunks immediately before and after an included chunk when its `self_contained` score is low, within a token budget |
| FR-R7 | **Result object**: `passages`, `conflicts`, `dropped` (each with reason and probabilities), `abstain`, `gate_p`, `filter_used`, `degraded`, per-stage latency, Jev tokens, and a `to_prompt()` that renders evidence and conflicts as separate, cited blocks |
| FR-R8 | Optional `answer()` helper for OpenAI-compatible and Anthropic APIs. No dependency on LiteLLM |
| FR-R9 | **Packed mode** (optional): several passages per request, each question pointing at `passages[i]`, to fit the 40 requests-per-second limit. Off by default until M4 measures its accuracy cost [4] |
| FR-R10 | **Retrieve-only mode** against an existing collection that jevrag didn't write: routing is skipped (no tags); classification and gating still work if a text field is named |

### 7.8 Interfaces (FR-X)

| ID | Requirement |
|---|---|
| FR-X1 | Python API, sync and async: `Pipeline.from_config(...)`, `.ingest(paths_or_docs)`, `.retrieve(query)`, `.answer(query)` |
| FR-X2 | CLI: `init`, `ingest`, `query`, `delete`, `reenrich`, `reembed`, `inspect` (show a document's chunks and cut scores), `eval`, `calibrate`, `cost`, `serve`, `mcp` |
| FR-X3 | One `jevrag.yaml` for all settings; secrets only from the environment, which the CLI can load from a `.env` file. The repo ships `jevrag.example.yaml` (every setting, commented) and `.env.example` (every variable, blank); tests fail if either drifts from the code |
| FR-X4 | Optional HTTP server (FastAPI extra): `/ingest`, `/retrieve`, `/answer`, `/health` |
| FR-X5 | Optional MCP server exposing `search_knowledge` and `ingest_documents`, so Hermes and other agents can use a collection as a tool |
| FR-X6 | LangChain `BaseRetriever` and LlamaIndex retriever wrappers |

### 7.9 Evaluation and tuning (FR-V)

| ID | Requirement |
|---|---|
| FR-V1 | `jevrag eval` on a labelled set (query, relevant `doc_id` or evidence span, or "unanswerable"). Metrics: evidence recall@5 and @10, nDCG@10, MRR, context tokens sent to the LLM, abstention precision and recall, latency p50/p90, Jev and embedding cost per 1,000 queries |
| FR-V2 | `jevrag calibrate` chooses each threshold from labelled data for a target (for example, drop at most 2% of truly relevant passages) and writes them to config with the model ID they were tuned on |
| FR-V3 | Built-in baselines on the same corpus and embedder: `fixed` 512/64, `structural`, `semantic-embedding`; no re-rank, cross-encoder (`bge-reranker-v2-m3`, local), Cohere Rerank |
| FR-V4 | `jevrag label`: a small terminal tool to label retrieved passages, with Jev's answers hidden to avoid anchoring |
| FR-V5 | Shadow mode for every Jev stage, so a live deployment logs what Jev would change before anyone switches it on |

## 8. Cost and speed

Estimates, not measurements; M2 replaces them.

**Ingestion (Jev).** Chunking sends each sentence about once as state (plus window overlap) and two short questions per sentence pair, roughly 2 to 5 times the document's tokens. Enrichment sends each chunk once plus its questions, roughly 1.5 to 2 times. At $0.042 per million, that is about **$0.15 to $0.30 of Jev per million document tokens**. Embedding with a small hosted model adds about $0.02 per million [UNVERIFIED: current list price varies by provider]. At 100,000 Jev tokens per second, one key ingests roughly 15,000 to 30,000 document tokens per second at best.

**Query (Jev).** Routing about 500 tokens; 30 passages at about 500 tokens each including questions, about 15,000; gate about 4,000. Roughly 20,000 tokens, or **about $0.0008 per query**.

**Latency.** TypeSafe reports 70 to 500 ms per request [vendor figure]; Jermes measured a 243 ms median for small requests over 100 live calls [15]. With concurrent classification, target p50 under 1.2 s and p90 under 2.5 s for the whole retrieval step, excluding the LLM.

**The binding limit is requests, not money.** At 40 requests per second [3], classifying 30 passages one per request allows only about 1.25 queries per second per key (30 passage requests plus routing and the gate). Packed mode (FR-R9) at 5 passages per request raises that to about 5 (8 requests per query instead of 32). Jermes also hit a 30-request window on a new Vercel account and temporary 503s on up to 20% of calls [15]. High-traffic users need an enterprise limit or packed mode.

## 9. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | **Fail soft at query time.** Any Jev error or timeout returns vector-ranked results marked `degraded=true`; the gate never abstains on an error |
| NFR-2 | **Fall back at ingestion.** Jev chunking failure falls back to `structural`; enrichment failure stores `unknown` tags and marks chunks for `reenrich` |
| NFR-3 | **Determinism.** Same inputs, config and cached answers produce the same chunks, IDs and routing |
| NFR-4 | **Security.** Keys from the environment only, never logged; traces redact text unless `trace.include_text` is set. Quarantined chunks stay out of prompts by default. Injection screening is one layer among several |
| NFR-5 | **Dependencies.** Python 3.10+. Core depends only on `httpx`, `pydantic` and `pyyaml`; every database, embedder, loader and server is an extra (`pip install "jev-rag-dynamic-chunking-and-retrieval[qdrant,openai]"`). Every dependency has an upper bound (`>=floor,<next_major`) |
| NFR-6 | **Observability.** JSON trace per ingest and query; optional OpenTelemetry spans |
| NFR-7 | **Tests.** Unit tests on recorded Jev fixtures, no network in default CI; adapter conformance in docker-compose; a nightly live Jev smoke test |
| NFR-8 | **Documentation.** README with features first, install, configure, a full ingestion guide and a full retrieval guide; results and known issues in linked pages; every claim cites its measurement |
| NFR-9 | **Licence.** MIT; public repository |

Evidence rule: the sample's citations are bare domains [1] and can't be checked. The repo's docs must cite primary pages (TypeSafe docs, database docs, measured runs), and vendor figures are labelled as vendor figures.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Jev chunking doesn't beat cheaper chunkers | M2 go/no-go; if it loses, `structural` becomes the default and Jev chunking is opt-in |
| Rate limits change during early access [3] | Packed mode, token-bucket limiter, cache, clear errors |
| A model update shifts probabilities | Pinned model ID; thresholds stored with the ID they were tuned on; `calibrate` re-runs |
| Planted instructions pass the screen [4] | Quarantine at ingest, drop at query, separate evidence block in the prompt, and advice to treat retrieved text as data in the LLM's system prompt |
| Adapter drift as SDKs change (Qdrant `search` removal is an example [11]) | Conformance suite, upper-bounded pins, nightly runs against latest releases |
| Router filters out the right answer | Confidence gating, top-2 fallback, unfiltered retry (FR-R1) |
| Non-English corpora | Documented as lower accuracy [3]; eval harness to measure before use |
| Vendor dependence on one model provider | Every stage has a code-only fallback; nothing stored is unreadable without Jev |

## 11. Milestones

Each milestone is a goal with a test that says it's done. Work moves to the next milestone as soon as the test passes. M1 to M3 are one short build, test and ship cycle; the first public release is M3, and M4 grows coverage after people are using it.

**Status (30 September 2026): M1 is built.** Done: Jev client for all three backends; Markdown, text, HTML, PDF and DOCX loaders; Jev, structural, fixed and semantic-embedding chunkers; enrichment with taxonomy and custom questions; OpenAI-compatible, Vercel AI Gateway, Ollama, sentence-transformers and callable embedders; memory, Qdrant, Chroma, pgvector and Pinecone adapters plus the LangChain bridge; the full retrieval path with `answer()`; CLI `init`, `check`, `ingest`, `query`, `inspect`, `delete`. The conformance suite passes on memory, Qdrant, Chroma, the LangChain bridge and pgvector (against real Postgres). Pinecone is deferred: the adapter is written and ships marked experimental, and live testing moves to M4 (no account yet). Moved out of M1: `reenrich` and `reembed` (to M4); tests use a keyword-driven fake Jev rather than recorded fixtures, plus an opt-in live test.

| Milestone | Goal | Content | Done when |
|---|---|---|---|
| **M1 Working core** | One document goes in and a grounded, cited answer comes out, end to end | Jev client (TypeSafe, OpenRouter, Vercel), parser, Jev chunker with `structural` fallback, enrichment, embedders (OpenAI, sentence-transformers, Ollama), first-release adapters (Section 7.6), the full retrieval path, CLI `init`/`ingest`/`query`/`inspect`, config, cache, traces | Conformance suite green on every first-release store; unit tests pass on recorded Jev fixtures; `jevrag ingest` then `jevrag query` works against each store |
| **M2 Measured** | Defaults set by measurement, not by guess | `jevrag eval` with the built-in baselines, run on small samples of BEIR SciFact and FiQA (retrieval and re-ranking) and QASPER (long-document chunking) [UNVERIFIED: dataset licences to confirm], plus the injection test set; question wordings and thresholds tuned; RESULTS.md written | Go/no-go per stage. Jev chunking is the default only if it beats the best baseline on evidence recall@10 on at least 2 of 3 sets. Classification is on by default only if it cuts context tokens sent to the LLM by at least 40% without lowering recall@10 by more than 2 points. A stage that misses its bar ships in `shadow` mode, with the result in RESULTS.md |
| **M3 Released** | People can install and use it | Public repo, README and KNOWN_ISSUES.md matching the shipped behaviour, PyPI package, tagged v1.0.0, CI running unit and conformance tests | `pip install` and the README quickstart work from a clean machine in under 10 minutes; every claim in the README links to RESULTS.md |
| **M4 Coverage** | Works with every popular vector database and agent framework | Pinecone conformance against a live index, remaining native adapters, hybrid search, remaining embedders (native Cohere, Voyage, Gemini, Mistral), LlamaIndex bridge, HTTP and MCP servers, LangChain and LlamaIndex retrievers, `reenrich`, `reembed`, `calibrate`, `label`, packed mode measured. Order set by what M3 users ask for | Conformance green on all 13 native stores; packed-mode accuracy cost documented; Hermes can query a collection through MCP |

Before M2 spends money: dry-run cost estimate, a single-document test, then the batch.

## 12. Success metrics

Targets are hypotheses until M2.

- Evidence recall@10 no lower than the best baseline; context tokens to the LLM at least 40% lower.
- Abstention: at least 80% of unanswerable queries abstain while no more than 5% of answerable ones do.
- Zero planted-instruction chunks in prompts on the injection test set.
- Jev cost at or under $0.001 per query and $0.30 per million ingested tokens.
- Adoption: an adapter for each of the 13 native stores passing conformance by M4.

## 13. Open questions

Decided in M1: jevrag has its own async `httpx` client (adapted from Jermes) rather than depending on the pre-1.0 `typesafe-sdk` [10], because the Vercel and OpenRouter backends are needed.


1. **Default embedder** for the quickstart: local sentence-transformers (no key, slower) or OpenAI (one more key).
2. **Hermes integration**: ship as a Jermes feature, a standalone Hermes plugin, or only the MCP server? Recommendation: standalone package first, MCP for Hermes, no Jermes coupling.

## Sources

- [1] https://docs.google.com/document/d/10k9hz7DK0ZPC4ZSIMhn42NsmcCUuA4mLgOpGe0PhnNU/edit — "Jev chunking and ingestion for RAG" (sample design and code)
- [2] https://docs.typesafe.ai/api — TypeSafe API reference (`POST /v1/systemone`, question and answer types, errors)
- [3] https://docs.typesafe.ai/models — TypeSafe Models (jev-1.13.0 price, rate limits, context, aliases, customisation, languages)
- [4] https://docs.typesafe.ai/model-jaggedness/jev-1.13 — Jev 1.13 jaggedness (known failure modes)
- [5] https://docs.typesafe.ai/cookbooks/classifying_rag_passages — Cookbook: Classifying RAG passages (four Nouls per passage, routing in code)
- [6] https://docs.typesafe.ai/cookbooks/rerank_typesafe — Cookbook: Re-ranking (CLERC, top-1 5% to 18%, top-10 38% to 62%)
- [7] https://docs.typesafe.ai/cookbooks/autoformat — Cookbook: Structure recovery (one Noul per line pair in one request; "mid-sentence" versus "same paragraph" wording)
- [8] https://docs.typesafe.ai/patterns/fan-out — Pattern: Speculative fan-out (many questions per request)
- [9] https://docs.typesafe.ai/cookbooks/parallel_questions — Cookbook: Parallel questions (batching 13 questions: 12.2x cheaper, 10.0x faster, same answers)
- [10] https://docs.typesafe.ai/sdk/python — TypeSafe Python SDK (`client.system_one`)
- [11] https://pypi.org/project/qdrant-client/1.19.1/ — qdrant-client 1.19.1 (inspected: `QdrantClient` has `query_points`, no `search`)
- [12] https://pypi.org/project/pinecone-client/ — pinecone-client (deprecated; the package is `pinecone`, 10.0.0)
- [13] https://www.mongodb.com/docs/vector-search/query/aggregation-stages/vector-search-stage/ — MongoDB `$vectorSearch` (pre-filter fields must be indexed as `filter`)
- [14] https://docs.pinecone.io/reference/api/database-limits — Pinecone database limits (per-record metadata limit; 40 KB per Pinecone community answers [UNVERIFIED on the limits page itself])
- [15] https://github.com/MersivMedia/jermes — Jermes (Jev backends, latency 243 ms median, Vercel rate-limit and 503 findings in docs/KNOWN_ISSUES.md)
- [16] https://pypi.org/project/chromadb/1.5.9/ — chromadb 1.5.9 (inspected: collection space defaults to `l2` when not set)
