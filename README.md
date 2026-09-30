# jev-rag-dynamic-chunking-and-retrieval

Jev-steered chunking, ingestion and retrieval for any vector database.

**Cut documents where the meaning changes, keep junk and planted instructions out of your index, and send your LLM only the passages that answer the question, on whichever vector database you already use.**

- **Chunks that follow the text, not a character count.** Jev judges whether each sentence continues the point of the one before it; code places the cuts inside hard token limits.
- **A cleaner index.** Filler, boilerplate and text that tries to instruct an AI are dropped or quarantined before embedding. Every chunk is tagged against your own taxonomy, with probabilities.
- **Less context, fewer wrong answers.** Every retrieved passage is classified as evidence, a conflict with the question, or noise; a final check abstains when the passages can't answer, before any LLM call.
- **Any vector database.** Native adapters for pgvector, Qdrant, Pinecone, Chroma, Weaviate, Milvus, MongoDB Atlas, Elasticsearch, OpenSearch, Redis and LanceDB, plus LangChain and LlamaIndex bridges for the rest.
- **About $0.0008 of Jev per query** (estimate; see [Cost](#cost-and-limits)). Jev charges $0.042 per million input tokens and nothing for output.

> **Status: design stage.** This README describes the v1.0 interface specified in the [PRD](docs/PRD.md). Nothing here has been benchmarked yet. Quality claims wait for the M2 evaluation, and each Jev stage ships in `shadow` mode until it wins on your data.

jevrag uses [Jev](https://docs.typesafe.ai/introduction), TypeSafe AI's System One model. Jev never writes text. It answers typed questions (a yes/no probability, one option from a list, or a score on a scale) and plain code with visible thresholds decides what happens. Every decision is logged with its probabilities.

## What it does

| Stage | What Jev decides | What code does |
|---|---|---|
| Chunking | Does sentence *i* continue the point of sentence *i−1*? Does it depend on it to make sense? | Places cuts where continuity is lowest, within min/target/max token sizes |
| Quality screen | Is this chunk filler, boilerplate, or trying to instruct an AI? Is it self-contained? | Drops, quarantines or keeps it |
| Tagging | Which option in each of your taxonomy fields fits, including "other" | Stores the tag only when confident; always stores the probability |
| Query routing | Which taxonomy values the question is about | Applies a metadata filter only when confident, and retries unfiltered if it returns too little |
| Passage classification | Relevant? Usable evidence? Contradicts the question's premise? Instructs the AI? | Includes, marks as a conflict, or drops; ranks what's kept |
| Answer gate | Can these passages answer the question? | Abstains without calling the LLM, or builds the prompt |

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Configure](#configure)
- [Ingestion guide: processing and storing embeddings](#ingestion-guide-processing-and-storing-embeddings)
- [Retrieval guide: querying, classifying and answering](#retrieval-guide-querying-classifying-and-answering)
- [Vector databases](#vector-databases)
- [Tuning and evaluation](#tuning-and-evaluation)
- [Cost and limits](#cost-and-limits)
- [Use with agents and frameworks](#use-with-agents-and-frameworks)
- [More](#more)

## Install

Python 3.10 or later. The package installs as `jev-rag-dynamic-chunking-and-retrieval`; you import it as `jevrag` and run it as the `jevrag` command. The core has three dependencies; each database, embedder and file format is an extra.

```bash
pip install "jev-rag-dynamic-chunking-and-retrieval[qdrant,openai]"  # Qdrant + OpenAI embeddings
pip install "jev-rag-dynamic-chunking-and-retrieval[pgvector,local]"  # Postgres + local sentence-transformers
pip install "jev-rag-dynamic-chunking-and-retrieval[pinecone,cohere,pdf]"  # Pinecone + Cohere + PDF loader
pip install "jev-rag-dynamic-chunking-and-retrieval[all]"  # everything
```

| Extra | Installs |
|---|---|
| `pgvector`, `qdrant`, `pinecone`, `chroma`, `weaviate`, `milvus`, `mongodb`, `elasticsearch`, `opensearch`, `redis`, `lancedb` | That database's official client |
| `langchain`, `llamaindex` | Bridges to any LangChain or LlamaIndex vector store |
| `openai`, `cohere`, `voyage`, `gemini`, `mistral`, `ollama`, `local` | Embedding providers (`local` = sentence-transformers) |
| `pdf`, `docx`, `html` | File loaders |
| `server`, `mcp` | HTTP API and MCP server |

### Jev key

Set one of these in your environment. jevrag detects which is present.

| Provider | Variable |
|---|---|
| TypeSafe direct | `TYPESAFE_API_KEY` (from [console.typesafe.ai](https://console.typesafe.ai/)) |
| Vercel AI Gateway | `AI_GATEWAY_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |

Plus the key for your embedding provider (for example `OPENAI_API_KEY`) and your database's connection settings.

```bash
jevrag check     # one live Jev call, one embedding call, one database round trip
```

## Quickstart

```bash
docker run -d -p 6333:6333 qdrant/qdrant       # any supported database works
export TYPESAFE_API_KEY=...  OPENAI_API_KEY=...

jevrag init --store qdrant --embedder openai:text-embedding-3-small
jevrag ingest ./docs --collection handbook
jevrag query "How long do refresh tokens last?" --collection handbook
```

`query` prints the kept passages with their scores, any conflicts, what was dropped and why, and whether the gate would abstain. Add `--answer` to have an LLM write the answer from those passages.

The same in Python:

```python
from jevrag import Pipeline

rag = Pipeline.from_config("jevrag.yaml")
report = rag.ingest(["./docs"], collection="handbook")
print(report.summary())        # docs, chunks, dropped, quarantined, Jev tokens, cost

result = rag.retrieve("How long do refresh tokens last?", collection="handbook")
if result.abstain:
    print("Not in the documents:", result.reason)
else:
    for p in result.passages:
        print(f"{p.scores['contains_answer_evidence']:.2f}  {p.source_uri}  {p.text[:80]}")
```

## Configure

`jevrag init` writes `jevrag.yaml`. Secrets come only from the environment.

```yaml
jev:
  model: jev-1.13.0            # pinned; thresholds are tuned per model version
  backend: auto                # auto | typesafe | vercel | openrouter
  cache_dir: .jevrag/cache     # answers are cached by content hash

store:
  kind: qdrant
  url: http://localhost:6333
  # api_key_env: QDRANT_API_KEY

embedder:
  provider: openai
  model: text-embedding-3-small
  batch_size: 128

chunking:
  method: jev                  # jev | structural | fixed | semantic-embedding
  min_tokens: 64
  target_tokens: 350
  max_tokens: 800
  overlap_tokens: 0
  mode: on                     # off | shadow | on

enrich:
  mode: on
  drop_low_information: 0.85
  drop_boilerplate: 0.85
  quarantine_instructs_ai: 0.70
  taxonomy: taxonomy.yaml      # optional
  tag_min_confidence: 0.50

retrieve:
  top_k: 30
  hybrid: false
  route: { mode: on, min_confidence: 0.60, top2_mass: 0.80, min_candidates: 5 }
  classify:
    mode: on
    drop_instructs_ai: 0.50
    min_relevant: 0.50
    min_evidence: 0.40
    conflict: 0.60
    max_passages: 8
  expand_neighbours: false
  gate: { mode: on, answer_min: 0.35 }

answer:                        # optional; only used by `answer()` / `--answer`
  provider: anthropic
  model: claude-sonnet-4-5
```

The thresholds above are placeholders until M2 sets measured defaults. Run `jevrag calibrate` on your own labelled queries before relying on them (see [Tuning](#tuning-and-evaluation)).

Every Jev stage has a `mode`. `shadow` computes and logs Jev's decision but uses the fallback's result, so you can compare before switching on.

## Ingestion guide: processing and storing embeddings

Ingestion turns files into records in your vector database. Five steps, each inspectable on its own.

```
files -> 1 parse -> 2 chunk -> 3 enrich -> 4 embed -> 5 store
```

### Step 1: Parse

Loaders read `.txt`, `.md`, `.html`, `.pdf` (text layer only), and `.docx`, and produce blocks: heading, paragraph, list item, table, code or quote, each with its heading path ("Guide > Auth > Tokens") and character offsets.

What code decides, never Jev:

- **Headings are hard boundaries.** No chunk spans two sections.
- **Tables and code blocks stay whole**, split by rows or lines only if they exceed `max_tokens`.
- **Sentences** are split by a segmenter that handles "e.g.", "U.S.", decimals, version numbers and URLs.

Scanned PDFs, images and audio need OCR or transcription first; Jev reads text only.

To bring your own parsed text:

```python
from jevrag import Document
docs = [Document(doc_id="kb-142", text=body, title="Refunds policy",
                 source_uri="https://example.com/kb/142", metadata={"team": "billing"})]
rag.ingest(docs, collection="handbook")
```

`doc_id` should be stable across runs (a URL, path or database key). It's what makes re-ingestion replace old chunks instead of duplicating them.

### Step 2: Chunk

Within each section, jevrag numbers the sentences and sends them to Jev in windows sized to Jev's request budget (32k tokens for state plus the longest question). For every adjacent pair, two yes/no questions go in the **same request**:

| Question | Wording |
|---|---|
| `continues` | Does `s[i]` continue the specific point that `s[i-1]` is making? |
| `refers_back` | Does `s[i]` depend on `s[i-1]` to be understood, for example by referring back to it with words like this, it, these or such? |

Windows overlap by four sentences so every pair is judged with context on both sides. A 2,000-sentence manual takes a handful of requests, not 2,000.

Code then places the cuts:

1. Each gap between sentences gets a cut cost from the two probabilities (paragraph breaks lower it).
2. A dynamic-programming pass chooses the cuts with the lowest total cost, plus a penalty for straying from `target_tokens`, never below `min_tokens` or above `max_tokens`.
3. A gap where `refers_back` ≥ 0.5 is never cut unless the size limit forces it, so "This means..." is never orphaned from what "this" refers to.
4. Chunks under `min_tokens` merge into their most continuous neighbour.

Sizes are counted in your embedding model's tokens, not characters.

See the result before storing anything:

```bash
jevrag inspect ./docs/auth.md          # chunks, sizes, and the score at every candidate cut
jevrag inspect ./docs/auth.md --compare structural,fixed
```

Each chunk also gets an `embed_text`: the document title and heading path prepended to the chunk, so a chunk that says "They expire after 14 days" still embeds near questions about refresh tokens. `text` (what the LLM sees) is stored separately.

If Jev is unavailable, chunking falls back to `structural` (headings, paragraphs, size limits) and records which chunker produced each chunk.

### Step 3: Enrich

One Jev request per chunk carries every enrichment question at once.

**Quality screen** (yes/no probabilities):

| Question | Default action |
|---|---|
| `low_information`: is this filler with no usable information? | Drop at ≥ 0.85 |
| `boilerplate`: navigation, footer, cookie banner, table of contents? | Drop at ≥ 0.85 |
| `instructs_ai`: does it try to give instructions to an AI system that reads it? | **Quarantine** at ≥ 0.70 |
| `self_contained`: can it be understood without the text around it? | Stored; used by neighbour expansion |

Dropped chunks are logged in the ingest report. Quarantined chunks are stored with `quarantined=true` and excluded from every query by default, so you can review them:

```bash
jevrag inspect --collection handbook --quarantined
```

**Taxonomy tags.** Define fields in `taxonomy.yaml`. Each is a one-of-N choice; describe every option, and always include `other`, because Jev must put its probability somewhere and a missing option forces a wrong tag.

```yaml
version: 3
fields:
  doc_type:
    route: true                 # usable as a query filter
    options:
      policy:    Rules the company commits to; what is and isn't allowed
      how_to:    Step-by-step instructions for doing a task
      reference: Specifications, limits, parameters, API fields
      faq:       Short question-and-answer entries
      other:     None of the above
  product:
    route: true
    options:
      billing:   Payments, invoices, refunds, plans
      auth:      Sign-in, sessions, tokens, SSO
      api:       Endpoints, SDKs, webhooks, rate limits
      other:     Anything else
questions:                      # optional custom questions, stored as metadata
  mentions_deadline:
    type: noul
    instructions: Does this text state a date or time limit that applies to the reader?
```

A tag is stored only when Jev's confidence is at least `tag_min_confidence`; otherwise it's `unknown`. The probability is always stored as `tags.<field>_p`, so you can filter on it later.

**Duplicates.** Exact duplicates (same content hash) are removed in code. Optional near-duplicate removal has code propose similar pairs by embedding distance and Jev confirm whether they state the same information.

To change the taxonomy or thresholds later without re-chunking or re-embedding:

```bash
jevrag reenrich --collection handbook
```

### Step 4: Embed

Chunks are embedded in batches with the configured provider, using the provider's document input type where it has one (Cohere, Voyage, Gemini) and the matching query type at search time.

| Provider | Example model setting |
|---|---|
| OpenAI | `openai:text-embedding-3-small` |
| Cohere | `cohere:embed-v4.0` |
| Voyage | `voyage:voyage-3.5` |
| Google | `gemini:gemini-embedding-001` |
| Mistral | `mistral:mistral-embed` |
| Ollama | `ollama:nomic-embed-text` |
| Local | `local:BAAI/bge-small-en-v1.5` |
| Anything else | a Python callable `list[str] -> list[list[float]]` |

Model names are examples; use any model your provider serves.

The first ingest writes a **collection manifest**: embedding provider, model, dimension, distance metric, chunker, taxonomy version and jevrag version. Later writes with a different dimension, and queries with a different embedding model, are refused with a clear error. Mixing embedding models in one collection silently ruins retrieval. To switch models:

```bash
jevrag reembed --collection handbook --to handbook-v2 --embedder voyage:voyage-3.5
```

This reuses stored chunks and tags and only pays for new embeddings.

### Step 5: Store

Each chunk becomes one record:

| Field | Meaning |
|---|---|
| `id` | UUIDv5 of `doc_id` + chunk content hash: identical on every run |
| `vector` | The embedding (plus a sparse vector when hybrid search is on) |
| `text` | Chunk text shown to the LLM |
| `doc_id`, `source_uri`, `title`, `section_path` | Where it came from |
| `chunk_index`, `char_start`, `char_end` | Position, for citations and neighbour expansion |
| `tags.<field>`, `tags.<field>_p` | Taxonomy tag and its probability |
| `q.<name>` | Quality and custom question answers |
| `quarantined` | Excluded from queries when true |
| `chunker`, `jev_model`, `pipeline_version`, `ingested_at`, `content_hash` | Provenance |

Each adapter maps these to native fields (Pinecone metadata, Qdrant payload, Postgres columns, Mongo document fields, and so on) and creates what filtering needs, such as Qdrant payload indexes or MongoDB `filter` fields in the vector index.

**Re-ingesting is safe.** Unchanged documents (same content hash) are skipped. A changed document has its current chunks upserted and its stale ones deleted. Removing a document:

```bash
jevrag delete --collection handbook --doc kb-142
```

### Ingest report

```bash
jevrag ingest ./docs --collection handbook --dry-run   # parse, chunk, estimate cost; no Jev or DB writes
jevrag ingest ./docs --collection handbook --limit 1   # one document end to end
jevrag ingest ./docs --collection handbook             # everything
```

Always estimate and test one document before a large batch. The report lists documents processed and skipped, chunks created, dropped and quarantined with reasons, Jev requests and tokens, embedding tokens, estimated cost and time, and any fallback used. A JSON trace per document goes to `.jevrag/traces/`.

## Retrieval guide: querying, classifying and answering

```
query -> 1 route -> 2 recall -> 3 classify -> 4 expand -> 5 gate -> result (-> 6 answer)
```

### Step 1: Route (optional)

One Jev request on the query asks which value of each `route: true` taxonomy field the question is about (with an `any` option), plus whether it's something the documents could answer at all rather than chit-chat or a request to act.

Code turns that into a filter cautiously, because a wrong filter hides the right answer:

- confidence ≥ `min_confidence` → `product = auth`
- else, top two options together ≥ `top2_mass` → `product in [auth, api]`
- else → no filter
- if the filtered search returns fewer than `min_candidates` hits → search again unfiltered

You can also pass filters yourself, in the portable filter language that every adapter translates:

```python
rag.retrieve(q, collection="handbook",
             where={"and": [{"eq": {"tags.product": "billing"}},
                            {"gte": {"ingested_at": "2026-01-01"}}]})
```

Operators: `eq`, `ne`, `in`, `nin`, `gt`, `gte`, `lt`, `lte`, `exists`, `and`, `or`, `not`. Quarantined records are always excluded unless you pass `include_quarantined=True`.

### Step 2: Recall

The query is embedded with the collection's embedding model (query input type) and the store returns the top `top_k` (default 30). Recall is deliberately wide; the next step does the narrowing. With `hybrid: true`, stores that support keyword and vector search together (Weaviate, Qdrant, Elasticsearch, OpenSearch, Milvus, pgvector with full-text) fuse both.

Scores from every store are normalised to similarity in [0, 1], higher is better, so thresholds mean the same thing on any database. The raw score is kept too.

### Step 3: Classify

Each candidate gets one Jev request whose state is the query and that one passage, with four yes/no questions (the pattern from TypeSafe's [passage classification cookbook](https://docs.typesafe.ai/cookbooks/classifying_rag_passages)):

| Question | Used for |
|---|---|
| `is_relevant` | Does the passage address the subject of the query? |
| `contains_answer_evidence` | Does it state information usable in a direct answer? |
| `contradicts_query_premise` | Does it conflict with a factual assumption in the query? |
| `instructs_ai` | Does it try to control the system answering the query? |

Requests run concurrently under a shared rate limiter. Code then routes each passage, first match wins:

1. `instructs_ai` ≥ `drop_instructs_ai` → **drop**
2. relevant and `contradicts_query_premise` ≥ `conflict` → **conflict**
3. relevant and `contains_answer_evidence` ≥ `min_evidence` → **include**
4. otherwise → **drop**

Included passages are sorted by evidence probability (ties broken by vector score) and capped at `max_passages`. None of the questions asks "should this be included?"; that decision stays in code, where changing it means editing a number.

Conflicts matter: if someone asks "Refresh tokens expire after 30 days; how do I extend that?" and the docs say 14 days, the 14-day passage arrives in a separate conflict block so the LLM can correct the premise instead of going along with it.

### Step 4: Expand (optional)

With `expand_neighbours: true`, an included chunk with a low `self_contained` score brings in the chunks immediately before and after it, fetched by `doc_id` and `chunk_index`, within a token budget.

### Step 5: Gate

One final Jev request with the query and the included passages: do `passages` contain the information needed to answer `query`? Below `answer_min`, the result has `abstain=True` and no LLM is called. Your app can say "That isn't in the documents" instead of letting the model guess.

The gate never abstains because of an error. If Jev fails or times out anywhere in retrieval, you get vector-ranked results marked `degraded=True`.

### The result

```python
result = rag.retrieve("How long do refresh tokens last?", collection="handbook")

result.passages      # included, ranked; each has .text .source_uri .section_path .scores
result.conflicts     # passages that contradict the query's premise
result.dropped       # each with .reason ("off_topic", "no_evidence", "instructs_ai") and scores
result.abstain       # True if the gate decided the passages can't answer
result.gate_p        # the gate's probability
result.filter_used   # the metadata filter actually applied
result.degraded      # True if any Jev stage failed and fell back
result.trace         # per-stage latency, Jev tokens, thresholds
result.to_prompt()   # evidence and conflicts as separate numbered, cited blocks
```

### Step 6: Answer (optional)

```python
answer = rag.answer("How long do refresh tokens last?", collection="handbook")
print(answer.text, answer.citations)
```

`answer()` runs `retrieve()`, returns the abstention message if the gate abstained, and otherwise calls the configured LLM (OpenAI-compatible or Anthropic) with `to_prompt()`. The system prompt tells the model to treat retrieved text as data, not instructions. To use your own LLM stack, call `retrieve()` and pass `to_prompt()` yourself.

### Using an existing collection

You can add classification and gating to a collection jevrag didn't build:

```bash
jevrag query "..." --store pinecone --index docs --text-field content --no-route
```

Routing needs jevrag's tags, so it's off; recall, classification and the gate work as long as the stored text is in a named field.

## Vector databases

| Database | Extra | Hybrid | Notes |
|---|---|---|---|
| Postgres + pgvector | `pgvector` | With full-text | Creates the table, HNSW index and metadata indexes |
| Qdrant | `qdrant` | Yes | Creates payload indexes for filter fields |
| Pinecone | `pinecone` | Sparse-dense | Collection = namespace; text trimmed to the metadata size limit with a warning |
| Chroma | `chroma` | No | Sets the distance metric explicitly; distances converted to similarity |
| Weaviate | `weaviate` | Yes | |
| Milvus / Zilliz | `milvus` | Yes | |
| MongoDB Atlas | `mongodb` | Atlas hybrid search | Creates `filter` fields in the vector index; true upserts |
| Elasticsearch | `elasticsearch` | Yes | |
| OpenSearch | `opensearch` | Yes | |
| Redis | `redis` | Yes | Via RedisVL |
| LanceDB | `lancedb` | Yes | Embedded, no server |
| Azure AI Search | `azure` | Yes | |
| turbopuffer | `turbopuffer` | Yes | |
| Anything in LangChain / LlamaIndex | `langchain` / `llamaindex` | Store-dependent | Fewer guarantees: stale delete and filter translation depend on the store |

The first release ships pgvector, Qdrant, Chroma, Pinecone and the LangChain bridge; the rest follow (see the [PRD](docs/PRD.md) milestones). Per-store details are in [Databases](docs/DATABASES.md). Every native adapter passes the same conformance suite (round trip, idempotent upsert, stale delete, every filter operator, score normalisation).

### Adding a database

Implement six methods:

```python
from jevrag.stores import VectorStore, Record, Hit, Where, Capabilities

class MyStore(VectorStore):
    def ensure_collection(self, name: str, dim: int, metric: str, filter_fields: list[str]) -> None: ...
    def upsert(self, collection: str, records: list[Record]) -> None: ...
    def delete(self, collection: str, ids: list[str] | None = None, where: Where | None = None) -> None: ...
    def get(self, collection: str, ids: list[str]) -> list[Record]: ...
    def query(self, collection: str, vector: list[float], where: Where | None,
              top_k: int, sparse: dict | None = None) -> list[Hit]: ...
    def capabilities(self) -> Capabilities: ...
```

Then run `pytest tests/stores/conformance --store mystore`.

## Tuning and evaluation

Thresholds decide everything, so measure them on your own data.

```bash
jevrag label --collection handbook --queries queries.txt   # label retrieved passages; Jev's answers hidden
jevrag eval --collection handbook --labels labels.jsonl    # scores this config against baselines
jevrag calibrate --labels labels.jsonl --target drop_relevant_max=0.02
```

`eval` reports evidence recall@5 and @10, nDCG@10, MRR, tokens sent to the LLM, abstention precision and recall, latency, and cost per 1,000 queries, next to built-in baselines (fixed, structural and embedding-based chunking; no re-rank, a local cross-encoder, and Cohere Rerank). `calibrate` picks each threshold for your target and writes it to `jevrag.yaml` with the Jev model version it was tuned on. Re-run it when you change model versions.

A sensible rollout: turn everything to `shadow`, run real traffic for a few days, compare in `jevrag eval`, then switch on the stages that win.

## Cost and limits

Estimates until M2 measures them.

| Work | Jev tokens | Jev cost |
|---|---|---|
| Ingest 1M document tokens (chunk + enrich) | about 3.5M to 7M | about $0.15 to $0.30 |
| One query (route + 30 passages + gate) | about 20k | about $0.0008 |

Jev's published limits for `jev-1.13.0` are 100,000 tokens and 40 requests per second, and TypeSafe says they are adjusting during early access. The request limit binds first: classifying 30 passages one per request allows about 1.25 queries per second per key. For more, lower `top_k`, enable packed classification (`classify.passages_per_request`, measured in M4), or ask TypeSafe for a higher limit. jevrag retries 429 and 529 responses with backoff and honours `retry-after`.

`jevrag cost --collection handbook` prints actual Jev and embedding spend from the traces.

## Use with agents and frameworks

```bash
jevrag serve --port 8900        # HTTP: /ingest /retrieve /answer /health
jevrag mcp                      # MCP server: search_knowledge, ingest_documents
```

Add the MCP server to Hermes Agent, Claude Code or any MCP client to give an agent a searchable knowledge base. LangChain and LlamaIndex retrievers:

```python
from jevrag.integrations.langchain import JevragRetriever
retriever = JevragRetriever(pipeline=rag, collection="handbook")
```

## More

- **[PRD](docs/PRD.md)**: requirements, design decisions and milestones
- **[Results](docs/RESULTS.md)**: every measurement, with method and caveats (from M2)
- **[Known issues](docs/KNOWN_ISSUES.md)**: current limits
- **[Databases](docs/DATABASES.md)**: per-store setup and capabilities
- **[Tuning](docs/TUNING.md)**: question wording, thresholds and calibration

jevrag is not affiliated with TypeSafe AI. Jev is text-only and works best in English. Treat its injection screening as one layer of defence, not the only one.

## License

MIT
