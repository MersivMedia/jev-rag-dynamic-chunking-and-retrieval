# jev-rag-dynamic-chunking-and-retrieval

Jev-steered chunking, ingestion and retrieval for any vector database.

**Cut documents where the meaning changes, keep junk and planted instructions out of your index, and send your LLM only the passages that answer the question, on the vector database you already use.**

- **Chunks that follow the text, not a character count.** Jev judges whether each sentence continues the point of the one before it; code places the cuts inside hard token limits. Headings, tables and code blocks are handled in code.
- **A cleaner index.** Filler and boilerplate are dropped, and text that tries to instruct an AI is quarantined, before anything is embedded. Every chunk can be tagged against your own taxonomy, with probabilities.
- **Less context, fewer wrong answers.** Each retrieved passage is classified as evidence, a conflict with the question, or noise. A final check abstains when the passages can't answer, before any LLM call.
- **Your database.** Adapters for Postgres + pgvector, Qdrant and Chroma, plus a bridge to any LangChain vector store, all held to one conformance suite. A Pinecone adapter is included as experimental.
- **Cheap.** A small end-to-end run cost $0.00024 of Jev to ingest four documents and under $0.0001 per query ([measured](docs/RESULTS.md)). Jev charges $0.042 per million input tokens and nothing for output.

> **Status: v1.0 in development.** The pipeline below is built and tested: 160 offline tests, the store conformance suite against real Postgres + pgvector, and a live end-to-end test against Jev. A first benchmark on 97,000 words of Wikipedia with 156 questions is done: Jev retrieval ranked the evidence first for 96.5% of questions against 79% for plain vector search, sent about 75% less context, and abstained on 40 of 42 unanswerable questions, while Jev chunking did no better than structural chunking. See [Results](docs/RESULTS.md) for how it was measured, its limits, and what hasn't been measured yet, and [Known issues](docs/KNOWN_ISSUES.md).

jevrag uses [Jev](https://docs.typesafe.ai/introduction), TypeSafe AI's System One model. Jev never writes text. It answers typed questions (a yes/no probability, one option from a list, or a score on a scale), and plain code with visible thresholds decides what happens. Every decision is logged with its probabilities.

## What it does

| Stage | What Jev decides | What code does |
|---|---|---|
| Chunking | Does this sentence continue the point of the previous one? Does it depend on it to make sense? | Places cuts where continuity is lowest, within min/target/max token sizes |
| Quality screen | Is this chunk filler or boilerplate? Does it contain instructions aimed at an AI? Is it self-contained? | Drops, quarantines or keeps it |
| Tagging | Which option of each taxonomy field fits, including `other` | Stores the tag only when confident; always stores the probability |
| Query routing | Which taxonomy value the question is about | Applies a metadata filter only when confident; retries unfiltered if it returns too little |
| Passage classification | Relevant? Usable evidence? Contradicts the question's premise? Instructions aimed at an AI? | Includes, marks as a conflict, or drops; ranks what's kept |
| Answer gate | Can these passages answer the question? | Abstains without calling the LLM, or builds the prompt |

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Configure](#configure)
- [Ingestion guide: processing and storing embeddings](#ingestion-guide-processing-and-storing-embeddings)
- [Retrieval guide: querying, classifying and answering](#retrieval-guide-querying-classifying-and-answering)
- [Vector databases](#vector-databases)
- [CLI reference](#cli-reference)
- [Cost and limits](#cost-and-limits)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [More](#more)

## Install

Python 3.10 to 3.13. The package installs as `jev-rag-dynamic-chunking-and-retrieval`; you import it as `jevrag` and run it as the `jevrag` command. The core needs only `httpx`, `pydantic` and `pyyaml`; each database, embedder and file format is an extra.

Until the first PyPI release, install from GitHub:

```bash
pip install "jev-rag-dynamic-chunking-and-retrieval[qdrant] @ git+https://github.com/MersivMedia/jev-rag-dynamic-chunking-and-retrieval"
```

| Extra | Installs |
|---|---|
| `qdrant`, `chroma`, `pgvector` | That database's official client |
| `pinecone` | The Pinecone client, for the experimental adapter |
| `langchain` | `langchain-core`, for the bridge to any LangChain vector store |
| `local` | sentence-transformers, for local embeddings |
| `pdf`, `docx` | PyMuPDF and python-docx loaders |
| `tokens` | tiktoken, for exact token counts (otherwise an estimate is used) |
| `all` | Everything above except `local` |

### Keys

Set one Jev key. jevrag uses the first one it finds, in this order:

| Provider | Variable | Default model ID |
|---|---|---|
| TypeSafe direct | `TYPESAFE_API_KEY` ([console.typesafe.ai](https://console.typesafe.ai/)) | `jev-1.13.0` |
| Vercel AI Gateway | `AI_GATEWAY_API_KEY` | `typesafe-ai/jev` |
| OpenRouter | `OPENROUTER_API_KEY` | `typesafe/jev-1.13` |

Then add the key for your embedding provider (`OPENAI_API_KEY`, or reuse `AI_GATEWAY_API_KEY` with the `gateway:` embedder) and your database's connection settings.

Put them in a `.env` file: [`.env.example`](.env.example) lists every variable jevrag reads, blank, with a note on each. The `jevrag` CLI loads `./.env` before every command. Variables already set in your shell win, blank lines in the file are ignored, and it warns if the file is readable by other users. Use `--env-file path` for another file or `--no-env-file` to skip it. The Python API doesn't read `.env` on its own; call `jevrag.envfile.load_env_file()` first if you want the same behaviour.

Without a Jev key everything still runs, but chunking falls back to `structural`, nothing is screened or tagged, and retrieval returns plain vector ranking marked `degraded`.

## Quickstart

```bash
docker run -d -p 6333:6333 qdrant/qdrant       # or any supported database

jevrag init --store qdrant --embedder openai:text-embedding-3-small
                                               # writes jevrag.yaml, plus a blank .env (chmod 600) if none exists
# edit .env: set TYPESAFE_API_KEY (or another Jev key) and OPENAI_API_KEY
jevrag check                                   # one Jev call, one embedding, one store round trip
jevrag ingest ./docs --collection handbook --dry-run
jevrag ingest ./docs --collection handbook
jevrag query "How long do refresh tokens last?" --collection handbook
```

`query` prints the kept passages with their evidence scores, any conflicts, the filter used and the gate's verdict. Add `-v` to see what was dropped and why, `--json` for machine output, or `--answer` to have an LLM write the answer from the passages.

The same in Python:

```python
from jevrag import Pipeline

rag = Pipeline.from_config("jevrag.yaml")
report = rag.ingest(["./docs"], collection="handbook")
print(report.summary())        # docs, chunks, dropped, quarantined, Jev requests, tokens, cost

result = rag.retrieve("How long do refresh tokens last?", collection="handbook")
if result.abstain:
    print("Not in the documents:", result.reason)
else:
    for p in result.passages:
        print(f"{p.scores['contains_answer_evidence']:.2f}  {p.citation}")
        print(p.text[:200])
```

Every sync method has an async twin (`aingest`, `aretrieve`, `aanswer`) for use inside an event loop.

## Configure

Two sample files at the repo root:

- [`jevrag.example.yaml`](jevrag.example.yaml): every setting with its default and a comment, plus a ready-to-uncomment block for each database. `jevrag init --full` writes the same file. Plain `jevrag init` writes the short version below.
- [`.env.example`](.env.example): every environment variable, blank.

Secrets never go in the YAML; it only names the variable to read (`api_key_env`, `dsn_env`). The short config:

```yaml
jev:
  backend: auto                # auto | typesafe | vercel | openrouter
  # model: jev-1.13.0          # pin a version; each backend has its own default ID
  cache_dir: .jevrag/cache     # answers are cached by content hash
  max_rps: 30                  # published limit is 40 requests/s

store:
  kind: qdrant
  url: http://localhost:6333
  # api_key_env: QDRANT_API_KEY

embedder:
  model: openai:text-embedding-3-small
  batch_size: 128

chunking:
  method: jev                  # jev | structural | fixed | semantic-embedding
  mode: on                     # on | shadow | off
  min_tokens: 64
  target_tokens: 350
  max_tokens: 800
  overlap_tokens: 0

enrich:
  mode: on
  drop_low_information: 0.85
  drop_boilerplate: 0.85
  quarantine_instructs_ai: 0.70
  tag_min_confidence: 0.50
  # taxonomy: taxonomy.yaml

retrieve:
  top_k: 30
  expand_neighbours: false
  route: { mode: on, min_confidence: 0.60, top2_mass: 0.80, min_candidates: 5 }
  classify:
    mode: on
    drop_instructs_ai: 0.70
    min_relevant: 0.50
    min_evidence: 0.40
    conflict: 0.60
    max_passages: 8
  gate: { mode: on, answer_min: 0.35 }

answer:                        # only used by answer() / `jevrag query --answer`
  provider: openai             # openai | anthropic | gateway
  model: gpt-4.1-mini
```

**The thresholds are starting points, not measured defaults.** They produced the right decisions on the small probes in [Results](docs/RESULTS.md), which is not the same as being tuned. Check them against your own documents with `jevrag query -v` before relying on them.

Every Jev stage has a `mode`. `shadow` computes and logs Jev's decision but acts on the fallback's result (structural chunks, keep every chunk, vector order, never abstain), so you can compare before switching it on. The shadow decisions appear in the ingest traces and in `result.trace`.

Unknown keys are rejected with an error, so a typo like `topk` fails loudly.

## Ingestion guide: processing and storing embeddings

Ingestion turns files into records in your vector database. Five steps, each inspectable on its own.

```
files -> 1 parse -> 2 chunk -> 3 enrich -> 4 embed -> 5 store
```

### Step 1: Parse

Loaders read `.md`, `.txt`, `.rst`, `.html`, `.pdf` (text layer only, needs the `pdf` extra) and `.docx` (needs `docx`). HTML is converted to Markdown with scripts, styles, navigation, headers, footers and asides removed. Each document becomes blocks (heading, paragraph, list item, table, code or quote), each with its heading path ("Guide > Auth > Tokens") and character offsets.

What code decides, never Jev:

- **Headings are hard boundaries.** No chunk spans two sections.
- **Tables and code blocks stay whole**, split by rows or lines only if they exceed `max_tokens`.
- **Sentences** are split by a segmenter that handles "e.g.", "U.S.", "Dr.", decimals, version numbers and URLs. A sentence longer than `max_tokens` is split at whitespace.

Scanned PDFs, images and audio need OCR or transcription first; Jev reads text only.

To bring your own parsed text:

```python
from jevrag import Document

docs = [Document(doc_id="kb-142", text=body, title="Refunds policy",
                 source_uri="https://example.com/kb/142", metadata={"team": "billing"})]
rag.ingest(docs, collection="handbook")
```

`doc_id` should be stable across runs (a URL, path or database key). It's what makes re-ingestion replace old chunks instead of duplicating them. For files, it's the path relative to the folder you ingest. `format` is `markdown` (default), `text` or `html`. Scalar `metadata` values are stored as `m_<key>` and can be filtered on.

### Step 2: Chunk

Within each section, jevrag sends Jev the section text (in windows sized to Jev's request budget) plus two yes/no questions for every adjacent pair of sentences, all in **one request per window**:

| Question | Wording |
|---|---|
| `continues` | In the document, does `sentence` continue the specific point that `previous` is making? |
| `refers_back` | Does `sentence` depend on `previous` to be understood, for example by referring back to it with words like this, it, these or such? |

Each question carries its own sentence pair. An earlier design referred to sentences by position (`sentences[4]`), and Jev answered those about the wrong sentences: mean error 0.63 on a labelled probe, against 0.07 with the pair inline ([Results](docs/RESULTS.md#boundary-question-format)). Set `chunking.boundary.style: keyed` to use named keys instead, about 25% fewer tokens at slightly lower accuracy on that probe.

Gaps that code already decides (after a heading, between sections) are never asked. A 2,000-sentence manual takes a handful of requests, not 2,000.

Code then places the cuts:

1. Each gap gets a cut cost: `0.6 × continues + 0.4 × refers_back`, reduced at paragraph breaks.
2. A dynamic-programming pass chooses the cuts with the lowest total cost plus a penalty for straying from `target_tokens`, and never exceeds `max_tokens`.
3. A gap where `refers_back` ≥ 0.5 is protected: it's only cut if the size limit forces it, so "This means..." stays with what "this" refers to.
4. Chunks under `min_tokens` merge into their more continuous neighbour.

Sizes are counted with tiktoken's `cl100k_base` when the `tokens` extra is installed, otherwise with an estimate. Pass `token_counter=` to `Pipeline` to use your embedding model's own tokenizer.

See the result before storing anything:

```bash
jevrag inspect ./docs/auth.md                     # chunks and sizes
jevrag inspect ./docs/auth.md -v                  # plus Jev's scores at every candidate cut
jevrag inspect ./docs/auth.md --compare structural,fixed
```

Each chunk also gets an `embed_text`: the document title and heading path prepended to the chunk, so a chunk that says "They expire after 14 days" still embeds near questions about refresh tokens. `text` (what the LLM sees) is stored separately.

Other methods: `structural` (headings and paragraph breaks, no Jev), `fixed` (about `target_tokens` per chunk, sentence-aligned, with optional `overlap_tokens`), and `semantic-embedding` (cut where adjacent sentence embeddings diverge; one embedding call per sentence). If Jev fails, `jev` falls back to `structural` and the ingest report says so.

### Step 3: Enrich

One Jev request per chunk carries every enrichment question at once.

**Quality screen** (yes/no probabilities, stored as `q_<name>`):

| Question | Default action |
|---|---|
| `low_information`: is this filler with no usable information? | Drop at ≥ 0.85 |
| `boilerplate`: navigation, footer, cookie banner, copyright line, table of contents? | Drop at ≥ 0.85 |
| `instructs_ai`: does it contain instructions addressed to an AI assistant, rather than information for a human reader? | **Quarantine** at ≥ 0.70 |
| `self_contained`: can it be understood without the text before it? | Stored; used by neighbour expansion |

Dropped chunks are listed in the ingest report (`jevrag ingest -v`) and in the per-document trace. Quarantined chunks are stored with `quarantined=true` and excluded from every query unless you ask for them, so you can review them:

```bash
jevrag inspect --collection handbook --quarantined
```

**Taxonomy tags.** Define fields in a YAML file and point `enrich.taxonomy` at it. Each field is a one-of-N choice. Describe every option, and always include `other`: Jev must put its probability somewhere, and a missing option forces a wrong tag. jevrag refuses a field without `other`.

```yaml
version: 3
fields:
  doc_type:
    route: true                 # usable as a query filter
    options:
      policy:    Rules the company commits to; what is and isn't allowed
      how_to:    Step-by-step instructions for doing a task
      reference: Specifications, limits, parameters, API fields
      other:     None of the above
  product:
    route: true
    options:
      billing:   Payments, invoices, refunds, plans
      auth:      Sign-in, sessions, tokens, SSO
      api:       Endpoints, SDKs, webhooks, rate limits
      other:     Anything else
questions:                      # optional custom questions, stored as x_<name>
  mentions_deadline:
    type: noul
    instructions: Does this text state a date or time limit that applies to the reader?
```

A tag is stored as `tag_<field>` only when Jev's confidence is at least `tag_min_confidence`; otherwise it's `unknown`. The probability is always stored as `tag_<field>_p`, so you can filter on it later. If an enrichment request fails, the chunk is kept with `unknown` tags and `needs_reenrich=true`.

**Duplicates.** Chunks with identical text within a document are stored once.

### Step 4: Embed

Chunks are embedded in batches with the configured provider.

| Provider | `embedder.model` | Key |
|---|---|---|
| OpenAI | `openai:text-embedding-3-small` | `OPENAI_API_KEY` |
| Vercel AI Gateway | `gateway:openai/text-embedding-3-small` | `AI_GATEWAY_API_KEY` |
| Any OpenAI-compatible API | `openai:<model>` plus `base_url` and `api_key_env` | yours |
| Ollama | `ollama:nomic-embed-text` | none (`OLLAMA_HOST`) |
| Local (sentence-transformers) | `local:BAAI/bge-small-en-v1.5` | none |
| Testing only | `hash:256` (deterministic, no semantic quality) | none |
| Anything else | a Python callable `list[str] -> list[list[float]]` passed as `embedder=` | yours |

Model names are examples; use any model your provider serves. Extra keys under `embedder:` (such as `base_url`, `api_key_env`, `dimensions`, `batch_size`) are passed to the provider.

The first ingest writes a **collection manifest**: embedding provider and model, dimension, distance metric, chunker, taxonomy version and jevrag version. Later ingests or queries with a different embedding model are refused with a clear error, because mixing embedding models in one collection silently ruins retrieval. To switch models, ingest into a new collection.

### Step 5: Store

Each chunk becomes one record:

| Field | Meaning |
|---|---|
| `id` | UUIDv5 of `doc_id` + chunk content hash: identical on every run and valid in every store |
| vector | The embedding of `embed_text` |
| text | Chunk text shown to the LLM |
| `doc_id`, `source_uri`, `title`, `section_path` | Where it came from |
| `chunk_index`, `char_start`, `char_end`, `tokens` | Position and size, for citations and neighbour expansion |
| `tag_<field>`, `tag_<field>_p` | Taxonomy tag and its probability |
| `q_<name>`, `x_<name>` | Quality and custom question answers |
| `m_<key>` | Your document metadata |
| `quarantined` | Excluded from queries when true |
| `chunker`, `jev_model`, `pipeline_version`, `ingested_at`, `content_hash`, `doc_hash` | Provenance |

Each adapter maps these to native fields (Qdrant payload, Postgres columns and `jsonb`, Chroma metadata, Pinecone metadata) and creates what filtering needs, such as Qdrant payload indexes and Postgres expression indexes.

**Re-ingesting is safe.** Unchanged documents (same text and title) are skipped. A changed document has its current chunks upserted and its stale ones deleted. `--force` reprocesses everything. Removing a document:

```bash
jevrag delete --collection handbook --doc kb-142
```

### Ingest report

```bash
jevrag ingest ./docs --collection handbook --dry-run   # parse and estimate Jev requests and cost; no Jev or DB writes
jevrag ingest ./docs --collection handbook --limit 1   # one document end to end
jevrag ingest ./docs --collection handbook -v          # everything, with per-document detail
```

Estimate and test one document before a large batch. The report lists documents ingested, skipped and failed, chunks stored, dropped and quarantined, Jev requests (and how many came from the cache), tokens and cost, embedding tokens, time, and any fallback used. A JSON trace per document goes to `.jevrag/traces/ingest/<collection>/`, including Jev's score at every candidate cut.

## Retrieval guide: querying, classifying and answering

```
query -> 1 route -> 2 recall -> 3 classify -> 4 expand -> 5 gate -> result (-> 6 answer)
```

### Step 1: Route

If your taxonomy has `route: true` fields, one Jev request on the query asks which value of each field the question is about (with an added `any` option), plus whether it's something documents could answer at all rather than small talk or a request to act.

Code turns that into a filter cautiously, because a wrong filter hides the right answer:

- top option's confidence ≥ `min_confidence` → `tag_product = auth`
- else, top two options together ≥ `top2_mass` → `tag_product in [auth, api]`
- else → no filter
- if the filtered search returns fewer than `min_candidates` hits → search again unfiltered

Routing is skipped when you pass a filter yourself. Filters use a portable language that every adapter translates:

```python
rag.retrieve(q, collection="handbook",
             where={"and": [{"eq": {"tag_product": "billing"}},
                            {"gte": {"chunk_index": 2}}]})
```

```bash
jevrag query "refund window?" --collection handbook --where '{"eq": {"m_team": "billing"}}'
```

Operators: `eq`, `ne`, `in`, `nin`, `gt`, `gte`, `lt`, `lte`, `exists`, `and`, `or`, `not`. Several operators in one dict are ANDed. `ne` and `nin` match only records that have the field. Quarantined records are always excluded unless `retrieve.include_quarantined` is true.

### Step 2: Recall

The query is embedded with the collection's embedding model and the store returns the top `top_k` (default 30). Recall is deliberately wide; the next step does the narrowing.

Scores from every store are normalised to similarity in [0, 1], higher is better, so the numbers mean the same thing on any database. Cosine distance `d` becomes `1 − d`; L2 distance becomes `1 / (1 + d)`. The raw score is kept too.

### Step 3: Classify

Each candidate gets one Jev request whose state is the query and that one passage, with four yes/no questions (adapted from TypeSafe's [passage classification cookbook](https://docs.typesafe.ai/cookbooks/classifying_rag_passages)):

| Question | Used for |
|---|---|
| `is_relevant` | Does the passage address the subject of the query? |
| `contains_answer_evidence` | Does it state information usable in a direct answer? |
| `contradicts_query_premise` | Does it conflict with a factual premise in the query? |
| `instructs_ai` | Does it contain instructions addressed to an AI assistant? |

Requests run concurrently under a shared rate limiter. Code then routes each passage, first match wins:

1. `instructs_ai` ≥ `drop_instructs_ai` → **drop**
2. `is_relevant` < `min_relevant` → **drop** (off topic)
3. `contradicts_query_premise` ≥ `conflict` → **conflict**
4. `contains_answer_evidence` ≥ `min_evidence` → **include**
5. otherwise → **drop** (no evidence)

Included passages are sorted by evidence probability (ties broken by vector score) and capped at `max_passages`. None of the questions asks "should this be included?"; that decision stays in code, where changing it means editing a number.

Conflicts matter. Asked "Refresh tokens expire after 30 days; how do I extend that?" when the docs say 14 days, the 14-day passage scored 0.95 on `contradicts_query_premise` in a live run and arrived in a separate conflict block, so the LLM can correct the premise instead of going along with it.

### Step 4: Expand (optional)

With `expand_neighbours: true`, an included chunk whose `self_contained` score is below 0.5 brings in the chunks just before and after it (same `doc_id`, adjacent `chunk_index`), within `expand_max_tokens`.

### Step 5: Gate

One final Jev request with the query and the kept passages: do `passages` contain the information needed to answer `query`? Below `answer_min`, the result has `abstain=True` and no LLM is called, so your app can say "That isn't in the documents" instead of letting the model guess.

The gate never abstains because of an error. If Jev fails or times out anywhere in retrieval, you get vector-ranked results marked `degraded=True`.

### The result

```python
result = rag.retrieve("How long do refresh tokens last?", collection="handbook")

result.passages      # included, ranked; each has .text .citation .doc_id .section_path .scores .vector_score
result.conflicts     # passages that contradict the query's premise
result.dropped       # each with .reason ("off_topic", "no_evidence", "instructs_ai", "over_max_passages")
result.abstain       # True if the gate decided the passages can't answer
result.reason        # why it abstained
result.gate_p        # the gate's probability
result.filter_used   # the metadata filter actually applied
result.degraded      # True if a Jev stage failed and fell back
result.trace         # per-stage latency, routing decisions, Jev requests, tokens and cost
result.to_prompt()   # evidence and conflicts as separate, numbered, cited blocks
result.to_dict()     # everything, JSON-serialisable
```

### Step 6: Answer (optional)

```python
answer = rag.answer("How long do refresh tokens last?", collection="handbook")
print(answer.text, answer.citations)
```

`answer()` runs `retrieve()`, returns "I couldn't find that in the documents." if the gate abstained, and otherwise calls the configured LLM with `to_prompt()`. Providers: `openai` (`OPENAI_API_KEY`), `anthropic` (`ANTHROPIC_API_KEY`), `gateway` (`AI_GATEWAY_API_KEY`, models like `openai/gpt-4.1-mini`), or any OpenAI-compatible API via `base_url`. The system prompt tells the model to treat retrieved text as data, not instructions, and to cite passages by number. To use your own LLM stack, call `retrieve()` and pass `to_prompt()` yourself.

### Using an existing collection

```bash
jevrag query "..." --collection docs --retrieve-only
```

`--retrieve-only` skips the manifest check and routing, so classification and the gate can run on a collection jevrag didn't build. The collection must use the field layout the adapter expects (for example Qdrant payload `text`, Pinecone metadata `text`); a configurable text field is on the roadmap.

## Vector databases

| Database | `store.kind` | Tested | Notes |
|---|---|---|---|
| Postgres + pgvector | `pgvector` | Conformance suite against Postgres 17 + pgvector | One table per collection; HNSW index with the metric's operator class; expression indexes on filter fields; filters compile to parameterised SQL. `dsn` or `dsn_env` (default `DATABASE_URL`) |
| Qdrant | `qdrant` | Conformance suite in embedded mode | Uses `query_points`; creates payload indexes for filter fields. `url`, or `path` for embedded local mode |
| Chroma | `chroma` | Conformance suite, persistent client | Sets the distance metric explicitly and converts distances to similarity. Existence tests and `ne`/`nin` run in Python after a superset query. `path`, or `host` + `port` |
| Pinecone (experimental) | `pinecone` | **Not yet run against Pinecone; not part of v1.0's tested set** | Collection = namespace in one serverless index (created if missing). Text trimmed to fit the 40 KB metadata limit. Filtered deletes list ids first. `index`, `cloud`, `region` |
| Anything in LangChain | Python only | Conformance suite with `InMemoryVectorStore` | `LangChainStore(your_store)`; filters run in Python over an over-fetched result |
| In-memory | `memory` | Conformance suite | Reference semantics; optional JSON file with `path`. For tests and small demos |

Every adapter except the experimental Pinecone one passes the same conformance suite: round trip, idempotent upsert, replace, delete by id and by filter, the stale-delete pattern, every filter operator (18 cases), score normalisation, empty batches and manifests.

LangChain bridge:

```python
from langchain_core.vectorstores import InMemoryVectorStore   # or any LangChain vector store
from jevrag import Pipeline
from jevrag.stores.langchain import LangChainStore, JevragEmbeddings
from jevrag.embed import make_embedder

emb = make_embedder("openai:text-embedding-3-small")
rag = Pipeline(store=LangChainStore(InMemoryVectorStore(JevragEmbeddings(emb))), embedder=emb)
```

### Adding a database

Subclass `VectorStore` and implement seven methods:

```python
from jevrag.stores import VectorStore, Capabilities
from jevrag.types import Record, Hit

class MyStore(VectorStore):
    kind = "mystore"
    def ensure_collection(self, name, dim, metric, filter_fields): ...
    def upsert(self, collection, records: list[Record]): ...
    def delete(self, collection, ids=None, where=None): ...
    def get(self, collection, ids) -> list[Record]: ...
    def list_ids(self, collection, where=None, limit=100_000) -> list[str]: ...
    def query(self, collection, vector, where, top_k) -> list[Hit]: ...
    def capabilities(self) -> Capabilities: ...
```

`where` arrives in the portable language; `jevrag.stores.filters.parse()` turns it into a small tree to translate, `push_down_not()` removes `not` for stores that lack it, and `matches()` is the reference evaluator every adapter must agree with. Manifests default to a sidecar file; override `get_manifest`/`put_manifest` to store them natively. Then add your store to the fixture in `tests/stores/test_conformance.py` and make it pass.

## CLI reference

| Command | What it does |
|---|---|
| `jevrag init --store <kind> --embedder <spec>` | Write a starter `jevrag.yaml` and, if missing, a blank `.env`. `--full` for every setting; `--force` overwrites the YAML (never `.env`) |
| `jevrag check` | One Jev call, one embedding, one store round trip |
| `jevrag ingest <paths...> --collection <name>` | Parse, chunk, enrich, embed, store. `--dry-run`, `--limit N`, `--force`, `-v`, `--json` |
| `jevrag query "<question>" --collection <name>` | Retrieve. `--where JSON`, `--top-k`, `--no-route`, `--retrieve-only`, `--answer`, `-v`, `--json` |
| `jevrag inspect <file>` | Show how a file would be chunked. `--compare structural,fixed`, `-v` for cut scores |
| `jevrag inspect --collection <name>` | List stored records with tags and scores. `--quarantined` |
| `jevrag delete --collection <name> --doc <doc_id>` | Delete one document's records |

`-c path/to/jevrag.yaml` selects a config file; `--env-file path` or `--no-env-file` controls `.env` loading (both go before the command). Set `JEVRAG_DEBUG=1` for full tracebacks.

## Cost and limits

Measured on the small live runs in [Results](docs/RESULTS.md), through Vercel AI Gateway:

| Work | Jev requests | Jev input tokens | Jev cost |
|---|---|---|---|
| Ingest 4 short documents (chunk + enrich) | 9 | 5,672 | $0.00024 |
| One query (route + classify up to 4 candidates + gate) | 3 to 5 | 1,250 to 2,060 | $0.00005 to $0.00009 |

Cost scales with `top_k`: each candidate passage is one classification request of roughly 400 tokens plus the passage. With the default `top_k: 30` and passages of about 350 tokens, expect roughly 25,000 input tokens, about $0.001 per query (an estimate, not yet measured at that size). Repeated inputs are served from the on-disk cache at no cost.

Jev's published limits for `jev-1.13.0` are 100,000 tokens and 40 requests per second, and TypeSafe says they are adjusting them during early access. The request limit binds first: with `top_k: 30`, one key handles about 1.25 queries per second. jevrag paces requests with a shared limiter (`jev.max_rps`, default 30), retries 408, 429, 5xx and 529 responses with backoff, and honours `retry-after`. For more throughput, lower `top_k` or ask TypeSafe for a higher limit.

## Testing

```bash
pip install -e ".[qdrant,chroma,pgvector,langchain,dev]"
pytest                                   # offline: fake Jev over httpx.MockTransport, no keys, no network

# pgvector conformance against a real database
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=jevrag pgvector/pgvector:pg17
JEVRAG_TEST_PG_DSN=postgresql://postgres:jevrag@localhost:5432/postgres pytest tests/stores -k pgvector

# live end-to-end against Jev (costs well under $0.01)
JEVRAG_LIVE=1 TYPESAFE_API_KEY=... OPENAI_API_KEY=... pytest tests/test_live.py
```

CI runs the offline suite on Python 3.10, 3.12 and 3.13, and the pgvector conformance suite against a Postgres service container.

## Roadmap

Planned, not in this release. Tracked in the [PRD](docs/PRD.md) milestones:

- **Measured defaults:** `jevrag eval` against baseline chunkers and re-rankers on public datasets, threshold calibration, and published results.
- **More databases:** Pinecone verified against a live index, then Weaviate, Milvus, MongoDB Atlas, Elasticsearch, OpenSearch, Redis, LanceDB, Azure AI Search, turbopuffer, and a LlamaIndex bridge.
- **More embedders:** native Cohere, Voyage, Gemini and Mistral clients with document/query input types.
- **Hybrid search**, near-duplicate removal across documents, packed multi-passage classification.
- **Maintenance commands:** `reenrich` (new taxonomy without re-embedding), `reembed` (new embedder into a new collection).
- **Serving:** HTTP server, MCP server for agents, LangChain and LlamaIndex retriever wrappers.

## More

- **[PRD](docs/PRD.md)**: requirements, design decisions and milestones
- **[Results](docs/RESULTS.md)**: every measurement so far, with method and caveats
- **[Known issues](docs/KNOWN_ISSUES.md)**: current limits

jevrag is not affiliated with TypeSafe AI. Jev reads text only and works best in English. Treat its injection screening as one layer of defence, never the only one.

## License

MIT
