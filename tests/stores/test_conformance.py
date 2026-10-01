"""Adapter conformance suite (FR-S6). Every store must pass every test here.

Runs against: memory, qdrant (embedded), chroma (ephemeral), langchain (InMemoryVectorStore)
always; pgvector when JEV_RETRIEVAL_TEST_PG_DSN is set; pinecone when JEV_RETRIEVAL_TEST_PINECONE_INDEX
and PINECONE_API_KEY are set (it creates and deletes a namespace).
"""

from __future__ import annotations

import math
import os
import uuid

import pytest

from jev_retrieval.stores import MemoryStore
from jev_retrieval.stores.base import BASE_FILTER_FIELDS
from jev_retrieval.stores.filters import FilterError
from jev_retrieval.types import Record

DIM = 4


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _records():
    rows = [
        ("a", [1, 0, 0, 0], {"doc_id": "d1", "chunk_index": 0, "tag": "billing", "score": 0.9, "flag": True}),
        ("b", [0.9, 0.1, 0, 0], {"doc_id": "d1", "chunk_index": 1, "tag": "auth", "score": 0.5, "flag": False}),
        ("c", [0, 1, 0, 0], {"doc_id": "d2", "chunk_index": 0, "tag": "billing", "score": 0.2, "flag": True}),
        ("d", [0, 0, 1, 0], {"doc_id": "d2", "chunk_index": 1, "tag": "api", "score": 0.7}),
        ("e", [0, 0, 0, 1], {"doc_id": "d3", "chunk_index": 0, "score": 0.1, "flag": False}),
    ]
    return [Record(str(uuid.uuid5(uuid.NAMESPACE_URL, k)), _unit(v), f"text {k}", {**m, "key": k})
            for k, v, m in rows]


def _key(recs_or_hits):
    return sorted(r.metadata["key"] for r in recs_or_hits)


STORES = ["memory", "qdrant", "chroma", "langchain"]
if os.environ.get("JEV_RETRIEVAL_TEST_PG_DSN"):
    STORES.append("pgvector")
if os.environ.get("JEV_RETRIEVAL_TEST_PINECONE_INDEX") and os.environ.get("PINECONE_API_KEY"):
    STORES.append("pinecone")


@pytest.fixture(params=STORES)
def store(request, tmp_path):
    kind = request.param
    name = f"conf_{uuid.uuid4().hex[:8]}"
    if kind == "memory":
        s = MemoryStore()
    elif kind == "qdrant":
        pytest.importorskip("qdrant_client")
        from jev_retrieval.stores.qdrant import QdrantStore
        s = QdrantStore(":memory:")
    elif kind == "chroma":
        pytest.importorskip("chromadb")
        from jev_retrieval.stores.chroma import ChromaStore
        s = ChromaStore(str(tmp_path / "chroma"))
    elif kind == "langchain":
        pytest.importorskip("langchain_core")
        from langchain_core.embeddings import FakeEmbeddings
        from langchain_core.vectorstores import InMemoryVectorStore
        from jev_retrieval.stores.langchain import LangChainStore
        s = LangChainStore(InMemoryVectorStore(FakeEmbeddings(size=DIM)))
        s.manifest_dir = str(tmp_path / "manifests")
    elif kind == "pgvector":
        from jev_retrieval.stores.pgvector import PgVectorStore
        s = PgVectorStore(os.environ["JEV_RETRIEVAL_TEST_PG_DSN"])
    elif kind == "pinecone":
        from jev_retrieval.stores.pinecone import PineconeStore
        s = PineconeStore(os.environ["JEV_RETRIEVAL_TEST_PINECONE_INDEX"])
    s.ensure_collection(name, DIM, "cosine", {**BASE_FILTER_FIELDS, "tag": "keyword", "score": "float",
                                               "flag": "bool", "key": "keyword"})
    s.upsert(name, _records())
    s.test_collection = name  # type: ignore[attr-defined]
    s.test_kind = kind  # type: ignore[attr-defined]
    yield s
    try:
        s.drop_collection(name)
    except Exception:
        pass
    s.close()


def _c(s):
    return s.test_collection


def test_round_trip(store):
    recs = _records()
    got = store.get(_c(store), [recs[0].id, recs[3].id, str(uuid.uuid4())])
    assert _key(got) == ["a", "d"]
    a = next(r for r in got if r.metadata["key"] == "a")
    assert a.text == "text a"
    assert a.metadata["doc_id"] == "d1" and a.metadata["chunk_index"] == 0 and a.metadata["flag"] is True
    assert a.metadata["score"] == pytest.approx(0.9)


def test_query_order_and_normalised_scores(store):
    hits = store.query(_c(store), _unit([1, 0, 0, 0]), None, 3)
    if store.test_kind == "langchain":  # FakeEmbeddings: LangChain re-embeds; ordering is the wrapped store's
        assert len(hits) == 3
        return
    assert [h.metadata["key"] for h in hits[:2]] == ["a", "b"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-3)
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_upsert_is_idempotent(store):
    store.upsert(_c(store), _records())
    assert len(store.list_ids(_c(store))) == 5


def test_upsert_replaces(store):
    r = _records()[0]
    r.text = "changed"
    r.metadata = {**r.metadata, "tag": "changed"}
    store.upsert(_c(store), [r])
    got = store.get(_c(store), [r.id])
    assert got[0].text == "changed" and got[0].metadata["tag"] == "changed"
    assert len(store.list_ids(_c(store))) == 5


def test_delete_by_ids_and_filter(store):
    recs = _records()
    store.delete(_c(store), ids=[recs[0].id])
    assert len(store.list_ids(_c(store))) == 4
    store.delete(_c(store), where={"eq": {"doc_id": "d2"}})
    assert sorted(store.list_ids(_c(store))) == sorted([recs[1].id, recs[4].id])
    store.delete(_c(store))  # neither ids nor where: deletes nothing
    assert len(store.list_ids(_c(store))) == 2


def test_stale_delete_pattern(store):
    """Re-ingesting d1 with one chunk: delete d1's ids that are no longer present."""
    recs = _records()
    keep = {recs[0].id}
    existing = set(store.list_ids(_c(store), {"eq": {"doc_id": "d1"}}))
    assert existing == {recs[0].id, recs[1].id}
    store.delete(_c(store), ids=sorted(existing - keep))
    assert set(store.list_ids(_c(store), {"eq": {"doc_id": "d1"}})) == keep


def test_empty_batches(store):
    store.upsert(_c(store), [])
    store.delete(_c(store), ids=[])
    assert store.get(_c(store), []) == []
    assert len(store.list_ids(_c(store))) == 5


FILTER_CASES = [
    ({"eq": {"tag": "billing"}}, ["a", "c"]),
    ({"ne": {"tag": "billing"}}, ["b", "d"]),  # e has no tag: comparisons need the field
    ({"in": {"tag": ["auth", "api"]}}, ["b", "d"]),
    ({"nin": {"tag": ["auth", "api"]}}, ["a", "c"]),
    ({"gt": {"score": 0.5}}, ["a", "d"]),
    ({"gte": {"score": 0.5}}, ["a", "b", "d"]),
    ({"lt": {"score": 0.2}}, ["e"]),
    ({"lte": {"score": 0.2}}, ["c", "e"]),
    ({"eq": {"flag": True}}, ["a", "c"]),
    ({"eq": {"flag": False}}, ["b", "e"]),
    ({"eq": {"chunk_index": 1}}, ["b", "d"]),
    ({"exists": "tag"}, ["a", "b", "c", "d"]),
    ({"and": [{"eq": {"doc_id": "d1"}}, {"eq": {"tag": "auth"}}]}, ["b"]),
    ({"or": [{"eq": {"doc_id": "d3"}}, {"eq": {"tag": "api"}}]}, ["d", "e"]),
    ({"not": {"eq": {"doc_id": "d1"}}}, ["c", "d", "e"]),
    ({"not": {"or": [{"eq": {"doc_id": "d1"}}, {"eq": {"doc_id": "d2"}}]}}, ["e"]),
    ({"not": {"exists": "tag"}}, ["e"]),
    ({"eq": {"doc_id": "d1"}, "gte": {"score": 0.6}}, ["a"]),  # implicit AND
]


@pytest.mark.parametrize("where,expected", FILTER_CASES, ids=[str(c[0]) for c in FILTER_CASES])
def test_filters(store, where, expected):
    try:
        ids = store.list_ids(_c(store), where)
    except FilterError:
        if not store.capabilities().not_exists and "exists" in str(where):
            pytest.skip(f"{store.test_kind} can't express existence filters (declared in capabilities)")
        raise
    got = _key(store.get(_c(store), ids)) if ids else []
    assert got == expected
    if store.test_kind != "langchain":
        hits = store.query(_c(store), _unit([1, 1, 1, 1]), where, 10)
        assert sorted(h.metadata["key"] for h in hits) == expected


def test_manifest(store):
    assert store.get_manifest(_c(store)) is None
    store.put_manifest(_c(store), {"embedder": "hash:4", "dim": DIM, "metric": "cosine"})
    assert store.get_manifest(_c(store))["dim"] == DIM
    assert len(store.list_ids(_c(store))) == 5  # the manifest is never a record
    hits = store.query(_c(store), _unit([1, 0, 0, 0]), None, 10)
    assert len(hits) == 5


def test_bad_filter_raises(store):
    with pytest.raises(FilterError):
        store.list_ids(_c(store), {"bogus": {"x": 1}})
