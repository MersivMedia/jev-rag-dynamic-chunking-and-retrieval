import asyncio

import pytest

from jev_retrieval import Document, Pipeline
from jev_retrieval.embed import make_embedder

lc = pytest.importorskip("langchain_core")


def test_langchain_bridge_stores_jev_retrievals_vectors():
    from langchain_core.vectorstores import InMemoryVectorStore
    from jev_retrieval.stores.langchain import JevRetrievalEmbeddings, LangChainStore

    emb = make_embedder("hash:64")
    store = InMemoryVectorStore(JevRetrievalEmbeddings(emb))
    rag = Pipeline(store=LangChainStore(store), embedder=emb, trace_dir=None)
    rag.ingest([Document(doc_id="a", title="Auth", text="# Tokens\n\nRefresh tokens expire after 14 days.")],
               collection="kb")
    row = next(iter(store.store.values()))
    expected = asyncio.run(emb.embed_documents(["Auth > Tokens\n\n" + row["text"]]))[0]
    assert row["vector"] == pytest.approx(expected)  # embed_text vector, not a re-embedding of bare text
    r = rag.retrieve("refresh tokens", collection="kb")
    assert r.passages and r.passages[0].doc_id == "a"


def test_pipeline_requires_an_embedder():
    with pytest.raises(ValueError, match="embedder"):
        Pipeline()


def test_embed_text_does_not_repeat_title():
    from jev_retrieval.chunk import ChunkConfig, chunk_document
    doc = Document(doc_id="g", title="Guide", text="# Guide\n\n## Tokens\n\nRefresh tokens expire after 14 days.")
    chunks, _ = asyncio.run(chunk_document(doc, ChunkConfig(method="structural")))
    assert chunks[0].embed_text.startswith("Guide > Tokens\n\n")
