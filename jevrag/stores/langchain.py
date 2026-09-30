"""Bridge to any LangChain ``VectorStore`` (``pip install ...[langchain]``).

Covers databases without a native jevrag adapter. Fewer guarantees than a
native adapter (Capabilities.native_filters = False):

* filters are applied **in Python** after over-fetching ``top_k * overfetch``
  candidates, so a very selective filter can return fewer than ``top_k``
* ids are passed through; stores that ignore caller ids break stale-delete
* scores come from ``similarity_search_with_relevance_scores`` (already 0..1 by
  LangChain's convention for most stores)
* the manifest lives in a sidecar file under ``.jevrag/manifests``

Pass a vector store whose embedding function returns the same vectors jevrag
computes, or wrap jevrag's embedder with :class:`JevragEmbeddings`.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from ..types import Hit, Record
from .base import Capabilities, VectorStore
from .filters import Where, matches, parse


class JevragEmbeddings:
    """LangChain ``Embeddings`` backed by a jevrag embedder.

    Wrap your store's embedding with this. When jevrag writes records it hands the
    vectors it already computed (from ``embed_text``: title + heading path + chunk)
    to this object, so stores that re-embed inside ``add_documents`` store jevrag's
    vectors instead of re-embedding the bare chunk text.
    """

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder
        self._precomputed: Optional[dict] = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pre = self._precomputed
        if pre is not None and all(t in pre for t in texts):
            return [pre[t] for t in texts]
        import asyncio
        return asyncio.run(self.embedder.embed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        import asyncio
        return asyncio.run(self.embedder.embed_query(text))


class LangChainStore(VectorStore):
    kind = "langchain"

    def __init__(self, store: Any, *, overfetch: int = 4, name: str = "langchain") -> None:
        self.store = store
        self.overfetch = max(1, overfetch)
        self.kind = f"langchain-{name}"
        self._ids: dict = {}

    def ensure_collection(self, name: str, dim: int, metric: str, filter_fields: Mapping[str, str]) -> None:
        return None  # the wrapped store owns its schema

    def collection_exists(self, name: str) -> bool:
        return self.get_manifest(name) is not None

    def _docs(self, records: Sequence[Record]) -> List[Any]:
        from langchain_core.documents import Document  # type: ignore
        return [Document(page_content=r.text, metadata={**r.metadata, "jevrag_id": r.id}, id=r.id) for r in records]

    def upsert(self, collection: str, records: Sequence[Record]) -> None:
        if not records:
            return
        ids = [r.id for r in records]
        try:
            self.store.delete(ids=ids)
        except Exception:
            pass
        vecs = [r.vector for r in records]
        emb = getattr(self.store, "embeddings", None) or getattr(self.store, "embedding", None)
        if hasattr(self.store, "add_embeddings") and all(v is not None for v in vecs):
            self.store.add_embeddings(list(zip([r.text for r in records], vecs)),
                                      metadatas=[{**r.metadata, "jevrag_id": r.id} for r in records], ids=ids)
        elif isinstance(emb, JevragEmbeddings) and all(v is not None for v in vecs):
            emb._precomputed = {r.text: list(r.vector or []) for r in records}
            try:
                self.store.add_documents(self._docs(records), ids=ids)
            finally:
                emb._precomputed = None
        else:  # the store embeds the chunk text itself with its own embedding function
            self.store.add_documents(self._docs(records), ids=ids)
        known = self._ids.setdefault(collection, {})
        for r in records:
            known[r.id] = dict(r.metadata)

    def delete(self, collection: str, ids: Optional[Sequence[str]] = None, where: Optional[Where] = None) -> None:
        if not ids and where is not None:
            ids = self.list_ids(collection, where)
        if ids:
            self.store.delete(ids=list(ids))
            for i in ids:
                self._ids.get(collection, {}).pop(i, None)

    def get(self, collection: str, ids: Sequence[str]) -> List[Record]:
        if not ids or not hasattr(self.store, "get_by_ids"):
            return []
        out = []
        for d in self.store.get_by_ids(list(ids)):
            meta = dict(d.metadata or {})
            rid = meta.pop("jevrag_id", None) or d.id
            out.append(Record(str(rid), None, d.page_content, meta))
        return out

    def list_ids(self, collection: str, where: Optional[Where] = None, limit: int = 100_000) -> List[str]:
        node = parse(where)
        known = self._ids.get(collection, {})
        if not known and hasattr(self.store, "store"):  # InMemoryVectorStore exposes its dict
            known = {k: dict(v.get("metadata") or {}) for k, v in self.store.store.items()}
        return [i for i, m in known.items() if matches(node, m)][:limit]

    def query(self, collection: str, vector: Sequence[float], where: Optional[Where], top_k: int) -> List[Hit]:
        node = parse(where)
        k = top_k * self.overfetch if node is not None else top_k
        if hasattr(self.store, "similarity_search_with_score_by_vector"):
            pairs = self.store.similarity_search_with_score_by_vector(list(vector), k=k)
        else:
            docs = self.store.similarity_search_by_vector(list(vector), k=k)
            pairs = [(d, 0.0) for d in docs]
        hits = []
        for d, s in pairs:
            meta = dict(d.metadata or {})
            rid = str(meta.pop("jevrag_id", None) or d.id)
            if not matches(node, meta):
                continue
            s = float(s)
            hits.append(Hit(rid, max(0.0, min(1.0, s)), s, d.page_content, meta))
            if len(hits) >= top_k:
                break
        return hits

    def capabilities(self) -> Capabilities:
        return Capabilities(native_filters=False, not_exists=True, max_batch=256, exact_stale_delete=False)
