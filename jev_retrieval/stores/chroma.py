"""Chroma adapter (``pip install ...[chroma]``, chromadb >= 1.5).

Sets the distance metric explicitly at creation (Chroma's default depends on
the attached embedding function, with L2 as the base default) and converts
distances to similarity. Collection names are padded to Chroma's 3-character
minimum.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..types import Hit, Record
from .base import Capabilities, VectorStore, similarity
from .filters import FilterError, Node, Where, matches, parse, push_down_not

_SPACE = {"cosine": "cosine", "dot": "ip", "l2": "l2"}
MANIFEST_KEY = "jev_retrieval_manifest"


class ChromaStore(VectorStore):
    kind = "chroma"

    def __init__(self, path: Optional[str] = None, *, host: Optional[str] = None, port: int = 8000,
                 client: Any = None) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise ImportError('Chroma needs: pip install "jev-rag-retrieval[chroma]"') from exc
        if client is not None:
            self.c = client
        elif host:
            self.c = chromadb.HttpClient(host=host, port=port)
        elif path:
            self.c = chromadb.PersistentClient(path=path)
        else:
            self.c = chromadb.EphemeralClient()
        self._cols: Dict[str, Any] = {}

    @staticmethod
    def _n(name: str) -> str:
        return name if len(name) >= 3 else f"{name}__"

    def _col(self, name: str) -> Any:
        if name not in self._cols:
            self._cols[name] = self.c.get_collection(self._n(name), embedding_function=None)
        return self._cols[name]

    def collection_exists(self, name: str) -> bool:
        try:
            self.c.get_collection(self._n(name), embedding_function=None)
            return True
        except Exception:
            return False

    def ensure_collection(self, name: str, dim: int, metric: str, filter_fields: Mapping[str, str]) -> None:
        col = self.c.get_or_create_collection(self._n(name), configuration={"hnsw": {"space": _SPACE[metric]}},
                                              embedding_function=None)
        self._cols[name] = col

    def drop_collection(self, name: str) -> None:
        try:
            self.c.delete_collection(self._n(name))
        except Exception:
            pass
        self._cols.pop(name, None)

    def _space(self, name: str) -> str:
        cfg = self._col(name).configuration
        hnsw = (cfg or {}).get("hnsw") or {}
        space = str(hnsw.get("space") or "l2")
        return {"ip": "dot"}.get(space, space)

    # -- filters --------------------------------------------------------------
    #
    # Chroma can't test field existence, and its $ne / $nin also match records
    # that lack the field. So the adapter pushes down a *superset* filter
    # (existence tests become "true") and applies the exact jev-retrieval semantics in
    # Python afterwards, over-fetching for queries when the push-down is lossy.

    def _where(self, node: Optional[Node]) -> Tuple[Optional[Dict[str, Any]], bool]:
        """(chroma where, exact). exact=False means results must be post-filtered."""
        node = push_down_not(node, allow_not_exists=True)
        lossy = False

        def w(n: Node) -> Optional[Dict[str, Any]]:  # None = "true"
            nonlocal lossy
            if n.op == "and":
                kids = [k for k in (w(a) for a in n.args) if k is not None]
                return None if not kids else (kids[0] if len(kids) == 1 else {"$and": kids})
            if n.op == "or":
                kids = [w(a) for a in n.args]
                if any(k is None for k in kids):
                    return None
                return kids[0] if len(kids) == 1 else {"$or": kids}
            if n.op in ("exists", "not"):
                lossy = True
                return None
            if n.op in ("ne", "nin"):
                lossy = True
            if n.op in ("gt", "gte", "lt", "lte") and isinstance(n.value, (str, bool)):
                raise FilterError(f"Chroma range filters need numbers, got {n.value!r} for {n.field}")
            val = list(n.value) if n.op in ("in", "nin") else n.value
            return {str(n.field): {f"${n.op}": val}}

        if node is None:
            return None, True
        out = w(node)
        return out, not lossy

    # -- records --------------------------------------------------------------

    def upsert(self, collection: str, records: Sequence[Record]) -> None:
        if not records:
            return
        col = self._col(collection)
        for i in range(0, len(records), 1000):
            b = records[i:i + 1000]
            col.upsert(ids=[r.id for r in b], embeddings=[list(r.vector or []) for r in b],
                       documents=[r.text for r in b], metadatas=[dict(r.metadata) or {"_": 0} for r in b])

    def delete(self, collection: str, ids: Optional[Sequence[str]] = None, where: Optional[Where] = None) -> None:
        col = self._col(collection)
        if ids:
            col.delete(ids=list(ids))
        elif where is not None:
            ids = self.list_ids(collection, where)
            if ids:
                col.delete(ids=ids)

    def get(self, collection: str, ids: Sequence[str]) -> List[Record]:
        if not ids:
            return []
        r = self._col(collection).get(ids=list(ids), include=["documents", "metadatas", "embeddings"])
        embs = r.get("embeddings")
        return [Record(i, list(map(float, embs[k])) if embs is not None else None, r["documents"][k] or "",
                       _clean(r["metadatas"][k]))
                for k, i in enumerate(r["ids"])]

    def list_ids(self, collection: str, where: Optional[Where] = None, limit: int = 100_000) -> List[str]:
        node = parse(where)
        cw, exact = self._where(node)
        if exact:
            return list(self._col(collection).get(where=cw, limit=limit, include=[])["ids"])
        r = self._col(collection).get(where=cw, include=["metadatas"])
        return [i for i, m in zip(r["ids"], r["metadatas"]) if matches(node, _clean(m))][:limit]

    def query(self, collection: str, vector: Sequence[float], where: Optional[Where], top_k: int) -> List[Hit]:
        col = self._col(collection)
        space = self._space(collection)
        n = col.count()
        if n == 0:
            return []
        node = parse(where)
        cw, exact = self._where(node)
        k = top_k if exact else min(n, max(top_k * 4, top_k + 20))
        r = col.query(query_embeddings=[list(vector)], n_results=min(k, n), where=cw,
                      include=["documents", "metadatas", "distances"])
        hits = []
        for idx, i in enumerate(r["ids"][0]):
            meta = _clean(r["metadatas"][0][idx])
            if not exact and not matches(node, meta):
                continue
            d = float(r["distances"][0][idx])
            if space == "dot":  # Chroma "ip" distance is 1 - dot
                score = similarity(1.0 - d, "dot", is_distance=False)
            elif space == "l2":  # Chroma "l2" is squared euclidean
                score = similarity(d ** 0.5, "l2", is_distance=True)
            else:
                score = similarity(d, "cosine", is_distance=True)
            hits.append(Hit(i, score, d, r["documents"][0][idx] or "", meta))
            if len(hits) >= top_k:
                break
        return hits

    # -- manifest in collection metadata ------------------------------------

    def get_manifest(self, collection: str) -> Optional[Dict[str, Any]]:
        if not self.collection_exists(collection):
            return None
        import json
        raw = (self._col(collection).metadata or {}).get(MANIFEST_KEY)
        return json.loads(raw) if raw else None

    def put_manifest(self, collection: str, manifest: Mapping[str, Any]) -> None:
        import json
        col = self._col(collection)
        meta = dict(col.metadata or {})
        meta[MANIFEST_KEY] = json.dumps(dict(manifest), sort_keys=True)
        col.modify(metadata=meta)

    def capabilities(self) -> Capabilities:
        return Capabilities(native_filters=True, not_exists=True, max_batch=1000)


def _clean(meta: Any) -> Dict[str, Any]:
    return {k: v for k, v in (meta or {}).items() if k != "_"}
