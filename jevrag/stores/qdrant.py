"""Qdrant adapter (``pip install ...[qdrant]``, qdrant-client >= 1.19).

Uses ``query_points`` (``search`` was removed from the client), creates payload
indexes for filter fields, stores the manifest in the collection's metadata
where supported and in a reserved sidecar point otherwise.
``url=":memory:"`` or ``path=...`` run Qdrant's embedded local mode.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..types import Hit, Record
from .base import Capabilities, VectorStore, similarity
from .filters import FilterError, Node, Where, parse, push_down_not

MANIFEST_ID = str(uuid.UUID("00000000-0000-0000-0000-00000000cafe"))
_TYPES = {"keyword": "KEYWORD", "integer": "INTEGER", "float": "FLOAT", "bool": "BOOL", "datetime": "DATETIME"}
_DIST = {"cosine": "COSINE", "dot": "DOT", "l2": "EUCLID"}


class QdrantStore(VectorStore):
    kind = "qdrant"

    def __init__(self, url: Optional[str] = None, *, api_key: Optional[str] = None, path: Optional[str] = None,
                 prefix: str = "", timeout: int = 30, client: Any = None) -> None:
        try:
            from qdrant_client import QdrantClient, models  # type: ignore
        except ImportError as exc:
            raise ImportError('Qdrant needs: pip install "jev-rag-dynamic-chunking-and-retrieval[qdrant]"') from exc
        self.m = models
        if client is not None:
            self.c = client
        elif url == ":memory:":
            self.c = QdrantClient(location=":memory:")
        elif path:
            self.c = QdrantClient(path=path)
        else:
            self.c = QdrantClient(url=url or "http://localhost:6333", api_key=api_key, timeout=timeout)
        self.prefix = prefix
        self._metric: Dict[str, str] = {}

    def _n(self, name: str) -> str:
        return f"{self.prefix}{name}"

    # -- collections ----------------------------------------------------------

    def collection_exists(self, name: str) -> bool:
        return bool(self.c.collection_exists(self._n(name)))

    def ensure_collection(self, name: str, dim: int, metric: str, filter_fields: Mapping[str, str]) -> None:
        m = self.m
        cname = self._n(name)
        if not self.c.collection_exists(cname):
            self.c.create_collection(cname, vectors_config=m.VectorParams(
                size=dim, distance=getattr(m.Distance, _DIST[metric])))
        self._metric[name] = metric
        import warnings
        for f, t in filter_fields.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # local mode warns that indexes are no-ops
                    self.c.create_payload_index(cname, field_name=f"meta.{f}",
                                                field_schema=getattr(m.PayloadSchemaType, _TYPES.get(t, "KEYWORD")))
            except Exception:  # already exists
                pass

    def drop_collection(self, name: str) -> None:
        if self.c.collection_exists(self._n(name)):
            self.c.delete_collection(self._n(name))

    def _metric_of(self, name: str) -> str:
        if name not in self._metric:
            info = self.c.get_collection(self._n(name))
            dist = str(info.config.params.vectors.distance).lower()
            self._metric[name] = "l2" if "euclid" in dist else ("dot" if "dot" in dist else "cosine")
        return self._metric[name]

    # -- filters --------------------------------------------------------------

    def _filter(self, node: Optional[Node], *, exclude_manifest: bool = True) -> Any:
        m = self.m
        node = push_down_not(node, allow_not_exists=True)

        def cond(n: Node) -> Any:
            key = f"meta.{n.field}"
            if n.op == "eq":
                return m.FieldCondition(key=key, match=m.MatchValue(value=n.value))
            if n.op == "in":
                return m.FieldCondition(key=key, match=m.MatchAny(any=list(n.value)))
            if n.op in ("gt", "gte", "lt", "lte"):
                if isinstance(n.value, (bool, str)):
                    raise FilterError(f"Qdrant range filters need numbers, got {n.value!r} for {n.field}")
                return m.FieldCondition(key=key, range=m.Range(**{n.op: n.value}))
            if n.op == "exists":
                return m.Filter(must_not=[m.IsNullCondition(is_null=m.PayloadField(key=key)),
                                          m.IsEmptyCondition(is_empty=m.PayloadField(key=key))])
            if n.op == "ne":  # must have the field and differ
                return m.Filter(must=[cond(Node("exists", field=n.field))],
                                must_not=[m.FieldCondition(key=key, match=m.MatchValue(value=n.value))])
            if n.op == "nin":
                return m.Filter(must=[cond(Node("exists", field=n.field))],
                                must_not=[m.FieldCondition(key=key, match=m.MatchAny(any=list(n.value)))])
            if n.op == "and":
                return m.Filter(must=[cond(a) for a in n.args])
            if n.op == "or":
                return m.Filter(should=[cond(a) for a in n.args])
            if n.op == "not":
                return m.Filter(must_not=[cond(n.args[0])])
            raise FilterError(f"Qdrant adapter can't express {n.op!r}")

        must = [cond(node)] if node is not None else []
        must_not = [m.HasIdCondition(has_id=[MANIFEST_ID])] if exclude_manifest else []
        if not must and not must_not:
            return None
        return m.Filter(must=must or None, must_not=must_not or None)

    # -- records --------------------------------------------------------------

    def upsert(self, collection: str, records: Sequence[Record]) -> None:
        if not records:
            return
        pts = [self.m.PointStruct(id=r.id, vector=list(r.vector or []), payload={"text": r.text, "meta": r.metadata})
               for r in records]
        for i in range(0, len(pts), 256):
            self.c.upsert(self._n(collection), points=pts[i:i + 256], wait=True)

    def delete(self, collection: str, ids: Optional[Sequence[str]] = None, where: Optional[Where] = None) -> None:
        m = self.m
        if ids:
            self.c.delete(self._n(collection), points_selector=m.PointIdsList(points=list(ids)), wait=True)
        elif where is not None:
            flt = self._filter(parse(where))
            self.c.delete(self._n(collection), points_selector=m.FilterSelector(filter=flt), wait=True)

    def get(self, collection: str, ids: Sequence[str]) -> List[Record]:
        if not ids:
            return []
        pts = self.c.retrieve(self._n(collection), ids=list(ids), with_payload=True, with_vectors=True)
        out = []
        for p in pts:
            if str(p.id) == MANIFEST_ID:
                continue
            payload = p.payload or {}
            vec = p.vector if isinstance(p.vector, list) else None
            out.append(Record(str(p.id), vec, payload.get("text", ""), dict(payload.get("meta") or {})))
        return out

    def list_ids(self, collection: str, where: Optional[Where] = None, limit: int = 100_000) -> List[str]:
        flt = self._filter(parse(where))
        out: List[str] = []
        offset = None
        while len(out) < limit:
            pts, offset = self.c.scroll(self._n(collection), scroll_filter=flt, limit=min(1000, limit - len(out)),
                                        offset=offset, with_payload=False, with_vectors=False)
            out.extend(str(p.id) for p in pts)
            if offset is None:
                break
        return out

    def query(self, collection: str, vector: Sequence[float], where: Optional[Where], top_k: int) -> List[Hit]:
        metric = self._metric_of(collection)
        res = self.c.query_points(self._n(collection), query=list(vector), query_filter=self._filter(parse(where)),
                                  limit=top_k, with_payload=True)
        hits = []
        for p in res.points:
            payload = p.payload or {}
            is_dist = metric == "l2"  # Qdrant returns similarity for cosine/dot and distance for euclid
            hits.append(Hit(str(p.id), similarity(float(p.score), metric, is_distance=is_dist), float(p.score),
                            payload.get("text", ""), dict(payload.get("meta") or {})))
        return hits

    # -- manifest: reserved point with a zero vector ----------------------------

    def get_manifest(self, collection: str) -> Optional[Dict[str, Any]]:
        if not self.c.collection_exists(self._n(collection)):
            return None
        pts = self.c.retrieve(self._n(collection), ids=[MANIFEST_ID], with_payload=True)
        return dict((pts[0].payload or {}).get("manifest") or {}) if pts else None

    def put_manifest(self, collection: str, manifest: Mapping[str, Any]) -> None:
        info = self.c.get_collection(self._n(collection))
        dim = info.config.params.vectors.size
        vec = [0.0] * dim
        vec[0] = 1e-6  # cosine rejects all-zero vectors
        self.c.upsert(self._n(collection), points=[self.m.PointStruct(
            id=MANIFEST_ID, vector=vec, payload={"manifest": dict(manifest), "meta": {"_manifest": True}})], wait=True)

    def capabilities(self) -> Capabilities:
        return Capabilities(native_filters=True, not_exists=True, max_batch=256)

    def close(self) -> None:
        try:
            self.c.close()
        except Exception:
            pass
