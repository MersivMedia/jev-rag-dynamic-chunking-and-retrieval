"""EXPERIMENTAL: not yet run against a live Pinecone index.

Pinecone adapter (``pip install ...[pinecone]``, the ``pinecone`` package; ``pinecone-client`` is deprecated).

A jevrag collection is a **namespace** in one serverless index. The index is
created if missing (``PINECONE_CLOUD`` / ``PINECONE_REGION``, default aws /
us-east-1). Text is stored in metadata, trimmed to fit Pinecone's per-record
metadata limit (40 KB) with a warning. Manifests are a reserved record per namespace.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..types import Hit, Record
from .base import Capabilities, VectorStore, similarity
from .filters import FilterError, Node, Where, parse, push_down_not

log = logging.getLogger("jevrag.pinecone")
MANIFEST_ID = str(uuid.UUID("00000000-0000-0000-0000-00000000cafe"))
MAX_METADATA_BYTES = 40_000
TEXT_BUDGET = 30_000


class PineconeStore(VectorStore):
    kind = "pinecone"

    def __init__(self, index: str, *, api_key: Optional[str] = None, cloud: Optional[str] = None,
                 region: Optional[str] = None, client: Any = None) -> None:
        try:
            from pinecone import Pinecone  # type: ignore
        except ImportError as exc:
            raise ImportError('Pinecone needs: pip install "jev-rag-dynamic-chunking-and-retrieval[pinecone]"') from exc
        self.pc = client or Pinecone(api_key=api_key or os.environ.get("PINECONE_API_KEY"))
        self.index_name = index
        self.cloud = cloud or os.environ.get("PINECONE_CLOUD", "aws")
        self.region = region or os.environ.get("PINECONE_REGION", "us-east-1")
        self._idx: Any = None
        self._metric: Optional[str] = None

    def _index(self) -> Any:
        if self._idx is None:
            self._idx = self.pc.Index(self.index_name)
        return self._idx

    def _has_index(self) -> bool:
        return bool(self.pc.has_index(self.index_name))

    def collection_exists(self, name: str) -> bool:
        return self._has_index() and self.get_manifest(name) is not None

    def ensure_collection(self, name: str, dim: int, metric: str, filter_fields: Mapping[str, str]) -> None:
        pm = {"cosine": "cosine", "dot": "dotproduct", "l2": "euclidean"}[metric]
        if not self._has_index():
            from pinecone import ServerlessSpec  # type: ignore
            self.pc.create_index(name=self.index_name, dimension=dim, metric=pm,
                                 spec=ServerlessSpec(cloud=self.cloud, region=self.region))
            for _ in range(60):
                st = self.pc.describe_index(self.index_name).status
                if (st.get("ready") if isinstance(st, dict) else getattr(st, "ready", False)):
                    break
                time.sleep(2)
        desc = self.pc.describe_index(self.index_name)
        if int(desc.dimension) != dim:
            raise ValueError(f"Pinecone index {self.index_name!r} has dimension {desc.dimension}, embedder gives {dim}")
        have = {"cosine": "cosine", "dotproduct": "dot", "euclidean": "l2"}.get(str(desc.metric), "cosine")
        if have != metric:
            raise ValueError(f"Pinecone index {self.index_name!r} uses {desc.metric}; this collection needs {pm}")
        self._metric = metric

    def drop_collection(self, name: str) -> None:
        try:
            self._index().delete(delete_all=True, namespace=name)
        except Exception:
            pass

    def _metric_of(self) -> str:
        if self._metric is None:
            desc = self.pc.describe_index(self.index_name)
            self._metric = {"cosine": "cosine", "dotproduct": "dot", "euclidean": "l2"}.get(str(desc.metric), "cosine")
        return self._metric

    # -- filters (Pinecone uses a Mongo-style language) -------------------------

    def _filter(self, node: Optional[Node]) -> Optional[Dict[str, Any]]:
        node = push_down_not(node, allow_not_exists=True)

        def f(n: Node) -> Dict[str, Any]:
            if n.op in ("and", "or"):
                return {f"${n.op}": [f(a) for a in n.args]}
            if n.op == "exists":
                return {str(n.field): {"$exists": True}}
            if n.op == "not":  # only `not exists` reaches here
                return {str(n.args[0].field): {"$exists": False}}
            if n.op in ("gt", "gte", "lt", "lte") and isinstance(n.value, (str, bool)):
                raise FilterError(f"Pinecone range filters need numbers, got {n.value!r} for {n.field}")
            if n.op in ("ne", "nin"):  # jevrag semantics: must have the field
                val = list(n.value) if n.op == "nin" else n.value
                return {"$and": [{str(n.field): {"$exists": True}}, {str(n.field): {f"${n.op}": val}}]}
            val = list(n.value) if n.op == "in" else n.value
            return {str(n.field): {f"${n.op}": val}}

        base = {"_manifest": {"$exists": False}}
        return {"$and": [base, f(node)]} if node is not None else base

    # -- records --------------------------------------------------------------

    @staticmethod
    def _meta(r: Record) -> Dict[str, Any]:
        meta = {k: v for k, v in r.metadata.items() if v is not None}
        text = r.text
        if len(text.encode("utf-8")) > TEXT_BUDGET:
            log.warning("record %s text trimmed to fit Pinecone's metadata limit", r.id)
            text = text.encode("utf-8")[:TEXT_BUDGET].decode("utf-8", errors="ignore")
        meta["text"] = text
        size = len(json.dumps(meta).encode("utf-8"))
        if size > MAX_METADATA_BYTES:
            raise ValueError(f"record {r.id} metadata is {size} bytes, over Pinecone's {MAX_METADATA_BYTES} limit")
        return meta

    def upsert(self, collection: str, records: Sequence[Record]) -> None:
        if not records:
            return
        vecs = [{"id": r.id, "values": list(r.vector or []), "metadata": self._meta(r)} for r in records]
        for i in range(0, len(vecs), 100):
            self._index().upsert(vectors=vecs[i:i + 100], namespace=collection)

    def delete(self, collection: str, ids: Optional[Sequence[str]] = None, where: Optional[Where] = None) -> None:
        if ids:
            for i in range(0, len(ids), 1000):
                self._index().delete(ids=list(ids[i:i + 1000]), namespace=collection)
        elif where is not None:
            # serverless indexes don't delete by filter: list matching ids first
            ids = self.list_ids(collection, where)
            if ids:
                self.delete(collection, ids=ids)

    def get(self, collection: str, ids: Sequence[str]) -> List[Record]:
        if not ids:
            return []
        res = self._index().fetch(ids=list(ids), namespace=collection)
        out = []
        for vid, v in (res.vectors or {}).items():
            meta = dict(v.metadata or {})
            if meta.pop("_manifest", None):
                continue
            text = meta.pop("text", "")
            out.append(Record(vid, list(v.values or []), text, meta))
        return out

    def list_ids(self, collection: str, where: Optional[Where] = None, limit: int = 100_000) -> List[str]:
        node = parse(where)
        if node is None:
            out: List[str] = []
            for page in self._index().list(namespace=collection):
                out.extend(i for i in page if i != MANIFEST_ID)
                if len(out) >= limit:
                    break
            return out[:limit]
        # Pinecone has no list-by-filter: a filtered query with a neutral vector returns every match
        # (scores ignored), up to Pinecone's top_k ceiling of 10,000 per call.
        dim = int(self.pc.describe_index(self.index_name).dimension)
        probe = [1e-6] + [0.0] * (dim - 1)
        res = self._index().query(vector=probe, top_k=min(limit, 10_000), namespace=collection,
                                  filter=self._filter(node), include_metadata=False, include_values=False)
        return [m.id for m in (res.matches or [])]

    def query(self, collection: str, vector: Sequence[float], where: Optional[Where], top_k: int) -> List[Hit]:
        metric = self._metric_of()
        res = self._index().query(vector=list(vector), top_k=top_k, namespace=collection,
                                  filter=self._filter(parse(where)), include_metadata=True)
        hits = []
        for mt in res.matches or []:
            meta = dict(mt.metadata or {})
            text = meta.pop("text", "")
            raw = float(mt.score)
            hits.append(Hit(mt.id, similarity(raw, metric, is_distance=metric == "l2"), raw, text, meta))
        return hits

    def get_manifest(self, collection: str) -> Optional[Dict[str, Any]]:
        if not self._has_index():
            return None
        res = self._index().fetch(ids=[MANIFEST_ID], namespace=collection)
        v = (res.vectors or {}).get(MANIFEST_ID)
        return json.loads(v.metadata["manifest"]) if v is not None and v.metadata else None

    def put_manifest(self, collection: str, manifest: Mapping[str, Any]) -> None:
        dim = int(self.pc.describe_index(self.index_name).dimension)
        vec = [0.0] * dim
        vec[0] = 1e-6
        self._index().upsert(vectors=[{"id": MANIFEST_ID, "values": vec, "metadata": {
            "_manifest": True, "manifest": json.dumps(dict(manifest), sort_keys=True)}}], namespace=collection)

    def capabilities(self) -> Capabilities:
        return Capabilities(native_filters=True, not_exists=True, max_batch=100, max_metadata_bytes=MAX_METADATA_BYTES)
