"""In-memory store: the reference implementation for semantics, tests and demos.

Persists to a JSON file when ``path`` is given (fine for a few thousand chunks;
use a real database beyond that).
"""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..types import Hit, Record
from .base import Capabilities, VectorStore, similarity
from .filters import Where, matches, parse


class MemoryStore(VectorStore):
    kind = "memory"

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(os.path.expanduser(path)) if path else None
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        if self.path and self.path.exists():
            self._data = json.loads(self.path.read_text())

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data))
            tmp.replace(self.path)

    def _col(self, name: str) -> Dict[str, Any]:
        if name not in self._data:
            raise KeyError(f"collection {name!r} does not exist")
        return self._data[name]

    def ensure_collection(self, name: str, dim: int, metric: str, filter_fields: Mapping[str, str]) -> None:
        with self._lock:
            if name not in self._data:
                self._data[name] = {"dim": dim, "metric": metric, "records": {}, "manifest": None}
                self._save()

    def collection_exists(self, name: str) -> bool:
        return name in self._data

    def drop_collection(self, name: str) -> None:
        with self._lock:
            self._data.pop(name, None)
            self._save()

    def upsert(self, collection: str, records: Sequence[Record]) -> None:
        with self._lock:
            col = self._col(collection)
            for r in records:
                if r.vector is None or len(r.vector) != col["dim"]:
                    raise ValueError(f"vector dimension must be {col['dim']}")
                col["records"][r.id] = {"vector": list(r.vector), "text": r.text, "metadata": dict(r.metadata)}
            self._save()

    def delete(self, collection: str, ids: Optional[Sequence[str]] = None, where: Optional[Where] = None) -> None:
        with self._lock:
            col = self._col(collection)
            if ids:
                for i in ids:
                    col["records"].pop(i, None)
            elif where is not None:
                node = parse(where)
                for i in [i for i, r in col["records"].items() if matches(node, r["metadata"])]:
                    del col["records"][i]
            self._save()

    def get(self, collection: str, ids: Sequence[str]) -> List[Record]:
        col = self._col(collection)
        return [Record(i, col["records"][i]["vector"], col["records"][i]["text"], dict(col["records"][i]["metadata"]))
                for i in ids if i in col["records"]]

    def list_ids(self, collection: str, where: Optional[Where] = None, limit: int = 100_000) -> List[str]:
        node = parse(where)
        col = self._col(collection)
        return [i for i, r in col["records"].items() if matches(node, r["metadata"])][:limit]

    def query(self, collection: str, vector: Sequence[float], where: Optional[Where], top_k: int) -> List[Hit]:
        col = self._col(collection)
        node = parse(where)
        metric = col["metric"]
        qn = math.sqrt(sum(x * x for x in vector)) or 1.0
        scored = []
        for i, r in col["records"].items():
            if not matches(node, r["metadata"]):
                continue
            v = r["vector"]
            if metric == "l2":
                raw = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, v)))
                score = similarity(raw, "l2", is_distance=True)
            else:
                dot = sum(a * b for a, b in zip(vector, v))
                raw = dot / (qn * (math.sqrt(sum(x * x for x in v)) or 1.0)) if metric == "cosine" else dot
                score = similarity(raw, metric, is_distance=False)
            scored.append((score, raw, i, r))
        scored.sort(key=lambda t: (-t[0], t[2]))
        return [Hit(i, s, raw, r["text"], dict(r["metadata"])) for s, raw, i, r in scored[:top_k]]

    def get_manifest(self, collection: str) -> Optional[Dict[str, Any]]:
        col = self._data.get(collection)
        return dict(col["manifest"]) if col and col.get("manifest") else None

    def put_manifest(self, collection: str, manifest: Mapping[str, Any]) -> None:
        with self._lock:
            self._col(collection)["manifest"] = dict(manifest)
            self._save()

    def capabilities(self) -> Capabilities:
        return Capabilities(native_filters=True, not_exists=True, max_batch=10_000)
