"""The store interface every adapter implements (FR-S1..S4).

Adapters are synchronous (most database SDKs are); the pipeline calls them in a
worker thread. Metadata is flat: str, int, float and bool values only.
"""

from __future__ import annotations

import json
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..types import Hit, Record
from .filters import Node, Where, parse

# Field -> type, for stores that need indexes to filter (Qdrant payload indexes,
# Postgres expression indexes). Taxonomy fields are added per collection.
BASE_FILTER_FIELDS: Dict[str, str] = {
    "doc_id": "keyword",
    "quarantined": "bool",
    "chunk_index": "integer",
    "chunker": "keyword",
    "content_hash": "keyword",
}

METRICS = ("cosine", "dot", "l2")


@dataclass
class Capabilities:
    native_filters: bool = True  # filters run in the database (else post-filtered in Python)
    not_exists: bool = True
    hybrid: bool = False
    max_batch: int = 256
    max_metadata_bytes: Optional[int] = None
    exact_stale_delete: bool = True


def similarity(raw: float, metric: str, *, is_distance: bool) -> float:
    """Normalise a raw score to similarity in [0, 1], higher is better (FR-S3).

    cosine/dot similarity -> clamp to [0, 1]; cosine distance d -> 1 - d, clamped;
    L2 distance d -> 1 / (1 + d).
    """
    if metric == "l2":
        d = raw if is_distance else -raw
        return 1.0 / (1.0 + max(0.0, d))
    s = 1.0 - raw if is_distance else raw
    if math.isnan(s):
        return 0.0
    return max(0.0, min(1.0, s))


def safe_name(name: str, max_len: int = 48) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_").lower() or "default"
    return s[:max_len]


class VectorStore(ABC):
    kind: str = "base"

    @abstractmethod
    def ensure_collection(self, name: str, dim: int, metric: str, filter_fields: Mapping[str, str]) -> None:
        """Create the collection (and whatever filtering needs) if missing. Idempotent."""

    @abstractmethod
    def upsert(self, collection: str, records: Sequence[Record]) -> None:
        ...

    @abstractmethod
    def delete(self, collection: str, ids: Optional[Sequence[str]] = None, where: Optional[Where] = None) -> None:
        """Delete by ids, or by filter. Neither given deletes nothing."""

    @abstractmethod
    def get(self, collection: str, ids: Sequence[str]) -> List[Record]:
        """Records for the ids that exist, in no particular order. Vectors may be omitted."""

    @abstractmethod
    def query(self, collection: str, vector: Sequence[float], where: Optional[Where], top_k: int) -> List[Hit]:
        """Top-k by similarity, filtered, best first, scores normalised."""

    @abstractmethod
    def list_ids(self, collection: str, where: Optional[Where] = None, limit: int = 100_000) -> List[str]:
        ...

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def collection_exists(self, name: str) -> bool:
        return self.get_manifest(name) is not None

    def drop_collection(self, name: str) -> None:
        raise NotImplementedError(f"{self.kind} can't drop collections")

    # -- manifest (FR-I3): default is a sidecar file; adapters override natively ---

    manifest_dir: str = ".jev-retrieval/manifests"

    def _manifest_path(self, collection: str) -> Path:
        return Path(os.path.expanduser(self.manifest_dir)) / f"{self.kind}-{safe_name(collection, 120)}.json"

    def get_manifest(self, collection: str) -> Optional[Dict[str, Any]]:
        p = self._manifest_path(collection)
        return json.loads(p.read_text()) if p.exists() else None

    def put_manifest(self, collection: str, manifest: Mapping[str, Any]) -> None:
        p = self._manifest_path(collection)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True))
        tmp.replace(p)

    def close(self) -> None:
        return None

    @staticmethod
    def _parse(where: Optional[Where]) -> Optional[Node]:
        return parse(where)
