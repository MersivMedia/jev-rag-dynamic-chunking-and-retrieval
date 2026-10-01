"""Vector store adapters. Import a store lazily with :func:`make_store`."""

from __future__ import annotations

import os
from typing import Any, Mapping

from .base import BASE_FILTER_FIELDS, Capabilities, VectorStore, similarity
from .filters import FilterError, Where, matches, parse
from .memory import MemoryStore

KINDS = ("memory", "qdrant", "chroma", "pgvector", "pinecone", "langchain")


def make_store(spec: Mapping[str, Any]) -> VectorStore:
    """Build a store from config: ``{kind: qdrant, url: ..., api_key_env: ...}``."""
    kind = str(spec.get("kind", "memory")).lower()
    key_env = spec.get("api_key_env")
    api_key = os.environ.get(key_env) if key_env else None
    if kind == "memory":
        return MemoryStore(spec.get("path"))
    if kind == "qdrant":
        from .qdrant import QdrantStore
        return QdrantStore(spec.get("url"), api_key=api_key or os.environ.get("QDRANT_API_KEY"),
                           path=spec.get("path"), prefix=spec.get("prefix", ""))
    if kind == "chroma":
        from .chroma import ChromaStore
        return ChromaStore(spec.get("path"), host=spec.get("host"), port=int(spec.get("port", 8000)))
    if kind == "pgvector":
        from .pgvector import PgVectorStore
        dsn = spec.get("dsn") or os.environ.get(spec.get("dsn_env", "DATABASE_URL"), "")
        return PgVectorStore(dsn, schema=spec.get("schema", "public"), table_prefix=spec.get("table_prefix", "jev_retrieval_"))
    if kind == "pinecone":
        import warnings
        warnings.warn("the Pinecone adapter is experimental: it has not been tested against a live index",
                      UserWarning, stacklevel=2)
        from .pinecone import PineconeStore
        if not spec.get("index"):
            raise ValueError("pinecone store needs 'index' (the index name)")
        return PineconeStore(spec["index"], api_key=api_key, cloud=spec.get("cloud"), region=spec.get("region"))
    if kind == "langchain":
        raise ValueError("the langchain bridge wraps a Python object: build LangChainStore(store) in code")
    raise ValueError(f"unknown store kind {kind!r}; choose from {KINDS}")


__all__ = ["BASE_FILTER_FIELDS", "Capabilities", "VectorStore", "similarity", "FilterError", "Where", "matches",
           "parse", "MemoryStore", "KINDS", "make_store"]
