"""Postgres + pgvector adapter (``pip install ...[pgvector]``).

One table per collection: ``id uuid``, ``embedding vector(dim)``, ``text``,
``metadata jsonb``. Creates the ``vector`` extension, an HNSW index with the
right operator class, and expression indexes on filter fields. Filters compile
to parameterised SQL; field names are validated, never interpolated raw.
Manifests live in a ``jev_retrieval_manifests`` table.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..types import Hit, Record
from .base import Capabilities, VectorStore, safe_name, similarity
from .filters import FilterError, Node, Where, parse

_OPS = {"cosine": ("vector_cosine_ops", "<=>"), "dot": ("vector_ip_ops", "<#>"), "l2": ("vector_l2_ops", "<->")}
_FIELD = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,62}$")


class PgVectorStore(VectorStore):
    kind = "pgvector"

    def __init__(self, dsn: Optional[str] = None, *, schema: str = "public", table_prefix: str = "jev_retrieval_",
                 conn: Any = None) -> None:
        try:
            import psycopg  # type: ignore
        except ImportError as exc:
            raise ImportError('pgvector needs: pip install "jev-rag-retrieval[pgvector]"') from exc
        self.schema = safe_name(schema)
        self.prefix = table_prefix
        self.conn = conn or psycopg.connect(dsn or os.environ.get("DATABASE_URL", ""), autocommit=True)
        self._metric: Dict[str, str] = {}
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f'CREATE TABLE IF NOT EXISTS "{self.schema}"."{self.prefix}manifests" '
                        "(collection text PRIMARY KEY, manifest jsonb NOT NULL)")
        from pgvector.psycopg import register_vector  # type: ignore
        register_vector(self.conn)

    def _t(self, name: str) -> str:
        return f'"{self.schema}"."{self.prefix}{safe_name(name)}"'

    def _tname(self, name: str) -> str:
        return f"{self.prefix}{safe_name(name)}"

    def collection_exists(self, name: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (f"{self.schema}.{self._tname(name)}",))
            return cur.fetchone()[0] is not None

    def ensure_collection(self, name: str, dim: int, metric: str, filter_fields: Mapping[str, str]) -> None:
        opclass, _ = _OPS[metric]
        t = self._t(name)
        base = self._tname(name)
        with self.conn.cursor() as cur:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {t} (id uuid PRIMARY KEY, embedding vector({int(dim)}) NOT NULL, "
                        "text text NOT NULL, metadata jsonb NOT NULL DEFAULT '{}'::jsonb)")
            cur.execute(f'CREATE INDEX IF NOT EXISTS "{base}_hnsw" ON {t} USING hnsw (embedding {opclass})')
            for f in filter_fields:
                if not _FIELD.match(f):  # validated: letters, digits, _ and . only, so safe to inline
                    continue
                cur.execute(f'CREATE INDEX IF NOT EXISTS "{base}_{safe_name(f, 30)}" ON {t} ((metadata->>\'{f}\'))')
        self._metric[name] = metric

    def drop_collection(self, name: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self._t(name)}")
            cur.execute(f'DELETE FROM "{self.schema}"."{self.prefix}manifests" WHERE collection = %s', (name,))

    def _metric_of(self, name: str) -> str:
        if name not in self._metric:
            m = self.get_manifest(name) or {}
            self._metric[name] = m.get("metric", "cosine")
        return self._metric[name]

    # -- filters --------------------------------------------------------------

    def _sql(self, node: Optional[Node]) -> Tuple[str, List[Any]]:
        return _Compiler().compile(node)

    # -- records --------------------------------------------------------------

    def upsert(self, collection: str, records: Sequence[Record]) -> None:
        if not records:
            return
        import numpy as np  # psycopg adapts numpy arrays to the vector type; declared in the [pgvector] extra
        t = self._t(collection)
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {t} (id, embedding, text, metadata) VALUES (%s, %s, %s, %s::jsonb) "
                "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, text = EXCLUDED.text, "
                "metadata = EXCLUDED.metadata",
                [(r.id, np.asarray(r.vector, dtype=np.float32), r.text, json.dumps(r.metadata)) for r in records])

    def delete(self, collection: str, ids: Optional[Sequence[str]] = None, where: Optional[Where] = None) -> None:
        t = self._t(collection)
        with self.conn.cursor() as cur:
            if ids:
                cur.execute(f"DELETE FROM {t} WHERE id = ANY(%s::uuid[])", (list(ids),))
            elif where is not None:
                sql, params = self._sql(parse(where))
                cur.execute(f"DELETE FROM {t} WHERE {sql}", params)

    def get(self, collection: str, ids: Sequence[str]) -> List[Record]:
        if not ids:
            return []
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id::text, embedding, text, metadata FROM {self._t(collection)} "
                        "WHERE id = ANY(%s::uuid[])", (list(ids),))
            return [Record(r[0], [float(x) for x in r[1].to_list()], r[2], dict(r[3] or {})) for r in cur.fetchall()]

    def list_ids(self, collection: str, where: Optional[Where] = None, limit: int = 100_000) -> List[str]:
        sql, params = self._sql(parse(where))
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id::text FROM {self._t(collection)} WHERE {sql} LIMIT %s", params + [limit])
            return [r[0] for r in cur.fetchall()]

    def query(self, collection: str, vector: Sequence[float], where: Optional[Where], top_k: int) -> List[Hit]:
        import numpy as np
        metric = self._metric_of(collection)
        _, op = _OPS[metric]
        sql, params = self._sql(parse(where))
        q = np.asarray(vector, dtype=np.float32)
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id::text, text, metadata, embedding {op} %s AS d FROM {self._t(collection)} "
                        f"WHERE {sql} ORDER BY embedding {op} %s LIMIT %s", [q] + params + [q, top_k])
            rows = cur.fetchall()
        hits = []
        for rid, text, meta, d in rows:
            d = float(d)
            if metric == "dot":  # <#> is negative inner product
                score = similarity(-d, "dot", is_distance=False)
            elif metric == "l2":
                score = similarity(d, "l2", is_distance=True)
            else:
                score = similarity(d, "cosine", is_distance=True)
            hits.append(Hit(rid, score, d, text, dict(meta or {})))
        return hits

    def get_manifest(self, collection: str) -> Optional[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(f'SELECT manifest FROM "{self.schema}"."{self.prefix}manifests" WHERE collection = %s',
                        (collection,))
            row = cur.fetchone()
        return dict(row[0]) if row else None

    def put_manifest(self, collection: str, manifest: Mapping[str, Any]) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f'INSERT INTO "{self.schema}"."{self.prefix}manifests" (collection, manifest) '
                        "VALUES (%s, %s::jsonb) ON CONFLICT (collection) DO UPDATE SET manifest = EXCLUDED.manifest",
                        (collection, json.dumps(dict(manifest))))

    def capabilities(self) -> Capabilities:
        return Capabilities(native_filters=True, not_exists=True, max_batch=1000)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


class _Compiler:
    """Filter AST -> (SQL, params) with placeholders and params in the same left-to-right order."""

    def __init__(self) -> None:
        self.params: List[Any] = []

    def p(self, v: Any) -> str:
        self.params.append(v)
        return "%s"

    def fieldname(self, f: Optional[str]) -> str:
        if not f or not _FIELD.match(f):
            raise FilterError(f"invalid field name {f!r}")
        return f

    def present(self, f: str) -> str:
        return f"(metadata ? {self.p(f)} AND jsonb_typeof(metadata->{self.p(f)}) <> 'null')"

    def eq(self, f: str, v: Any) -> str:
        if isinstance(v, bool):
            return f"(jsonb_typeof(metadata->{self.p(f)}) = 'boolean' AND (metadata->{self.p(f)})::boolean = {self.p(v)})"
        if isinstance(v, (int, float)):
            return (f"(jsonb_typeof(metadata->{self.p(f)}) = 'number' AND "
                    f"(metadata->>{self.p(f)})::double precision = {self.p(float(v))})")
        return f"(jsonb_typeof(metadata->{self.p(f)}) = 'string' AND metadata->>{self.p(f)} = {self.p(v)})"

    def rng(self, f: str, sym: str, v: Any) -> str:
        if isinstance(v, bool):
            raise FilterError("range filters on booleans are not supported")
        if isinstance(v, (int, float)):
            return (f"(jsonb_typeof(metadata->{self.p(f)}) = 'number' AND "
                    f"(metadata->>{self.p(f)})::double precision {sym} {self.p(float(v))})")
        return f"(jsonb_typeof(metadata->{self.p(f)}) = 'string' AND metadata->>{self.p(f)} {sym} {self.p(v)})"

    def rec(self, n: Node) -> str:
        if n.op == "and":
            return "(" + " AND ".join(self.rec(a) for a in n.args) + ")"
        if n.op == "or":
            return "(" + " OR ".join(self.rec(a) for a in n.args) + ")"
        if n.op == "not":
            return f"(NOT {self.rec(n.args[0])})"
        f = self.fieldname(n.field)
        if n.op == "exists":
            return self.present(f)
        if n.op == "eq":
            return self.eq(f, n.value)
        if n.op == "ne":
            return f"({self.present(f)} AND NOT {self.eq(f, n.value)})"
        if n.op == "in":
            return "(" + " OR ".join(self.eq(f, v) for v in n.value) + ")"
        if n.op == "nin":
            return f"({self.present(f)} AND NOT (" + " OR ".join(self.eq(f, v) for v in n.value) + "))"
        sym = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[n.op]
        return self.rng(f, sym, n.value)

    def compile(self, node: Optional[Node]) -> Tuple[str, List[Any]]:
        if node is None:
            return "TRUE", []
        sql = self.rec(node)
        return sql, self.params
