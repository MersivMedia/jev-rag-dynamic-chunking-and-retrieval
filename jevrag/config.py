"""``jevrag.yaml``: every setting in one file; secrets only from the environment (FR-X3)."""

from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from .chunk import BoundaryConfig, ChunkConfig
from .enrich import EnrichConfig, Taxonomy
from .jev import JevConfig
from .retrieve import AnswerConfig, RetrieveConfig

DEFAULT_FILE = "jevrag.yaml"

TEMPLATE = """\
# jevrag configuration. Secrets come from the environment only.
# Thresholds are starting points, not measured defaults: tune them on your own data.

jev:
  backend: auto                # auto | typesafe | vercel | openrouter (auto = first key found)
  # model: jev-1.13.0          # pin a version; each backend has its own default ID
  cache_dir: .jevrag/cache     # answers are cached by content hash
  max_rps: 30                  # published limit is 40 requests/s

store:
{store}

embedder:
  model: {embedder}
  batch_size: 128

chunking:
  method: jev                  # jev | structural | fixed | semantic-embedding
  mode: on                     # on | shadow | off
  min_tokens: 64
  target_tokens: 350
  max_tokens: 800
  overlap_tokens: 0

enrich:
  mode: on
  drop_low_information: 0.85
  drop_boilerplate: 0.85
  quarantine_instructs_ai: 0.70
  tag_min_confidence: 0.50
  # taxonomy: taxonomy.yaml

retrieve:
  top_k: 30
  expand_neighbours: false
  route: {{ mode: on, min_confidence: 0.60, top2_mass: 0.80, min_candidates: 5 }}
  classify:
    mode: on
    drop_instructs_ai: 0.70
    min_relevant: 0.50
    min_evidence: 0.40
    conflict: 0.60
    max_passages: 8
  gate: {{ mode: on, answer_min: 0.35 }}

answer:                        # only used by answer() / `jevrag query --answer`
  provider: openai             # openai | anthropic | gateway
  model: gpt-4.1-mini
"""

STORE_TEMPLATES = {
    "memory": "  kind: memory\n  path: .jevrag/memory-store.json",
    "qdrant": "  kind: qdrant\n  url: http://localhost:6333\n  # api_key_env: QDRANT_API_KEY",
    "chroma": "  kind: chroma\n  path: .jevrag/chroma        # or host: localhost + port: 8000",
    "pgvector": "  kind: pgvector\n  dsn_env: DATABASE_URL        # postgresql://user:pass@host:5432/db",
    "pinecone": "  kind: pinecone\n  index: jevrag                # collection = namespace in this index\n"
                "  # api_key_env: PINECONE_API_KEY",
}


def _build(cls: Any, data: Optional[Mapping[str, Any]]) -> Any:
    """Build a config dataclass from a dict, recursing into nested dataclass fields."""
    values = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} settings: {sorted(unknown)}")
    defaults = cls()  # every config dataclass has full defaults
    kwargs = {}
    for name, value in values.items():
        current = getattr(defaults, name)
        if is_dataclass(current) and isinstance(value, Mapping):
            kwargs[name] = _build(type(current), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _mode(v: Any) -> Any:
    # YAML reads bare on/off as booleans
    if v is True:
        return "on"
    if v is False:
        return "off"
    return v


def _fix_modes(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: (_mode(v) if k == "mode" else _fix_modes(v)) for k, v in d.items()}
    return d


class Config:
    def __init__(self, raw: Optional[Mapping[str, Any]] = None, base_dir: str = ".") -> None:
        data: Dict[str, Any] = _fix_modes(dict(raw or {}))
        unknown = set(data) - {"jev", "store", "embedder", "chunking", "enrich", "retrieve", "answer"}
        if unknown:
            raise ValueError(f"unknown config sections: {sorted(unknown)}")
        self.raw = data
        self.base_dir = base_dir
        self.jev = _build(JevConfig, data.get("jev"))
        self.store: Dict[str, Any] = dict(data.get("store") or {"kind": "memory"})
        emb = dict(data.get("embedder") or {})
        self.embedder_spec = emb.pop("model", None) or emb.pop("spec", None)
        self.embedder_options = emb
        ch = dict(data.get("chunking") or {})
        boundary = ch.pop("boundary", None)
        self.chunking = _build(ChunkConfig, ch)
        if boundary:
            self.chunking.boundary = _build(BoundaryConfig, boundary)
        en = dict(data.get("enrich") or {})
        tax_path = en.pop("taxonomy", None)
        self.enrich = _build(EnrichConfig, en)
        if tax_path:
            p = tax_path if os.path.isabs(tax_path) else os.path.join(base_dir, tax_path)
            self.enrich.taxonomy = Taxonomy.load(p)
        self.retrieve = _build(RetrieveConfig, data.get("retrieve"))
        self.answer = _build(AnswerConfig, data.get("answer"))

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        p = Path(path or DEFAULT_FILE)
        if not p.exists():
            if path:
                raise FileNotFoundError(f"config file not found: {p}")
            return cls({})
        with open(p, encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh) or {}, base_dir=str(p.parent))


def render_template(store: str = "memory", embedder: str = "openai:text-embedding-3-small") -> str:
    if store not in STORE_TEMPLATES:
        raise ValueError(f"unknown store {store!r}; choose from {sorted(STORE_TEMPLATES)}")
    return TEMPLATE.format(store=STORE_TEMPLATES[store], embedder=embedder)
