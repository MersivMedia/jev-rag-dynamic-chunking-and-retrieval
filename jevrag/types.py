"""Core data types shared by every stage."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Fixed namespace so record IDs are identical on every machine and every run.
RECORD_NAMESPACE = uuid.UUID("5b0f3a52-8f0e-4b8e-9d2a-6a1e2c7b9f40")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalise_text(text: str) -> str:
    """Normalise line endings and trailing whitespace; used for content hashes."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip("\n")


def record_id(doc_id: str, chunk_hash: str) -> str:
    """UUIDv5 of (doc_id, chunk content hash): valid in every supported store."""
    return str(uuid.uuid5(RECORD_NAMESPACE, f"{doc_id}\x1f{chunk_hash}"))


@dataclass
class Document:
    """A document to ingest.

    ``doc_id`` must be stable across runs (URL, path or database key): it is what
    lets re-ingestion replace old chunks instead of duplicating them.
    ``format`` is ``text``, ``markdown`` or ``html``.
    """

    text: str
    doc_id: Optional[str] = None
    title: str = ""
    source_uri: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    format: str = "markdown"

    def resolved_id(self) -> str:
        if self.doc_id:
            return self.doc_id
        if self.source_uri:
            return self.source_uri
        return "sha256:" + sha256(normalise_text(self.text))[:32]


@dataclass
class Block:
    """A structural block of a parsed document, with offsets into its text."""

    kind: str  # heading | paragraph | list_item | table | code | quote
    start: int
    end: int
    section_path: List[str]
    level: int = 0  # heading level for headings


@dataclass
class Unit:
    """The smallest piece the chunker places cuts between."""

    start: int
    end: int
    tokens: int
    kind: str  # sentence | table | code | heading
    block: int
    first_in_block: bool
    section: int


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_index: int
    char_start: int
    char_end: int
    section_path: List[str]
    chunker: str
    tokens: int
    embed_text: str = ""
    title: str = ""
    source_uri: str = ""
    # q_<name> -> probability or value; tag_<field> -> option; tag_<field>_p -> probability
    answers: Dict[str, Any] = field(default_factory=dict)
    quarantined: bool = False
    dropped_reason: Optional[str] = None

    @property
    def content_hash(self) -> str:
        return sha256(normalise_text(self.text))

    @property
    def id(self) -> str:
        return record_id(self.doc_id, self.content_hash)


@dataclass
class Record:
    """What a store adapter writes: logical fields, flat scalar metadata."""

    id: str
    vector: Optional[List[float]]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    id: str
    score: float  # normalised similarity in [0, 1], higher is better
    raw_score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
