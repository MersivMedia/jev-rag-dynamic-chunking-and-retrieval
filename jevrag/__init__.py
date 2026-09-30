"""Jev-steered chunking, ingestion and retrieval for any vector database."""

__version__ = "1.0.0.dev0"

from .types import Chunk, Document  # noqa: E402
from .pipeline import IngestReport, Pipeline  # noqa: E402
from .retrieve.result import Passage, RetrievalResult  # noqa: E402

__all__ = [
    "__version__",
    "Chunk",
    "Document",
    "IngestReport",
    "Passage",
    "Pipeline",
    "RetrievalResult",
]
