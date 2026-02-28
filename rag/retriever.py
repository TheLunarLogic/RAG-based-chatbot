"""
Thread-scoped retriever store.

Stores one FAISS retriever per chat thread, ensuring strict isolation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ── Private stores (keyed by str(thread_id)) ──
_retrievers: Dict[str, Any] = {}
_metadata: Dict[str, dict] = {}


def get_retriever(thread_id: Optional[str]):
    """Return the FAISS retriever for *thread_id*, or None."""
    if not thread_id:
        return None
    return _retrievers.get(str(thread_id))


def set_retriever(thread_id: str, retriever, meta: dict) -> None:
    """Store a retriever and its metadata for *thread_id*."""
    key = str(thread_id)
    _retrievers[key] = retriever
    _metadata[key] = meta


def has_document(thread_id: str) -> bool:
    """Check whether a document has been indexed for *thread_id*."""
    return str(thread_id) in _retrievers


def get_metadata(thread_id: str) -> dict:
    """Return ingestion metadata (filename, pages, chunks) for *thread_id*."""
    return _metadata.get(str(thread_id), {})
