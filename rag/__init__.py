"""
rag package — RAG pipeline built on LangGraph + FAISS + Groq.

Public API
----------
- ``chatbot``               Compiled LangGraph chatbot
- ``ingest_pdf(...)``        Index a PDF for a thread
- ``retrieve_all_threads()`` List all checkpointed thread IDs
- ``get_metadata(tid)``      Get ingestion metadata for a thread
"""

from rag.graph import chatbot, retrieve_all_threads          # noqa: F401
from rag.ingestion import ingest_pdf                         # noqa: F401
from rag.retriever import get_metadata, has_document         # noqa: F401
