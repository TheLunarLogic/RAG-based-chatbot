"""
PDF ingestion: load, split, embed, and store as a FAISS retriever.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import embeddings
from rag.retriever import set_retriever


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Build a FAISS retriever from *file_bytes* and register it for *thread_id*.

    Returns a summary dict: ``{filename, documents, chunks}``.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()

        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        ).split_documents(docs)

        retriever = FAISS.from_documents(chunks, embeddings).as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        name = filename or os.path.basename(tmp_path)
        meta = {"filename": name, "documents": len(docs), "chunks": len(chunks)}

        set_retriever(thread_id, retriever, meta)
        return meta
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
