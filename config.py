"""
Shared configuration: LLM and embedding models.

All environment variables are loaded here via dotenv.
Other modules should import from this file instead of
re-initializing models.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# ── Groq LLM ──
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=1024,
)

# ── Embedding model ──
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ── Deepgram ──
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
