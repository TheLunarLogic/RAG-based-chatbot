"""
LangGraph graph definition: state, chat node, checkpointer, compilation.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from config import llm
from rag.retriever import get_retriever, get_metadata


# ── State ──
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Chat node ──
def chat_node(state: ChatState, config=None):
    """
    RAG chat node:
      - Retrieves context from the PDF indexed for this thread.
      - Passes context + user question to the LLM.
      - If no PDF is indexed, asks the user to upload one.
    """
    # Normalize thread_id to str for consistent dict lookup
    thread_id = None
    if config and isinstance(config, dict):
        raw_id = config.get("configurable", {}).get("thread_id")
        thread_id = str(raw_id) if raw_id is not None else None

    user_message = state["messages"][-1].content if state["messages"] else ""

    # Strict per-thread retriever lookup
    retriever = get_retriever(thread_id)

    if retriever is not None:
        docs = retriever.invoke(user_message)
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        source_file = get_metadata(str(thread_id)).get("filename", "unknown")

        system_message = SystemMessage(
            content=(
                f"You are a helpful assistant. Answer the user's question using ONLY "
                f"the context below from the document '{source_file}'. "
                f"If the answer is not in the context, say so.\n\n"
                f"Context:\n{context}"
            )
        )
    else:
        system_message = SystemMessage(
            content=(
                "You are a helpful assistant. No PDF document has been uploaded yet. "
                "Please ask the user to upload a PDF so you can answer questions about it."
            )
        )

    messages = [system_message, *state["messages"]]
    response = llm.invoke(messages, config=config)
    return {"messages": [response]}


# ── Checkpointer ──
_conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=_conn)

# ── Compile graph ──
_graph = StateGraph(ChatState)
_graph.add_node("chat_node", chat_node)
_graph.add_edge(START, "chat_node")
_graph.add_edge("chat_node", END)

chatbot = _graph.compile(checkpointer=checkpointer)


# ── Helpers ──
def retrieve_all_threads() -> list[str]:
    """Return all thread_ids that have at least one checkpoint."""
    threads = set()
    for cp in checkpointer.list(None):
        threads.add(cp.config["configurable"]["thread_id"])
    return list(threads)
