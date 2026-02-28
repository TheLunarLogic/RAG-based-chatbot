"""
RAG PDF Chatbot — Streamlit entry point.

Run with:  streamlit run app.py
"""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from rag import chatbot, ingest_pdf, retrieve_all_threads, get_metadata
from voice import speech_to_text, text_to_speech


# ═══════════════════════ Session helpers ═══════════════════════

def _generate_thread_id() -> str:
    return str(uuid.uuid4())


def _add_thread(thread_id: str) -> None:
    tid = str(thread_id)
    if tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(tid)


def _reset_chat() -> None:
    tid = _generate_thread_id()
    st.session_state["thread_id"] = tid
    _add_thread(tid)
    st.session_state["message_history"] = []


def _load_conversation(thread_id: str) -> list:
    tid = str(thread_id)
    state = chatbot.get_state(config={"configurable": {"thread_id": tid}})
    return state.values.get("messages", [])


def _handle_user_input(user_text: str, thread_key: str) -> str:
    """
    Send *user_text* through the RAG pipeline, display the streamed
    response, and return the full AI answer.
    """
    st.session_state["message_history"].append(
        {"role": "user", "content": user_text}
    )
    with st.chat_message("user"):
        st.markdown(user_text)

    config = {
        "configurable": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):

        def _stream():
            for chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_text)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content

        ai_message = st.write_stream(_stream())

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = get_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

    return ai_message


# ═══════════════════ Session initialization ═══════════════════

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = _generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

_add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None


# ═══════════════════════════ Sidebar ══════════════════════════

st.sidebar.title("RAG PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    _reset_chat()
    st.rerun()

if thread_docs:
    latest = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest.get('filename')}` "
        f"({latest.get('chunks')} chunks from {latest.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for tid in threads:
        if st.sidebar.button(str(tid), key=f"side-thread-{tid}"):
            selected_thread = tid


# ══════════════════════ Main chat area ════════════════════════

st.title("RAG PDF Chatbot")

# ── Chat history ──
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Voice input ──
col1, col2 = st.columns([1, 5])
with col1:
    speak_clicked = st.button("🎤 Speak")

if speak_clicked:
    with st.spinner("🎙️ Recording for 5 seconds…"):
        transcript = speech_to_text(duration=5)

    if transcript and not transcript.startswith("["):
        ai_response = _handle_user_input(transcript, thread_key)
        with st.spinner("🔊 Generating audio…"):
            audio_bytes = text_to_speech(ai_response)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
    elif transcript.startswith("["):
        st.error(transcript)
    else:
        st.warning("No speech detected. Please try again.")

# ── Text input ──
user_input = st.chat_input("Ask about your document")
if user_input:
    _handle_user_input(user_input, thread_key)

st.divider()

# ── Thread switching ──
if selected_thread:
    sel_tid = str(selected_thread)
    st.session_state["thread_id"] = sel_tid
    messages = _load_conversation(sel_tid)

    st.session_state["message_history"] = [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content,
        }
        for m in messages
    ]
    st.session_state["ingested_docs"].setdefault(sel_tid, {})
    st.rerun()
