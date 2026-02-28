# RAG PDF Chatbot with Voice

A conversational RAG (Retrieval-Augmented Generation) chatbot that answers questions about uploaded PDF documents — with optional voice input/output via Deepgram.

---

## Features

- **PDF Q&A** — Upload a PDF, ask questions, get answers grounded in the document.
- **Voice Input** — Click 🎤 Speak to ask questions via microphone (Deepgram STT, nova-3).
- **Voice Output** — AI responses are spoken back via Deepgram TTS (aura-asteria-en).
- **Multi-thread** — Multiple chat threads with strict per-thread document isolation.
- **Conversation Memory** — Powered by LangGraph with SQLite checkpointing.
- **Streaming** — Real-time token streaming for assistant responses.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — `llama-3.1-8b-instant` |
| Embeddings | HuggingFace — `BAAI/bge-small-en-v1.5` |
| Vector Store | FAISS (CPU) |
| Orchestration | LangGraph + SQLite checkpointer |
| STT | Deepgram nova-3 |
| TTS | Deepgram aura-asteria-en |
| Frontend | Streamlit |

---

## Project Structure

```
Agent/
├── app.py              # Streamlit entry point
├── config.py           # LLM, embeddings, API keys
├── rag/
│   ├── __init__.py     # Public API
│   ├── retriever.py    # Thread-scoped retriever store
│   ├── ingestion.py    # PDF → FAISS indexing
│   └── graph.py        # LangGraph state, chat node, graph
├── voice/
│   ├── __init__.py     # Public API
│   ├── stt.py          # Speech-to-text (Deepgram)
│   └── tts.py          # Text-to-speech (Deepgram)
├── .env                # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create conda environment

```bash
conda create -n rag python=3.11 -y
conda activate rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
```

- Get a Groq API key at [console.groq.com](https://console.groq.com)
- Get a Deepgram API key at [console.deepgram.com](https://console.deepgram.com)

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. **Upload a PDF** — Use the sidebar file uploader.
2. **Ask questions** — Type in the chat input or click 🎤 Speak.
3. **Switch threads** — Click past conversations in the sidebar to resume.
4. **New chat** — Click "New Chat" to start a fresh thread with a new document.

---

## Notes

- Voice recording uses `sounddevice` which records from the **local machine's microphone**. This works when running Streamlit on localhost.
- The `DEEPGRAM_API_KEY` is optional — text chat works without it, only voice features require it.
- Conversation history is persisted in `chatbot.db` (SQLite). Delete this file to reset all threads.
