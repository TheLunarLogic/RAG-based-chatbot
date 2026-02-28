"""
Text-to-speech via Deepgram aura-asteria-en REST API.
"""

from __future__ import annotations

import requests

from config import DEEPGRAM_API_KEY

_TTS_URL = "https://api.deepgram.com/v1/speak"


def text_to_speech(text: str) -> bytes | None:
    """
    Convert *text* to speech and return raw MP3 bytes.

    Use with ``st.audio(mp3_bytes, format="audio/mp3")`` in Streamlit.
    Returns ``None`` on error.
    """
    if not DEEPGRAM_API_KEY or not text or not text.strip():
        return None

    try:
        resp = requests.post(
            _TTS_URL,
            params={"model": "aura-asteria-en"},
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"text": text},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"[TTS error] {e}")
        return None
