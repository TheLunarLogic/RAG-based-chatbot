"""
Speech-to-text via Deepgram nova-3 REST API.
"""

from __future__ import annotations

import io

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

from config import DEEPGRAM_API_KEY

_STT_URL = "https://api.deepgram.com/v1/listen"


def speech_to_text(duration: int = 5, sample_rate: int = 16000) -> str:
    """
    Record *duration* seconds from the default microphone and return
    the Deepgram transcript.

    Returns an empty string if nothing was detected, or an error string
    wrapped in ``[brackets]`` on failure.
    """
    if not DEEPGRAM_API_KEY:
        return "[Error] DEEPGRAM_API_KEY not set in .env"

    # 1. Record audio
    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16,
        )
        sd.wait()
    except Exception as e:
        return f"[Recording error] {e}"

    # 2. Convert to WAV bytes in memory
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    # 3. Call Deepgram STT
    try:
        resp = requests.post(
            _STT_URL,
            params={"model": "nova-3", "smart_format": "true"},
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            },
            data=wav_bytes,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
        )
    except Exception as e:
        return f"[STT error] {e}"
