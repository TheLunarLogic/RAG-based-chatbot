"""
voice package — Deepgram STT + TTS utilities.

Public API
----------
- ``speech_to_text(duration, sample_rate)``  Record mic → transcript
- ``text_to_speech(text)``                   Text → MP3 bytes
"""

from voice.stt import speech_to_text   # noqa: F401
from voice.tts import text_to_speech   # noqa: F401
