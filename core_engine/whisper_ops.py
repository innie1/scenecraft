"""
Local transcription using whisper.cpp (via pywhispercpp bindings).
No API keys, no network calls, no PyTorch. This is the "find where I
said this word" feature: transcribe() gives you timestamped segments,
search_transcript() finds which segment a word/phrase falls in.
"""

import subprocess
import tempfile
from pathlib import Path

from pywhispercpp.model import Model

_model_cache: dict[str, Model] = {}


def _get_model(model_name: str = "base") -> Model:
    """Whisper models load once and are reused across calls — loading is
    the slow part, not transcribing, so we cache by model size."""
    if model_name not in _model_cache:
        _model_cache[model_name] = Model(model_name)
    return _model_cache[model_name]


def _extract_audio(video_path: str) -> str:
    """whisper.cpp wants 16kHz mono WAV. Pull audio out of the video first."""
    tmp_wav = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", tmp_wav,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return tmp_wav


def transcribe(video_path: str, model_name: str = "base") -> list[dict]:
    """
    Returns a list of {start, end, text} segments with real timestamps.
    Runs entirely locally — no API call, no internet needed once the
    model file is downloaded the first time.
    """
    model = _get_model(model_name)
    wav_path = _extract_audio(video_path)
    try:
        segments = model.transcribe(wav_path)
    finally:
        Path(wav_path).unlink(missing_ok=True)

    return [
        {
            "start": round(seg.t0 / 100, 2),  # pywhispercpp gives centiseconds
            "end": round(seg.t1 / 100, 2),
            "text": seg.text.strip(),
        }
        for seg in segments
    ]


def search_transcript(segments: list[dict], query: str) -> list[dict]:
    """Find every segment containing the query (case-insensitive) and
    return them with their timestamps — this is what powers 'where did
    I say this word' and jumps the player to that point."""
    q = query.lower().strip()
    if not q:
        return []
    return [seg for seg in segments if q in seg["text"].lower()]
