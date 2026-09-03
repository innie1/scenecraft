"""
Client for a local Ollama server — the only "AI" in this app that
actually reasons rather than pattern-matches. No network calls beyond
localhost, no API key, no cost per call. Used as a fallback when the
composer's deterministic patterns (see desktop_app/main.py run_command)
don't match, and to turn a text description into motion-graphics code
(see motion_graphics_ops.py).

This module only talks HTTP to Ollama's local REST API — no UI, no
ffmpeg, no pywebview, importable and testable on its own.
"""

import json
import urllib.request
import urllib.error

DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5-coder:3b"


class OllamaUnavailableError(Exception):
    pass


def is_available(host: str = DEFAULT_HOST, timeout: float = 1.5) -> bool:
    """True if an Ollama server is actually reachable right now. Cheap
    enough to call before every fallback attempt — no need to cache."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def list_models(host: str = DEFAULT_HOST, timeout: float = 3.0) -> list[str]:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []


def generate(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: float = 120.0,
) -> str:
    """One-shot text generation (not chat/streaming — simplest thing that
    works for both the composer fallback and motion-graphics code gen).
    Raises OllamaUnavailableError if the server can't be reached at all,
    so callers can distinguish "not running" from "returned garbage"."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError) as e:
        raise OllamaUnavailableError(f"Could not reach Ollama at {host}: {e}") from e
    return data.get("response", "")


def _extract_json(text: str) -> dict | None:
    """Models wrap JSON in prose/code fences more often than not — pull
    out the first {...} block and parse that instead of demanding a
    perfectly bare JSON response."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


COMMAND_SYSTEM_PROMPT = """You translate a video editor command into ONE JSON action. \
Reply with ONLY a JSON object, no prose, no markdown fences.

Valid actions and their JSON shape:
- {"action": "cut", "start": <seconds>, "end": <seconds>} — REMOVES/TRIMS a
  range of the video. Any command about trimming, cutting, or removing a
  portion is "cut", never "seek" — "seek" is ONLY for moving the playhead
  to look at a moment, not for editing anything out.
- {"action": "seek", "time": <seconds>} — just moves the playhead, nothing
  is edited.
- {"action": "effect", "effect": "brightness"|"contrast"|"saturation", "delta": <number>} —
  delta is a SMALL relative nudge, always between -0.5 and 0.5 (e.g. 0.15
  for "a bit more", 0.4 for "a lot more/much brighter"), never a large
  number.
- {"action": "effect", "effect": "grayscale", "value": true}
- {"action": "unknown"}

Convert spoken time to seconds exactly: "a minute fifteen" = 75 (60+15),
"a minute thirty" = 90, "two minutes" = 120. Never guess a rounder number
than the math gives.

Examples:
"trim the first 3 seconds off" -> {"action": "cut", "start": 0, "end": 3}
"remove the last part, from 1:20 to the end isn't needed, cut 1:20 to 1:35" -> {"action": "cut", "start": 80, "end": 95}
"jump to a minute fifteen" -> {"action": "seek", "time": 75}
"take me to 30 seconds in" -> {"action": "seek", "time": 30}
"desaturate this a bit" -> {"action": "effect", "effect": "saturation", "delta": -0.15}
"pump up the contrast" -> {"action": "effect", "effect": "contrast", "delta": 0.3}

If the command doesn't clearly map to one of these, reply {"action": "unknown"}. \
Never invent an action outside this list."""


def interpret_command(command_text: str, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> dict:
    """Fallback interpreter for composer commands the deterministic
    patterns didn't match. Returns a dict with at least an "action" key;
    {"action": "unknown"} (or a dict lacking "action" entirely, if the
    model's output couldn't be parsed at all) means it couldn't help
    either — callers should fall back to the normal "didn't recognize
    that" message, not assume this always succeeds."""
    raw = generate(command_text, system=COMMAND_SYSTEM_PROMPT, model=model, host=host)
    parsed = _extract_json(raw)
    return parsed if parsed is not None else {"action": "unknown", "raw": raw}
