"""
Turns a raw pasted script into scenes for the guided recording flow.
Pure text parsing — no UI, no ffmpeg, no pywebview. This is deterministic
(not AI): it recognizes explicit "Scene N" markers first, falling back
to blank-line-separated paragraphs, with acting/direction notes pulled
out of bracketed lines like "[speak slowly, smile]".
"""

import re

_SCENE_MARKER = re.compile(r"^\s*scene\s+\d+\s*[:\-]?\s*$", re.IGNORECASE)
_DIRECTION_LINE = re.compile(r"^\[(.+)\]$")


def parse_script(raw_text: str) -> list[dict]:
    """
    Returns a list of {id, text, direction, recorded_path} dicts — the
    same shape as project_state.ScriptScene, as plain dicts so callers
    can build ScriptScene(**scene) directly.
    """
    text = (raw_text or "").strip()
    if not text:
        return []

    lines = text.splitlines()
    has_markers = any(_SCENE_MARKER.match(line) for line in lines)
    blocks = _split_on_markers(lines) if has_markers else _split_on_blank_lines(lines)

    scenes = []
    for i, block in enumerate(blocks, start=1):
        direction_parts = []
        content_lines = []
        for line in block:
            stripped = line.strip()
            if not stripped:
                continue
            m = _DIRECTION_LINE.match(stripped)
            if m:
                direction_parts.append(m.group(1))
            else:
                content_lines.append(stripped)

        content = " ".join(content_lines).strip()
        direction = " ".join(direction_parts).strip()
        if not content and not direction:
            continue
        scenes.append({
            "id": f"scene_{i}",
            "text": content,
            "direction": direction,
            "recorded_path": None,
        })
    return scenes


def _split_on_markers(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if _SCENE_MARKER.match(line):
            if current:
                blocks.append(current)
            current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _split_on_blank_lines(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    block: list[str] = []
    for line in lines:
        if line.strip() == "":
            if block:
                blocks.append(block)
                block = []
        else:
            block.append(line)
    if block:
        blocks.append(block)
    return blocks
