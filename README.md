# Scenecraft

Offline-first AI-assisted video editor. Built as a standalone desktop app,
separate from agentie-v2. No paid APIs in the engine — everything runs
locally.

## Structure

```
core_engine/      Pure Python. FFmpeg operations + project state (JSON).
                   No UI code, no MCP code. Both the desktop app and the
                   future MCP server call into this and only this.
desktop_app/       pywebview shell — the offline app itself.
                   main.py exposes core_engine functions to the UI as a
                   JS-callable API (window.pywebview.api.*).
                   ui/index.html is the editor screen: open video, preview,
                   trim scene by start/end seconds, cut.
mcp_server/        Empty for now. Built later, once the offline app works.
                   Will expose core_engine as MCP tools so Agentie or any
                   other agent can drive the same engine over stdio.
```

## Run it

```
pip install -r requirements.txt
python desktop_app/main.py
```

This opens a native window. Click "Open video", pick a file, set a
start/end time, click "Cut scene" — it calls ffmpeg directly and saves the
scene to `~/Scenecraft/<project name>/scenes/`.

## What's already working (tested, not just written)

- `core_engine/ffmpeg_ops.py` — `probe()` and `cut_clip()` were run against
  a real generated test video during scaffolding: cut a 3-second scene out
  of a 10-second clip, verified the output duration. This works.
- Fast cut uses stream copy (no re-encode, near-instant); falls back to
  re-encode automatically if the cut point isn't on a keyframe.

## What's NOT built yet (in order of what to do next)

1. **Whisper integration** (`core_engine/whisper_ops.py`) — transcription
   and "find where I said this word" search. Use whisper.cpp, not
   openai-whisper — see size notes below.
2. **Export/render pipeline** — currently `export()` just re-encodes the
   whole source; needs to stitch the actual scene list from
   `project_state.py` into one output.
3. **Packaging** — PyInstaller + Inno Setup, same pipeline as agentie-v2.
   Bundle ffmpeg + whisper.cpp binaries so users don't need to install
   anything separately.
4. **MCP server layer** — once the offline app is solid, wrap
   `core_engine` functions as MCP tools in `mcp_server/`.

## Size notes

Use **whisper.cpp**, not openai-whisper (PyTorch). PyTorch alone adds
600MB-1GB to the installer for no benefit here — whisper.cpp does the same
transcription job in ~5-10MB plus a quantized model (75-500MB depending on
size chosen). Target for v1: ~300-400MB installer using the "base" model.
