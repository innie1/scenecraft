"""
Scenecraft — offline desktop app.
This is the ONLY place that touches pywebview. It exposes core_engine
functions to the UI as a JS-callable API. The UI never talks to ffmpeg
directly — it always goes through this Api class.
"""

import sys
import os
import re
import json
import uuid
import hashlib
import http.server
import socketserver
import threading
import urllib.parse
import mimetypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import webview
from core_engine import (
    ffmpeg_ops, project_state, whisper_ops, script_ops, llm_ops,
    motion_graphics_ops, link_ops,
)

SCENECRAFT_ROOT = Path.home() / "Scenecraft"
SETTINGS_PATH = SCENECRAFT_ROOT / "settings.json"

# Local models the picker offers. Not every entry is necessarily pulled —
# get_model_status() cross-references this against llm_ops.list_models()
# so the UI can show "(not installed)" rather than silently doing nothing.
AVAILABLE_LOCAL_MODELS = [
    {"id": "qwen2.5-coder:3b", "label": "Qwen2.5 Coder 3B"},
    {"id": "gemma4:e2b-it-q4_K_M", "label": "Gemma 4 (e2b)"},
]


def _strip_markdown_fences(text: str) -> str:
    """Local models wrap code in ```html fences more often than not,
    even when told not to — strip them rather than fail on it."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

# WebView2 on Windows won't reliably play file:// video (no seeking, often
# no rendering at all). Instead we serve video files ourselves over plain
# HTTP with Range support, and point <video src> at that. Only paths this
# app has explicitly registered (via Api._media_url) are servable — this
# is a local, no-auth server, so it must not become an arbitrary local
# file server just because it's bound to 127.0.0.1.
_ALLOWED_MEDIA_PATHS: set[str] = set()

# One-time upload slots for the guided-recording flow: the browser
# MediaRecorder produces a Blob in JS and POSTs it straight to this
# server (rather than round-tripping through the JSON pywebview bridge,
# which is a poor fit for tens-of-MB video). A token maps to exactly one
# server-chosen destination path — the client never gets to choose where
# a POST writes, only which pre-registered slot it fills.
_UPLOAD_TOKENS: dict[str, str] = {}


class _RangeRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep stdout clean; errors still show via send_error

    def do_HEAD(self):
        self._serve(send_body=False)

    def do_GET(self):
        self._serve(send_body=True)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/upload":
            self.send_error(404, "Not found")
            return
        query = urllib.parse.parse_qs(parsed.query)
        token = query.get("token", [None])[0]
        dest = _UPLOAD_TOKENS.pop(token, None) if token else None
        if not dest:
            self.send_error(403, "Invalid or already-used upload token")
            return

        dest_path = Path(dest)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        content_length = int(self.headers.get("Content-Length", 0))
        with open(dest_path, "wb") as f:
            remaining = content_length
            chunk_size = 256 * 1024
            while remaining > 0:
                chunk = self.rfile.read(min(chunk_size, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)

        body = json.dumps({"ok": True, "path": str(dest_path)}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve(self, send_body: bool):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/media":
            self.send_error(404, "Not found")
            return
        query = urllib.parse.parse_qs(parsed.query)
        raw = query.get("path", [None])[0]
        if not raw:
            self.send_error(400, "Missing path")
            return
        file_path = urllib.parse.unquote(raw)
        if file_path not in _ALLOWED_MEDIA_PATHS:
            self.send_error(403, "Not registered for serving")
            return
        p = Path(file_path)
        if not p.is_file():
            self.send_error(404, "File not found")
            return

        file_size = p.stat().st_size
        content_type = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        range_header = self.headers.get("Range")

        start, end = 0, file_size - 1
        status = 200
        if range_header:
            try:
                _, rng = range_header.split("=", 1)
                start_str, end_str = rng.split("-", 1)
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                end = min(end, file_size - 1)
                if start > end or start < 0:
                    raise ValueError
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Access-Control-Allow-Origin", "*")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        if not send_body:
            return
        chunk_size = 256 * 1024
        with open(p, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(chunk_size, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    return
                remaining -= len(chunk)


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def _start_media_server() -> int:
    """Starts the range-request media server on a free localhost port in a
    background thread, returns the port it bound to."""
    server = _ThreadingHTTPServer(("127.0.0.1", 0), _RangeRequestHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return port


class Api:
    def __init__(self, media_port: int):
        self.project: project_state.Project | None = None
        self.project_path: str | None = None
        self.media_port = media_port
        # set while the assistant is waiting on an answer to something it
        # asked (see chat()); None means no question is outstanding
        self.pending_flow: dict | None = None

    def _progress(self, stage: str):
        """Tell the UI what this call is actually doing right now.

        Everything the Director does is one blocking bridge call, so
        without this the user stares at a spinner for a minute with no
        idea whether it's thinking, rendering or encoding. Best-effort by
        design: if there's no window (tests, headless runs) or the bridge
        is busy, progress reporting must never break the actual work."""
        try:
            js = f"window.scenecraftProgress && window.scenecraftProgress({json.dumps(stage)})"
            webview.windows[0].evaluate_js(js)
        except Exception:
            pass

    def _media_url(self, path: str | None) -> str | None:
        if not path:
            return None
        _ALLOWED_MEDIA_PATHS.add(path)
        return f"http://127.0.0.1:{self.media_port}/media?path={urllib.parse.quote(path, safe='')}"

    def _playable_path(self, source: str) -> str:
        """A file that plays fine in VLC can still fail to render at all
        in the embedded browser if its codec isn't one Chromium decodes
        natively (HEVC footage from phones/cameras is the common case).
        Returns the original path when it's already browser-safe, or a
        cached H.264/AAC proxy (generated once) when it isn't. Only ever
        used for the <video> src — cut_scene/transcribe/export always
        read `source` itself, at full original quality."""
        try:
            info = ffmpeg_ops.probe(source)
        except Exception:
            return source
        if ffmpeg_ops.is_browser_playable(info):
            return source
        proxy_name = hashlib.sha1(source.encode("utf-8")).hexdigest() + ".mp4"
        proxy_path = str(SCENECRAFT_ROOT / "_proxies" / proxy_name)
        if not Path(proxy_path).exists():
            ffmpeg_ops.make_browser_proxy(source, proxy_path)
        return proxy_path

    def _tracks_payload(self):
        return [
            {
                "kind": t.kind,
                "clips": [vars(c) | {"duration": c.duration, "end_time": c.end_time} for c in t.clips],
            }
            for t in self.project.tracks
        ]

    def _project_payload(self):
        if not self.project:
            return None
        return {
            "name": self.project.name,
            "source_video": self.project.source_video,
            "tracks": self._tracks_payload(),
            "transcript": self.project.transcript,
            "script": [vars(s) for s in self.project.script],
            "aspect_ratio": self.project.aspect_ratio,
        }

    def _save_project(self):
        if self.project and self.project_path:
            project_state.save(self.project, self.project_path)

    def save_project(self):
        """Explicit save, used by the editor's back button before it
        navigates away — mirrors the auto-save that already runs after
        cut_scene/transcribe, as a safety net."""
        self._save_project()
        return {"ok": True}

    def _generate_project_name(self) -> str:
        n = 1
        while (SCENECRAFT_ROOT / f"Project {n}").exists():
            n += 1
        return f"Project {n}"

    def new_project(self):
        """No prompt, no file picker — creates an empty, auto-named project
        and saves it immediately so there's a real project.json on disk to
        come back to. The user imports a video afterward, from inside the
        editor (see import_video)."""
        name = self._generate_project_name()
        project = project_state.Project(name=name, source_video="")
        path = str(SCENECRAFT_ROOT / project.name / "project.json")
        project_state.save(project, path)

        self.project = project
        self.project_path = path
        return {"path": None, "info": None, "project": self._project_payload()}

    def import_video(self):
        """Opens the video picker (multi-select) and adds each result to
        the video track, in order, at the end of whatever's already
        there — picking several files stitches them together on the
        timeline. The last one imported becomes the project's active
        source for cut/transcribe."""
        if not self.project:
            return {"error": "No project loaded."}
        result = webview.windows[0].create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            file_types=("Video files (*.mp4;*.mov;*.mkv;*.avi;*.webm)", "All files (*.*)"),
        )
        if not result:
            return None

        video_track = self.project.track("video")
        last_info = None
        for source in result:
            try:
                info = ffmpeg_ops.probe(source)
            except Exception as e:
                return {"error": f"Couldn't read {source}: {e}"}
            last_info = info
            self.project.source_video = source
            timeline_start = max((c.end_time for c in video_track.clips), default=0.0)
            clip = project_state.Clip(
                id=f"import_{len(video_track.clips) + 1}",
                source_path=source,
                in_point=0.0,
                out_point=info["duration_seconds"],
                start_time=timeline_start,
            )
            self.project.add_clip(clip, track="video")

        self._save_project()
        try:
            playable = self._playable_path(self.project.source_video)
        except Exception as e:
            return {"error": f"Imported, but couldn't prepare it for preview: {e}"}
        return {"path": self._media_url(playable), "info": last_info, "project": self._project_payload()}

    def add_audio_clip(self, start_time: float | None = None):
        """Opens an audio picker and adds the result to the audio track.
        start_time=None appends after whatever's already there; a number
        places it at that exact position (e.g. the composer's "add music
        here" uses the current playhead)."""
        if not self.project:
            return {"error": "No project loaded."}
        result = webview.windows[0].create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("Audio files (*.mp3;*.wav;*.m4a;*.aac;*.flac;*.ogg)", "All files (*.*)"),
        )
        if not result:
            return None
        source = result[0]
        try:
            info = ffmpeg_ops.probe(source)
        except Exception as e:
            return {"error": str(e)}

        audio_track = self.project.track("audio")
        placed_at = (
            start_time if start_time is not None
            else max((c.end_time for c in audio_track.clips), default=0.0)
        )
        clip = project_state.Clip(
            id=f"audio_{len(audio_track.clips) + 1}",
            source_path=source,
            in_point=0.0,
            out_point=info["duration_seconds"],
            start_time=max(0.0, placed_at),
        )
        self.project.add_clip(clip, track="audio")
        self._save_project()
        return {"ok": True, "project": self._project_payload()}

    def set_aspect_ratio(self, ratio: str):
        """"auto" (the default) exports at the first video clip's own
        resolution — matches "adjust to the first uploaded video". Any
        other value pins export to a fixed canvas (see
        ffmpeg_ops.ASPECT_RATIOS) with each clip letterboxed to fit."""
        if not self.project:
            return {"error": "No project loaded."}
        if ratio != "auto" and ratio not in ffmpeg_ops.ASPECT_RATIOS:
            return {"error": f"Unknown aspect ratio {ratio!r}."}
        self.project.aspect_ratio = ratio
        self._save_project()
        return {"ok": True, "aspect_ratio": ratio}

    def rename_project(self, name: str):
        if not self.project:
            return {"error": "No project loaded."}
        name = name.strip()
        if not name:
            return {"error": "Project name is required."}
        if name == self.project.name:
            return {"ok": True, "name": self.project.name}

        new_dir = SCENECRAFT_ROOT / name
        if new_dir.exists():
            return {"error": f"A project named {name!r} already exists."}

        old_dir = SCENECRAFT_ROOT / self.project.name
        if self.project_path and old_dir.is_dir():
            old_dir.rename(new_dir)
        self.project.name = name
        self.project_path = str(new_dir / "project.json")
        self._save_project()
        return {"ok": True, "name": self.project.name}

    def list_projects(self):
        """Names + paths of saved projects under ~/Scenecraft."""
        if not SCENECRAFT_ROOT.exists():
            return []
        projects = []
        for entry in sorted(SCENECRAFT_ROOT.iterdir()):
            project_file = entry / "project.json"
            if not project_file.exists():
                continue
            try:
                proj = project_state.load(str(project_file))
            except Exception:
                continue
            projects.append({"name": proj.name, "path": str(project_file)})
        return projects

    def open_project(self, path: str):
        try:
            project = project_state.load(path)
        except Exception as e:
            return {"error": str(e)}

        self.project = project
        self.project_path = path
        try:
            info = ffmpeg_ops.probe(project.source_video)
            playable = self._playable_path(project.source_video)
        except Exception:
            info = None
            playable = project.source_video
        return {"path": self._media_url(playable), "info": info, "project": self._project_payload()}

    def cut_scene(self, start: float, end: float):
        """Cuts [start, end) out of the source video into its own file and
        places it as a new clip on the video track, appended after
        whatever's already there."""
        if not self.project:
            return {"error": "No project loaded."}
        if not self.project.source_video:
            return {"error": "Import a video first."}
        if start < 0 or end <= start:
            return {"error": f"Invalid range: start must be >= 0 and end must be after start (got {start}-{end})."}
        try:
            source_duration = ffmpeg_ops.probe(self.project.source_video)["duration_seconds"]
        except Exception:
            source_duration = None
        if source_duration is not None and start >= source_duration:
            return {"error": f"Start ({start}s) is at or past the video's length ({source_duration:.1f}s)."}
        video_track = self.project.track("video")
        out_dir = SCENECRAFT_ROOT / self.project.name / "scenes"
        scene_id = f"scene_{len(video_track.clips) + 1}"
        out_path = str(out_dir / f"{scene_id}.mp4")
        self._progress("Cutting the clip")
        try:
            ffmpeg_ops.cut_clip(self.project.source_video, out_path, start, end)
        except Exception as e:
            return {"error": str(e)}

        timeline_start = max((c.end_time for c in video_track.clips), default=0.0)
        clip = project_state.Clip(
            id=scene_id,
            source_path=out_path,
            in_point=0.0,
            out_point=end - start,
            start_time=timeline_start,
        )
        self.project.add_clip(clip, track="video")
        self._save_project()
        return {"scene_id": scene_id, "path": out_path}

    def get_tracks(self):
        if not self.project:
            return []
        return self._tracks_payload()

    def move_clip(self, track_kind: str, clip_id: str, start_time: float):
        if not self.project:
            return {"error": "No project loaded."}
        track = self.project.track(track_kind)
        clip = next((c for c in track.clips if c.id == clip_id), None)
        if not clip:
            return {"error": f"Clip {clip_id!r} not found on track {track_kind!r}."}
        clip.start_time = max(0.0, start_time)
        track.clips.sort(key=lambda c: c.start_time)
        self._save_project()
        return {"ok": True, "start_time": clip.start_time}

    # ---- color correction / effects (real ffmpeg eq= + hue=, no AI) ----
    _EFFECT_BOUNDS = {
        "brightness": (-1.0, 1.0, 0.0),
        "contrast": (0.0, 3.0, 1.0),
        "saturation": (0.0, 3.0, 1.0),
    }

    def _find_clip(self, clip_id: str):
        if not self.project:
            return None
        for track in self.project.tracks:
            for clip in track.clips:
                if clip.id == clip_id:
                    return clip
        return None

    def set_clip_effect(self, clip_id: str, effect: str, value):
        if not self.project:
            return {"error": "No project loaded."}
        clip = self._find_clip(clip_id)
        if not clip:
            return {"error": f"Clip {clip_id!r} not found."}
        if effect == "grayscale":
            clip.effects["grayscale"] = bool(value)
        elif effect in self._EFFECT_BOUNDS:
            lo, hi, _ = self._EFFECT_BOUNDS[effect]
            clip.effects[effect] = max(lo, min(hi, float(value)))
        else:
            return {"error": f"Unknown effect {effect!r}."}
        self._save_project()
        return {"ok": True, "effects": clip.effects, "project": self._project_payload()}

    def reset_clip_effects(self, clip_id: str):
        if not self.project:
            return {"error": "No project loaded."}
        clip = self._find_clip(clip_id)
        if not clip:
            return {"error": f"Clip {clip_id!r} not found."}
        clip.effects = {}
        self._save_project()
        return {"ok": True, "project": self._project_payload()}

    def _target_video_clips(self, selected_clip_id: str | None):
        """Effect commands from the composer apply to the selected clip
        if there is one, otherwise to every video clip — useful since
        the common case is a single clip and selecting one first is
        extra friction for no benefit."""
        video_clips = self.project.track("video").clips
        if selected_clip_id:
            match = next((c for c in video_clips if c.id == selected_clip_id), None)
            if match:
                return [match]
        return video_clips

    def _adjust_effect(self, selected_clip_id: str | None, effect: str, delta: float) -> int:
        lo, hi, default = self._EFFECT_BOUNDS[effect]
        clips = self._target_video_clips(selected_clip_id)
        for c in clips:
            current = c.effects.get(effect, default)
            c.effects[effect] = max(lo, min(hi, current + delta))
        self._save_project()
        return len(clips)

    def transcribe(self):
        if not self.project:
            return {"error": "No project loaded."}
        if not self.project.source_video:
            return {"error": "Import a video first."}
        self._progress("Transcribing the audio")
        try:
            segments = whisper_ops.transcribe(self.project.source_video)
        except Exception as e:
            return {"error": str(e)}
        self.project.transcript = segments
        self._save_project()
        return {"transcript": segments}

    def transcribe_link(self, url: str):
        """Transcribe a video or audio URL without importing it first —
        paste a link into the composer and get its transcript back.

        The transcript replaces the project's current one (it's the thing
        search and caption generation read from), so this is transcribing
        *that* link, not merging it into an existing transcript."""
        if not self.project:
            return {"error": "No project loaded."}
        url = link_ops.find_url(url or "") or (url or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            return {"error": "That doesn't look like a link. Paste a full http(s) video or audio URL."}

        self._progress("Fetching the audio")
        try:
            path, title = link_ops.fetch_audio(url, SCENECRAFT_ROOT / self.project.name / "linked")
        except link_ops.LinkFetchError as e:
            return {"error": str(e)}

        self._progress("Transcribing the audio")
        try:
            segments = whisper_ops.transcribe(path)
        except Exception as e:
            return {"error": f"Downloaded it, but transcribing failed: {e}"}

        self.project.transcript = segments
        self._save_project()
        return {"ok": True, "title": title, "path": path, "transcript": segments}

    def search_transcript(self, query: str):
        if not self.project:
            return []
        if not query or not query.strip():
            return self.project.transcript
        return whisper_ops.search_transcript(self.project.transcript, query)

    def generate_captions(self):
        """Turns the transcript into caption clips on the text track —
        one per segment, positioned/timed to match, so export burns them
        in via the existing drawtext pipeline. Transcribes first if that
        hasn't run yet. Safely re-runnable: previously auto-generated
        captions (id prefix "caption_") are replaced, not duplicated.

        Known limitation: segment timestamps are relative to the original
        source recording, not the edited timeline. This lines up
        correctly for a source that hasn't been cut/rearranged yet;
        captions generated after cutting will be out of sync with the
        cut clips' new positions, since a cut clip doesn't carry where it
        came from in the original — regenerate captions before cutting,
        not after, until that's tracked."""
        if not self.project:
            return {"error": "No project loaded."}
        if not self.project.transcript:
            if not self.project.source_video:
                return {"error": "Import a video first."}
            try:
                self.project.transcript = whisper_ops.transcribe(self.project.source_video)
            except Exception as e:
                return {"error": str(e)}

        text_track = self.project.track("text")
        text_track.clips = [c for c in text_track.clips if not c.id.startswith("caption_")]
        for i, seg in enumerate(self.project.transcript):
            text_track.clips.append(project_state.Clip(
                id=f"caption_{i + 1}",
                source_path="",
                in_point=0.0,
                out_point=seg["end"] - seg["start"],
                start_time=seg["start"],
                label=seg["text"],
            ))
        text_track.clips.sort(key=lambda c: c.start_time)

        self._save_project()
        return {"ok": True, "count": len(self.project.transcript), "project": self._project_payload()}

    def export_project(self):
        if not self.project:
            return {"error": "No project loaded."}
        result = webview.windows[0].create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=f"{self.project.name}_export.mp4",
            file_types=("MP4 video (*.mp4)",),
        )
        if not result:
            return None
        out_path = result[0] if isinstance(result, (list, tuple)) else result
        self._progress("Rendering the final video")
        try:
            ffmpeg_ops.export(self.project, out_path)
        except Exception as e:
            return {"error": str(e)}
        return {"path": out_path}

    # ---- motion graphics: local AI writes the animation, Playwright renders it ----
    _MOTION_GRAPHIC_SYSTEM_PROMPT = (
        "You write a single self-contained HTML document (inline <style> and <script>, "
        "no external resources) that renders a CSS/JS animation matching the user's "
        "description. It must start playing automatically on page load, with no click "
        "or other user interaction required. Reply with ONLY the raw HTML — no markdown "
        "code fences, no explanation before or after it."
    )

    def generate_motion_graphic(self, description: str, duration: float = 3.0):
        """The one genuinely AI-generated feature in this app: a local
        Ollama model writes an HTML/CSS/JS animation for the given
        description, a headless browser (Playwright) plays it back and
        captures frames, ffmpeg encodes those into a clip that lands on
        the video track like anything else. Slow on modest hardware —
        the model alone can take tens of seconds — and a small local
        code model's output quality is well below what a large hosted
        model or a real generative-video API would produce; this is
        genuinely offline and free, not a substitute for those."""
        if not self.project:
            return {"error": "No project loaded."}
        if not description or not description.strip():
            return {"error": "Describe what you want the motion graphic to show."}
        if not llm_ops.is_available():
            return {"error": "Local AI (Ollama) isn't running. Install/start Ollama to generate motion graphics."}

        self._progress("Designing the animation")
        try:
            html = llm_ops.generate(description.strip(), system=self._MOTION_GRAPHIC_SYSTEM_PROMPT, model=self._active_model())
        except llm_ops.OllamaUnavailableError as e:
            return {"error": str(e)}
        html = _strip_markdown_fences(html)
        if "<html" not in html.lower():
            return {"error": "The local AI didn't return usable HTML for that description — try rephrasing, or a simpler description."}

        out_dir = SCENECRAFT_ROOT / self.project.name / "motion_graphics"
        out_path = str(out_dir / f"motiongraphic_{uuid.uuid4().hex[:8]}.mp4")
        self._progress("Rendering frames")
        try:
            motion_graphics_ops.render_html_animation(html, out_path, duration=duration)
        except Exception as e:
            return {"error": f"Failed to render the animation: {e}"}

        video_track = self.project.track("video")
        timeline_start = max((c.end_time for c in video_track.clips), default=0.0)
        clip = project_state.Clip(
            id=f"motiongraphic_{len(video_track.clips) + 1}",
            source_path=out_path,
            in_point=0.0,
            out_point=duration,
            start_time=timeline_start,
            label=description.strip()[:60],
        )
        self.project.add_clip(clip, track="video")
        self._save_project()
        try:
            playable = self._playable_path(out_path)
        except Exception:
            playable = out_path
        return {"ok": True, "path": self._media_url(playable), "project": self._project_payload()}

    # ---- composer: local, offline pattern-matched commands ----
    # No network, no API key. Recognizes a fixed set of phrasings and maps
    # them onto the same Api methods the UI buttons already call — this
    # is not a general NLU/LLM, so unrecognized phrasing (e.g. "change the
    # color", "add effects" — there's no color/effects engine at all yet)
    # returns a clear "not supported" message instead of guessing.
    def run_command(self, text: str, current_time: float = 0.0, selected_clip_id: str | None = None):
        if not self.project:
            return {"error": "No project loaded."}
        raw = (text or "").strip()
        if not raw:
            return {"error": "Type a command first."}
        t = raw.lower()

        # A link is unambiguous, and checked first so a URL that happens to
        # contain a word like "export" can't be read as a command.
        linked_url = link_ops.find_url(raw)
        if linked_url:
            result = self.transcribe_link(linked_url)
            if "error" in result:
                return {"action": "transcribe", "message": result["error"], "result": result}
            n = len(result["transcript"])
            return {
                "action": "transcribe",
                "message": f"Transcribed “{result['title']}” — {n} segments. Open Transcript to read or search it.",
                "result": result,
            }

        def num(s: str) -> float:
            return float(s)

        # "cut the first N seconds"
        m = re.search(r"\bcut\s+(?:the\s+)?first\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?\b", t)
        if m:
            n = num(m.group(1))
            result = self.cut_scene(0.0, n)
            return {"action": "cut", "message": f"Cut the first {n:g}s." if "error" not in result else result["error"], "result": result}

        # "cut the next N seconds" — relative to the current playhead
        m = re.search(r"\bcut\s+(?:the\s+)?next\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?\b", t)
        if m:
            n = num(m.group(1))
            result = self.cut_scene(current_time, current_time + n)
            return {"action": "cut", "message": f"Cut {n:g}s from {current_time:g}s." if "error" not in result else result["error"], "result": result}

        # "cut from A to B" / "cut A to B" / "cut A-B"
        m = re.search(r"\bcut\s+(?:from\s+)?(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?\s*(?:to|-|–)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?\b", t)
        if m:
            a, b = num(m.group(1)), num(m.group(2))
            result = self.cut_scene(a, b)
            return {"action": "cut", "message": f"Cut {a:g}s to {b:g}s." if "error" not in result else result["error"], "result": result}

        # "cut at N (seconds)" — a single point only tells us where to
        # start; cut N seconds from there, same as "cut the next N seconds"
        m = re.search(r"\bcut\s+at\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?\b", t)
        if m:
            n = num(m.group(1))
            result = self.cut_scene(current_time, current_time + n)
            return {"action": "cut", "message": f"Cut {n:g}s from {current_time:g}s." if "error" not in result else result["error"], "result": result}

        # "add music" / "add audio" / "add a song" [here]
        m = re.search(r"\badd\s+(?:a\s+|some\s+)?(?:music|audio|song)(?:\s+track)?\b(.*)$", t)
        if m:
            here = "here" in m.group(1)
            result = self.add_audio_clip(current_time if here else None)
            if result is None:
                return {"action": "add_audio", "message": "Cancelled — no audio file selected.", "result": None}
            return {"action": "add_audio", "message": "Added audio clip." if "error" not in result else result["error"], "result": result}

        # "go to 1:23" (mm:ss) — checked before the plain-seconds form below
        m = re.search(r"\b(?:go to|take me to|jump to|seek to)\s+(\d+):(\d+)\b", t)
        if m:
            target = int(m.group(1)) * 60 + int(m.group(2))
            return {"action": "seek", "message": f"Jumped to {m.group(1)}:{m.group(2)}.", "result": {"time": target}}

        # "go to 90 seconds" / "take me to 45" / "jump to 12s"
        m = re.search(r"\b(?:go to|take me to|jump to|seek to)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?\b", t)
        if m:
            target = num(m.group(1))
            return {"action": "seek", "message": f"Jumped to {target:g}s.", "result": {"time": target}}

        # color/effects — targets the selected clip if there is one,
        # otherwise every video clip
        if re.search(r"\b(?:brighter|brighten|increase brightness)\b", t):
            n = self._adjust_effect(selected_clip_id, "brightness", 0.15)
            return {"action": "effect", "message": f"Increased brightness on {n} clip(s).", "result": {"project": self._project_payload()}}
        if re.search(r"\b(?:darker|darken|decrease brightness)\b", t):
            n = self._adjust_effect(selected_clip_id, "brightness", -0.15)
            return {"action": "effect", "message": f"Decreased brightness on {n} clip(s).", "result": {"project": self._project_payload()}}
        if re.search(r"\b(?:more contrast|increase contrast)\b", t):
            n = self._adjust_effect(selected_clip_id, "contrast", 0.2)
            return {"action": "effect", "message": f"Increased contrast on {n} clip(s).", "result": {"project": self._project_payload()}}
        if re.search(r"\b(?:less contrast|decrease contrast|flatten)\b", t):
            n = self._adjust_effect(selected_clip_id, "contrast", -0.2)
            return {"action": "effect", "message": f"Decreased contrast on {n} clip(s).", "result": {"project": self._project_payload()}}
        if re.search(r"\b(?:more saturation|increase saturation|more vibrant|saturate)\b", t):
            n = self._adjust_effect(selected_clip_id, "saturation", 0.3)
            return {"action": "effect", "message": f"Increased saturation on {n} clip(s).", "result": {"project": self._project_payload()}}
        if re.search(r"\b(?:less saturation|decrease saturation|desaturate)\b", t):
            n = self._adjust_effect(selected_clip_id, "saturation", -0.3)
            return {"action": "effect", "message": f"Decreased saturation on {n} clip(s).", "result": {"project": self._project_payload()}}
        if re.search(r"\b(?:black and white|grayscale|greyscale)\b", t):
            clips = self._target_video_clips(selected_clip_id)
            for c in clips:
                c.effects["grayscale"] = True
            self._save_project()
            return {"action": "effect", "message": f"Made {len(clips)} clip(s) black and white.", "result": {"project": self._project_payload()}}
        if re.search(r"\b(?:reset (?:effects|color)|remove effects|clear effects)\b", t):
            clips = self._target_video_clips(selected_clip_id)
            for c in clips:
                c.effects = {}
            self._save_project()
            return {"action": "effect", "message": f"Reset effects on {len(clips)} clip(s).", "result": {"project": self._project_payload()}}

        # "add captions" / "add subtitles" / "generate captions"
        if re.search(r"\b(?:add|generate)\s+(?:captions|subtitles)\b", t):
            result = self.generate_captions()
            return {"action": "captions", "message": f"Added {result.get('count', 0)} caption(s)." if "error" not in result else result["error"], "result": result}

        # "import a video" / "add a video"
        if re.search(r"\bimport\b", t) or re.search(r"\badd\s+(?:a\s+)?video\b", t):
            result = self.import_video()
            if result is None:
                return {"action": "import", "message": "Cancelled — no video selected.", "result": None}
            return {"action": "import", "message": "Imported video." if "error" not in result else result["error"], "result": result}

        # "transcribe" / "generate a transcript"
        if re.search(r"\btranscribe\b", t) or re.search(r"\bgenerate\s+(?:a\s+)?transcript\b", t):
            result = self.transcribe()
            return {"action": "transcribe", "message": "Transcribed." if "error" not in result else result["error"], "result": result}

        # "write a script about X" / "write me a script for X" / "create a script about X"
        m = re.search(r"\bwrite\b(?:\s+me)?\s+a\s+script\s+(?:about|for|on)\s+(.+)$", t) or \
            re.search(r"\b(?:create|generate)\s+a\s+script\s+(?:about|for|on)\s+(.+)$", t)
        if m:
            topic = raw[m.start(1):m.end(1)]
            result = self.write_script(topic)
            if "error" in result:
                return {"action": "write_script", "message": result["error"], "result": result}
            n = len(result["script"])
            return {"action": "write_script", "message": f"Wrote a {n}-scene script about {topic!r}. Open the Script panel to review it.", "result": result}

        # "search for X" / "find X" / "find where I said X"
        m = re.search(r"\b(?:search|find)\b(?:\s+for)?\s+(?:where\s+i\s+said\s+)?(.+)$", t)
        if m:
            query = m.group(1).strip(" \"'")
            matches = self.search_transcript(query)
            return {"action": "search", "message": f"Found {len(matches)} match(es) for {query!r}.", "result": {"query": query, "matches": matches}}

        # "export"
        if re.search(r"\bexport\b", t):
            result = self.export_project()
            if result is None:
                return {"action": "export", "message": "Cancelled — no save location chosen.", "result": None}
            return {"action": "export", "message": f"Exported to {result.get('path')}." if "error" not in result else result["error"], "result": result}

        # "generate a motion graphic of X" / "create an animation showing X"
        m = re.search(r"\b(?:generate|create|make)\s+(?:a\s+)?(?:motion graphic|animation)\s+(?:of|showing|with)\s+(.+)$", t)
        if m:
            description = raw[m.start(1):m.end(1)]
            result = self.generate_motion_graphic(description)
            return {"action": "motion_graphic", "message": "Generated motion graphic." if "error" not in result else result["error"], "result": result}

        # Local AI (Ollama) fallback — only reached if nothing above
        # matched. Silently skipped if Ollama isn't running; the
        # deterministic patterns above always run first and are instant,
        # this never blocks or slows down the common case.
        if llm_ops.is_available():
            try:
                interpretation = llm_ops.interpret_command(raw, model=self._active_model())
            except llm_ops.OllamaUnavailableError:
                interpretation = {"action": "unknown"}
            action = interpretation.get("action")
            if action == "cut" and "start" in interpretation and "end" in interpretation:
                result = self.cut_scene(float(interpretation["start"]), float(interpretation["end"]))
                return {"action": "cut", "message": "Cut it (via local AI)." if "error" not in result else result["error"], "result": result}
            if action == "seek" and "time" in interpretation:
                return {"action": "seek", "message": "Jumped there (via local AI).", "result": {"time": float(interpretation["time"])}}
            if action == "effect" and interpretation.get("effect") == "grayscale":
                clips = self._target_video_clips(selected_clip_id)
                for c in clips:
                    c.effects["grayscale"] = True
                self._save_project()
                return {"action": "effect", "message": f"Made {len(clips)} clip(s) black and white (via local AI).", "result": {"project": self._project_payload()}}
            if action == "effect" and interpretation.get("effect") in self._EFFECT_BOUNDS and "delta" in interpretation:
                n = self._adjust_effect(selected_clip_id, interpretation["effect"], float(interpretation["delta"]))
                return {"action": "effect", "message": f"Adjusted {interpretation['effect']} (via local AI) on {n} clip(s).", "result": {"project": self._project_payload()}}

        return {
            "error": (
                f"Didn't recognize that command: {raw!r}. This composer only understands a fixed set of "
                "phrasings (no AI interpretation, no network) — try things like \"cut the first 5 seconds\", "
                "\"cut at 3 seconds\", \"add music\", \"add captions\", \"go to 1:23\", \"brighter\"/\"darker\", "
                "\"more contrast\"/\"less contrast\", \"more saturation\"/\"less saturation\", "
                "\"black and white\", \"reset effects\", \"generate a motion graphic of X\", "
                "\"transcribe\", a pasted video/audio link to transcribe, \"search for fox\", or \"export\". If a local AI (Ollama) is "
                "running, phrasings close to these are also understood even if not word-for-word."
            )
        }

    # ---- conversation: the Director talks, it doesn't just parse ----
    #
    # run_command above is a fixed phrasebook — great when you know the
    # phrasing, useless for "hello" or a half-formed idea. chat() wraps it:
    # exact commands still hit the instant offline path, anything else gets
    # a real reply, and requests that are missing details become questions
    # instead of errors.

    _CHAT_SYSTEM_PROMPT = (
        "You are the assistant inside a video editing app. Reply with ONLY a JSON "
        "object — no prose, no markdown fences.\n\n"
        'Shape: {"intent": <intent>, "reply": <one or two short friendly sentences>}\n\n'
        "Intents:\n"
        '- "chat" — greetings, thanks, small talk, or a general question. Put your '
        'actual conversational answer in "reply".\n'
        '- "motion_graphic" — they want an animation or motion graphic made.\n'
        '- "script" — they want a script written.\n'
        '- "captions" — they want captions or subtitles on the video.\n'
        '- "transcribe" — they want the speech transcribed to text.\n'
        '- "export" — they want to export, render or save out the video.\n'
        '- "unknown" — anything else.\n\n'
        "Examples:\n"
        '"hello" -> {"intent": "chat", "reply": "Hey! What are we making today?"}\n'
        '"how are you?" -> {"intent": "chat", "reply": "Doing well and ready to edit. What do you need?"}\n'
        '"what can you do" -> {"intent": "chat", "reply": "I can write scripts, cut clips, add captions, build motion graphics and export."}\n'
        '"thanks" -> {"intent": "chat", "reply": "Anytime."}\n'
        '"generate a motion graphic" -> {"intent": "motion_graphic", "reply": "Sure."}\n'
        '"make me an animation" -> {"intent": "motion_graphic", "reply": "Sure."}\n'
        '"write me a script" -> {"intent": "script", "reply": "Happy to."}\n'
        '"put subtitles on this" -> {"intent": "captions", "reply": "On it."}\n'
    )

    # Requests that need details we don't have yet become a short interview
    # rather than an error. Each entry is the questions to ask, in order.
    _FLOWS = {
        "motion_graphic": [
            ("description", "What should the motion graphic show?"),
            ("style", "Any particular style or colours? (say \"skip\" for a clean default)"),
        ],
        "script": [
            ("topic", "What should the script be about?"),
        ],
    }

    _CANCEL_WORDS = {"cancel", "never mind", "nevermind", "stop", "forget it", "no"}
    _SKIP_WORDS = {"skip", "none", "no preference", "whatever", "any", "you choose", "up to you"}

    def chat(self, message: str, current_time: float = 0.0, selected_clip_id: str | None = None):
        """Single entry point for the Director panel."""
        text = (message or "").strip()
        if not text:
            return {"action": "chat", "message": ""}
        if not self.project:
            return {"action": "chat", "error": "No project loaded."}

        # 1. A question is outstanding — this message answers it.
        if self.pending_flow:
            return self._continue_flow(text)

        # 2. Exact commands keep the fast, deterministic, offline path.
        result = self.run_command(text, current_time, selected_clip_id)
        if "error" not in result:
            return result

        # 3. Nothing matched — actually talk about it.
        return self._converse(text, result["error"])

    def _converse(self, text: str, fallback_error: str):
        if not llm_ops.is_available():
            return {"action": "chat", "error": fallback_error}
        self._progress("Thinking")
        try:
            raw = llm_ops.generate(text, system=self._CHAT_SYSTEM_PROMPT, model=self._active_model())
        except llm_ops.OllamaUnavailableError:
            return {"action": "chat", "error": fallback_error}

        parsed = llm_ops._extract_json(raw) or {}
        intent = str(parsed.get("intent") or "unknown").strip()
        reply = str(parsed.get("reply") or "").strip()

        if intent in self._FLOWS:
            return self._start_flow(intent, reply)
        # Intents that map straight onto an existing command need no details.
        passthrough = {"captions": "add captions", "transcribe": "transcribe", "export": "export"}
        if intent in passthrough:
            return self.run_command(passthrough[intent], 0.0, None)
        if intent == "chat" and reply:
            return {"action": "chat", "message": reply}
        return {"action": "chat", "error": fallback_error}

    def _start_flow(self, name: str, preamble: str = ""):
        self.pending_flow = {"name": name, "slots": {}, "index": 0}
        slot, question = self._FLOWS[name][0]
        # the model's own "Sure." reads as filler in front of a question
        return {"action": "chat", "message": question, "awaiting": slot}

    def _continue_flow(self, text: str):
        flow = self.pending_flow
        questions = self._FLOWS[flow["name"]]
        slot, _ = questions[flow["index"]]

        if text.strip().lower() in self._CANCEL_WORDS:
            self.pending_flow = None
            return {"action": "chat", "message": "No problem — dropped it. What else?"}

        flow["slots"][slot] = "" if text.strip().lower() in self._SKIP_WORDS else text.strip()
        flow["index"] += 1

        if flow["index"] < len(questions):
            next_slot, next_question = questions[flow["index"]]
            return {"action": "chat", "message": next_question, "awaiting": next_slot}

        slots = flow["slots"]
        self.pending_flow = None
        return self._run_flow(flow["name"], slots)

    def _run_flow(self, name: str, slots: dict):
        if name == "motion_graphic":
            description = slots.get("description", "")
            style = slots.get("style", "")
            if style:
                description = f"{description}, in this style: {style}"
            result = self.generate_motion_graphic(description)
            message = ("Made it — it's on the timeline."
                       if "error" not in result else result["error"])
            return {"action": "motion_graphic", "message": message, "result": result}

        if name == "script":
            result = self.write_script(slots.get("topic", ""))
            if "error" in result:
                return {"action": "write_script", "message": result["error"], "result": result}
            n = len(result["script"])
            return {
                "action": "write_script",
                "message": f"Wrote a {n}-scene script. Open Script & record to shoot it.",
                "result": result,
            }

        return {"action": "chat", "error": f"Don't know how to finish {name!r}."}

    # ---- settings: local-only storage, e.g. for a future API key ----
    def get_settings(self):
        if not SETTINGS_PATH.exists():
            return {}
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def set_api_key(self, api_key: str):
        """Stores an API key locally in ~/Scenecraft/settings.json. Nothing
        in the app currently sends this anywhere — every AI feature here
        (composer fallback, script writing, motion graphics) runs on the
        local model selected via set_local_model, not this key."""
        settings = self.get_settings()
        settings["api_key"] = api_key.strip()
        SCENECRAFT_ROOT.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
        return {"ok": True}

    def _active_model(self) -> str:
        return self.get_settings().get("local_model") or llm_ops.DEFAULT_MODEL

    def set_local_model(self, model_id: str):
        """Persists which local Ollama model the composer fallback, script
        writing, and motion-graphics generation should use. Doesn't check
        that it's actually installed — a bad choice just surfaces as a
        normal Ollama error on the next call, same as any other model."""
        settings = self.get_settings()
        settings["local_model"] = model_id
        SCENECRAFT_ROOT.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
        return {"ok": True}

    def get_model_status(self):
        """Everything the UI needs to show which AI model is actually
        active right now and whether it's local (it's always local —
        this app makes no cloud AI calls) and actually installed, rather
        than assuming the composer's AI fallback silently works."""
        available = llm_ops.is_available()
        installed = llm_ops.list_models() if available else []
        active = self._active_model()
        return {
            "is_local": True,
            "ollama_available": available,
            "active_model": active,
            "active_model_installed": active in installed,
            "installed_models": installed,
            "known_models": AVAILABLE_LOCAL_MODELS,
        }

    # ---- guided recording: script -> scenes -> teleprompter -> capture ----
    def set_script(self, raw_text: str):
        """Parses a pasted script into scenes (deterministic text parsing,
        see core_engine/script_ops.py — no AI). Replaces any existing
        script; already-recorded takes for scenes that still exist by id
        would be preserved by a real diff, but a fresh parse always
        produces fresh scene ids, so re-pasting a script starts recording
        over. That's intentional: editing the script mid-shoot should be
        a deliberate reset, not a silent partial merge."""
        if not self.project:
            return {"error": "No project loaded."}
        scenes = script_ops.parse_script(raw_text)
        if not scenes:
            return {"error": "Couldn't find any scenes in that script. Separate scenes with a blank line, or label them 'Scene 1:', 'Scene 2:', etc."}
        self.project.script = [project_state.ScriptScene(**s) for s in scenes]
        self._save_project()
        return {"ok": True, "script": [vars(s) for s in self.project.script]}

    _SCRIPT_WRITER_SYSTEM_PROMPT = (
        "You write a short video script for someone to record themselves reading "
        "on camera. Format it EXACTLY like this, with a blank line between scenes:\n\n"
        "Scene 1:\n[smile warmly, speak slowly]\nWhat to say, as natural spoken sentences.\n\n"
        "Scene 2:\n[lean in, sound excited]\nMore spoken lines.\n\n"
        "Every scene needs its own bracketed direction describing HOW to deliver that "
        "specific scene's lines (tone, pace, gesture, expression) — never reuse the same "
        "words 'direction' or 'acting' as the direction itself, and never repeat the exact "
        "same bracketed text across two scenes. Write 3 to 6 scenes. Keep each scene's "
        "spoken lines short — a sentence or two, the length someone can actually say in one "
        "take. Reply with ONLY the script in that format, no title, no explanation before or "
        "after it."
    )

    def write_script(self, topic: str):
        """The local model writes a script for the given topic/description
        (not the whisper transcript's script_ops parser — this generates
        new spoken lines from scratch), formatted for set_script() to
        parse straight into scenes ready for guided recording."""
        if not self.project:
            return {"error": "No project loaded."}
        if not topic or not topic.strip():
            return {"error": "Say what the script should be about."}
        if not llm_ops.is_available():
            return {"error": "Local AI (Ollama) isn't running. Install/start Ollama to write a script."}
        self._progress("Writing the script")
        try:
            raw = llm_ops.generate(topic.strip(), system=self._SCRIPT_WRITER_SYSTEM_PROMPT, model=self._active_model())
        except llm_ops.OllamaUnavailableError as e:
            return {"error": str(e)}
        raw = _strip_markdown_fences(raw)
        return self.set_script(raw)

    def create_upload_slot(self, scene_id: str):
        """Registers a one-time destination for the next recording
        upload and returns the URL to POST it to. See _UPLOAD_TOKENS."""
        if not self.project:
            return {"error": "No project loaded."}
        token = uuid.uuid4().hex
        dest_path = str(SCENECRAFT_ROOT / self.project.name / "recordings" / f"{scene_id}_{token[:8]}.webm")
        _UPLOAD_TOKENS[token] = dest_path
        return {"upload_url": f"http://127.0.0.1:{self.media_port}/upload?token={token}"}

    def finish_scene_recording(self, scene_id: str, file_path: str):
        """Called once the browser has finished POSTing a take to the
        upload slot above. Marks the scene recorded and places the
        capture on the video track (full length, appended), immediately
        editable the same as any imported clip."""
        if not self.project:
            return {"error": "No project loaded."}
        try:
            info = ffmpeg_ops.probe(file_path)
        except Exception as e:
            return {"error": str(e)}

        for scene in self.project.script:
            if scene.id == scene_id:
                scene.recorded_path = file_path
                break

        video_track = self.project.track("video")
        timeline_start = max((c.end_time for c in video_track.clips), default=0.0)
        clip = project_state.Clip(
            id=f"rec_{scene_id}",
            source_path=file_path,
            in_point=0.0,
            out_point=info["duration_seconds"],
            start_time=timeline_start,
        )
        self.project.add_clip(clip, track="video")
        self._save_project()

        try:
            playable = self._playable_path(file_path)
        except Exception:
            playable = file_path
        return {"ok": True, "preview_path": self._media_url(playable), "project": self._project_payload()}


def main():
    media_port = _start_media_server()
    api = Api(media_port=media_port)
    ui_path = Path(__file__).resolve().parent / "ui" / "index.html"
    webview.create_window(
        "Scenecraft", str(ui_path), js_api=api, width=1100, height=720, min_size=(800, 560)
    )
    webview.start(debug=os.environ.get("SCENECRAFT_DEBUG") == "1")


if __name__ == "__main__":
    main()
