"""
Scenecraft — offline desktop app.
This is the ONLY place that touches pywebview. It exposes core_engine
functions to the UI as a JS-callable API. The UI never talks to ffmpeg
directly — it always goes through this Api class.
"""

import sys
import os
import http.server
import socketserver
import threading
import urllib.parse
import mimetypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import webview
from core_engine import ffmpeg_ops, project_state, whisper_ops

SCENECRAFT_ROOT = Path.home() / "Scenecraft"

# WebView2 on Windows won't reliably play file:// video (no seeking, often
# no rendering at all). Instead we serve video files ourselves over plain
# HTTP with Range support, and point <video src> at that. Only paths this
# app has explicitly registered (via Api._media_url) are servable — this
# is a local, no-auth server, so it must not become an arbitrary local
# file server just because it's bound to 127.0.0.1.
_ALLOWED_MEDIA_PATHS: set[str] = set()


class _RangeRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # keep stdout clean; errors still show via send_error

    def do_HEAD(self):
        self._serve(send_body=False)

    def do_GET(self):
        self._serve(send_body=True)

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

    def _media_url(self, path: str | None) -> str | None:
        if not path:
            return None
        _ALLOWED_MEDIA_PATHS.add(path)
        return f"http://127.0.0.1:{self.media_port}/media?path={urllib.parse.quote(path, safe='')}"

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
        """Opens the video picker and adds the result to the video track
        (full length, at the end of whatever's already there), and makes
        it the project's active source for cut/transcribe."""
        if not self.project:
            return {"error": "No project loaded."}
        result = webview.windows[0].create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("Video files (*.mp4;*.mov;*.mkv;*.avi)", "All files (*.*)"),
        )
        if not result:
            return None
        source = result[0]
        try:
            info = ffmpeg_ops.probe(source)
        except Exception as e:
            return {"error": str(e)}

        self.project.source_video = source
        video_track = self.project.track("video")
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
        return {"path": self._media_url(source), "info": info, "project": self._project_payload()}

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
        except Exception:
            info = None
        return {"path": self._media_url(project.source_video), "info": info, "project": self._project_payload()}

    def cut_scene(self, start: float, end: float):
        """Cuts [start, end) out of the source video into its own file and
        places it as a new clip on the video track, appended after
        whatever's already there."""
        if not self.project:
            return {"error": "No project loaded."}
        if not self.project.source_video:
            return {"error": "Import a video first."}
        video_track = self.project.track("video")
        out_dir = SCENECRAFT_ROOT / self.project.name / "scenes"
        scene_id = f"scene_{len(video_track.clips) + 1}"
        out_path = str(out_dir / f"{scene_id}.mp4")
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

    def transcribe(self):
        if not self.project:
            return {"error": "No project loaded."}
        if not self.project.source_video:
            return {"error": "Import a video first."}
        try:
            segments = whisper_ops.transcribe(self.project.source_video)
        except Exception as e:
            return {"error": str(e)}
        self.project.transcript = segments
        self._save_project()
        return {"transcript": segments}

    def search_transcript(self, query: str):
        if not self.project:
            return []
        if not query or not query.strip():
            return self.project.transcript
        return whisper_ops.search_transcript(self.project.transcript, query)

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
        try:
            ffmpeg_ops.export(self.project, out_path)
        except Exception as e:
            return {"error": str(e)}
        return {"path": out_path}


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
