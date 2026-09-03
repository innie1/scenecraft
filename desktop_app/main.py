"""
Scenecraft — offline desktop app.
This is the ONLY place that touches pywebview. It exposes core_engine
functions to the UI as a JS-callable API. The UI never talks to ffmpeg
directly — it always goes through this Api class.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import webview
from core_engine import ffmpeg_ops, project_state, whisper_ops

SCENECRAFT_ROOT = Path.home() / "Scenecraft"


class Api:
    def __init__(self):
        self.project: project_state.Project | None = None
        self.project_path: str | None = None

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

    def pick_video_file(self):
        """Quick, unnamed open — used by the plain 'Open video' entry point.
        Not persisted to disk; use start_project() to create a saved project."""
        result = webview.windows[0].create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("Video files (*.mp4;*.mov;*.mkv;*.avi)", "All files (*.*)"),
        )
        if not result:
            return None
        source = result[0]
        self.project = project_state.Project(
            name=Path(source).stem, source_video=source
        )
        self.project_path = None
        info = ffmpeg_ops.probe(source)
        return {"path": source, "info": info, "project": self._project_payload()}

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

    def start_project(self, name: str):
        """Prompt-driven flow: name is collected by the UI first, then this
        opens the video picker, creates the project, and saves it immediately."""
        if not name or not name.strip():
            return {"error": "Project name is required."}
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

        project = project_state.Project(name=name.strip(), source_video=source)
        path = str(SCENECRAFT_ROOT / project.name / "project.json")
        project_state.save(project, path)

        self.project = project
        self.project_path = path
        return {"path": source, "info": info, "project": self._project_payload()}

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
        return {"path": project.source_video, "info": info, "project": self._project_payload()}

    def cut_scene(self, start: float, end: float):
        """Cuts [start, end) out of the source video into its own file and
        places it as a new clip on the video track, appended after
        whatever's already there."""
        if not self.project:
            return {"error": "No project loaded. Pick a video first."}
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
            return {"error": "No project loaded. Pick a video first."}
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
    api = Api()
    ui_path = Path(__file__).resolve().parent / "ui" / "index.html"
    webview.create_window(
        "Scenecraft", str(ui_path), js_api=api, width=1100, height=720, min_size=(800, 560)
    )
    webview.start(debug=os.environ.get("SCENECRAFT_DEBUG") == "1")


if __name__ == "__main__":
    main()
