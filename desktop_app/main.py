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
from core_engine import ffmpeg_ops, project_state

SCENECRAFT_ROOT = Path.home() / "Scenecraft"


class Api:
    def __init__(self):
        self.project: project_state.Project | None = None
        self.project_path: str | None = None

    def _project_payload(self):
        if not self.project:
            return None
        return {
            "name": self.project.name,
            "source_video": self.project.source_video,
            "scenes": [vars(s) for s in self.project.scenes],
        }

    def _save_project(self):
        if self.project and self.project_path:
            project_state.save(self.project, self.project_path)

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
        if not self.project:
            return {"error": "No project loaded. Pick a video first."}
        out_dir = SCENECRAFT_ROOT / self.project.name / "scenes"
        scene_id = f"scene_{len(self.project.scenes) + 1}"
        out_path = str(out_dir / f"{scene_id}.mp4")
        try:
            ffmpeg_ops.cut_clip(self.project.source_video, out_path, start, end)
        except Exception as e:
            return {"error": str(e)}
        scene = project_state.Scene(
            id=scene_id, source_path=out_path, start=start, end=end
        )
        self.project.add_scene(scene)
        self._save_project()
        return {"scene_id": scene_id, "path": out_path}

    def get_scenes(self):
        if not self.project:
            return []
        return [vars(s) for s in self.project.scenes]


def main():
    api = Api()
    ui_path = Path(__file__).resolve().parent / "ui" / "index.html"
    webview.create_window(
        "Scenecraft", str(ui_path), js_api=api, width=1100, height=720, min_size=(800, 560)
    )
    webview.start(debug=os.environ.get("SCENECRAFT_DEBUG") == "1")


if __name__ == "__main__":
    main()
