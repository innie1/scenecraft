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


class Api:
    def __init__(self):
        self.project: project_state.Project | None = None

    def pick_video_file(self):
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
        info = ffmpeg_ops.probe(source)
        return {"path": source, "info": info}

    def cut_scene(self, start: float, end: float):
        if not self.project:
            return {"error": "No project loaded. Pick a video first."}
        out_dir = Path.home() / "Scenecraft" / self.project.name / "scenes"
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
