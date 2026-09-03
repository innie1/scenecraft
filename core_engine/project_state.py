"""
Project state — the scene list and timeline for one editing project.
Plain JSON on disk. No AI, no UI. This is the "memory" of a single
project that both the desktop app and (later) the MCP server read/write.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict, field


@dataclass
class Scene:
    id: str
    source_path: str
    start: float
    end: float
    label: str = ""


@dataclass
class Project:
    name: str
    source_video: str
    scenes: list[Scene] = field(default_factory=list)

    def add_scene(self, scene: Scene) -> None:
        self.scenes.append(scene)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        scenes = [Scene(**s) for s in d.get("scenes", [])]
        return cls(name=d["name"], source_video=d["source_video"], scenes=scenes)


def save(project: Project, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(project.to_dict(), f, indent=2)


def load(path: str) -> Project:
    with open(path) as f:
        return Project.from_dict(json.load(f))
