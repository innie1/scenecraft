"""
Project state — the multi-track timeline for one editing project.
Plain JSON on disk. No AI, no UI. This is the "memory" of a single
project that both the desktop app and (later) the MCP server read/write.

A Project has a fixed set of named tracks (video, audio, text). Each
track holds an ordered list of Clips. A Clip references a media file
(source_path), an in/out trim range within that file, and a start_time
that positions it on the project's shared timeline — tracks are real
lanes, not just a flat ordered list.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict, field

SCHEMA_VERSION = 2

TRACK_KINDS = ("video", "audio", "text")


@dataclass
class Clip:
    id: str
    source_path: str
    in_point: float
    out_point: float
    start_time: float
    label: str = ""
    # Color adjustments applied at export (see ffmpeg_ops.export). Keys:
    # "brightness" (-1..1), "contrast" (0..3), "saturation" (0..3), each
    # relative to ffmpeg's eq= neutral values (0, 1, 1); "grayscale" (bool).
    # Missing keys mean untouched, not zero — an empty dict is a no-op.
    effects: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.out_point - self.in_point

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


@dataclass
class Track:
    kind: str
    clips: list[Clip] = field(default_factory=list)

    def add_clip(self, clip: Clip) -> None:
        self.clips.append(clip)
        self.clips.sort(key=lambda c: c.start_time)


def _default_tracks() -> list[Track]:
    return [Track(kind=kind) for kind in TRACK_KINDS]


@dataclass
class ScriptScene:
    """One scene in the guided-recording script: what to say, optional
    acting/direction notes, and the recording once captured (None until
    then — recording can span multiple sessions/days)."""
    id: str
    text: str
    direction: str = ""
    recorded_path: str | None = None


@dataclass
class Project:
    name: str
    source_video: str
    tracks: list[Track] = field(default_factory=_default_tracks)
    transcript: list[dict] = field(default_factory=list)
    script: list[ScriptScene] = field(default_factory=list)
    aspect_ratio: str = "auto"  # "auto" | "16:9" | "9:16" | "1:1" | "4:5" | "4:3" — see ffmpeg_ops.ASPECT_RATIOS
    schema_version: int = SCHEMA_VERSION

    def track(self, kind: str) -> Track:
        """Get the named track, creating it if this project predates it."""
        for t in self.tracks:
            if t.kind == kind:
                return t
        new_track = Track(kind=kind)
        self.tracks.append(new_track)
        return new_track

    def add_clip(self, clip: Clip, track: str = "video") -> None:
        self.track(track).add_clip(clip)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        version = d.get("schema_version", 1)
        if version < 2:
            return cls._from_legacy_v1(d)

        tracks = [
            Track(kind=t["kind"], clips=[Clip(**c) for c in t.get("clips", [])])
            for t in d.get("tracks", [])
        ]
        if not tracks:
            tracks = _default_tracks()

        return cls(
            name=d["name"],
            source_video=d["source_video"],
            tracks=tracks,
            transcript=d.get("transcript", []),
            script=[ScriptScene(**s) for s in d.get("script", [])],
            aspect_ratio=d.get("aspect_ratio", "auto"),
        )

    @classmethod
    def _from_legacy_v1(cls, d: dict) -> "Project":
        """v1 project.json stored a flat `scenes` list — no tracks, no
        timeline position. Each legacy Scene's source_path already points
        at an ffmpeg-cut file (a standalone clip starting at its own 0),
        so it migrates onto the video track with in_point=0 and
        out_point=duration; scene.start/scene.end were timecodes into the
        *original* source video, not this file, so they're only used here
        to recover each clip's duration. Timeline position is assigned by
        placing the migrated clips back-to-back in their original order,
        since v1 had no concept of a timeline position to preserve.
        """
        video_track = Track(kind="video")
        cursor = 0.0
        for legacy_scene in d.get("scenes", []):
            duration = legacy_scene["end"] - legacy_scene["start"]
            clip = Clip(
                id=legacy_scene["id"],
                source_path=legacy_scene["source_path"],
                in_point=0.0,
                out_point=duration,
                start_time=cursor,
                label=legacy_scene.get("label", ""),
            )
            video_track.clips.append(clip)
            cursor += duration

        return cls(
            name=d["name"],
            source_video=d["source_video"],
            tracks=[video_track, Track(kind="audio"), Track(kind="text")],
            transcript=d.get("transcript", []),
        )


def save(project: Project, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(project.to_dict(), f, indent=2)


def load(path: str) -> Project:
    with open(path) as f:
        return Project.from_dict(json.load(f))
