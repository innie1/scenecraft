"""
Core video operations, built directly on FFmpeg via subprocess.
No UI code here, no MCP code here. This module doesn't know or care
who's calling it — the desktop app calls it directly today, and the
MCP server will call these same functions later. Keep it that way.
"""

import subprocess
import shutil
import json
from pathlib import Path


class FFmpegNotFoundError(Exception):
    pass


def _ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegNotFoundError(
            "ffmpeg was not found on this system. Scenecraft bundles its own "
            "ffmpeg binary in packaged builds, but it's missing in this dev "
            "environment. Install ffmpeg or point FFMPEG_PATH at a binary."
        )
    return path


def probe(input_path: str) -> dict:
    """Return basic metadata about a video file: duration, resolution, fps."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise FFmpegNotFoundError("ffprobe not found alongside ffmpeg.")
    cmd = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    duration = float(data.get("format", {}).get("duration", 0))

    return {
        "duration_seconds": round(duration, 3),
        "width": video_stream.get("width") if video_stream else None,
        "height": video_stream.get("height") if video_stream else None,
        "fps": _parse_fps(video_stream.get("r_frame_rate")) if video_stream else None,
        "codec": video_stream.get("codec_name") if video_stream else None,
    }


def _parse_fps(rate_str: str | None) -> float | None:
    if not rate_str:
        return None
    num, _, den = rate_str.partition("/")
    try:
        return round(float(num) / float(den or 1), 2)
    except (ValueError, ZeroDivisionError):
        return None


def cut_clip(input_path: str, output_path: str, start: float, end: float) -> str:
    """
    Cut a clip from start to end (seconds). Uses stream copy (-c copy) so it's
    near-instant and lossless — no re-encoding. This is the fast path for
    scene trimming. Falls back to re-encode automatically if stream copy
    fails (which can happen if the cut point isn't on a keyframe).
    """
    ffmpeg = _ffmpeg_path()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    duration = end - start

    fast_cmd = [
        ffmpeg, "-y", "-ss", str(start), "-i", input_path,
        "-t", str(duration), "-c", "copy", output_path,
    ]
    result = subprocess.run(fast_cmd, capture_output=True, text=True)

    if result.returncode != 0 or not Path(output_path).exists():
        # Fallback: re-encode. Slower, but always works regardless of keyframes.
        reencode_cmd = [
            ffmpeg, "-y", "-ss", str(start), "-i", input_path,
            "-t", str(duration), "-c:v", "libx264", "-c:a", "aac", output_path,
        ]
        subprocess.run(reencode_cmd, capture_output=True, text=True, check=True)

    return output_path


def export(input_path: str, output_path: str, format: str = "mp4") -> str:
    """Render the final export. Placeholder for now — will grow to accept
    a full scene/timeline list once project_state.py is wired in."""
    ffmpeg = _ffmpeg_path()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", input_path, "-c:v", "libx264", "-c:a", "aac", output_path]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return output_path
