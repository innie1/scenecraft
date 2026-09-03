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
    audio_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"),
        None,
    )
    duration = float(data.get("format", {}).get("duration", 0))

    return {
        "duration_seconds": round(duration, 3),
        "width": video_stream.get("width") if video_stream else None,
        "height": video_stream.get("height") if video_stream else None,
        "fps": _parse_fps(video_stream.get("r_frame_rate")) if video_stream else None,
        "codec": video_stream.get("codec_name") if video_stream else None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
    }


# Codecs a Chromium-based <video> element (WebView2, the browser this app
# actually renders in) can decode natively. A file that plays fine in VLC
# can still fail to render at all here if it's outside this set (common
# with HEVC/H.265 phone/camera footage, or unusual audio codecs) — VLC
# uses its own much broader software decoder stack; the embedded browser
# doesn't. is_browser_playable()/make_browser_proxy() exist to paper over
# that gap for *playback only* — cut/export always operate on the
# original file, never the proxy.
_BROWSER_SAFE_VIDEO_CODECS = {"h264", "vp8", "vp9", "av1"}
_BROWSER_SAFE_AUDIO_CODECS = {"aac", "opus", "vorbis", "mp3", None}


def is_browser_playable(info: dict) -> bool:
    return info.get("codec") in _BROWSER_SAFE_VIDEO_CODECS and info.get("audio_codec") in _BROWSER_SAFE_AUDIO_CODECS


def make_browser_proxy(input_path: str, output_path: str) -> str:
    """Transcodes to H.264/AAC MP4 so the embedded browser can actually
    play it. Only for preview/playback — never used for cut/export, which
    always read the original file at its original quality."""
    ffmpeg = _ffmpeg_path()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create a playable proxy for {input_path}:\n{result.stderr}")
    return output_path


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


def _escape_drawtext(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def _wrap_caption_text(text: str, video_width: int, fontsize: int) -> str:
    """drawtext doesn't auto-wrap — a long caption just runs off both
    edges of the frame. Greedy word-wrap to roughly fit the frame width,
    using ~0.55*fontsize as a rough average glyph width (good enough for
    typical fonts; drawtext still centers/sizes the box from the actual
    rendered text, this only decides where line breaks go)."""
    avg_char_width = fontsize * 0.55
    max_chars_per_line = max(10, int((video_width * 0.9) / avg_char_width))

    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        added_len = len(word) + (1 if current else 0)
        if current and current_len + added_len > max_chars_per_line:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += added_len
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _find_font_file() -> str | None:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            # ffmpeg filter syntax needs ':' and '\' escaped inside the arg
            return candidate.replace("\\", "/").replace(":", "\\:")
    return None


# Fixed render resolutions for each supported export frame — long edge
# pinned to 1080/1920 so quality is consistent regardless of source
# footage. "auto" (not in this dict) instead matches the first video
# clip's own resolution, the original single-clip behavior.
ASPECT_RATIOS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "4:3": (1440, 1080),
}


def resolve_canvas_size(aspect_ratio: str, ref_width: int | None, ref_height: int | None) -> tuple[int, int]:
    if aspect_ratio in ASPECT_RATIOS:
        return ASPECT_RATIOS[aspect_ratio]
    return (ref_width or 1280, ref_height or 720)


def _clip_effects_filter(effects: dict) -> str | None:
    """Builds the ffmpeg filter chain for a clip's color adjustments, or
    None if it has none set. Missing keys are neutral (0 brightness, 1
    contrast/saturation) — an empty dict produces None, not a no-op eq=
    call."""
    if not effects:
        return None
    eq_params = []
    if "brightness" in effects:
        eq_params.append(f"brightness={effects['brightness']}")
    if "contrast" in effects:
        eq_params.append(f"contrast={effects['contrast']}")
    if "saturation" in effects:
        eq_params.append(f"saturation={effects['saturation']}")
    parts = []
    if eq_params:
        parts.append("eq=" + ":".join(eq_params))
    if effects.get("grayscale"):
        parts.append("hue=s=0")
    return ",".join(parts) if parts else None


def export(project, output_path: str) -> str:
    """
    Composite a Project's tracks into one rendered file.

    `project` is a project_state.Project (duck-typed here — this module
    only reads .track(kind)/.tracks and Clip's fields, so it stays
    decoupled from project_state and independently testable).

    - The video track drives the timeline length: its clips are placed
      back-to-back at their start_time, with black/silence inserted to
      fill any gap between them.
    - The audio track is mixed on top of the video track's own audio
      (each audio clip delayed to its start_time), not a replacement —
      adding music shouldn't silently drop the original dialogue.
    - The text track becomes drawtext overlays, each visible only during
      its clip's [start_time, end_time) window.

    Known v1 simplifications: video-track clips are assumed to each have
    their own audio stream (silent source video isn't handled specially),
    and overlapping video clips aren't resolved — later start_time simply
    concatenates after, so overlaps will just play back-to-back rather
    than actually overlapping.
    """
    ffmpeg = _ffmpeg_path()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    video_track = project.track("video")
    video_clips = sorted(video_track.clips, key=lambda c: c.start_time)
    if not video_clips:
        raise ValueError("Cannot export: the video track has no clips.")

    ref_info = probe(video_clips[0].source_path)
    width, height = resolve_canvas_size(getattr(project, "aspect_ratio", "auto"), ref_info["width"], ref_info["height"])
    fps = ref_info["fps"] or 30

    inputs: list[list[str]] = []

    def add_input(args: list[str]) -> int:
        idx = len(inputs)
        inputs.append(args)
        return idx

    filter_parts: list[str] = []
    video_labels: list[str] = []
    audio_labels: list[str] = []
    cursor = 0.0

    for clip in video_clips:
        if clip.start_time > cursor + 1e-6:
            gap = clip.start_time - cursor
            gv_idx = add_input(["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={gap}:r={fps}"])
            ga_idx = add_input(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"])
            gv_label, ga_label = f"gv{len(video_labels)}", f"ga{len(audio_labels)}"
            filter_parts.append(f"[{gv_idx}:v]trim=duration={gap},setpts=PTS-STARTPTS[{gv_label}]")
            filter_parts.append(f"[{ga_idx}:a]atrim=duration={gap},asetpts=PTS-STARTPTS[{ga_label}]")
            video_labels.append(gv_label)
            audio_labels.append(ga_label)
            cursor += gap

        v_idx = add_input(["-i", clip.source_path])
        v_label, a_label = f"v{len(video_labels)}", f"a{len(audio_labels)}"
        # Every clip is normalized to the canvas size, not just the "auto"
        # case — concat requires matching dimensions across all inputs
        # anyway, so this also fixes clips of genuinely different source
        # resolutions being concatenated together.
        effects_filter = _clip_effects_filter(getattr(clip, "effects", None) or {})
        effects_stage = f",{effects_filter}" if effects_filter else ""
        filter_parts.append(
            f"[{v_idx}:v]trim=start={clip.in_point}:end={clip.out_point},setpts=PTS-STARTPTS{effects_stage},"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black[{v_label}]"
        )
        filter_parts.append(
            f"[{v_idx}:a]atrim=start={clip.in_point}:end={clip.out_point},asetpts=PTS-STARTPTS[{a_label}]"
        )
        video_labels.append(v_label)
        audio_labels.append(a_label)
        cursor += clip.duration

    timeline_duration = cursor

    n = len(video_labels)
    filter_parts.append("".join(f"[{l}]" for l in video_labels) + f"concat=n={n}:v=1:a=0[basev]")
    filter_parts.append("".join(f"[{l}]" for l in audio_labels) + f"concat=n={n}:v=0:a=1[baseaudio]")

    final_audio_label = "baseaudio"
    audio_track = project.track("audio")
    if audio_track.clips:
        mix_labels = ["baseaudio"]
        for i, clip in enumerate(audio_track.clips):
            a_idx = add_input(["-i", clip.source_path])
            trimmed, delayed = f"aoT{i}", f"aoD{i}"
            filter_parts.append(
                f"[{a_idx}:a]atrim=start={clip.in_point}:end={clip.out_point},asetpts=PTS-STARTPTS[{trimmed}]"
            )
            delay_ms = int(clip.start_time * 1000)
            filter_parts.append(f"[{trimmed}]adelay={delay_ms}|{delay_ms}[{delayed}]")
            mix_labels.append(delayed)
        filter_parts.append(
            "".join(f"[{l}]" for l in mix_labels) + f"amix=inputs={len(mix_labels)}:duration=first[mixedaudio]"
        )
        final_audio_label = "mixedaudio"

    final_video_label = "basev"
    text_track = project.track("text")
    if any(c.label for c in text_track.clips):
        font_file = _find_font_file()
        if not font_file:
            raise RuntimeError(
                "No usable font file found for text overlays "
                f"(checked: {', '.join(_FONT_CANDIDATES)})."
            )
        fontsize = 36
        for i, clip in enumerate(text_track.clips):
            if not clip.label:
                continue
            next_label = f"txt{i}"
            wrapped = _wrap_caption_text(clip.label, width, fontsize)
            text = _escape_drawtext(wrapped)
            filter_parts.append(
                f"[{final_video_label}]drawtext=fontfile='{font_file}':text='{text}':fontcolor=white:fontsize={fontsize}:"
                f"box=1:boxcolor=black@0.5:boxborderw=8:x=(w-text_w)/2:y=h-th-40:line_spacing=4:"
                f"enable='between(t,{clip.start_time},{clip.end_time})'[{next_label}]"
            )
            final_video_label = next_label

    filter_complex = ";".join(filter_parts)

    cmd = [ffmpeg, "-y"]
    for input_args in inputs:
        cmd.extend(input_args)
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", f"[{final_video_label}]",
        "-map", f"[{final_audio_label}]",
        "-t", str(timeline_duration),
        "-c:v", "libx264", "-c:a", "aac",
        output_path,
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg export failed:\n{result.stderr}")
    return output_path
