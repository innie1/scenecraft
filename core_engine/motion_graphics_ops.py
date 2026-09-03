"""
Renders an HTML/CSS/JS animation to a video file by driving a headless
browser frame-by-frame and encoding the captures with ffmpeg. This is
the actual rendering half of "describe a motion graphic and generate
it" — the description-to-HTML half lives in llm_ops.py's local Ollama
call. No paid API, no cloud rendering: Playwright (browser automation)
+ ffmpeg, both already local dependencies.

No UI, no pywebview — importable and testable on its own, same pattern
as ffmpeg_ops/whisper_ops/script_ops.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright


class FFmpegNotFoundError(Exception):
    pass


def render_html_animation(
    html: str,
    output_path: str,
    duration: float,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> str:
    """
    `html` must be a complete, self-contained HTML document (its own
    <style>/<script>) that animates itself via CSS animations/transitions
    or JS (requestAnimationFrame, timers) as soon as it loads — nothing
    here drives the animation's clock, it just takes `fps * duration`
    screenshots spaced by real wall-clock time while the page plays and
    stitches them into a video. Keep it non-interactive: nothing here
    clicks or scrolls the page.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FFmpegNotFoundError("ffmpeg was not found on this system.")

    frame_count = max(1, round(duration * fps))
    frame_interval_ms = 1000 / fps

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        html_path = tmp_dir / "animation.html"
        html_path.write_text(html, encoding="utf-8")

        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": width, "height": height})
                page.goto(html_path.as_uri())
                for i in range(frame_count):
                    page.wait_for_timeout(frame_interval_ms)
                    page.screenshot(path=str(tmp_dir / f"frame_{i:05d}.png"))
            finally:
                browser.close()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg, "-y", "-framerate", str(fps),
            "-i", str(tmp_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to encode animation frames:\n{result.stderr}")

    return output_path
