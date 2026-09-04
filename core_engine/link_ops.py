"""
Pull media down from a URL so it can be transcribed like any local file.

Wraps yt-dlp, which covers both hosted video/audio pages and plain direct
media links (its generic extractor handles a bare .mp3/.mp4 URL), so the
composer only needs one code path for "here's a link, transcribe it".

Only audio is fetched — transcription is the point, and the audio-only
stream is a fraction of the download.
"""

import re
from pathlib import Path

_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

# Plenty of hosts (Wikimedia among them) reject yt-dlp's default agent
# outright with a 403, so present a normal browser one.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class LinkFetchError(Exception):
    pass


def find_url(text: str) -> str | None:
    """First URL in a piece of text, so "get me the transcript of <url>"
    works as well as pasting the bare link."""
    match = _URL_RE.search(text or "")
    if not match:
        return None
    # trailing sentence punctuation is almost never part of the URL
    return match.group(0).rstrip(".,;:!?)]}'\"")


def _readable(error: Exception) -> str:
    """yt-dlp packs an error onto one line: an "ERROR:" prefix, an
    "[extractor] id:" tag, the actual reason, then a "(caused by ...)"
    restatement of the same thing. Keep the reason, drop the rest."""
    text = str(error).replace("\x1b", "").split("\n")[0]
    text = re.sub(r"^\s*ERROR:\s*", "", text)
    text = re.sub(r"^\[[^\]]+\]\s*[^:]{0,80}?:\s*", "", text)  # "[generic] name: "
    text = re.split(r"\s*\(caused by ", text)[0]
    text = text.strip().rstrip(";,")
    if len(text) > 200:
        text = text[:197].rstrip() + "…"
    return text or "Couldn't fetch that link."


def fetch_audio(url: str, out_dir) -> tuple[str, str]:
    """Download the best available audio for `url`.

    Returns (path, title). Raises LinkFetchError with something a person
    can act on — a dead link, a private video, or a site yt-dlp can't read.
    """
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover - depends on the install
        raise LinkFetchError(
            "yt-dlp isn't installed, so links can't be fetched. Install it with: pip install yt-dlp"
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # a link that happens to point into a playlist should fetch the one
        # item, not silently start downloading a hundred of them
        "noplaylist": True,
        "http_headers": {"User-Agent": _USER_AGENT},
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            if info.get("entries"):
                info = info["entries"][0]
            path = ydl.prepare_filename(info)
    except LinkFetchError:
        raise
    except Exception as e:
        raise LinkFetchError(_readable(e)) from e

    if not Path(path).exists():
        raise LinkFetchError("The download finished but no media file turned up.")
    return path, (info.get("title") or Path(path).stem)
