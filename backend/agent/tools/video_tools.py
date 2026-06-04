import json
import os
import shutil
import subprocess
from pathlib import Path

from ...config import settings


def _find_binary(name: str) -> str:
    """Resolve ffmpeg/ffprobe even when the current shell has not reloaded PATH yet."""
    binary = shutil.which(name)
    if binary:
        return binary

    executable = f"{name}.exe" if os.name == "nt" else name
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        winget_packages = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if winget_packages.exists():
            matches = sorted(winget_packages.glob(f"**/{executable}"))
            for match in matches:
                if match.is_file():
                    return str(match)

    raise RuntimeError(
        f"{name} executable was not found. Install FFmpeg and make sure {name} is on PATH."
    )


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{detail}") from exc


def _parse_fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" not in value:
        fps = float(value)
        return fps if fps > 0 else None

    num, den = value.split("/", 1)
    denominator = float(den)
    if denominator == 0:
        return None

    fps = float(num) / denominator
    return fps if fps > 0 else None


def get_video_fps(video_path: str) -> float:
    """Return a video's native FPS via ffprobe."""
    ffprobe = _find_binary("ffprobe")
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            video_path,
        ]
    )
    data = json.loads(result.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found: {video_path}")

    stream = streams[0]
    fps = _parse_fraction(stream.get("avg_frame_rate")) or _parse_fraction(
        stream.get("r_frame_rate")
    )
    if fps is None:
        raise RuntimeError(f"Could not determine native FPS for video: {video_path}")
    return fps


def extract_frames(video_path: str, out_dir: Path, fps: float | None = None) -> list[Path]:
    """Extract frames at the given fps using ffmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in out_dir.glob("*.jpg"):
        old_frame.unlink()

    target_fps = float(fps if fps is not None else settings.SAMPLE_FPS)
    if target_fps <= 0:
        raise ValueError(f"fps must be positive, got {target_fps}")

    ffmpeg = _find_binary("ffmpeg")
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            f"fps={target_fps}",
            "-q:v",
            "2",
            str(out_dir / "%04d.jpg"),
            "-y",
        ]
    )

    frames = sorted(out_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return frames


def compose_video(frames_dir: Path, out_path: Path, fps: float | None = None) -> None:
    """Compose jpg frames into a browser-compatible H.264 MP4 using ffmpeg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames found in directory: {frames_dir}")

    target_fps = float(fps if fps is not None else settings.SAMPLE_FPS)
    if target_fps <= 0:
        raise ValueError(f"fps must be positive, got {target_fps}")

    ffmpeg = _find_binary("ffmpeg")
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(target_fps),
            "-i",
            str(frames_dir / "%04d.jpg"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_range=tv,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-movflags",
            "+faststart",
            str(out_path),
            "-y",
        ]
    )

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Could not compose video: {out_path}")
