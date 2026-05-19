import json
import subprocess
from pathlib import Path

from ...config import settings


def get_video_fps(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "json", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    num, den = data["streams"][0]["r_frame_rate"].split("/")
    return float(num) / float(den)


def extract_frames(video_path: str, out_dir: Path, fps: float | None = None) -> list[Path]:
    """Extract frames at the given fps (default: SAMPLE_FPS for detection)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target_fps = fps if fps is not None else settings.SAMPLE_FPS
    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={target_fps}",
            str(out_dir / "%04d.jpg"),
            "-y", "-loglevel", "error",
        ],
        check=True,
    )
    return sorted(out_dir.glob("*.jpg"))


def compose_video(frames_dir: Path, out_path: Path, fps: float | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_fps = fps if fps is not None else settings.SAMPLE_FPS
    subprocess.run(
        [
            "ffmpeg",
            "-framerate", str(target_fps),
            "-i", str(frames_dir / "%04d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(out_path),
            "-y", "-loglevel", "error",
        ],
        check=True,
    )
