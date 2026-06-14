"""SAM3 temporal masking mode helpers."""
from __future__ import annotations

SAM3_MODE_NORMAL = "normal"
SAM3_MODE_PRECISION = "precision"

_NORMAL_ALIASES = {
    "",
    "normal",
    "standard",
    "default",
    "balanced",
    "fast",
    "regular",
    "보통",
    "일반",
}
_PRECISION_ALIASES = {
    "precision",
    "precise",
    "accurate",
    "quality",
    "full",
    "full-frame",
    "full-frames",
    "full_frame",
    "full_frames",
    "all-frame",
    "all-frames",
    "all_frame",
    "all_frames",
    "per-frame",
    "per_frame",
    "정밀",
}


def normalize_sam3_mode(value: str | None) -> str:
    """Return the canonical SAM3 temporal mode."""
    raw = (value or SAM3_MODE_NORMAL).strip().lower()
    if raw in _NORMAL_ALIASES:
        return SAM3_MODE_NORMAL
    if raw in _PRECISION_ALIASES:
        return SAM3_MODE_PRECISION
    raise ValueError("sam3_mode must be 'normal' or 'precision'")


def sam3_mode_label(mode: str | None) -> str:
    normalized = normalize_sam3_mode(mode)
    return "보통" if normalized == SAM3_MODE_NORMAL else "정밀"


def sam3_mode_description(mode: str | None, keyframe_interval: int = 3) -> str:
    normalized = normalize_sam3_mode(mode)
    if normalized == SAM3_MODE_PRECISION:
        return "모든 프레임을 segmentation합니다"
    interval = max(1, int(keyframe_interval))
    return f"{interval}프레임 간격으로 segmentation하고 중간 프레임은 보간합니다"
