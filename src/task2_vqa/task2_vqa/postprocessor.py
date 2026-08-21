"""Answer cleanup and AIC 2026 Task 2 submission formatting."""

from __future__ import annotations

import re
from typing import Any, Optional


_ANSWER_PREFIX = re.compile(r"^\s*(?:answer|final answer|đáp\s*án)\s*:\s*", re.IGNORECASE)


def clean_answer(answer: Any, max_length: int = 200) -> str:
    """Normalize a short VLM answer without changing its semantic content."""
    text = re.sub(r"\s+", " ", "" if answer is None else str(answer)).strip()
    text = _ANSWER_PREFIX.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    text = text.strip().rstrip(".").rstrip()
    if not text:
        raise ValueError("VQA answer cannot be empty.")
    if max_length < 1:
        raise ValueError("max_length must be positive.")
    return text[:max_length].rstrip()


def build_submission_record(
    video_id: str,
    frame_idx: int,
    answer: Any,
    score: Optional[float] = None,
) -> dict[str, Any]:
    """Build the JSON contract consumed by the shared CSV exporter."""
    normalized_video_id = str(video_id or "").strip()
    if not normalized_video_id:
        raise ValueError("video_id cannot be empty.")
    try:
        normalized_frame_idx = int(frame_idx)
    except (TypeError, ValueError) as exc:
        raise ValueError("frame_idx must be an integer.") from exc
    if normalized_frame_idx < 0:
        raise ValueError("frame_idx cannot be negative.")

    record: dict[str, Any] = {
        "video_id": normalized_video_id,
        "frame_idx": normalized_frame_idx,
        "answer": clean_answer(answer),
    }
    if score is not None:
        record["score"] = float(score)
    return record
