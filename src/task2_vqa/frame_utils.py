"""Extract frames directly from video files for Task 2 (Visual Q&A).

BTC / Task 1 only provide `video_id` + `frame_idx` (an index into the original
video) -- there is no pre-extracted image on disk in general. These helpers
seek directly into the `.mp4` file to fetch the exact frame(s) needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import cv2
from PIL import Image

try:
    from .config import MAX_SIDE
except ImportError:  # allows running this file directly for local testing
    from config import MAX_SIDE


def resize_cap(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Resize so the longest side is <= max_side, keeping aspect ratio. Never upscales."""
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return img
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def extract_frame(video_path: str, frame_idx: int) -> Image.Image:
    """Fetch exactly 1 frame by index from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
    finally:
        cap.release()
    if not ok:
        raise IndexError(f"Frame {frame_idx} is out of range for {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return resize_cap(Image.fromarray(frame_rgb))


def extract_frame_window(
    video_path: str, center_frame_idx: int, window: int = 2, step: int = 5
) -> List[Image.Image]:
    """
    Fetch several consecutive frames around center_frame_idx, `step` frames apart.
    window=2, step=5 -> offsets [-10,-5,0,+5,+10] around the center.
    Use this when a question needs to see motion/action, not just a static state.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: List[Image.Image] = []
        for offset in range(-window, window + 1):
            fid = center_frame_idx + offset * step
            if fid < 0 or fid >= total_frames:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, frame_bgr = cap.read()
            if ok:
                frames.append(resize_cap(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))))
    finally:
        cap.release()
    if not frames:
        raise IndexError(f"Could not fetch any frame around {center_frame_idx} in {video_path}")
    return frames


def resolve_video_path(video_dir: str, video_id: str) -> str:
    """video_id has no extension -- find the matching file in video_dir."""
    for ext in (".mp4", ".MOV", ".mov", ".avi", ".mkv"):
        p = os.path.join(video_dir, f"{video_id}{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No video found for video_id={video_id} in {video_dir}")
