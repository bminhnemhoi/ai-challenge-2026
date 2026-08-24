"""Task 3 — TRAKE, powered by CHRONOS.

Assigned member: Member 4.

The task: given N ordered event descriptions belonging to one action, find one
video and N frames f1 < f2 < ... < fN, one per event.

The official score (rules 2.1.3) shapes the whole design:

    R-Score = 0                                   if the video is wrong
            = (1/N) * sum_j I(frame_j in [s_j,e_j])  otherwise

so the video is all-or-nothing while the events earn partial credit, and the
per-event windows are "thường rất ngắn, thông thường là dưới 10 frame".

Two consequences:

1. Only ONE video is ever worth submitting — every row should use the best
   candidate video, because a row on any other video scores exactly zero.
2. Frames should be perturbed around the aligned answer, since a 10-frame
   window and ~55-frame keyframe spacing mean the raw keyframe index usually
   misses even when the alignment is right.

The previous implementation took the independent top-1 keyframe per event and
sorted them.  That is the "no-DP baseline"; measured on a synthetic benchmark
with planted ground truth it reaches 20% sequence accuracy against CHRONOS's
88%, because independent maxima ignore both the ordering constraint and the
fact that the events belong to one contiguous action.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .alignment import adaptive_lambda, chronos_search
from .decomposer import RuleBasedDecomposer, detect_first_occurrence
from .scoring import fuse


class TRAKEEngine:
    """Temporal event alignment for Task 3.

    Wraps the CHRONOS dynamic-programming core with the repo's SigLIP-2 index.
    Pass an already-loaded :class:`~src.core.kis_engine.KISEngine` to share the
    index instead of loading a second ~820 MB copy.
    """

    def __init__(self, data_dir: str = "data", engine=None, decomposer=None) -> None:
        self.data_dir = data_dir
        self.engine = engine
        self.decomposer = decomposer or RuleBasedDecomposer()
        self._videos: Optional[Dict[str, dict]] = None

    # ------------------------------------------------------------------ load
    def load_index(self) -> "TRAKEEngine":
        if self.engine is None:
            from src.core.kis_engine import KISEngine

            self.engine = KISEngine(self.data_dir)
        if not getattr(self.engine, "_loaded", False):
            self.engine.load()
        self._build_video_ranges()
        return self

    def _build_video_ranges(self) -> None:
        """Contiguous global keyframe ranges, which the O(N*T) DP requires.

        The index is already grouped by video and ordered within each video, so
        the ranges are derived rather than assumed — a video whose rows are not
        contiguous would silently corrupt the alignment, so it raises instead.
        """
        vid = self.engine.video_id
        ranges: Dict[str, dict] = {}
        start = 0
        for i in range(1, len(vid) + 1):
            if i == len(vid) or vid[i] != vid[start]:
                v = str(vid[start])
                if v in ranges:
                    raise ValueError(
                        f"video {v} occupies more than one block of the index; "
                        "CHRONOS needs one contiguous range per video"
                    )
                ranges[v] = {"s_v": start, "e_v": i - 1}
                start = i
        self._videos = ranges

    # --------------------------------------------------------------- scoring
    def _score_matrix(self, events: Sequence[str]) -> np.ndarray:
        """(N, T) fused, row-z-scored score matrix over every keyframe.

        The excluded frames (channel bumpers, black frames) are masked AFTER
        fusing, with the sentinel the DP recognises.  Masking before the
        z-score would be catastrophic rather than merely wrong: ``fuse``
        standardises each row, so a few thousand -1e4 entries would dominate
        the mean and standard deviation and squash every real similarity into
        a hair-thin band around zero — the DP would then see no signal at all.
        """
        from .alignment import NEG

        rows = [self.engine.similarities(self.engine.query_vector(e)) for e in events]
        S = fuse(np.stack(rows, axis=0), mode="visual")
        S[:, ~self.engine.valid] = NEG
        return S

    # ----------------------------------------------------------------- align
    def align_sequence(
        self,
        event_descriptions: Sequence[str],
        video_id: Optional[str] = None,
        first_occurrence: Optional[bool] = None,
        top_k: int = 5,
        min_gap: int = 2,
        align_mode: str = "ordered",
    ) -> List[Dict[str, Any]]:
        """Align ordered event descriptions to one video's keyframe timeline.

        Returns candidates best-first; each carries ``video_id``,
        ``sequence_frames`` (the submittable frame indices, ascending),
        ``gids`` and per-event scores.
        """
        if self.engine is None or self._videos is None:
            self.load_index()
        events = [e for e in event_descriptions if str(e).strip()]
        if not events:
            return []

        S = self._score_matrix(events)
        mu = 2.0 if (first_occurrence if first_occurrence is not None else False) else 0.0
        videos = self._videos
        if video_id is not None:
            if video_id not in videos:
                raise ValueError(f"unknown video_id {video_id!r}")
            videos = {video_id: videos[video_id]}

        results, lam = chronos_search(
            S,
            videos,
            lam=None,
            g=min_gap,
            mu=mu,
            topk=top_k,
            align_mode=align_mode,
        )

        out: List[Dict[str, Any]] = []
        for r in results:
            gids = r["gids"]
            order = r.get("perm") or list(range(len(gids)))
            per_event = sorted(
                (
                    {
                        "idx": int(order[pos]) + 1,
                        "gid": int(g),
                        "frame_idx": int(self.engine.frame_idx[g]),
                        "pts_time": float(self.engine.pts_time[g]),
                        "score": float(S[min(int(order[pos]), S.shape[0] - 1), g]),
                    }
                    for pos, g in enumerate(gids)
                ),
                key=lambda e: e["idx"],
            )
            out.append(
                {
                    "video_id": r["video_id"],
                    "score": float(r["score"]),
                    "lambda_used": float(lam),
                    "events": per_event,
                    "gids": [int(g) for g in gids],
                    # answer columns follow EVENT order, which is what the
                    # grader compares against [s_j, e_j] for event j
                    "sequence_frames": [e["frame_idx"] for e in per_event],
                }
            )
        return out

    # ---------------------------------------------------------------- decode
    def decompose(self, raw_query: str):
        """Split a raw Vietnamese TRAKE prompt into ordered event descriptions."""
        dq = self.decomposer.decompose(raw_query)
        dq.first_occurrence = dq.first_occurrence or detect_first_occurrence(raw_query)
        return dq

    def answer(self, raw_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Raw query text straight to ranked aligned sequences."""
        dq = self.decompose(raw_query)
        return self.align_sequence(
            [e.text for e in dq.events],
            first_occurrence=dq.first_occurrence,
            top_k=top_k,
        )
