"""TrakeEngine — entry point of Task 3 (CHRONOS).

Wires the layers together::

    raw_query -> EDL (decomposer) -> MCS (scoring.fuse) -> TAP (chronos_search)
              -> VVR (verifier)   -> API response (spec section 8.3)

All heavy dependencies (text encoder, LLM, VLM, OCR/ASR searchers) are
injected as callables so the engine itself stays testable offline.

Index conventions, stated once because they differ by layer:

* the API/UI is **1-based** — ``events[].idx`` starts at 1 and ``anchors``
  keys refer to that idx (spec 8.2/8.3);
* :func:`~task3_trake.alignment.apply_anchors` is **0-based** (spec 6.6).

The conversion happens here, at the boundary, and nowhere else.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .alignment import adaptive_lambda, chronos_search
from .data_loader import TrakeData
from .decomposer import DecomposedQuery, Event, RuleBasedDecomposer, detect_first_occurrence
from .formatter import format_submission
from .scoring import SCORING_MODES, channel_matrix, fuse, visual_scores
from .verifier import VLMClient, verify_and_rerank

TextEncoder = Callable[[Sequence[str]], np.ndarray]
HitSearcher = Callable[[str], Dict[int, float]]  # hint -> {gid: score}


@dataclass
class TrakeConfig:
    """Runtime defaults (mirrors config/trake.yaml, spec section 11)."""

    mode: str = "balanced"
    weights: Optional[Dict[str, float]] = None  # None -> use SCORING_MODES[mode]
    align_mode: str = "ordered"
    lam: Optional[float] = None  # None == "auto"
    lambda_bounds: Tuple[float, float] = (0.005, 0.2)  # relative to score scale
    lambda_kappa: float = 0.5
    min_gap: int = 2
    earliness_mu: float = 2.0  # in score-std units, see alignment.chronos_search
    topk_videos: int = 20
    verify_enabled: bool = False
    topk_verify: int = 3
    swap_penalty: float = 0.5

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrakeConfig":
        """Load defaults from YAML.

        Every fallback reads the dataclass default rather than repeating a
        literal, so a config file that omits a key can never disagree with
        ``TrakeConfig()``.
        """
        import yaml

        d = cls()  # the single source of truth for every default
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        scoring = raw.get("scoring", {})
        align = raw.get("alignment", {})
        verify = raw.get("verify", {})
        lam_raw = align.get("lambda", "auto")
        lam = None if (lam_raw in (None, "auto")) else float(lam_raw)
        bounds = align.get("lambda_bounds", list(d.lambda_bounds))
        return cls(
            mode=scoring.get("mode", d.mode),
            weights=scoring.get("weights", d.weights),
            align_mode=align.get("align_mode", d.align_mode),
            lam=lam,
            lambda_bounds=(float(bounds[0]), float(bounds[1])),
            lambda_kappa=float(align.get("lambda_kappa", d.lambda_kappa)),
            min_gap=int(align.get("min_gap", d.min_gap)),
            earliness_mu=float(align.get("earliness_mu", d.earliness_mu)),
            topk_videos=int(align.get("topk_videos", d.topk_videos)),
            swap_penalty=float(align.get("swap_penalty", d.swap_penalty)),
            verify_enabled=bool(verify.get("enabled", d.verify_enabled)),
            topk_verify=max(1, int(verify.get("topk_verify", d.topk_verify))),
        )


class TrakeEngine:
    """Orchestrator for the TRAKE search flow.

    Parameters
    ----------
    data          : loaded corpus (:class:`TrakeData`).
    text_encoder  : ``texts -> (N, D) array``; must match the corpus encoder.
                    Optional if callers always pass ``query_vectors``.
    decomposer    : object with ``decompose(raw) -> DecomposedQuery``;
                    defaults to :class:`RuleBasedDecomposer`.
    ocr_searcher / asr_searcher : ``hint -> {gid: score}`` sparse retrievers
                    (Elasticsearch in production); optional.
    vlm           : VLM client for VVR; optional.
    """

    def __init__(
        self,
        data: TrakeData,
        text_encoder: Optional[TextEncoder] = None,
        decomposer=None,
        ocr_searcher: Optional[HitSearcher] = None,
        asr_searcher: Optional[HitSearcher] = None,
        vlm: Optional[VLMClient] = None,
        config: Optional[TrakeConfig] = None,
    ) -> None:
        self.data = data
        self.text_encoder = text_encoder
        self.decomposer = decomposer or RuleBasedDecomposer()
        self.ocr_searcher = ocr_searcher
        self.asr_searcher = asr_searcher
        self.vlm = vlm
        self.config = config or TrakeConfig()
        # a yaml that turns verification on without a VLM would otherwise fail
        # every single query at request time; fail at startup instead
        if self.config.verify_enabled and self.vlm is None:
            raise ValueError(
                "config enables verification (verify.enabled) but no VLM client was given"
            )

    # ------------------------------------------------------------------ EDL
    def _resolve_events(
        self,
        raw_query: Optional[str],
        context: Optional[str],
        events: Optional[Sequence],
        first_occurrence: Optional[bool],
    ) -> DecomposedQuery:
        if events:
            evs: List[Event] = []
            for k, ev in enumerate(events):
                if isinstance(ev, Event):
                    evs.append(ev)
                elif isinstance(ev, dict):
                    if not str(ev.get("text", "")).strip():
                        raise ValueError(f"event {k + 1} has no 'text'")
                    raw_idx, raw_w = ev.get("idx"), ev.get("weight")
                    # array position IS the event number: `query_vectors` and
                    # `anchors` are both positional, so silently reordering
                    # here would score each event against another's vector and
                    # pin anchors to the wrong event.  A conflicting explicit
                    # idx is ambiguous input, so reject it rather than guess.
                    if raw_idx is not None and int(raw_idx) != k + 1:
                        raise ValueError(
                            f"event {k + 1} declares idx={int(raw_idx)}: events must be "
                            f"supplied in order, with idx matching their position (1-based)"
                        )
                    evs.append(
                        Event(
                            idx=k + 1,
                            text=str(ev["text"]),
                            ook=bool(ev.get("ook", False)),
                            ook_term=ev.get("ook_term"),
                            ocr_hint=ev.get("ocr_hint"),
                            asr_hint=ev.get("asr_hint"),
                            # 'or 1.0' would rewrite an explicit weight of 0
                            # into full weight — the opposite of the request
                            weight=1.0 if raw_w is None else float(raw_w),
                        )
                    )
                else:
                    evs.append(Event(idx=k + 1, text=str(ev)))
            if first_occurrence is None:
                # the operator may hand-edit the events but keep the raw query;
                # the "thời điểm đầu tiên" evidence in it still applies
                first = detect_first_occurrence(raw_query or context or "")
            else:
                first = bool(first_occurrence)
            return DecomposedQuery(
                context=context or (raw_query or ""),
                first_occurrence=first,
                events=evs,
                source="manual",
            )
        if not raw_query:
            raise ValueError("either `events` or `raw_query` must be provided")
        dq = self.decomposer.decompose(raw_query)
        if first_occurrence is not None:
            dq.first_occurrence = bool(first_occurrence)
        if context:
            dq.context = context
        return dq

    # ------------------------------------------------------------------ MCS
    def _build_scores(
        self,
        dq: DecomposedQuery,
        query_vectors: Optional[np.ndarray],
        mode: str,
        weights: Optional[Dict[str, float]],
    ) -> np.ndarray:
        T = self.data.T
        if query_vectors is not None:
            U = np.asarray(query_vectors, dtype=np.float32)
        else:
            if self.text_encoder is None:
                raise ValueError("no text_encoder configured and no query_vectors given")
            U = np.asarray(self.text_encoder([e.text for e in dq.events]), dtype=np.float32)
        if U.shape[0] != len(dq.events):
            raise ValueError(f"got {U.shape[0]} query vectors for {len(dq.events)} events")

        S_vis = visual_scores(U, self.data.embeddings)

        S_ocr = None
        if self.ocr_searcher is not None and any(e.ocr_hint for e in dq.events):
            S_ocr = channel_matrix(
                [self.ocr_searcher(e.ocr_hint) if e.ocr_hint else None for e in dq.events], T
            )
        S_asr = None
        if self.asr_searcher is not None and any(e.asr_hint for e in dq.events):
            S_asr = channel_matrix(
                [self.asr_searcher(e.asr_hint) if e.asr_hint else None for e in dq.events], T
            )

        return fuse(
            S_vis,
            S_ocr,
            S_asr,
            mode=mode,
            weights=weights,
            event_weights=[e.weight for e in dq.events],
        )

    # ------------------------------------------------------------- prefilter
    def _prefilter_videos(self, context: str, prefilter_top: int) -> List[str]:
        """Risk R6: cheap context pass to cut the corpus to top-K videos.

        Skipped when there is no usable signal — ranking the corpus against an
        embedding of the empty string would truncate it arbitrarily and could
        drop the correct video.
        """
        if self.text_encoder is None or not (context or "").strip():
            return list(self.data.videos)
        u_ctx = np.asarray(self.text_encoder([context]), dtype=np.float32)
        s_ctx = visual_scores(u_ctx, self.data.embeddings)[0]
        ranked = sorted(
            self.data.videos.items(),
            key=lambda kv: -float(s_ctx[kv[1]["s_v"] : kv[1]["e_v"] + 1].max()),
        )
        return [vid for vid, _ in ranked[:prefilter_top]]

    # ----------------------------------------------------------------- search
    def search(
        self,
        raw_query: Optional[str] = None,
        context: Optional[str] = None,
        events: Optional[Sequence] = None,
        mode: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
        align_mode: Optional[str] = None,
        lam: Optional[float] = None,
        min_gap: Optional[int] = None,
        earliness: Optional[float] = None,
        first_occurrence: Optional[bool] = None,
        anchors: Optional[Dict] = None,
        topk: Optional[int] = None,
        verify: Optional[bool] = None,
        window: Optional[int] = None,
        query_vectors: Optional[np.ndarray] = None,
        candidate_videos: Optional[Sequence[str]] = None,
        prefilter_top: Optional[int] = None,
    ) -> dict:
        """Run the full TRAKE flow; returns the API response (spec 8.3).

        ``anchors`` keys are 1-based event indices matching ``events[].idx``.
        """
        t0 = time.perf_counter()
        cfg = self.config
        explicit_mode = mode is not None
        mode = mode or cfg.mode
        align_mode = align_mode or cfg.align_mode
        min_gap = cfg.min_gap if min_gap is None else int(min_gap)
        topk = cfg.topk_videos if topk is None else int(topk)
        verify = cfg.verify_enabled if verify is None else bool(verify)
        # check before doing any work, and unconditionally — not after the
        # search, where an empty result set would hide the misconfiguration
        if verify and self.vlm is None:
            raise ValueError("verify=true but no VLM client is configured")
        if lam is None:
            lam = cfg.lam  # may still be None == adaptive

        # weight precedence: explicit request > config yaml (only when the
        # request did not name a different mode) > the mode's preset
        if weights is None and not explicit_mode:
            weights = cfg.weights

        # L1 — decompose
        dq = self._resolve_events(raw_query, context, events, first_occurrence)

        # L2 — score matrix
        S = self._build_scores(dq, query_vectors, mode, weights)

        # earliness prior only when the query asks for "first occurrence"
        mu_cfg = cfg.earliness_mu if earliness is None else float(earliness)
        mu = mu_cfg if dq.first_occurrence else 0.0

        # anchors: 1-based at the API boundary -> 0-based for the DP
        anchors_int: Optional[Dict[int, int]] = None
        if anchors:
            anchors_int = {}
            for k, v in anchors.items():
                ik = int(k)
                if ik < 1:
                    raise ValueError(f"anchor event index {k!r} must be 1-based (>= 1)")
                if ik > len(dq.events):
                    raise ValueError(f"anchor event index {ik} exceeds {len(dq.events)} events")
                gid = int(v)
                # validate here, not deep inside apply_anchors: the forced-video
                # lookup below would otherwise index the keyframe table first
                # and surface an IndexError as a 500
                if not 0 <= gid < self.data.T:
                    raise ValueError(f"anchor gid {gid} out of range [0, {self.data.T})")
                anchors_int[ik - 1] = gid

        # candidate restriction (explicit list wins over context prefilter)
        video_ids: Optional[Sequence[str]] = candidate_videos
        if video_ids is None and prefilter_top:
            video_ids = self._prefilter_videos(dq.context, int(prefilter_top))
        if anchors_int and video_ids is not None:
            # an anchor is a statement of certainty; never let a prefilter drop
            # the very video the operator pointed at
            forced = {self.data.video_of_gid(g) for g in anchors_int.values()}
            video_ids = list(dict.fromkeys(list(video_ids) + [v for v in forced if v]))

        # resolve lambda here so the configured bounds actually apply
        if lam is None:
            # g must match the search: a probe allowed to place events 2 apart
            # reports a tempo the real DP cannot achieve at a larger min_gap,
            # and lambda comes out an order of magnitude too strong.
            lam = adaptive_lambda(
                S,
                self.data.videos,
                kappa=cfg.lambda_kappa,
                lo=cfg.lambda_bounds[0],
                hi=cfg.lambda_bounds[1],
                g=min_gap,
            )

        # L3 — DP search
        results_raw, lam_used = chronos_search(
            S,
            self.data.videos,
            lam=lam,
            g=min_gap,
            mu=mu,
            topk=topk,
            align_mode=align_mode,
            window=window,
            swap_penalty=cfg.swap_penalty,
            anchors=anchors_int,
            video_ids=video_ids,
        )

        results = [self._build_result(r, S, dq) for r in results_raw]

        # L4 — VVR
        verified = False
        if verify and results:
            results = verify_and_rerank(
                results, [e.text for e in dq.events], self.vlm, topk_verify=cfg.topk_verify
            )
            verified = True

        return {
            "query_id": f"trake_{int(time.time())}",
            "decomposed": dq.to_dict(),
            "mode": mode,
            # the weights actually fused — an explicit dict silently overrides
            # `mode`, so reporting the mode alone would misdescribe the run
            # (and corrupt any ablation logged from the API)
            "weights_used": dict(weights) if weights else dict(SCORING_MODES[mode]),
            "align_mode": align_mode,
            "lambda_used": float(lam_used),
            "min_gap": min_gap,
            "earliness": mu,
            "verified": verified,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
            "results": results,
        }

    def _build_result(self, raw: dict, S: np.ndarray, dq: DecomposedQuery) -> dict:
        """Assemble one API record.

        ``raw["gids"]`` is chronological.  Under the order-relaxing modes the
        k-th frame does NOT belong to event k, so the permutation decides
        which score row to read and which ``idx`` to report.

        ``events`` is returned ordered by EVENT NUMBER, and the answer columns
        (``sequence_frames``, ``submission_line``) follow that same order —
        the grader expects column k to hold the frame for event k.  Emitting
        them in chronological order would silently transpose the answer
        whenever the winning alignment used a swap.  The chronological reading
        is still available as ``gids_chronological`` for the UI grid.
        """
        perm = raw.get("perm")
        order = perm if perm else list(range(len(raw["gids"])))
        degraded = False
        events_meta: List[dict] = []
        for pos, gid in enumerate(raw["gids"]):
            row = int(order[pos]) if pos < len(order) else pos
            row = min(max(row, 0), S.shape[0] - 1)
            meta = self.data.lookup(gid)
            degraded = degraded or bool(meta.get("degraded"))
            events_meta.append(
                {
                    "idx": dq.events[row].idx if row < len(dq.events) else row + 1,
                    "gid": int(gid),
                    "frame_idx": meta["frame_idx"],
                    "ts_ms": meta["ts_ms"],
                    "shot_id": meta["shot_id"],
                    "score": float(S[row, gid]),
                    "thumb": f"/kf/{raw['video_id']}/{meta['frame_idx']}.jpg",
                }
            )
        by_event = sorted(events_meta, key=lambda e: e["idx"])
        rec = {
            "video_id": raw["video_id"],
            "score": float(raw["score"]),
            "events": by_event,
            "gids_chronological": [int(g) for g in raw["gids"]],
            "sequence_frames": [ev["frame_idx"] for ev in by_event],
        }
        if degraded:
            # keyframe ordinals are not video frame indices; emitting a
            # submission line here would guarantee a wrong answer
            rec["submission_line"] = None
            rec["warning"] = "no keyframe table loaded: frame_idx is a keyframe ordinal"
        else:
            rec["submission_line"] = format_submission(raw["video_id"], by_event)
        if perm:
            rec["event_order"] = [int(x) for x in perm]
        return rec
