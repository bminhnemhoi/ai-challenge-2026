"""L3 — TAP: Temporal Alignment by dynamic Programming.

Core of Task 3.  Solves the *ordered subsequence selection* problem:
given a score matrix ``S`` of shape ``(N, T)`` (N events x T global keyframes)
pick, inside one video's range ``[s_v, e_v]``, indices ``t_1 < t_2 < ... < t_N``
(with a minimum gap ``g``) maximizing::

    sum_i S[i, t_i]  -  lambda * (t_N - t_1 accumulated pairwise)  -  mu * earliness(t_1)

Recurrence (spec section 6.1)::

    DP[1, t] = S[1, t] - mu * (t - s_v) / (e_v - s_v)
    DP[i, t] = S[i, t] + max_{tau <= t - g} ( DP[i-1, tau] - lambda * (t - tau) )

The ``-lambda*(t - tau)`` term factorises into ``-lambda*t + lambda*tau`` so a
running maximum over ``DP[i-1, tau] + lambda*tau`` gives O(N*T) total work
(spec section 6.2).

Tie-break rule ("first occurrence" semantics, spec section 6.4): whenever two
predecessors give the same value, the EARLIEST tau must win.  Both
implementations therefore update the running maximum only on a STRICT
improvement (``>``), and the final end frame is taken with ``np.argmax`` which
returns the first (earliest) maximum.

Deviation from the spec, deliberate: section 6.4 writes the update as
``cand > running_max + 1e-9``, treating sub-epsilon differences as ties.  This
implementation uses an exact ``>`` instead, for two reasons.  First, the tie
that actually occurs in video data — a static shot whose consecutive keyframes
carry identical embeddings — produces *bit-identical* scores, which exact
``>`` already resolves to the earliest frame; and with ``lam = 0`` the whole
prefix term stays bit-identical too.  Second, an epsilon margin cannot be made
to agree between the sequential and vectorized forms: the reference compares
against the value at the currently chosen tau while ``np.maximum.accumulate``
tracks the true running maximum, so the two select different predecessors
whenever a skipped candidate sits within epsilon above the chosen one.  Exact
comparison keeps :func:`chronos_align` and :func:`chronos_align_ref`
provably identical, which the fuzz test in ``tests/test_alignment.py`` checks
on every masked/unmasked random matrix.

Design constraint (spec section 15): this module imports nothing but NumPy /
SciPy — no models, no network, no database.  It maps matrices to indices and
is testable in milliseconds.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:  # SciPy only needed for the `unordered` mode (Hungarian assignment)
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - environment without scipy
    _HAS_SCIPY = False

NEG = -1e9
_NEG_THRESHOLD = NEG / 2  # anything below this is "invalid / masked"

AlignResult = Tuple[float, Optional[List[int]]]


def _earliness_prior(
    mu: float,
    s_v: int,
    L: int,
    prior_span: Optional[Tuple[int, int]],
) -> np.ndarray:
    """Ramp subtracted from the first event's row (spec section 6.1).

    ``prior_span`` is ``(origin, extent)`` — the range the prior is measured
    against, which is NOT always the range being searched.  When a caller
    scans a sub-range (``strict_window_align``), measuring earliness from the
    sub-range start would restart the ramp at every window and cancel the
    prior entirely; passing the whole video keeps candidates from different
    windows on one comparable scale.
    """
    idx = np.arange(L, dtype=np.float64)
    if prior_span is None:
        return mu * (idx / max(L - 1, 1))
    origin, extent = prior_span
    return mu * ((idx + s_v - origin) / max(extent, 1))


def _validate_inputs(S: np.ndarray, s_v: int, e_v: int, g: int) -> int:
    if S.ndim != 2:
        raise ValueError(f"S must be 2-D (N, T), got shape {S.shape}")
    T = S.shape[1]
    if not (0 <= s_v <= e_v < T):
        raise ValueError(f"invalid video range [{s_v}, {e_v}] for T={T}")
    # g < 1 would allow the same keyframe for two consecutive events, which
    # violates the required strict ordering f1 < f2 < ... < fN.
    return max(1, int(g))


def chronos_align_ref(
    S: np.ndarray,
    s_v: int,
    e_v: int,
    lam: float,
    g: int = 2,
    mu: float = 0.0,
    prior_span: Optional[Tuple[int, int]] = None,
) -> AlignResult:
    """Reference (readable, loop-based) implementation of the DP.

    Kept as the ground truth for regression tests; use :func:`chronos_align`
    (vectorized, identical output) in production.

    Parameters
    ----------
    S    : (N, T) score matrix, z-scored per row.
    s_v  : first global keyframe index of the video (inclusive).
    e_v  : last  global keyframe index of the video (inclusive).
    lam  : temporal penalty (>= 0).
    g    : minimum gap (in keyframes) between consecutive events, >= 1.
    mu   : earliness prior; 0.0 disables it.
    prior_span : optional ``(origin, extent)`` the prior is measured against;
           defaults to the searched range.  See :func:`_earliness_prior`.

    Returns
    -------
    (score, [gid_1..gid_N]) or (NEG, None) if the video is too short / masked.
    """
    g = _validate_inputs(S, s_v, e_v, g)
    N = S.shape[0]
    L = e_v - s_v + 1
    if L < (N - 1) * g + 1:
        return NEG, None

    Sv = S[:, s_v : e_v + 1].astype(np.float64)

    dp = np.full((N, L), NEG, dtype=np.float64)
    back = np.full((N, L), -1, dtype=np.int64)

    # --- base case: first event, with earliness prior -----------------------
    dp[0] = Sv[0] - _earliness_prior(mu, s_v, L, prior_span)

    # --- recurrence with running max ---------------------------------------
    for i in range(1, N):
        run_val = -np.inf
        run_arg = -1
        for t in range(L):
            tau = t - g  # the predecessor index newly unlocked at time t
            if tau >= 0 and dp[i - 1, tau] > _NEG_THRESHOLD:
                cand = dp[i - 1, tau] + lam * tau
                if cand > run_val:  # STRICT '>' => earliest tie-break
                    run_val, run_arg = cand, tau
            if run_arg >= 0:
                dp[i, t] = Sv[i, t] - lam * t + run_val
                back[i, t] = run_arg

    # --- pick the end point, earliest among ties ----------------------------
    last = dp[N - 1]
    best = float(last.max())
    if best <= _NEG_THRESHOLD:
        return NEG, None
    t_end = int(np.argmax(last))  # np.argmax returns the FIRST maximum

    # --- backtrack ----------------------------------------------------------
    path = [t_end]
    t = t_end
    for i in range(N - 1, 0, -1):
        t = int(back[i, t])
        path.append(t)
    path.reverse()
    return best, [p + s_v for p in path]


def chronos_align(
    S: np.ndarray,
    s_v: int,
    e_v: int,
    lam: float,
    g: int = 2,
    mu: float = 0.0,
    prior_span: Optional[Tuple[int, int]] = None,
) -> AlignResult:
    """Vectorized DP, output-identical to :func:`chronos_align_ref`.

    The inner loop over t is replaced by ``np.maximum.accumulate`` on
    ``prefix[tau] = DP[i-1, tau] + lam * tau``; the earliest-argmax is
    recovered by forward-filling the positions where the prefix strictly
    improves on its running maximum (exactly mirroring the reference's
    strict-'>' update rule).
    """
    g = _validate_inputs(S, s_v, e_v, g)
    N = S.shape[0]
    L = e_v - s_v + 1
    if L < (N - 1) * g + 1:
        return NEG, None

    Sv = S[:, s_v : e_v + 1].astype(np.float64)
    idx = np.arange(L, dtype=np.float64)

    back = np.full((N, L), -1, dtype=np.int64)

    dp_prev = Sv[0] - _earliness_prior(mu, s_v, L, prior_span)

    for i in range(1, N):
        prefix = np.where(dp_prev > _NEG_THRESHOLD, dp_prev + lam * idx, -np.inf)

        cummax = np.maximum.accumulate(prefix)
        prev_max = np.empty(L, dtype=np.float64)
        prev_max[0] = -np.inf
        prev_max[1:] = cummax[:-1]
        is_new = prefix > prev_max  # strict improvement == earliest tie-break
        run_arg = np.maximum.accumulate(np.where(is_new, np.arange(L), -1))

        dp = np.full(L, NEG, dtype=np.float64)
        bk = np.full(L, -1, dtype=np.int64)
        if L > g:
            M = cummax[: L - g]  # running max over tau in [0, t-g]
            A = run_arg[: L - g]
            ok = np.isfinite(M) & (A >= 0)
            vals = Sv[i, g:] - lam * idx[g:] + M
            dp[g:] = np.where(ok, vals, NEG)
            bk[g:] = np.where(ok, A, -1)
        back[i] = bk
        dp_prev = dp

    best = float(dp_prev.max())
    if best <= _NEG_THRESHOLD:
        return NEG, None
    t_end = int(np.argmax(dp_prev))

    path = [t_end]
    t = t_end
    for i in range(N - 1, 0, -1):
        t = int(back[i, t])
        path.append(t)
    path.reverse()
    return best, [p + s_v for p in path]


def estimate_tempo(
    S: np.ndarray,
    videos: Optional[Dict[str, dict]] = None,
    quantile: float = 0.5,
    default_gap: float = 8.0,
    top_videos: int = 5,
) -> float:
    """Estimate the typical keyframe gap between consecutive events.

    Gap statistics are collected inside the strongest candidate videos
    (ranked by the per-video sum of row maxima) rather than from one global
    argmax, which is far more robust on a large corpus.  ``top_videos`` stays
    small: only a handful of candidates actually contain the event chain, and
    videos without it contribute arbitrary noise gaps.

    Returns ``default_gap`` when there is nothing to measure (N < 2, no
    positive gaps) instead of producing NaN as the spec draft did.
    """
    N = S.shape[0]
    if N < 2:
        return default_gap

    gaps: List[float] = []
    if videos:
        strength = []
        for meta in videos.values():
            sl = S[:, meta["s_v"] : meta["e_v"] + 1]
            if sl.shape[1] == 0:
                continue
            strength.append((float(sl.max(axis=1).sum()), meta["s_v"], meta["e_v"]))
        strength.sort(key=lambda r: -r[0])
        for _, s_v, e_v in strength[:top_videos]:
            peaks = np.sort(np.argmax(S[:, s_v : e_v + 1], axis=1))
            d = np.diff(peaks)
            gaps.extend(d[d > 0].tolist())
    else:
        peaks = np.sort([int(np.argmax(S[i])) for i in range(N)])
        d = np.diff(peaks)
        gaps.extend(d[d > 0].tolist())

    return float(np.quantile(gaps, quantile)) if gaps else default_gap


def estimate_tempo_dp(
    S: np.ndarray,
    videos: Dict[str, dict],
    probe_lam_rel: float = 0.005,
    g: int = 2,
    top_videos: int = 3,
    default_gap: float = 8.0,
    quantile: float = 0.5,
) -> float:
    """Estimate the event tempo from a first-pass ALIGNMENT (two-pass calibration).

    :func:`estimate_tempo` reads the gaps between independent per-event
    argmaxes, which is fragile: a single event whose global maximum lands on a
    decoy elsewhere in the video inflates the apparent tempo.  Measured on the
    internal benchmark, that estimator over-estimated tight-tempo queries by
    4x (true median gap 5.5 keyframes, estimated 22.0).

    Running the DP once with a neutral probe penalty and measuring the gaps of
    the chain it returns is self-consistent — the probe alignment already
    obeys ordering and minimum gap, so its frames are drawn from one coherent
    chain rather than from N unrelated maxima.  The same measurement drops the
    median |log ratio| error from 0.255 to 0.005.

    Cost is ``top_videos`` single-video DP runs, negligible beside the
    corpus-wide scan that follows.
    """
    if S.shape[0] < 2 or not videos:
        return default_gap
    sigma = score_scale(S)
    strength = []
    for meta in videos.values():
        sl = S[:, meta["s_v"] : meta["e_v"] + 1]
        if sl.shape[1] == 0:
            continue
        strength.append((float(sl.max(axis=1).sum()), int(meta["s_v"]), int(meta["e_v"])))
    strength.sort(key=lambda r: -r[0])

    gaps: List[float] = []
    for _, s_v, e_v in strength[:top_videos]:
        _, path = chronos_align(S, s_v, e_v, lam=probe_lam_rel * sigma, g=g)
        if path and len(path) > 1:
            gaps.extend(np.diff(path).tolist())
    return float(np.quantile(gaps, quantile)) if gaps else default_gap


def score_scale(S: np.ndarray, max_samples: int = 200_000) -> float:
    """Typical magnitude of one unit of score, ignoring NEG-masked entries.

    Equals ~1.0 for a z-scored matrix and ~0.1 for raw cosine similarities.
    Anchored rows are almost entirely NEG and must not inflate the estimate.

    Long rows are subsampled down to ``max_samples`` values: a standard
    deviation does not need 1.2M samples, and this call sits on the per-query
    hot path.  Sampling uses evenly spaced indices rather than a fixed stride
    so the estimate cannot phase-lock onto a periodic keyframe structure —
    at the design size the natural stride would be 6, uncomfortably close to
    the mandated 4+ keyframes per shot (R3), and a fixed phase would then
    always land on the same position within every shot.
    """
    T = S.shape[1]
    if T > max_samples:
        cols = np.linspace(0, T - 1, max_samples).astype(np.int64)
        cols = np.unique(cols)
    else:
        cols = slice(None)
    scales = []
    for i in range(S.shape[0]):
        row = S[i, cols]
        row = row[row > _NEG_THRESHOLD]
        if row.size > 1:
            scales.append(float(row.std()))
    if not scales:
        return 1.0
    med = float(np.median(scales))
    return med if med > 1e-9 else 1.0


def adaptive_lambda(
    S: np.ndarray,
    videos: Optional[Dict[str, dict]] = None,
    kappa: float = 0.5,
    quantile: float = 0.5,
    lo: float = 0.005,
    hi: float = 0.2,
    default_gap: float = 8.0,
    top_videos: int = 5,
    two_pass: bool = True,
    g: int = 2,
    probe_lam_rel: float = 0.005,
) -> float:
    """Estimate the event tempo and derive lambda (spec section 6.3).

    ``lambda = clip(kappa / median_gap, lo, hi) * score_scale(S)``

    A small median gap between event peaks yields a strong penalty (the DP
    should keep the chain tight); a large gap yields a weak one.

    **The penalty must be expressed in the score matrix's own units.**  The
    spec's constant (``0.02 / gap``) and bounds ``[0.0005, 0.02]`` were
    calibrated on raw cosine similarities, but CHRONOS feeds the DP a
    per-row z-scored matrix — also spec-mandated — whose values are roughly an
    order of magnitude larger.  Applying the raw-cosine constant there makes
    the penalty effectively negligible: on the internal benchmark it produced
    lambda ~ 0.0006 and 70% Sequence Accuracy, against 84% for a fixed
    lambda = 0.01, i.e. the "adaptive" mechanism was actively harmful.
    Multiplying by :func:`score_scale` makes the same formula correct at both
    scales, and ``lo``/``hi`` are consequently expressed in units of score
    standard deviation per keyframe.

    ``kappa`` is the tempo constant: with a z-scored matrix a chain whose
    events sit ``med`` keyframes apart pays about ``kappa`` per hop, so the
    penalty stays a fixed fraction of a one-sigma score difference regardless
    of tempo.  ``bench/sweep.py`` measures a flat optimum over kappa in
    [0.4, 0.6] on the internal benchmark (0.05-0.15 are pinned by the lower
    clip bound, 1.0 starts damaging loose-tempo queries); the default sits
    mid-plateau.  Re-run that sweep when moving to the real corpus.

    ``two_pass`` selects :func:`estimate_tempo_dp` (a probe alignment, far
    more accurate) over the cheap argmax-gap estimator; set it to False only
    when no video ranges are available or the extra probe is unaffordable.
    ``quantile``, ``top_videos`` and ``g`` are honoured by both estimators —
    ``g`` in particular must match the minimum gap the real search will use,
    or the probe reports a tempo the search cannot actually achieve.

    ``probe_lam_rel`` is the probe's own penalty and is deliberately NOT tied
    to ``lo``: raising the clip floor would otherwise also tighten the probe,
    which reports a shorter tempo, which raises lambda again — positive
    feedback that pushes the estimate away from the measured optimum.
    """
    if two_pass and videos:
        med = estimate_tempo_dp(
            S,
            videos,
            probe_lam_rel=probe_lam_rel,
            g=g,
            top_videos=top_videos,
            default_gap=default_gap,
            quantile=quantile,
        )
    else:
        med = estimate_tempo(S, videos, quantile, default_gap, top_videos)
    rel = float(np.clip(kappa / max(med, 1.0), lo, hi))
    return rel * score_scale(S)


def apply_anchors(S: np.ndarray, anchors: Dict[int, int], anchor_score: float = 10.0) -> np.ndarray:
    """AIL — Anchor Lock (spec section 6.6).

    ``anchors`` maps event index (0-based) -> global keyframe id.  The DP is
    forced through those exact points: every other keyframe of an anchored
    event is masked to NEG.  Videos not containing the anchored gid become
    infeasible for that event, which collapses the search space.

    Returns a modified COPY of S; the original is untouched.
    """
    N, T = S.shape
    out = S.copy()
    for i, gid in anchors.items():
        i, gid = int(i), int(gid)
        if not 0 <= i < N:
            raise ValueError(f"anchor event index {i} out of range [0, {N})")
        if not 0 <= gid < T:
            raise ValueError(f"anchor gid {gid} out of range [0, {T})")
        out[i, :] = NEG
        out[i, gid] = anchor_score
    return out


# ---------------------------------------------------------------------------
# Extended alignment modes (spec section 6.7)
# ---------------------------------------------------------------------------


def strict_window_align(
    S: np.ndarray,
    s_v: int,
    e_v: int,
    lam: float,
    window: int,
    g: int = 2,
    mu: float = 0.0,
) -> AlignResult:
    """`strict_window` mode: additionally require ``t_N - t_1 <= window``.

    Implemented as sliding ordered-DP sub-ranges ``[a, a + window]`` (length
    ``window + 1`` keyframes, i.e. maximum span exactly ``window``) with
    stride ``window // 4`` — dense enough that any solution of span
    <= 3/4 * window is fully covered by at least one window; a mild
    approximation near window edges.

    Every window is aligned with the earliness prior measured against the
    WHOLE VIDEO, not against the window.  Measuring it per window would reset
    the ramp to zero at each window start, so every candidate chain could be
    evaluated in a window beginning at its own first frame and would pay no
    prior at all — mu would be silently inert in this mode, and scores from
    different windows would not be comparable.
    """
    L = e_v - s_v + 1
    span = (s_v, max(L - 1, 1))
    if window >= L - 1:  # the whole video already satisfies the span bound
        return chronos_align(S, s_v, e_v, lam, g, mu, prior_span=span)

    # Exactness shortcut: if the UNCONSTRAINED optimum already satisfies the
    # span bound it is provably the constrained optimum too (the constrained
    # optimum can never exceed the unconstrained one).  This also removes the
    # stride's blind spot for spans in (3W/4, W] whenever the global optimum
    # is the answer.
    sc_free, path_free = chronos_align(S, s_v, e_v, lam, g, mu, prior_span=span)
    if path_free is not None and path_free[-1] - path_free[0] <= window:
        return sc_free, path_free

    stride = max(1, window // 4)
    starts = list(range(s_v, e_v - window + 1, stride))
    if not starts or starts[-1] < e_v - window:
        starts.append(e_v - window)
    best: AlignResult = (NEG, None)
    for a in starts:
        b = min(a + window, e_v)  # inclusive: span within [a, b] is <= window
        sc, path = chronos_align(S, a, b, lam, g, mu, prior_span=span)
        if path is not None and sc > best[0]:  # strict '>' keeps earliest window
            best = (sc, path)
    return best


def unordered_align(
    S: np.ndarray,
    s_v: int,
    e_v: int,
    window: Optional[int] = None,
    mu: float = 0.0,
) -> Tuple[float, Optional[List[int]], Optional[Tuple[int, ...]]]:
    """`unordered` mode: Hungarian assignment of events to keyframes.

    Used when the query lists events without a stated order (risk R5).
    Assigns each event to a distinct keyframe maximizing the total score,
    optionally inside a sliding window.  Minimum-gap is NOT enforced in this
    mode, and neither is the temporal penalty ``lambda`` — there is no
    "consecutive pair" to penalise once the order is unknown.

    ``window`` is a SPAN bound, matching :func:`strict_window_align` and the
    ``chronos_search`` docstring: an assignment may span at most ``window``
    keyframes.

    ``mu`` applies the earliness prior to every assigned frame rather than to
    a distinguished first event, since no event is distinguished here — but
    divided by N, so the worst case (every event at the very end of the video)
    still costs exactly ``mu``, matching the contract ``chronos_search``
    documents and the value the sweep calibrated.  Being a per-column
    constant, it leaves the assignment structure intact.

    Returns ``(score, gids, perm)``: ``gids`` sorted ascending
    (chronological) and ``perm[k]`` = original event index matched at
    position k of that chronological sequence.  An assignment touching any
    masked (NEG) cell — e.g. a video that does not contain an anchored gid —
    is rejected as infeasible, matching ordered-mode semantics.
    """
    if not _HAS_SCIPY:  # pragma: no cover
        raise RuntimeError("scipy is required for unordered_align")
    N = S.shape[0]
    L = e_v - s_v + 1
    if L < N:
        return NEG, None, None
    Sv = S[:, s_v : e_v + 1].astype(np.float64)
    if mu:
        # divided by N: the assignment pays one ramp per event, so without
        # this the worst case would cost N*mu instead of the documented mu
        Sv = Sv - (mu / N) * (np.arange(L, dtype=np.float64) / max(L - 1, 1))

    def _assign(a: int, b: int):
        """Best assignment inside the inclusive range [a, b], or None."""
        Wv = Sv[:, a : b + 1]
        if Wv.shape[1] < N:
            return None
        rows, cols = linear_sum_assignment(-Wv)
        # infeasible if any assigned cell is masked (a single NEG entry can
        # never be offset by z-score/anchor-scale values)
        if np.any(Wv[rows, cols] <= _NEG_THRESHOLD):
            return None
        pairs = sorted(
            ((int(c) + a + s_v, int(r)) for r, c in zip(rows, cols)), key=lambda p: p[0]
        )
        return (
            float(Wv[rows, cols].sum()),
            [g_ for g_, _ in pairs],
            tuple(r_ for _, r_ in pairs),
        )

    if window is None or window >= L - 1:
        return _assign(0, L - 1) or (NEG, None, None)

    # Exactness shortcut, as in strict_window_align: an unconstrained optimum
    # that already fits the span IS the constrained optimum.
    free = _assign(0, L - 1)
    if free and free[1][-1] - free[1][0] <= window:
        return free

    # stride window//4 (matching strict_window_align) covers every span up to
    # 3W/4; window//2 left spans in (W/2, W] findable only by lucky alignment
    stride = max(1, window // 4)
    starts = list(range(0, L - window, stride))
    if not starts or starts[-1] < L - 1 - window:
        starts.append(L - 1 - window)

    best_score, best_gids, best_perm = NEG, None, None
    for a in starts:
        found = _assign(a, min(a + window, L - 1))  # inclusive: span <= window
        if found and found[0] > best_score:
            best_score, best_gids, best_perm = found
    return best_score, best_gids, best_perm


def soft_order_align(
    S: np.ndarray,
    s_v: int,
    e_v: int,
    lam: float,
    g: int = 2,
    mu: float = 0.0,
    swap_penalty: float = 0.5,
) -> Tuple[float, Optional[List[int]], Tuple[int, ...]]:
    """`soft_order` mode: allow one adjacent transposition at a fixed penalty.

    Tries the identity order plus every single adjacent swap of event rows,
    runs the ordered DP on each, and returns the best penalized score.

    Returns ``(score, gids, perm)`` where ``perm[k]`` is the original event
    index matched at position k of the (chronologically increasing) path.

    The row permutation is applied to the VIDEO SLICE, not to the full-width
    matrix: ``S[list(perm)]`` copies all T columns, and this function runs N
    permutations per video across V videos, so permuting first would memcpy
    the whole corpus O(V * N^2) times per query while the DP only ever reads
    one video's range.
    """
    N = S.shape[0]
    identity = tuple(range(N))
    candidates: List[Tuple[Tuple[int, ...], int]] = [(identity, 0)]
    for k in range(N - 1):
        p = list(identity)
        p[k], p[k + 1] = p[k + 1], p[k]
        candidates.append((tuple(p), 1))

    Sv = S[:, s_v : e_v + 1]
    L = Sv.shape[1]
    best_score, best_path, best_perm = NEG, None, identity
    for perm, n_swaps in candidates:
        # align within the slice, then shift the local path back to global ids
        sc, path = chronos_align(
            Sv[list(perm)], 0, L - 1, lam, g, mu, prior_span=(0, max(L - 1, 1))
        )
        if path is None:
            continue
        sc -= swap_penalty * n_swaps
        if sc > best_score:
            best_score = sc
            best_path = [p + s_v for p in path]
            best_perm = perm
    return best_score, best_path, best_perm


# ---------------------------------------------------------------------------
# Corpus-wide search
# ---------------------------------------------------------------------------


def chronos_search(
    S: np.ndarray,
    videos: Dict[str, dict],
    lam: Optional[float] = None,
    g: int = 2,
    mu: float = 0.0,
    topk: int = 20,
    align_mode: str = "ordered",
    window: Optional[int] = None,
    swap_penalty: float = 0.5,
    anchors: Optional[Dict[int, int]] = None,
    video_ids: Optional[Iterable[str]] = None,
    use_ref: bool = False,
) -> Tuple[List[dict], float]:
    """Scan the whole corpus, return top-k videos with their frame sequences.

    Parameters
    ----------
    S          : (N, T) fused, row-z-scored score matrix over ALL keyframes.
    videos     : {video_id: {"s_v": int, "e_v": int, ...}} — global ranges.
    lam        : temporal penalty in ABSOLUTE score units; None -> derived by
                 :func:`adaptive_lambda` (which is already scale-aware).
    mu         : earliness prior in units of SCORE STANDARD DEVIATION — the
                 maximum penalty, applied to a chain starting at the very end
                 of the video, is ``mu`` sigma.  Converted to absolute units
                 here via :func:`score_scale`, for the same reason lambda is:
                 the mandated per-row z-scoring changes the score magnitude by
                 about an order of magnitude, so a raw-cosine-calibrated
                 constant would be silently inert.
    align_mode : "ordered" | "unordered" | "strict_window" | "soft_order".
    window     : span constraint (keyframes) for strict_window / unordered.
    anchors    : {event_idx: gid} hard constraints (AIL).
    video_ids  : optional iterable restricting the scan (candidate prefilter).
    use_ref    : run the loop reference implementation (debugging only).

    Returns ``(results, lambda_used)`` where each result is
    ``{"video_id", "score", "gids"}`` — plus ``"perm"`` for the order-relaxing
    modes (``soft_order``, ``unordered``), where ``perm[k]`` is the original
    event index matched at chronological position k.
    """
    if anchors:
        S = apply_anchors(S, anchors)
    if lam is None:
        lam = adaptive_lambda(S, videos, g=g)
    if mu:
        mu = mu * score_scale(S)

    if video_ids is not None:
        allowed = set(video_ids)
        items = [(v, m) for v, m in videos.items() if v in allowed]
    else:
        items = list(videos.items())

    align = chronos_align_ref if use_ref else chronos_align

    out: List[dict] = []
    for vid, meta in items:
        s_v, e_v = int(meta["s_v"]), int(meta["e_v"])
        perm: Optional[Tuple[int, ...]] = None
        if align_mode == "ordered":
            sc, path = align(S, s_v, e_v, lam, g, mu)
        elif align_mode == "strict_window":
            if window is None:
                raise ValueError("strict_window mode requires `window`")
            sc, path = strict_window_align(S, s_v, e_v, lam, window, g, mu)
        elif align_mode == "unordered":
            sc, path, perm = unordered_align(S, s_v, e_v, window, mu)
        elif align_mode == "soft_order":
            sc, path, perm = soft_order_align(S, s_v, e_v, lam, g, mu, swap_penalty)
        else:
            raise ValueError(f"unknown align_mode: {align_mode!r}")
        if path is not None:
            rec = {"video_id": vid, "score": sc, "gids": path}
            if perm is not None:
                rec["perm"] = list(perm)
            out.append(rec)

    out.sort(key=lambda r: -r["score"])
    return out[:topk], lam
