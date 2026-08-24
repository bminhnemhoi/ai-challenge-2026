"""Tests for L3 — TAP (the DP core).

The most safety-critical behaviors per the spec:
* earliest tie-break (section 6.4) — strict '>' semantics;
* min-gap enforcement and strict temporal ordering;
* vectorized implementation identical to the loop reference (section 6.5 note);
* anchor lock forcing the path (section 6.6);
* extended modes (section 6.7).
"""

import numpy as np
import pytest

from src.task3_trake.alignment import (
    NEG,
    adaptive_lambda,
    apply_anchors,
    chronos_align,
    chronos_align_ref,
    chronos_search,
    estimate_tempo,
    score_scale,
    soft_order_align,
    strict_window_align,
    unordered_align,
)

BOTH = [chronos_align, chronos_align_ref]


def make_S(n, t, base=0.0):
    return np.full((n, t), base, dtype=np.float64)


# ---------------------------------------------------------------- basic DP


@pytest.mark.parametrize("align", BOTH)
def test_simple_two_peaks(align):
    S = make_S(2, 10)
    S[0, 2] = 5.0
    S[1, 7] = 4.0
    score, path = align(S, 0, 9, lam=0.01, g=2)
    assert path == [2, 7]
    assert score == pytest.approx(5.0 + 4.0 - 0.01 * (7 - 2), abs=1e-9)


@pytest.mark.parametrize("align", BOTH)
def test_video_offset(align):
    """Global gids must be returned relative to the whole corpus, not the video."""
    S = make_S(2, 30)
    S[0, 12] = 3.0
    S[1, 20] = 3.0
    score, path = align(S, 10, 25, lam=0.0, g=2)
    assert path == [12, 20]


@pytest.mark.parametrize("align", BOTH)
def test_ordering_constraint_rejects_reversed(align):
    """Best peaks in reversed order must not be selected out of order."""
    S = make_S(2, 12, base=0.0)
    S[0, 8] = 9.0  # event 1 late
    S[1, 3] = 9.0  # event 2 early — the pair (8, 3) is illegal
    S[0, 1] = 2.0
    S[1, 4] = 2.0
    _, path = align(S, 0, 11, lam=0.0, g=2)
    assert path is not None
    assert path[0] < path[1]
    # best LEGAL combination: event1 at its small peak (1), event2 at its big one (3)
    assert path == [1, 3]


@pytest.mark.parametrize("align", BOTH)
def test_min_gap_enforced(align):
    S = make_S(2, 10)
    S[0, 4] = 5.0
    S[1, 5] = 5.0  # adjacent — illegal with g=2
    S[1, 8] = 1.0
    _, path = align(S, 0, 9, lam=0.0, g=2)
    assert path is not None
    assert path[1] - path[0] >= 2


@pytest.mark.parametrize("align", BOTH)
def test_infeasible_video(align):
    S = make_S(4, 8)
    score, path = align(S, 0, 4, lam=0.01, g=2)  # L=5 < (4-1)*2+1=7
    assert score == NEG and path is None


@pytest.mark.parametrize("align", BOTH)
def test_single_event(align):
    S = make_S(1, 10)
    S[0, 3] = 2.0
    S[0, 6] = 2.0  # equal later peak — earliest must win
    score, path = align(S, 0, 9, lam=0.01, g=2)
    assert path == [3]
    assert score == pytest.approx(2.0)


# ----------------------------------------------------------- tie-breaking


@pytest.mark.parametrize("align", BOTH)
def test_tiebreak_earliest_predecessor(align):
    """With lam=0, equal-score predecessors: the EARLIEST tau must be chosen."""
    S = make_S(2, 12)
    S[0, 2] = 3.0
    S[0, 5] = 3.0  # same value, later
    S[1, 9] = 4.0
    _, path = align(S, 0, 11, lam=0.0, g=2)
    assert path == [2, 9]


@pytest.mark.parametrize("align", BOTH)
def test_tiebreak_earliest_end(align):
    """Two identical full sequences: the earlier one must be returned even with mu=0."""
    S = make_S(2, 60)
    for start in (2, 40):
        S[0, start] = 3.0
        S[1, start + 4] = 3.0
    _, path = align(S, 0, 59, lam=0.0, g=2, mu=0.0)
    assert path == [2, 6]


@pytest.mark.parametrize("align", BOTH)
def test_bit_identical_scores_pick_earliest(align):
    """The tie that actually happens in video: a static shot of equal frames.

    Documents why the spec's 1e-9 epsilon is unnecessary here — real ties are
    bit-identical, which exact '>' already resolves to the earliest frame.
    """
    S = make_S(2, 40)
    S[0, 10:15] = 3.0  # five identical keyframes (static shot)
    S[1, 25:30] = 3.0
    _, path = align(S, 0, 39, lam=0.0, g=2)
    assert path == [10, 25]


@pytest.mark.parametrize("align", BOTH)
def test_positive_lambda_prefers_compact_pair(align):
    """With lam>0 ties in raw score resolve to the tighter (later-start) pair."""
    S = make_S(2, 30)
    S[0, 2] = 3.0
    S[0, 20] = 3.0
    S[1, 24] = 3.0
    _, path = align(S, 0, 29, lam=0.05, g=2)
    assert path == [20, 24]  # gap 4 beats gap 22


@pytest.mark.parametrize("align", BOTH)
def test_earliness_prior_flips_choice(align):
    """mu strictly prefers the early sequence when the late one is slightly better.

    lam > 0 is needed so the chain stays cohesive (mu only penalizes t_1;
    with lam=0 the DP could legally mix an early t_1 with a late t_2).
    """
    S = make_S(2, 100)
    S[0, 5] = 3.00
    S[1, 10] = 3.00
    S[0, 80] = 3.001  # marginally better but late
    S[1, 85] = 3.001
    _, path_no_mu = align(S, 0, 99, lam=0.001, g=2, mu=0.0)
    assert path_no_mu == [80, 85]
    _, path_mu = align(S, 0, 99, lam=0.001, g=2, mu=0.05)
    assert path_mu == [5, 10]


# ------------------------------------------------------------------ anchors


def test_anchor_forces_path():
    S = make_S(2, 20)
    S[0, 3] = 5.0
    S[0, 10] = 4.0
    S[1, 15] = 5.0
    anchored = apply_anchors(S, {0: 10})
    _, path = chronos_align(anchored, 0, 19, lam=0.0, g=2)
    assert path[0] == 10
    # original S untouched
    assert S[0, 3] == 5.0


def test_anchor_excludes_other_videos():
    S = make_S(2, 40)
    S[0, 5] = 9.0
    S[1, 10] = 9.0  # strong sequence in video A [0..19]
    S[0, 25] = 1.0
    S[1, 30] = 1.0  # weak sequence in video B [20..39]
    videos = {"A": {"s_v": 0, "e_v": 19}, "B": {"s_v": 20, "e_v": 39}}
    results, _ = chronos_search(S, videos, lam=0.0, anchors={0: 25})
    assert results[0]["video_id"] == "B"
    assert results[0]["gids"][0] == 25
    # video A must be infeasible for event 0 (its row is NEG there)
    assert all(r["video_id"] != "A" for r in results)


def test_anchor_validation():
    S = make_S(2, 10)
    with pytest.raises(ValueError):
        apply_anchors(S, {5: 3})
    with pytest.raises(ValueError):
        apply_anchors(S, {0: 99})


# ------------------------------------------------- ref vs vectorized fuzz


def test_fuzz_equivalence_ref_vs_vectorized():
    rng = np.random.default_rng(7)
    for trial in range(120):
        N = int(rng.integers(1, 6))
        L = int(rng.integers(max(2, (N - 1) * 3 + 1), 90))
        S = rng.standard_normal((N, L))
        # random NEG masking (simulates anchors) on ~5% of entries
        mask = rng.random((N, L)) < 0.05
        S[mask] = NEG
        lam = float(rng.uniform(0.0, 0.05))
        g = int(rng.integers(1, 4))
        mu = float(rng.choice([0.0, 0.05]))
        s1, p1 = chronos_align_ref(S, 0, L - 1, lam, g, mu)
        s2, p2 = chronos_align(S, 0, L - 1, lam, g, mu)
        assert p1 == p2, f"trial {trial}: paths differ {p1} vs {p2}"
        assert s1 == pytest.approx(s2, abs=1e-9), f"trial {trial}"


def test_fuzz_path_properties():
    rng = np.random.default_rng(11)
    for _ in range(60):
        N = int(rng.integers(2, 6))
        L = int(rng.integers((N - 1) * 4 + 2, 120))
        S = rng.standard_normal((N, L))
        g = int(rng.integers(1, 4))
        _, path = chronos_align(S, 0, L - 1, float(rng.uniform(0, 0.03)), g)
        assert path is not None and len(path) == N
        for a, b in zip(path, path[1:]):
            assert b - a >= g


# ------------------------------------------------------------ adaptive lam


def test_adaptive_lambda_bounds_and_nan_safety():
    rng = np.random.default_rng(3)
    S = rng.standard_normal((4, 200))  # unit-scale, like a z-scored matrix
    sigma = score_scale(S)
    lam = adaptive_lambda(S)
    # bounds are relative to the score scale, so compare in those units
    assert 0.005 * sigma <= lam <= 0.2 * sigma

    # all rows peak at the same index -> all gaps zero -> must not be NaN
    S2 = np.zeros((3, 50))
    S2[:, 10] = 5.0
    lam2 = adaptive_lambda(S2)
    assert np.isfinite(lam2)

    # single event -> default gap fallback
    S3 = rng.standard_normal((1, 50))
    lam3 = adaptive_lambda(S3)
    assert np.isfinite(lam3) and 0.005 * score_scale(S3) <= lam3 <= 0.2 * score_scale(S3)


def test_adaptive_lambda_tempo_direction():
    """Tight event spacing must produce a larger lambda than loose spacing."""
    S_tight = np.zeros((3, 400))
    S_loose = np.zeros((3, 400))
    for i in range(3):
        S_tight[i, 100 + 2 * i] = 5.0
        S_loose[i, 50 + 100 * i] = 5.0
    assert adaptive_lambda(S_tight) > adaptive_lambda(S_loose)


def test_adaptive_lambda_scales_with_score_magnitude():
    """The penalty must follow the score matrix's own units.

    Regression: the spec's raw-cosine constant applied to a z-scored matrix
    produced a penalty ~10x too weak, making adaptive lambda worse than a
    hand-picked fixed one.
    """
    rng = np.random.default_rng(31)
    S = rng.standard_normal((3, 300))
    lam_unit = adaptive_lambda(S)
    lam_tenth = adaptive_lambda(0.1 * S)  # e.g. raw cosine instead of z-scores
    assert lam_tenth == pytest.approx(0.1 * lam_unit, rel=1e-6)


def test_score_scale_ignores_masked_rows():
    """An anchored row is almost entirely NEG and must not distort the scale."""
    rng = np.random.default_rng(32)
    S = rng.standard_normal((2, 500))
    anchored = apply_anchors(S, {0: 100})
    assert score_scale(anchored) == pytest.approx(score_scale(S[1:]), rel=0.3)
    assert np.isfinite(adaptive_lambda(anchored))


def test_adaptive_lambda_honours_min_gap():
    """The probe must run at the same min_gap as the real search.

    Regression: the probe defaulted to g=2 no matter what the caller asked
    for, so it reported a tempo the search could not achieve and lambda came
    out an order of magnitude too strong.
    """
    S = np.zeros((3, 600))
    for i in range(3):
        S[i, 100 + 4 * i] = 5.0  # a tight chain the probe can only follow at g<=4
    videos = {"A": {"s_v": 0, "e_v": 599}}
    lam_tight = adaptive_lambda(S, videos, g=2)
    lam_wide = adaptive_lambda(S, videos, g=25)
    assert lam_wide < lam_tight


def test_score_scale_not_phase_locked():
    """Sampling must not lock onto a periodic keyframe structure."""
    T, period = 600_000, 6
    S = np.ones((1, T))
    S[0, ::period] = 100.0  # a systematically different frame every `period`
    # a fixed stride equal to the period would sample only the outliers
    assert score_scale(S) < 60.0


def test_estimate_tempo_recovers_planted_gaps():
    S = np.zeros((4, 500))
    for i in range(4):
        S[i, 50 + 20 * i] = 5.0
    videos = {"A": {"s_v": 0, "e_v": 499}}
    assert estimate_tempo(S, videos) == pytest.approx(20.0)
    assert estimate_tempo(np.zeros((1, 50))) == pytest.approx(8.0)  # N<2 fallback


# ------------------------------------------------------------------ search


def _two_video_S():
    S = make_S(2, 60)
    videos = {"A": {"s_v": 0, "e_v": 29}, "B": {"s_v": 30, "e_v": 59}}
    S[0, 5] = 5.0
    S[1, 12] = 5.0  # strong in A
    S[0, 35] = 2.0
    S[1, 44] = 2.0  # weak in B
    return S, videos


def test_search_ranks_correct_video_first():
    S, videos = _two_video_S()
    results, lam = chronos_search(S, videos, lam=0.001)
    assert results[0]["video_id"] == "A"
    assert results[0]["gids"] == [5, 12]
    assert len(results) == 2
    assert results[0]["score"] > results[1]["score"]
    assert lam == 0.001


def test_search_adaptive_lambda_returned():
    S, videos = _two_video_S()
    _, lam = chronos_search(S, videos, lam=None)
    assert lam > 0 and np.isfinite(lam)


def test_search_video_filter():
    S, videos = _two_video_S()
    results, _ = chronos_search(S, videos, lam=0.001, video_ids=["B"])
    assert [r["video_id"] for r in results] == ["B"]


def test_search_topk():
    S, videos = _two_video_S()
    results, _ = chronos_search(S, videos, lam=0.001, topk=1)
    assert len(results) == 1


# ------------------------------------------------------------ extra modes


def test_strict_window_excludes_spread_solution():
    S = make_S(2, 200)
    S[0, 10] = 5.0
    S[1, 150] = 5.0  # high but spread over 140 frames
    S[0, 60] = 3.0
    S[1, 70] = 3.0  # compact alternative
    _, path = strict_window_align(S, 0, 199, lam=0.0, window=30, g=2)
    assert path == [60, 70]


def test_strict_window_span_exactly_equal_to_window_is_legal():
    """Spec 6.7 defines the constraint as t_N - t_1 <= W, inclusive.

    Regression: windows were built as [a, a+W-1], whose maximum reachable
    span is W-1, so a legal span-W optimum was unreachable at every stride.
    """
    S = make_S(2, 40)
    S[0, 0] = 10.0
    S[1, 10] = 10.0  # span exactly 10
    score, path = strict_window_align(S, 0, 39, lam=0.0, window=10, g=2)
    assert path == [0, 10]
    assert score == pytest.approx(20.0)


def test_strict_window_finds_optimum_missed_by_stride():
    """Spans in (3W/4, W] must still be found via the exactness shortcut."""
    S = make_S(2, 200)
    S[0, 5] = 8.0
    S[1, 41] = 8.0  # span 36 with W=40 -> stride 10 would skip it
    score, path = strict_window_align(S, 0, 199, lam=0.0, window=40, g=2)
    assert path == [5, 41]
    assert score == pytest.approx(16.0)


def test_strict_window_earliness_prior_is_video_relative():
    """mu must still prefer the earlier chain inside strict_window mode.

    Regression: the prior was measured from each sliding window's start, so
    every candidate could be scored in a window beginning at its own first
    frame and paid no prior at all — mu was silently inert in this mode.
    """
    S = make_S(2, 200)
    S[0, 10] = 3.0
    S[1, 15] = 3.0  # early chain
    S[0, 150] = 3.001
    S[1, 155] = 3.001  # late duplicate, marginally better
    S[0, 60] = 5.0
    S[1, 199] = 5.0  # wide decoy: defeats the exactness shortcut

    _, path_no_mu = strict_window_align(S, 0, 199, lam=0.001, window=20, g=2, mu=0.0)
    assert path_no_mu == [150, 155]
    _, path_mu = strict_window_align(S, 0, 199, lam=0.001, window=20, g=2, mu=1.0)
    assert path_mu == [10, 15]


def test_strict_window_scores_comparable_across_windows():
    """Two identical chains at different depths must not score identically.

    The window-relative prior also made scores incomparable BETWEEN windows,
    which corrupts the cross-video ranking inside one chronos_search.
    """
    S = make_S(2, 200)
    S[0, 5] = 4.0
    S[1, 10] = 4.0
    sc_early, _ = strict_window_align(S, 0, 199, lam=0.0, window=20, g=2, mu=2.0)
    S2 = make_S(2, 200)
    S2[0, 170] = 4.0
    S2[1, 175] = 4.0
    sc_late, _ = strict_window_align(S2, 0, 199, lam=0.0, window=20, g=2, mu=2.0)
    assert sc_early > sc_late


def test_strict_window_respects_bound_when_free_optimum_too_wide():
    """The shortcut must not leak an over-wide unconstrained solution."""
    S = make_S(2, 200)
    S[0, 10] = 9.0
    S[1, 190] = 9.0  # unconstrained winner, span 180
    S[0, 100] = 4.0
    S[1, 110] = 4.0  # legal alternative, span 10
    _, path = strict_window_align(S, 0, 199, lam=0.0, window=20, g=2)
    assert path is not None
    assert path[-1] - path[0] <= 20


def test_unordered_assigns_all_events():
    S = make_S(3, 50)
    # events described in the "wrong" order relative to time
    S[0, 40] = 5.0
    S[1, 10] = 5.0
    S[2, 25] = 5.0
    score, path, perm = unordered_align(S, 0, 49)
    assert path == [10, 25, 40]  # chronological output
    assert score == pytest.approx(15.0)
    # perm[k] = original event index at chronological position k
    assert perm == (1, 2, 0)


def test_unordered_rejects_anchor_violating_video():
    """A video lacking the anchored gid must be infeasible, as in ordered mode.

    Regression: the Hungarian assignment happily routed through NEG-masked
    cells, so every non-anchored video came back with a garbage path at score
    ~ -1e9 and padded the top-k list.
    """
    S = make_S(2, 40)
    S[0, 5] = 9.0
    S[1, 10] = 9.0  # strong sequence in video A [0..19]
    S[0, 25] = 1.0
    S[1, 30] = 1.0  # weak sequence in video B [20..39]
    videos = {"A": {"s_v": 0, "e_v": 19}, "B": {"s_v": 20, "e_v": 39}}
    results, _ = chronos_search(S, videos, lam=0.0, align_mode="unordered", anchors={0: 25})
    assert [r["video_id"] for r in results] == ["B"]
    assert results[0]["gids"][0] == 25
    assert results[0]["score"] > 0  # not a masked -1e9 sentinel


def test_unordered_window_is_a_span_like_strict_window():
    """Both modes take `window` as the span bound t_N - t_1 <= W (spec 6.7).

    Regression: unordered built windows of `window` KEYFRAMES, whose maximum
    span is W-1, so the two modes disagreed by one for the same input.
    """
    S = make_S(2, 40)
    S[0, 0] = 10.0
    S[1, 10] = 10.0  # span exactly 10
    score, path, _ = unordered_align(S, 0, 39, window=10)
    assert path == [0, 10]
    assert score == pytest.approx(20.0)


def test_unordered_earliness_prior_prefers_earlier_assignment():
    S = make_S(2, 200)
    S[0, 10] = 3.0
    S[1, 15] = 3.0
    S[0, 150] = 3.05
    S[1, 155] = 3.05  # slightly better but late
    _, path_no_mu, _ = unordered_align(S, 0, 199)
    assert path_no_mu == [150, 155]
    _, path_mu, _ = unordered_align(S, 0, 199, mu=1.0)
    assert path_mu == [10, 15]


def test_unordered_matches_brute_force_under_a_span_bound():
    """The windowed assignment must be optimal, not merely feasible.

    Regression: a stride of window//2 with no exactness shortcut left spans in
    (W/2, W] findable only by lucky alignment — 4.7% of random cases came back
    suboptimal.
    """
    import itertools

    rng = np.random.default_rng(11)
    for _ in range(120):
        N = int(rng.integers(2, 4))
        L = int(rng.integers(N + 2, 16))
        window = int(rng.integers(1, L))
        mu = float(rng.choice([0.0, 1.0]))
        S = rng.standard_normal((N, L))
        got, path, _ = unordered_align(S, 0, L - 1, window=window, mu=mu)

        ramp = (mu / N) * (np.arange(L) / max(L - 1, 1))
        best = NEG
        for cols in itertools.permutations(range(L), N):
            if max(cols) - min(cols) > window:
                continue
            best = max(best, sum(S[i, c] - ramp[c] for i, c in enumerate(cols)))
        if path is None:
            assert best <= NEG / 2, "reported infeasible but a legal assignment exists"
        else:
            assert got >= best - 1e-9, f"suboptimal: {got} < {best}"
            assert path[-1] - path[0] <= window


@pytest.mark.parametrize("window", [None, 0, 1, 2, 5, 20, 100])
@pytest.mark.parametrize("L,N", [(2, 2), (3, 2), (5, 3), (10, 3), (21, 2)])
def test_unordered_edge_cases_never_crash(L, N, window):
    """Degenerate window/length combinations must degrade, not raise."""
    S = np.random.default_rng(L * 100 + N).standard_normal((N, L))
    score, path, perm = unordered_align(S, 0, L - 1, window=window)
    if path is None:
        assert score == NEG
        return
    assert len(path) == N == len(set(path))
    assert path == sorted(path)
    assert perm is not None and sorted(perm) == list(range(N))
    if window is not None:
        assert path[-1] - path[0] <= window


def test_unordered_perm_reported_by_search():
    S = make_S(2, 30)
    S[0, 20] = 5.0
    S[1, 8] = 5.0
    videos = {"A": {"s_v": 0, "e_v": 29}}
    results, _ = chronos_search(S, videos, lam=0.0, align_mode="unordered")
    assert results[0]["gids"] == [8, 20]
    assert results[0]["perm"] == [1, 0]


def test_soft_order_recovers_adjacent_swap():
    S = make_S(2, 40)
    # true temporal order is event2 (t=10) then event1 (t=20)
    S[0, 20] = 5.0
    S[1, 10] = 5.0
    score, path, perm = soft_order_align(S, 0, 39, lam=0.0, g=2, swap_penalty=0.5)
    assert path == [10, 20]
    assert perm == (1, 0)
    assert score == pytest.approx(10.0 - 0.5)


def test_soft_order_slice_equals_full_matrix_permutation():
    """Permuting the video slice must equal permuting the whole matrix.

    soft_order_align permutes ``S[:, s_v:e_v+1]`` instead of ``S`` to avoid
    copying every column of a 1.2M-wide matrix N times per video.  That is
    only legitimate if the result is bit-identical, including the earliness
    prior (which is measured in local coordinates and shifted back) and
    NEG-masked anchors.
    """

    def reference(S, s_v, e_v, lam, g, mu, swap_penalty):
        N = S.shape[0]
        identity = tuple(range(N))
        cands = [(identity, 0)]
        for k in range(N - 1):
            p = list(identity)
            p[k], p[k + 1] = p[k + 1], p[k]
            cands.append((tuple(p), 1))
        best = (NEG, None, identity)
        for perm, n in cands:
            sc, path = chronos_align(S[list(perm)], s_v, e_v, lam, g, mu)
            if path is None:
                continue
            sc -= swap_penalty * n
            if sc > best[0]:
                best = (sc, path, perm)
        return best

    rng = np.random.default_rng(2026)
    for _ in range(60):
        N = int(rng.integers(2, 5))
        T = int(rng.integers(40, 200))
        S = rng.standard_normal((N, T))
        if rng.random() < 0.3:  # simulate anchors
            S[rng.integers(0, N), rng.random(T) < 0.9] = NEG
        s_v = int(rng.integers(0, T // 3))
        e_v = int(rng.integers(s_v + (N - 1) * 3 + 2, T))
        lam = float(rng.uniform(0, 0.05))
        g = int(rng.integers(1, 4))
        mu = float(rng.choice([0.0, 0.5, 2.0]))
        sp = float(rng.choice([0.0, 0.5]))

        got = soft_order_align(S, s_v, e_v, lam, g, mu, sp)
        want = reference(S, s_v, e_v, lam, g, mu, sp)
        assert got[1] == want[1]
        assert got[2] == want[2]
        if got[1] is not None:
            assert got[0] == pytest.approx(want[0], abs=1e-9)


def test_unordered_earliness_worst_case_costs_exactly_mu():
    """The prior is per-assignment, not per-event.

    Regression: subtracting the ramp from all N rows charged N*mu in the worst
    case, breaking the contract chronos_search documents and the value the
    sweep calibrated.
    """
    L, N, mu = 200, 4, 1.0

    def forced(positions):
        """Only `positions` are assignable, so the ramp cost is exact."""
        S = np.full((N, L), NEG)
        for i, p in enumerate(positions):
            S[i, p] = 5.0
        return S

    early = forced(range(N))  # chain pinned at the very start
    late = forced(range(L - N, L))  # chain pinned at the very end
    sc_early, path_early, _ = unordered_align(early, 0, L - 1, mu=mu)
    sc_late, path_late, _ = unordered_align(late, 0, L - 1, mu=mu)
    assert path_early == list(range(N))
    assert path_late == list(range(L - N, L))
    # worst case costs mu in total, NOT N * mu
    assert sc_early - sc_late == pytest.approx(mu, abs=0.05)


def test_soft_order_no_swap_when_ordered():
    S = make_S(2, 40)
    S[0, 10] = 5.0
    S[1, 20] = 5.0
    _, path, perm = soft_order_align(S, 0, 39, lam=0.0, g=2)
    assert path == [10, 20]
    assert perm == (0, 1)


def test_search_mode_dispatch():
    S, videos = _two_video_S()
    for mode, kwargs in [
        ("ordered", {}),
        ("unordered", {}),
        ("strict_window", {"window": 20}),
        ("soft_order", {}),
    ]:
        results, _ = chronos_search(S, videos, lam=0.001, align_mode=mode, **kwargs)
        assert results, mode
        assert results[0]["video_id"] == "A", mode
    with pytest.raises(ValueError):
        chronos_search(S, videos, lam=0.001, align_mode="bogus")


def test_search_strict_window_requires_window():
    S, videos = _two_video_S()
    with pytest.raises(ValueError):
        chronos_search(S, videos, lam=0.001, align_mode="strict_window")


# ------------------------------------------------------------- edge cases


@pytest.mark.parametrize("align", BOTH)
def test_gap_below_one_is_clamped(align):
    """g=0 would allow duplicate frames; it must be clamped to 1."""
    S = make_S(2, 10)
    S[0, 4] = 5.0
    S[1, 4] = 5.0
    _, path = align(S, 0, 9, lam=0.0, g=0)
    assert path is not None
    assert path[0] < path[1]


@pytest.mark.parametrize("align", BOTH)
def test_invalid_range_raises(align):
    S = make_S(2, 10)
    with pytest.raises(ValueError):
        align(S, 5, 20, lam=0.01)
    with pytest.raises(ValueError):
        align(S, -1, 5, lam=0.01)


@pytest.mark.parametrize("align", BOTH)
def test_fully_masked_video_returns_none(align):
    S = np.full((2, 20), NEG)
    score, path = align(S, 0, 19, lam=0.01, g=2)
    assert path is None and score == NEG
