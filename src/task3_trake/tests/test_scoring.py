"""Tests for L2 — MCS (multi-channel scoring)."""

import numpy as np
import pytest

from src.task3_trake.scoring import (
    SCORING_MODES,
    SPARSE_Z_CAP,
    blend_ook_vector,
    channel_matrix,
    fuse,
    sparse_to_dense,
    visual_scores,
    zscore_rows,
)


def test_zscore_rows_properties():
    rng = np.random.default_rng(0)
    S = rng.normal(loc=3.0, scale=2.0, size=(4, 500))
    Z = zscore_rows(S)
    assert np.allclose(Z.mean(axis=1), 0.0, atol=1e-4)
    assert np.allclose(Z.std(axis=1), 1.0, atol=1e-3)


def test_zscore_constant_row_is_zero():
    S = np.full((2, 100), 7.0)
    Z = zscore_rows(S)
    assert np.allclose(Z, 0.0)
    assert np.all(np.isfinite(Z))


def test_zscore_normalizes_difficulty():
    """An 'easy' row (high mean) must not dominate a 'hard' row after z-scoring."""
    easy = np.random.default_rng(1).normal(0.8, 0.05, size=200)
    hard = np.random.default_rng(2).normal(0.1, 0.05, size=200)
    easy[50] = 0.99
    hard[70] = 0.35  # 5-sigma peak for the hard event
    Z = zscore_rows(np.stack([easy, hard]))
    assert Z[1, 70] > 2.0  # the hard event's peak survives normalization
    assert abs(Z[0].mean()) < 0.1


# -------------------------------------------------------------- visual GEMM


def test_visual_scores_chunked_matches_direct():
    rng = np.random.default_rng(4)
    U = rng.standard_normal((3, 16)).astype(np.float32)
    E32 = rng.standard_normal((1000, 16)).astype(np.float32)
    E32 /= np.linalg.norm(E32, axis=1, keepdims=True)
    E = E32.astype(np.float16)

    S_chunked = visual_scores(U, E, chunk_size=137)
    S_direct = visual_scores(U, E, chunk_size=10**9)
    assert np.allclose(S_chunked, S_direct, atol=1e-6)
    assert S_chunked.shape == (3, 1000)
    assert np.abs(S_chunked).max() <= 1.0 + 1e-3


def test_visual_scores_float32_fastpath_matches_float16():
    rng = np.random.default_rng(41)
    U = rng.standard_normal((2, 12)).astype(np.float32)
    E32 = rng.standard_normal((500, 12)).astype(np.float32)
    E32 /= np.linalg.norm(E32, axis=1, keepdims=True)
    S32 = visual_scores(U, E32)
    S16 = visual_scores(U, E32.astype(np.float16))
    assert np.allclose(S32, S16, atol=2e-3)


def test_visual_scores_dim_mismatch():
    with pytest.raises(ValueError):
        visual_scores(np.zeros((2, 8), np.float32), np.zeros((10, 16), np.float16))


# ------------------------------------------------------------ sparse inputs


def test_sparse_to_dense():
    row = sparse_to_dense({3: 1.0, 7: 0.5}, T=10)
    assert row[3] == 1.0 and row[7] == 0.5
    assert row.sum() == pytest.approx(1.5)
    with pytest.raises(ValueError):
        sparse_to_dense({99: 1.0}, T=10)


def test_channel_matrix_none_rows():
    M = channel_matrix([{1: 1.0}, None, {}], T=5)
    assert M.shape == (3, 5)
    assert np.allclose(M[1], 0.0) and np.allclose(M[2], 0.0)


# ------------------------------------------------------------------- fusion


def test_fuse_visual_only_equals_zscore():
    rng = np.random.default_rng(5)
    S_vis = rng.standard_normal((2, 300)).astype(np.float32)
    fused = fuse(S_vis, mode="visual")
    assert np.allclose(fused, zscore_rows(S_vis), atol=1e-6)


def test_fuse_weight_renormalization_on_missing_channels():
    """`balanced` without OCR/ASR must degrade to pure (z-scored) visual."""
    rng = np.random.default_rng(6)
    S_vis = rng.standard_normal((2, 300)).astype(np.float32)
    fused = fuse(S_vis, None, None, mode="balanced")
    assert np.allclose(fused, zscore_rows(S_vis), atol=1e-6)


def test_fuse_per_row_renormalization():
    """An event with no ASR hint keeps full visual amplitude.

    Regression: whole-matrix renormalization scaled hint-less rows by
    0.5/(0.5+0.4), so an event's amplitude depended on whether OTHER events
    happened to have hints.
    """
    rng = np.random.default_rng(60)
    S_vis = rng.standard_normal((2, 500)).astype(np.float32)
    S_asr = channel_matrix([{10: 1.0, 11: 0.9, 12: 0.8}, None], T=500)  # only event 1
    fused = fuse(S_vis, None, S_asr, mode="speech")
    base = zscore_rows(S_vis)
    assert np.allclose(fused[1], base[1], atol=1e-5)  # row 2 untouched
    assert not np.allclose(fused[0], base[0], atol=1e-5)  # row 1 got ASR signal


def test_fuse_sparse_channel_is_capped():
    """One lexical hit must not overwhelm the visual channel at corpus scale."""
    T = 200_000
    rng = np.random.default_rng(7)
    S_vis = rng.standard_normal((1, T)).astype(np.float32)
    S_asr = channel_matrix([{1234: 1.0}], T=T)

    capped = fuse(S_vis, None, S_asr, mode="speech")
    uncapped = fuse(S_vis, None, S_asr, mode="speech", z_cap=0.0)

    w = SCORING_MODES["speech"]
    expected_cap = w["asr"] / (w["visual"] + w["asr"]) * SPARSE_Z_CAP
    # the hit's contribution is bounded by the cap ...
    assert capped[0, 1234] < expected_cap + 3.0
    # ... whereas the raw z-score of a 1-in-200k spike is enormous
    assert uncapped[0, 1234] > 100.0


def test_sparse_cap_preserves_hit_ranking():
    """Bounding the amplitude must not flatten the retrieval scores.

    Regression: a hard clip saturated every hit to exactly z_cap, turning the
    lexical channel into a binary hit/no-hit flag that rewarded an
    Elasticsearch noise hit as much as the true match.
    """
    T = 100_000
    rng = np.random.default_rng(72)
    S_vis = rng.standard_normal((1, T)).astype(np.float32)
    S_asr = channel_matrix([{100: 1.0, 200: 0.6, 300: 0.3}], T=T)
    fused = fuse(S_vis, None, S_asr, mode="speech")

    strong = fused[0, 100] - S_vis[0, 100]
    medium = fused[0, 200] - S_vis[0, 200]
    weak = fused[0, 300] - S_vis[0, 300]
    assert strong > medium > weak  # ordering survives the cap
    # and the amplitude is still bounded well below the uncapped z (~183)
    assert fused[0, 100] < 10.0


def test_fuse_dense_channel_not_capped():
    """A dense channel (not sparse) keeps its natural z-scale."""
    rng = np.random.default_rng(71)
    S_vis = rng.standard_normal((1, 1000)).astype(np.float32)
    S_dense = rng.standard_normal((1, 1000)).astype(np.float32)
    a = fuse(S_vis, None, S_dense, mode="speech")
    b = fuse(S_vis, None, S_dense, mode="speech", z_cap=0.0)
    assert np.allclose(a, b, atol=1e-6)


def test_fuse_event_weights_boost_positive_only():
    """w > 1 must amplify rewards without deepening penalties.

    Regression: plain multiplication made a hard OOK event's below-mean frames
    MORE repulsive, demoting the very video the weight was meant to promote.
    """
    rng = np.random.default_rng(8)
    S_vis = rng.standard_normal((2, 100)).astype(np.float32)
    fused = fuse(S_vis, mode="visual", event_weights=[1.0, 2.0])
    base = zscore_rows(S_vis)
    assert np.allclose(fused[0], base[0], atol=1e-6)
    pos, neg = base[1] > 0, base[1] <= 0
    assert np.allclose(fused[1][pos], 2.0 * base[1][pos], atol=1e-5)
    assert np.allclose(fused[1][neg], base[1][neg], atol=1e-6)
    with pytest.raises(ValueError):
        fuse(S_vis, mode="visual", event_weights=[1.0])


def test_fuse_event_weights_below_one_shrink_symmetrically():
    """De-weighting must also remove the event's power to veto a video.

    Regression: w < 1 shrank only the rewards, so an event the operator asked
    to matter LESS kept its full penalty; at w = 0 it degenerated into a pure
    penalty term instead of leaving the objective.
    """
    rng = np.random.default_rng(81)
    S_vis = rng.standard_normal((2, 200)).astype(np.float32)
    base = zscore_rows(S_vis)

    half = fuse(S_vis, mode="visual", event_weights=[1.0, 0.5])
    assert np.allclose(half[1], 0.5 * base[1], atol=1e-5)

    off = fuse(S_vis, mode="visual", event_weights=[1.0, 0.0])
    assert np.allclose(off[1], 0.0, atol=1e-6)  # fully out of the objective

    with pytest.raises(ValueError):
        fuse(S_vis, mode="visual", event_weights=[1.0, -0.5])


def test_fuse_errors():
    with pytest.raises(ValueError):
        fuse(np.zeros((2, 10)), mode="bogus")
    with pytest.raises(ValueError):
        fuse(None, None, None, mode="visual")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fuse(np.zeros((2, 10)), None, np.zeros((2, 9)), mode="speech")


def test_fuse_normalize_false_passthrough():
    S = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    assert np.allclose(fuse(S, mode="visual", normalize=False), S)


def test_modes_sum_to_one():
    for mode, w in SCORING_MODES.items():
        assert sum(w.values()) == pytest.approx(1.0), mode


# ----------------------------------------------------------------- OOK blend


def test_blend_ook_vector():
    t = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    img = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v = blend_ook_vector(t, img)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)
    assert v[0] / v[1] == pytest.approx(0.6 / 0.4, rel=1e-5)


def test_blend_ook_vector_multiple_exemplars():
    t = np.array([1.0, 0.0], dtype=np.float32)
    imgs = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    v = blend_ook_vector(t, imgs)
    assert v[0] / v[1] == pytest.approx(0.6 / 0.4, rel=1e-5)
    with pytest.raises(ValueError):
        blend_ook_vector(t, np.zeros(3, dtype=np.float32))
