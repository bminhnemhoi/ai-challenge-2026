"""Tests for L0 access — contiguity validation (R1) and index repair (R2)."""

import json

import numpy as np
import pytest

from src.task3_trake.data_loader import (
    IndexRemapper,
    KeyframeTable,
    TrakeData,
    validate_contiguity,
)


def _table(video_ids, frame_idx):
    n = len(video_ids)
    return KeyframeTable(
        video_id=np.array(video_ids, dtype=object),
        frame_idx=np.array(frame_idx, dtype=np.int64),
        ts_ms=np.array([f * 40 for f in frame_idx], dtype=np.int64),
        shot_id=np.zeros(n, dtype=np.int64),
        pos_in_shot=np.zeros(n, dtype=np.float32),
    )


# ------------------------------------------------------------- contiguity


def test_validate_ok():
    videos = {"A": {"s_v": 0, "e_v": 4}, "B": {"s_v": 5, "e_v": 9}}
    assert validate_contiguity(videos, T=10) == []


def test_validate_detects_gap_overlap_and_bounds():
    assert validate_contiguity({"A": {"s_v": 1, "e_v": 4}}, T=5)  # doesn't start at 0
    assert validate_contiguity(
        {"A": {"s_v": 0, "e_v": 4}, "B": {"s_v": 6, "e_v": 9}}, T=10
    )  # gap
    assert validate_contiguity(
        {"A": {"s_v": 0, "e_v": 5}, "B": {"s_v": 5, "e_v": 9}}, T=10
    )  # overlap
    assert validate_contiguity({"A": {"s_v": 0, "e_v": 7}}, T=10)  # short tail
    assert validate_contiguity({}, T=10)


def test_trakedata_strict_raises_on_broken_index():
    E = np.zeros((10, 4), dtype=np.float16)
    with pytest.raises(ValueError):
        TrakeData(E, {"A": {"s_v": 0, "e_v": 3}, "B": {"s_v": 6, "e_v": 9}})
    d = TrakeData(E, {"A": {"s_v": 0, "e_v": 3}, "B": {"s_v": 6, "e_v": 9}}, strict=False)
    assert d.index_problems


# -------------------------------------------------------------- remapper


def test_remapper_rebuilds_contiguous_index():
    # rows interleaved across videos, unsorted frame order
    table = _table(["B", "A", "B", "A"], [30, 0, 0, 30])
    rm = IndexRemapper.from_keyframes(table)
    # first-appearance order: B then A; frames sorted inside each video
    new = rm.apply_to_table(table)
    assert list(new.video_id) == ["B", "B", "A", "A"]
    assert list(new.frame_idx) == [0, 30, 0, 30]
    assert rm.videos == {"B": {"s_v": 0, "e_v": 1}, "A": {"s_v": 2, "e_v": 3}}
    assert validate_contiguity(rm.videos, T=4) == []

    E = np.arange(8, dtype=np.float32).reshape(4, 2)
    E2 = rm.apply_to_embeddings(E)
    # embedding rows follow their keyframes
    assert list(E2[:, 0]) == [E[2, 0], E[0, 0], E[1, 0], E[3, 0]]
    # old_to_new is the inverse permutation
    assert np.array_equal(rm.old_to_new[rm.order], np.arange(4))


# ---------------------------------------------------------------- loading


def test_load_roundtrip(tmp_path):
    E = np.random.default_rng(0).standard_normal((6, 4)).astype(np.float16)
    np.save(tmp_path / "embeddings.npy", E)
    videos = {"A": {"s_v": 0, "e_v": 2, "fps": 25.0, "n_frames": 90},
              "B": {"s_v": 3, "e_v": 5, "fps": 25.0, "n_frames": 90}}
    (tmp_path / "videos.json").write_text(json.dumps(videos), encoding="utf-8")
    np.savez(
        tmp_path / "keyframes.npz",
        video_id=np.array(["A", "A", "A", "B", "B", "B"], dtype=object),
        frame_idx=np.array([0, 30, 60, 0, 30, 60]),
        ts_ms=np.array([0, 1200, 2400, 0, 1200, 2400]),
        shot_id=np.array([0, 0, 1, 2, 2, 3]),
        pos_in_shot=np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
    )
    data = TrakeData.load(tmp_path)
    assert data.T == 6 and data.dim == 4
    data32 = TrakeData.load(tmp_path, upcast=True)
    assert data32.embeddings.dtype == np.float32
    assert np.allclose(data32.embeddings, np.asarray(E, dtype=np.float32))
    rec = data.lookup(4)
    assert rec == {
        "gid": 4,
        "video_id": "B",
        "frame_idx": 30,
        "ts_ms": 1200,
        "shot_id": 2,
        "pos_in_shot": 1.0,
        "degraded": False,
    }
    assert data.video_of_gid(1) == "A"
    with pytest.raises(IndexError):
        data.lookup(99)


def test_lookup_without_keyframe_table_is_flagged_degraded():
    """The ordinal is NOT a frame index — callers must be able to tell."""
    E = np.zeros((6, 4), dtype=np.float16)
    videos = {"A": {"s_v": 0, "e_v": 2}, "B": {"s_v": 3, "e_v": 5}}
    data = TrakeData(E, videos)
    rec = data.lookup(4)
    assert rec["video_id"] == "B"
    assert rec["frame_idx"] == 1  # keyframe ordinal inside the video
    assert rec["degraded"] is True


def test_norm_stats_detects_unnormalized_embeddings():
    """R2: visual_scores never renormalizes the corpus, so this must be checked."""
    rng = np.random.default_rng(3)
    good = rng.standard_normal((50, 8)).astype(np.float32)
    good /= np.linalg.norm(good, axis=1, keepdims=True)
    videos = {"A": {"s_v": 0, "e_v": 49}}
    assert TrakeData(good, videos).norm_stats()["l2_normalized"] is True

    bad = good * 3.0
    stats = TrakeData(bad, videos).norm_stats()
    assert stats["l2_normalized"] is False
    assert stats["max_deviation"] > 1.0


def test_density_stats_flags_thin_shots():
    """R3: the spec requires at least 4 keyframes per shot."""
    E = np.zeros((6, 4), dtype=np.float16)
    videos = {"A": {"s_v": 0, "e_v": 5}}
    thin = _table(["A"] * 6, list(range(6)))
    thin.shot_id = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)
    stats = TrakeData(E, videos, thin).density_stats()
    assert stats["available"] is True
    assert stats["min_per_shot"] == 2
    assert stats["shots_below_min"] == 1
    assert stats["meets_r3"] is False
    assert TrakeData(E, videos).density_stats()["meets_r3"] is False


def test_density_stats_unsegmented_corpus_fails_r3():
    """A corpus with no shot labels must not report R3 as satisfied."""
    E = np.zeros((6, 4), dtype=np.float16)
    videos = {"A": {"s_v": 0, "e_v": 5}}
    unlabelled = _table(["A"] * 6, list(range(6)))
    unlabelled.shot_id = np.full(6, -1, dtype=np.int64)
    stats = TrakeData(E, videos, unlabelled).density_stats()
    assert stats["shot_ids"] is False
    assert stats["unlabelled"] == 6
    assert stats["meets_r3"] is False


def test_density_stats_keys_shots_per_video():
    """Shot ids repeat across videos; counting them globally merges shots."""
    E = np.zeros((8, 4), dtype=np.float16)
    videos = {"A": {"s_v": 0, "e_v": 3}, "B": {"s_v": 4, "e_v": 7}}
    t = _table(["A"] * 4 + ["B"] * 4, list(range(8)))
    t.shot_id = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)  # shot 0 in both
    stats = TrakeData(E, videos, t).density_stats()
    assert stats["n_shots"] == 2  # not 1
    assert stats["min_per_shot"] == 4
    assert stats["meets_r3"] is True


def test_keyframe_table_length_mismatch():
    E = np.zeros((6, 4), dtype=np.float16)
    videos = {"A": {"s_v": 0, "e_v": 5}}
    table = _table(["A", "A"], [0, 30])
    with pytest.raises(ValueError):
        TrakeData(E, videos, table)


def test_jsonl_loading(tmp_path):
    rows = [
        {"gid": 1, "video_id": "A", "frame_idx": 30, "ts_ms": 1200},
        {"gid": 0, "video_id": "A", "frame_idx": 0, "ts_ms": 0},
    ]
    p = tmp_path / "keyframes.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    t = KeyframeTable.load(p)
    assert list(t.frame_idx) == [0, 30]  # sorted by gid
