"""End-to-end tests for TrakeEngine on a tiny planted corpus."""

import numpy as np
import pytest

from src.task3_trake.data_loader import KeyframeTable, TrakeData  # noqa: F401
from src.task3_trake.chronos_engine import TrakeConfig, TrakeEngine
from src.task3_trake.verifier import MockVLM

D = 8


def _unit(v):
    return v / np.linalg.norm(v)


_DIR_RNG = np.random.default_rng(42)
DIRS = {
    "ev1": _unit(_DIR_RNG.standard_normal(D)),
    "ev2": _unit(_DIR_RNG.standard_normal(D)),
    "ctx": _unit(_DIR_RNG.standard_normal(D)),
}


def encoder(texts):
    return np.stack([DIRS[t] for t in texts]).astype(np.float32)


def make_data():
    """Two videos of 40 keyframes each.  A holds a strong (ev1@5, ev2@15)
    sequence, B a weaker copy (ev1@45, ev2@55).

    Seeded per call, not from a shared generator: tests that build their own
    corpus must all see the SAME one, or a result that depends on the corpus
    silently depends on test execution order instead.
    """
    T = 80
    rng = np.random.default_rng(1234)
    E = rng.standard_normal((T, D)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)

    def plant(gid, key, w):
        E[gid] = _unit((1 - w) * E[gid] + w * DIRS[key])

    plant(5, "ev1", 1.0)
    plant(15, "ev2", 1.0)
    plant(45, "ev1", 0.6)
    plant(55, "ev2", 0.6)

    videos = {
        "A": {"s_v": 0, "e_v": 39, "fps": 25.0, "n_frames": 1200},
        "B": {"s_v": 40, "e_v": 79, "fps": 25.0, "n_frames": 1200},
    }
    local = np.concatenate([np.arange(40), np.arange(40)])
    table = KeyframeTable(
        video_id=np.array(["A"] * 40 + ["B"] * 40, dtype=object),
        frame_idx=(local * 30).astype(np.int64),
        ts_ms=(local * 1200).astype(np.int64),
        shot_id=(np.arange(T) // 8).astype(np.int64),
        pos_in_shot=np.zeros(T, dtype=np.float32),
    )
    return TrakeData(E.astype(np.float16), videos, table)


@pytest.fixture(scope="module")
def engine():
    return TrakeEngine(make_data(), text_encoder=encoder)


def test_end_to_end_manual_events(engine):
    resp = engine.search(events=["ev1", "ev2"], mode="visual", lam=0.001)
    assert resp["results"], "no results returned"
    top = resp["results"][0]
    assert top["video_id"] == "A"
    assert [e["gid"] for e in top["events"]] == [5, 15]
    assert top["sequence_frames"] == [150, 450]
    assert top["submission_line"] == "A,150,450"
    # response schema (spec 8.3)
    for key in ("query_id", "decomposed", "lambda_used", "latency_ms", "results"):
        assert key in resp
    ev = top["events"][0]
    for key in ("idx", "gid", "frame_idx", "ts_ms", "shot_id", "score", "thumb"):
        assert key in ev
    assert ev["thumb"] == "/kf/A/150.jpg"
    assert resp["decomposed"]["source"] == "manual"


def test_raw_query_rule_decomposition(engine):
    # rule-based path: "context: e1; e2" — encoder knows these exact tokens
    resp = engine.search(raw_query="bối cảnh: ev1; ev2", mode="visual", lam=0.001)
    assert resp["decomposed"]["source"] == "rule"
    assert resp["results"][0]["video_id"] == "A"


def test_anchor_redirects_to_other_video(engine):
    """Anchor keys are 1-based, matching events[].idx in the API (spec 8.2)."""
    resp = engine.search(
        events=["ev1", "ev2"], mode="visual", lam=0.001, anchors={"1": 45}
    )
    top = resp["results"][0]
    assert top["video_id"] == "B"
    assert top["events"][0]["gid"] == 45
    assert all(r["video_id"] != "A" for r in resp["results"])


def test_anchor_second_event_is_one_based(engine):
    """anchors={'2': gid} must pin the SECOND event, not the first."""
    resp = engine.search(
        events=["ev1", "ev2"], mode="visual", lam=0.001, anchors={"2": 55}
    )
    top = resp["results"][0]
    assert top["video_id"] == "B"
    assert top["events"][1]["gid"] == 55


@pytest.mark.parametrize("bad", [{"0": 5}, {"3": 5}, {"-1": 5}])
def test_anchor_index_validation(engine, bad):
    with pytest.raises(ValueError):
        engine.search(events=["ev1", "ev2"], mode="visual", anchors=bad)


def test_anchored_video_survives_candidate_restriction(engine):
    """An anchor is a statement of certainty; a prefilter must not drop it."""
    resp = engine.search(
        events=["ev1", "ev2"],
        mode="visual",
        lam=0.001,
        anchors={"1": 45},           # gid 45 lives in video B
        candidate_videos=["A"],      # ... which the caller excluded
    )
    assert resp["results"][0]["video_id"] == "B"


def test_earliness_applied_only_when_first_occurrence(engine):
    r1 = engine.search(events=["ev1", "ev2"], first_occurrence=True, lam=0.001)
    r2 = engine.search(events=["ev1", "ev2"], first_occurrence=False, lam=0.001)
    assert r1["earliness"] == pytest.approx(TrakeConfig().earliness_mu)  # score-std units
    assert r2["earliness"] == 0.0


def test_query_vectors_bypass_encoder():
    eng = TrakeEngine(make_data())  # no text encoder at all
    U = encoder(["ev1", "ev2"])
    resp = eng.search(events=["ev1", "ev2"], query_vectors=U, mode="visual", lam=0.001)
    assert resp["results"][0]["video_id"] == "A"
    with pytest.raises(ValueError):
        eng.search(events=["ev1", "ev2"], mode="visual")  # no vectors, no encoder
    with pytest.raises(ValueError):
        eng.search(events=["ev1", "ev2"], query_vectors=U[:1], mode="visual")


def test_no_query_at_all(engine):
    with pytest.raises(ValueError):
        engine.search()


def test_candidate_video_restriction(engine):
    resp = engine.search(
        events=["ev1", "ev2"], mode="visual", lam=0.001, candidate_videos=["B"]
    )
    assert [r["video_id"] for r in resp["results"]] == ["B"]


def test_asr_fusion_path():
    hits = {"ev2-asr": {15: 1.0}}
    eng = TrakeEngine(
        make_data(),
        text_encoder=encoder,
        asr_searcher=lambda hint: hits.get(hint, {}),
    )
    events = [
        {"text": "ev1"},
        {"text": "ev2", "asr_hint": "ev2-asr"},
    ]
    resp = eng.search(events=events, mode="speech", lam=0.001)
    assert resp["results"][0]["video_id"] == "A"
    assert resp["results"][0]["events"][1]["gid"] == 15


def test_verify_attaches_vlm_fields():
    vlm = MockVLM(judge=lambda img, txt: 0.9)
    eng = TrakeEngine(make_data(), text_encoder=encoder, vlm=vlm)
    resp = eng.search(events=["ev1", "ev2"], mode="visual", lam=0.001, verify=True)
    top = resp["results"][0]
    assert resp["verified"] is True
    assert "final_score" in top
    assert top["events"][0]["vlm_conf"] == pytest.approx(0.9)
    assert vlm.calls  # the VLM was actually consulted


def test_verify_without_vlm_is_reported_not_silently_skipped(engine):
    """Pressing the verify toggle with no VLM must not look like a clean run."""
    with pytest.raises(ValueError, match="no VLM"):
        engine.search(events=["ev1", "ev2"], mode="visual", lam=0.001, verify=True)


def test_verified_flag_false_by_default(engine):
    resp = engine.search(events=["ev1", "ev2"], mode="visual", lam=0.001)
    assert resp["verified"] is False


def test_soft_order_answer_columns_follow_event_number():
    """Under a winning swap the answer columns must stay in EVENT order.

    Column k of the submission belongs to event k.  Emitting the frames in
    chronological order would silently transpose the answer whenever the
    alignment swapped two events.
    """
    eng = TrakeEngine(make_data(), text_encoder=encoder)
    # supplied in the reverse of their temporal order: E1="ev2" (gid 15),
    # E2="ev1" (gid 5)
    resp = eng.search(
        events=["ev2", "ev1"], mode="visual", lam=0.0, align_mode="soft_order"
    )
    top = resp["results"][0]
    assert top["video_id"] == "A"
    assert top["event_order"] == [1, 0]
    # events and answer columns are ordered by event number ...
    assert [e["idx"] for e in top["events"]] == [1, 2]
    assert [e["gid"] for e in top["events"]] == [15, 5]
    assert top["sequence_frames"] == [450, 150]
    assert top["submission_line"] == "A,450,150"
    # ... while the chronological reading stays available for the UI grid
    assert top["gids_chronological"] == [5, 15]
    assert all(e["score"] > 1.0 for e in top["events"])


def test_ordered_mode_columns_are_both_orders_at_once(engine):
    """With no swap, event order and chronological order coincide."""
    resp = engine.search(events=["ev1", "ev2"], mode="visual", lam=0.001)
    top = resp["results"][0]
    assert [e["idx"] for e in top["events"]] == [1, 2]
    assert top["sequence_frames"] == [150, 450]
    assert top["gids_chronological"] == [5, 15]
    assert "event_order" not in top


def test_degraded_corpus_suppresses_submission_line():
    """Keyframe ordinals must never be emitted as a submission line."""
    full = make_data()
    bare = TrakeData(full.embeddings, full.videos)  # no keyframe table
    eng = TrakeEngine(bare, text_encoder=encoder)
    resp = eng.search(events=["ev1", "ev2"], mode="visual", lam=0.001)
    top = resp["results"][0]
    assert top["submission_line"] is None
    assert "warning" in top


def test_request_weights_override_mode_preset():
    eng = TrakeEngine(make_data(), text_encoder=encoder)
    custom = {"visual": 1.0, "ocr": 0.0, "asr": 0.0}
    resp = eng.search(events=["ev1", "ev2"], mode="balanced", weights=custom, lam=0.001)
    assert resp["results"][0]["video_id"] == "A"
    # the response must report what was actually fused, not just the mode name
    assert resp["weights_used"] == custom
    assert resp["mode"] == "balanced"


def test_weights_used_reports_the_mode_preset_by_default(engine):
    from src.task3_trake.scoring import SCORING_MODES

    resp = engine.search(events=["ev1", "ev2"], mode="speech", lam=0.001)
    assert resp["weights_used"] == SCORING_MODES["speech"]


def test_config_weights_apply_only_when_mode_is_not_requested():
    cfg = TrakeConfig(weights={"visual": 1.0, "ocr": 0.0, "asr": 0.0})
    eng = TrakeEngine(make_data(), text_encoder=encoder, config=cfg)
    from src.task3_trake.scoring import SCORING_MODES

    implicit = eng.search(events=["ev1", "ev2"], lam=0.001)
    assert implicit["weights_used"] == cfg.weights
    explicit = eng.search(events=["ev1", "ev2"], mode="text_heavy", lam=0.001)
    assert explicit["weights_used"] == SCORING_MODES["text_heavy"]


def test_lambda_bounds_from_config_are_applied():
    """Regression: lambda_bounds was parsed and tested but never plumbed."""
    cfg = TrakeConfig(lam=None, lambda_bounds=(0.005, 0.006))
    eng = TrakeEngine(make_data(), text_encoder=encoder, config=cfg)
    resp = eng.search(events=["ev1", "ev2"], mode="visual")
    assert 0.005 <= resp["lambda_used"] <= 0.006


def test_empty_event_text_rejected(engine):
    with pytest.raises(ValueError):
        engine.search(events=[{"text": "  "}, {"text": "ev2"}])


def test_explicit_zero_weight_is_honoured(engine):
    """weight=0 must mean zero, not 'absent' (the `or 1.0` trap)."""
    resp = engine.search(
        events=[{"text": "ev1"}, {"text": "ev2", "weight": 0.0}], mode="visual", lam=0.001
    )
    assert resp["decomposed"]["events"][1]["weight"] == 0.0


def test_client_event_idx_follows_array_position(engine):
    """Array position IS the event number; a conflicting idx is rejected.

    Regression: sorting client events by their declared idx silently
    transposed caller-supplied query_vectors (consumed positionally) and
    resolved anchors against the post-sort numbering.
    """
    resp = engine.search(
        events=[{"text": "ev1", "idx": 1}, {"text": "ev2", "idx": 2}],
        mode="visual",
        lam=0.001,
    )
    assert [e["idx"] for e in resp["decomposed"]["events"]] == [1, 2]
    assert [e["idx"] for e in resp["results"][0]["events"]] == [1, 2]

    with pytest.raises(ValueError, match="idx"):
        engine.search(events=[{"text": "ev1", "idx": 3}, {"text": "ev2", "idx": 4}])
    with pytest.raises(ValueError, match="idx"):
        engine.search(events=[{"text": "ev2", "idx": 2}, {"text": "ev1", "idx": 1}])


def test_query_vectors_stay_aligned_with_supplied_events():
    """The k-th vector must score the k-th event, always."""
    eng = TrakeEngine(make_data())
    U = encoder(["ev2", "ev1"])  # E1 = "ev2" (gid 15), E2 = "ev1" (gid 5)
    resp = eng.search(
        events=["ev2", "ev1"],
        query_vectors=U,
        mode="visual",
        lam=0.0,
        align_mode="soft_order",  # the true order is E2 then E1
    )
    top = resp["results"][0]
    by_idx = {e["idx"]: e["gid"] for e in top["events"]}
    assert by_idx == {1: 15, 2: 5}


class _RecordingVLM:
    def __init__(self):
        self.asked = []

    def verify(self, image_ref, event_text):
        self.asked.append((image_ref, event_text))
        return {"match": True, "confidence": 1.0, "reason": "ok"}


def test_vvr_asks_about_the_right_event_under_swap():
    """End-to-end: soft_order + verify must not transpose the VLM questions."""
    rec = _RecordingVLM()
    eng = TrakeEngine(
        make_data(), text_encoder=encoder, vlm=rec, config=TrakeConfig(topk_verify=1)
    )
    resp = eng.search(
        events=["ev2", "ev1"], mode="visual", lam=0.0, align_mode="soft_order", verify=True
    )
    assert resp["results"][0]["video_id"] == "A"
    # E1 = "ev2" at gid 15 (frame 450); E2 = "ev1" at gid 5 (frame 150)
    assert rec.asked == [("/kf/A/450.jpg", "ev2"), ("/kf/A/150.jpg", "ev1")]


@pytest.mark.parametrize(
    "align_mode,extra",
    [
        ("ordered", {}),
        ("soft_order", {}),
        ("unordered", {}),
        ("strict_window", {"window": 30}),
    ],
)
def test_answer_contract_holds_in_every_align_mode(align_mode, extra):
    """One contract for all four modes: events by idx, columns to match, right VLM pairs."""
    rec = _RecordingVLM()
    eng = TrakeEngine(
        make_data(), text_encoder=encoder, vlm=rec, config=TrakeConfig(topk_verify=1)
    )
    resp = eng.search(
        events=["ev2", "ev1"],
        mode="visual",
        lam=0.0,
        align_mode=align_mode,
        verify=True,
        **extra,
    )
    top = resp["results"][0]
    idxs = [e["idx"] for e in top["events"]]
    assert idxs == sorted(idxs) == list(range(1, len(idxs) + 1))
    assert top["sequence_frames"] == [e["frame_idx"] for e in top["events"]]
    assert top["submission_line"] == top["video_id"] + "," + ",".join(
        str(e["frame_idx"]) for e in top["events"]
    )
    texts = {1: "ev2", 2: "ev1"}
    assert rec.asked == [(e["thumb"], texts[e["idx"]]) for e in top["events"]]


def test_min_gap_changes_adaptive_lambda(engine):
    """Regression: the engine dropped g, so min_gap had no effect on lambda."""
    tight = engine.search(events=["ev1", "ev2"], mode="visual", min_gap=2)
    wide = engine.search(events=["ev1", "ev2"], mode="visual", min_gap=14)
    assert wide["lambda_used"] < tight["lambda_used"]


def test_out_of_range_anchor_gid_is_a_value_error(engine):
    """Must not surface as an IndexError (HTTP 500) via the forced-video lookup."""
    with pytest.raises(ValueError, match="out of range"):
        engine.search(events=["ev1", "ev2"], mode="visual", anchors={"1": 999_999})
    with pytest.raises(ValueError, match="out of range"):
        engine.search(
            events=["ev1", "ev2"],
            mode="visual",
            anchors={"1": 999_999},
            candidate_videos=["A"],
        )


def test_verify_enabled_without_vlm_fails_at_construction():
    """A bad yaml must break at startup, not on every query."""
    with pytest.raises(ValueError, match="verify"):
        TrakeEngine(make_data(), text_encoder=encoder, config=TrakeConfig(verify_enabled=True))


def test_from_yaml_defaults_match_dataclass(tmp_path):
    """An empty config must reproduce TrakeConfig() exactly."""
    p = tmp_path / "empty.yaml"
    p.write_text("{}", encoding="utf-8")
    assert TrakeConfig.from_yaml(p) == TrakeConfig()


def test_config_from_yaml(tmp_path):
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parents[1] / "config" / "trake.yaml"
    cfg = TrakeConfig.from_yaml(cfg_path)
    assert cfg.mode == "balanced"
    assert cfg.lam is None  # "auto"
    assert cfg.min_gap == 2
    assert cfg.earliness_mu == pytest.approx(2.0)
    assert cfg.lambda_bounds == (0.005, 0.2)
    assert cfg.lambda_kappa == pytest.approx(0.5)
    # the shipped yaml must not silently disagree with the dataclass
    assert cfg == TrakeConfig()
    assert cfg.verify_enabled is False


def test_lambda_used_reported(engine):
    resp = engine.search(events=["ev1", "ev2"], mode="visual")  # lam None -> adaptive
    assert 0.004 <= resp["lambda_used"] <= 0.21  # relative bounds x score scale ~1
