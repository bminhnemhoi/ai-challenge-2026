"""Tests for L4 — VVR (VLM verification rerank)."""

import pytest

from src.task3_trake.verifier import MockVLM, verify_and_rerank


def _results():
    def res(vid, score):
        return {
            "video_id": vid,
            "score": score,
            "events": [
                {"thumb": f"/kf/{vid}/100.jpg"},
                {"thumb": f"/kf/{vid}/200.jpg"},
            ],
        }

    return [res("A", 10.0), res("B", 9.5), res("C", 0.0)]


def test_mock_vlm_clamps_and_records():
    vlm = MockVLM(judge=lambda img, txt: 1.7)
    v = vlm.verify("/kf/x.jpg", "an event")
    assert v["confidence"] == 1.0 and v["match"] is True
    assert vlm.calls == [("/kf/x.jpg", "an event")]


def test_rerank_flips_on_vlm_evidence():
    """B is a close CHRONOS runner-up; unanimous VLM support must lift it above A."""
    vlm = MockVLM(judge=lambda img, txt: 1.0 if "/B/" in img else 0.0)
    out = verify_and_rerank(_results(), ["e1", "e2"], vlm, topk_verify=3)
    assert [r["video_id"] for r in out] == ["B", "A", "C"]
    # final = 0.7*norm + 0.3*mean(conf): B = 0.7*0.95 + 0.3 = 0.965 > A = 0.7
    assert out[0]["final_score"] == pytest.approx(0.965, abs=1e-6)
    assert out[1]["final_score"] == pytest.approx(0.7, abs=1e-6)
    for ev in out[0]["events"]:
        assert ev["vlm_match"] is True and ev["vlm_conf"] == 1.0


def test_rerank_tail_untouched():
    results = _results() + [
        {"video_id": "D", "score": -5.0, "events": [{"thumb": "t"}, {"thumb": "t"}]}
    ]
    out = verify_and_rerank(results, ["e1", "e2"], MockVLM(), topk_verify=3)
    assert out[-1]["video_id"] == "D"
    assert "final_score" not in out[-1]
    assert "vlm_conf" not in out[-1]["events"][0]


def test_rerank_degenerate_scores():
    results = [
        {"video_id": v, "score": 1.0, "events": [{"thumb": v}]} for v in ("A", "B")
    ]
    out = verify_and_rerank(results, ["e"], MockVLM(), topk_verify=2)
    # equal scores -> both normalized to 1.0, order preserved (stable sort)
    assert [r["video_id"] for r in out] == ["A", "B"]


def test_rerank_empty():
    assert verify_and_rerank([], ["e"], MockVLM()) == []


def test_rerank_topk_zero_is_a_noop():
    """topk_verify=0 (settable in trake.yaml) must not crash on min([])."""
    results = _results()
    out = verify_and_rerank(results, ["e1", "e2"], MockVLM(), topk_verify=0)
    assert out == results
    assert all("final_score" not in r for r in out)


class Recorder:
    """VLM stand-in that records exactly which (image, text) pairs it was asked."""

    def __init__(self):
        self.asked = []

    def verify(self, image_ref, event_text):
        self.asked.append((image_ref, event_text))
        return {"match": True, "confidence": 1.0, "reason": "ok"}


def test_rerank_pairs_each_frame_with_its_own_event_text():
    """Pairing follows events[].idx, not position and not event_order.

    Regression (twice over): first positional pairing asked "does E1's frame
    show E2?" for every swapped candidate; then re-applying the CHRONOLOGICAL
    event_order to an events list already sorted by event number transposed
    the pairs right back.  Both deflate the confidence of a CORRECT sequence.
    """
    rec = Recorder()
    # what the engine actually emits: events sorted by idx, event_order still
    # describing the chronological reading
    result = {
        "video_id": "A",
        "score": 1.0,
        "event_order": [1, 0],
        "events": [
            {"idx": 1, "thumb": "frame_late"},   # E1 happens to be later
            {"idx": 2, "thumb": "frame_early"},
        ],
    }
    verify_and_rerank([result], ["text_E1", "text_E2"], rec, topk_verify=1)
    assert rec.asked == [("frame_late", "text_E1"), ("frame_early", "text_E2")]


def test_rerank_tolerates_missing_idx():
    """Hand-built records without idx must not crash the verifier."""
    rec = Recorder()
    result = {"video_id": "A", "score": 1.0, "events": [{"thumb": "t1"}, {"thumb": "t2"}]}
    out = verify_and_rerank([result], ["e1", "e2"], rec, topk_verify=1)
    assert len(rec.asked) == 2
    assert "final_score" in out[0]
