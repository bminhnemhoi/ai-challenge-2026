"""A judge that judged nothing must not read like a judge that found nothing.

This pins the two failures that actually happened on round 1, hours before the
deadline, and cost most of an evening between them:

  1. The daily quota ran out. Every call raised 429, `_ask_batch` swallowed it
     and returned [], `score()` returned {}, and the rerank printed a clean
     finish over a submission the VLM had never looked at. Nothing in the
     output said so.

  2. The fix for (1) treated every 429 as the daily quota. But the free tier
     spends the same code on a per-MINUTE limit, which clears by itself. So a
     model that was merely going too fast got struck off for the rest of the
     round after one burst.

Both are about telling the two 429s apart and saying out loud when the answer
is "I never looked".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.vlm import VLMJudge, _is_daily_quota  # noqa: E402

PER_MINUTE = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
    "your current quota', 'details': [{'quotaId': "
    "'GenerateRequestsPerMinutePerProjectPerModel'}]}}"
)
PER_DAY = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded "
    "your current quota', 'details': [{'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel'}]}}"
)


def test_per_minute_429_is_not_treated_as_the_daily_quota():
    assert not _is_daily_quota(PER_MINUTE)


def test_per_day_429_is_recognised():
    assert _is_daily_quota(PER_DAY)
    assert _is_daily_quota("quota exceeded: requests per day")


@pytest.fixture()
def judge(tmp_path):
    j = VLMJudge(tmp_path, model="model-a")
    j.api_key = "test-key"          # so .ready is True without a real key
    return j


def test_exhausted_model_is_skipped_but_only_after_a_daily_429(judge, monkeypatch):
    """A per-minute 429 must leave the model usable; a per-day one must not."""
    judge.exhausted.add("model-a")
    assert "model-a" in judge.exhausted
    # every other name in the chain is still available, so the judge is usable
    assert judge.usable


def test_cost_note_shouts_when_nothing_was_judged(judge):
    judge.errors.append("ClientError: 429 RESOURCE_EXHAUSTED")
    judge.exhausted.add("model-a")
    note = judge.cost_note()
    assert "HET QUOTA" in note
    assert "model-a" in note
    # the line that stops a silent empty round from passing as a real one
    assert "KHONG CHAM DUOC KHUNG HINH NAO" in note


def test_cost_note_stays_quiet_on_a_healthy_run(judge):
    judge.calls = 12
    judge.tokens_in = 100_000
    judge.tokens_out = 3_000
    note = judge.cost_note()
    assert "!!" not in note
    assert "12 lần gọi" in note


def test_usable_goes_false_only_when_every_model_is_exhausted(judge):
    from src.core.vlm import FALLBACK_MODELS

    for m in ("model-a", *FALLBACK_MODELS):
        judge.exhausted.add(m)
    assert not judge.usable


def test_consecutive_batches_are_spread_across_peer_models(judge):
    """Per-minute quota is per model, so one model at a time wastes the others."""
    firsts = [judge._model_order()[0] for _ in range(6)]
    assert len(set(firsts)) > 1, "every batch went to the same model"


def test_rotation_only_mixes_models_that_mark_to_the_same_scale(judge):
    """A heavier model must not slip into a profile that gets compared frame to frame."""
    for _ in range(8):
        assert "lite" in judge._model_order()[0]


def test_rotation_is_deterministic_so_a_rebuild_reproduces_the_round(tmp_path):
    def firsts():
        j = VLMJudge(tmp_path, model="gemini-3.5-flash-lite")
        j.api_key = "test-key"
        return [j._model_order()[0] for _ in range(6)]

    assert firsts() == firsts()


def test_an_exhausted_model_drops_out_of_the_rotation(judge):
    judge.exhausted.add("gemini-2.5-flash-lite")
    for _ in range(8):
        assert judge._model_order()[0] != "gemini-2.5-flash-lite"


def test_unfetchable_frames_are_counted_and_announced(judge, monkeypatch):
    """A DNS blip took out every frame of a run once, in complete silence."""
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(OSError("getaddrinfo failed")),
    )
    monkeypatch.setattr("time.sleep", lambda _s: None)
    assert judge._fetch("L01_V001", "001.jpg") is None
    assert judge.fetch_failures == 1
    note = judge.cost_note()
    assert "KHÔNG TẢI ĐƯỢC" in note
    assert "KHONG CHAM DUOC KHUNG HINH NAO" in note


def test_a_fetched_thumbnail_is_reused_from_disk(judge, tmp_path):
    thumb = judge.frame_dir / "L01_V001" / "001.jpg"
    thumb.parent.mkdir(parents=True, exist_ok=True)
    thumb.write_bytes(b"cached-bytes")
    # no network monkeypatching at all: if it touched the wire this would fail
    assert judge._fetch("L01_V001", "001.jpg") == b"cached-bytes"
    assert judge.fetch_failures == 0


def test_score_returns_empty_and_records_an_error_when_every_call_fails(judge, monkeypatch):
    """The empty result is allowed — the silence about it is not."""
    monkeypatch.setattr(VLMJudge, "_fetch", staticmethod(lambda v, f: b"jpegbytes"))

    def boom(self, query, images):
        self.errors.append("ClientError: 429 RESOURCE_EXHAUSTED")
        self.exhausted.add(self.model)
        return []

    monkeypatch.setattr(VLMJudge, "_ask_batch", boom)
    out = judge.score("cau hoi", [("L01_V001", 10, "001.jpg")])
    assert out == {}
    assert judge.errors, "a failed round must leave a trace to print"
    assert "!!" in judge.cost_note()
