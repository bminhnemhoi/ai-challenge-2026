"""The spoken channel — and the discipline about where it is allowed to act.

Measured on the 60 ground-truth queries, folding transcripts into the score is
negative at every weight and only +0.5% (noise) even gated on decisive evidence.
Those queries are pure visual-scene descriptions that nobody says out loud, so
the measurement is honest about them and silent about the topical queries the
real round contains. The disposition that follows: transcripts inform the
operator, they do not rank.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.transcripts import TranscriptIndex, tokenise

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = ROOT.parent / "transcripts_full"
CAPTIONS = ROOT / "data" / "captions"


def test_bigrams_are_produced_for_vietnamese():
    """Vietnamese writes syllables, so the meaningful unit is often two tokens.

    "măng tây" (asparagus) unigrams to "măng" + "tây"; "tây" alone matches every
    video that says "phương Tây", "hành tây" or "Tây Ninh".
    """
    toks = tokenise("măng tây chiên bột")
    assert "măng_tây" in toks
    assert "chiên_bột" in toks
    assert "măng" in toks


def test_stopwords_and_broadcast_boilerplate_are_dropped():
    toks = tokenise("kính chào quý vị đến với chương trình 60 giây")
    assert "chương" not in toks and "quý" not in toks
    assert "giây" in toks


def test_empty_and_odd_input_do_not_raise():
    assert tokenise("") == []
    assert tokenise(None) == []
    idx = TranscriptIndex().build()
    assert idx.n_videos == 0
    assert idx.score_videos("bất cứ thứ gì") == {}
    assert idx.best_segment("x", "nope") is None


def test_a_video_can_be_indexed_and_found():
    idx = TranscriptIndex()
    idx.add("V1", [{"start": 0.0, "text": "hôm nay chúng ta làm món củ năng om nấm chay"}],
            title="CỦ NĂNG OM NẤM CHAY")
    idx.add("V2", [{"start": 0.0, "text": "món thịt kho tàu đậm đà"}], title="THỊT KHO TÀU")
    idx.build()
    scores = idx.score_videos("củ năng nấm chay")
    assert scores.get("V1", 0) > scores.get("V2", 0)
    at, text = idx.best_segment("củ năng", "V1")
    assert at == 0.0 and "củ năng" in text


def test_segments_carry_timestamps_so_a_hit_localises_a_moment():
    """A title says which video; only a timestamp says when — which is what TRAKE needs."""
    idx = TranscriptIndex()
    idx.add("V1", [
        {"start": 0.0, "text": "mở đầu chương trình"},
        {"start": 60.0, "text": "bây giờ chúng ta cắt củ năng thành hạt lựu"},
        {"start": 120.0, "text": "kết thúc"},
    ])
    idx.build()
    at, _text = idx.best_segment("cắt củ năng hạt lựu", "V1")
    assert at == 60.0, "the passage, not the start of the video"


def test_both_transcript_formats_load():
    """The team's dumps are {segments:[{start,text}]}; fetch_captions writes [[t,text]]."""
    idx = TranscriptIndex()
    idx.add("A", [{"start": 1.0, "text": "một hai ba"}])
    idx.add("B", [[2.0, "bốn năm sáu"]])
    idx.build()
    assert idx.n_videos == 2
    assert idx.segments["B"][0] == (2.0, "bốn năm sáu")


def test_length_normalisation_stops_long_bulletins_dominating():
    """A 20-minute bulletin has 30x the words of a cooking clip; BM25's b handles it."""
    idx = TranscriptIndex()
    idx.add("SHORT", [{"start": 0.0, "text": "củ năng"}])
    idx.add("LONG", [{"start": 0.0, "text": "củ năng " + "chuyện khác " * 400}])
    idx.build()
    s = idx.score_videos("củ năng")
    assert s["SHORT"] > s["LONG"]


@pytest.mark.skipif(not TRANSCRIPTS.is_dir(), reason="no transcripts on this machine")
def test_the_real_corpus_finds_what_the_visual_index_missed():
    """Two cases checked by hand against the round-1 set.

    query-p1-4 describes battered, deep-fried asparagus; the visual ranking put
    a stir-fry first and MĂNG TÂY CHIÊN BIA third. query-p1-18 needs a dish with
    mushroom, water chestnut and tofu; CỦ NĂNG OM NẤM CHAY was not in the visual
    top six at all, and only two videos in the whole 873 mention "củ năng".
    """
    idx = TranscriptIndex().load_dir(TRANSCRIPTS, CAPTIONS)
    assert idx.n_videos > 500

    fried = idx.score_videos("măng tây tẩm bột chiên ngập dầu")
    assert fried, "no video mentions asparagus?"
    assert max(fried, key=fried.get) == "L26_V194"

    chestnut = idx.score_videos("củ năng")
    top3 = sorted(chestnut, key=chestnut.get, reverse=True)[:3]
    assert "L26_V012" in top3, f"CỦ NĂNG OM NẤM CHAY should surface, got {top3}"


@pytest.mark.skipif(not TRANSCRIPTS.is_dir(), reason="no transcripts on this machine")
def test_searching_is_fast_enough_to_use_inside_a_round():
    import time

    idx = TranscriptIndex().load_dir(TRANSCRIPTS, CAPTIONS)
    t0 = time.time()
    for _ in range(20):
        idx.score_videos("măng tây chiên bột trứng")
    per = (time.time() - t0) / 20
    assert per < 0.25, f"{per*1000:.0f} ms per query is too slow for interactive use"
