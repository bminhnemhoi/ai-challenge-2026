"""Tests for the official scoring / packaging module.

Every scoring assertion here is taken from a worked example in the BTC rules
document, so a failure means we have drifted from the published rules.
"""

import zipfile
from pathlib import Path

import pytest

from src.core.submission import (
    MAX_ROWS,
    RANK_THRESHOLDS,
    AllocationPlan,
    Candidate,
    allocate_kis_rows,
    allocate_trake_rows,
    csv_name_for_query,
    final_score,
    frame_ladder,
    package_submission,
    r_at_k,
    r_score_kis,
    r_score_qa,
    r_score_trake,
    score_bucket,
    verify_submission_zip,
    write_query_csv,
)

# --------------------------------------------------------------- R-Score
# Rules 2.1.1: GT is L01_V001, frames 500-510.


def test_kis_rscore_matches_the_rulebook_example():
    assert r_score_kis("L01_V001", 505, "L01_V001", (500, 510)) == 1.0
    assert r_score_kis("L01_V001", 600, "L01_V001", (500, 510)) == 0.0  # right video, wrong frame
    assert r_score_kis("L02_V003", 505, "L01_V001", (500, 510)) == 0.0  # wrong video


def test_kis_rscore_boundaries_are_inclusive():
    assert r_score_kis("V", 500, "V", (500, 510)) == 1.0
    assert r_score_kis("V", 510, "V", (500, 510)) == 1.0
    assert r_score_kis("V", 499, "V", (500, 510)) == 0.0
    assert r_score_kis("V", 511, "V", (500, 510)) == 0.0


def test_qa_rscore_needs_frame_and_answer():
    # rules 2.1.2 example: L05_V005, [800,900], answer "màu xanh"
    ok = dict(gt_video="L05_V005", gt_span=(800, 900), gt_answer="màu xanh")
    assert r_score_qa("L05_V005", 888, "màu xanh", **ok) == 1.0
    assert r_score_qa("L05_V005", 888, "màu trắng", **ok) == 0.0  # wrong answer
    assert r_score_qa("L06_V007", 888, "màu xanh", **ok) == 0.0  # wrong video
    assert r_score_qa("L05_V005", 950, "màu xanh", **ok) == 0.0  # frame outside window


def test_trake_rscore_matches_the_rulebook_example():
    # rules 2.1.3: L10_V010 with four windows; the team answers 101,156,203,251
    spans = [(95, 105), (145, 155), (195, 205), (245, 255)]
    assert r_score_trake("L10_V010", [101, 156, 203, 251], "L10_V010", spans) == pytest.approx(0.75)
    assert r_score_trake("L10_V010", [101, 150, 203, 251], "L10_V010", spans) == pytest.approx(1.0)


def test_trake_wrong_video_scores_zero_even_with_perfect_frames():
    spans = [(95, 105), (145, 155)]
    assert r_score_trake("OTHER", [100, 150], "L10_V010", spans) == 0.0


# ----------------------------------------------------------- Final Score


def test_final_score_matches_the_rulebook_worked_example():
    """Rules 2.2: first row 0.5, row 3 is 0.8 (the max), row 15 is 0.6."""
    scores = [0.5, 0.0, 0.8] + [0.0] * 11 + [0.6] + [0.0] * 84
    assert r_at_k(scores, 1) == 0.5
    assert r_at_k(scores, 5) == 0.8
    assert final_score(scores) == pytest.approx((0.5 + 0.8 + 0.8 + 0.8 + 0.8) / 5)
    assert final_score(scores) == pytest.approx(0.74)


def test_extra_wrong_rows_can_never_reduce_the_score():
    """R@k is a maximum over a prefix — the reason we always submit 100 rows."""
    base = [0.0, 1.0] + [0.0] * 8
    padded = base + [0.0] * 90
    assert final_score(padded) >= final_score(base)


def test_score_buckets_are_a_step_function_of_the_first_hit():
    assert score_bucket(1) == pytest.approx(1.0)
    assert score_bucket(2) == score_bucket(5) == pytest.approx(0.8)
    assert score_bucket(6) == score_bucket(20) == pytest.approx(0.6)
    assert score_bucket(21) == score_bucket(50) == pytest.approx(0.4)
    assert score_bucket(51) == score_bucket(100) == pytest.approx(0.2)
    assert score_bucket(101) == 0.0
    assert score_bucket(None) == 0.0


def test_moving_a_hit_within_a_bucket_is_worth_nothing():
    """Where tuning pays, and where it does not."""
    assert score_bucket(15) == score_bucket(10)  # no gain
    assert score_bucket(5) - score_bucket(6) == pytest.approx(0.2)  # real gain


# --------------------------------------------------------- frame ladders


def test_ladder_is_ordered_by_distance_so_truncation_keeps_the_best():
    assert frame_ladder(1000, 5, step=10) == [1000, 990, 1010, 980, 1020]


def test_ladder_respects_video_bounds():
    assert all(x >= 0 for x in frame_ladder(5, 8, step=10, lo=0))
    assert all(x <= 100 for x in frame_ladder(95, 8, step=10, lo=0, hi=100))


def test_ladder_step_covers_any_window_at_least_that_wide():
    """A window of `step`+1 frames inside the ladder span must contain an id."""
    step = 10
    ids = set(frame_ladder(1000, 11, step=step))
    span = max(ids) - min(ids)
    for s in range(1000 - span // 2, 1000 + span // 2 - step):
        window = range(s, s + step + 1)
        assert any(i in ids for i in window), f"window starting {s} not covered"


def test_ladder_deduplicates():
    assert len(set(frame_ladder(50, 12, step=10, lo=0, hi=60))) == len(
        frame_ladder(50, 12, step=10, lo=0, hi=60)
    )


# ------------------------------------------------------------ allocation


def _cands(n, spacing=5000):
    return [
        Candidate(f"V{i:03d}", 1000 + i * spacing, score=1.0 - i * 0.01, video_last_frame=999_999)
        for i in range(n)
    ]


def test_allocation_puts_the_single_best_keyframe_at_rank_one():
    rows = allocate_kis_rows(_cands(30))
    assert rows[0] == ("V000", 1000)


def test_allocation_fills_the_full_budget_when_candidates_allow():
    rows = allocate_kis_rows(_cands(60))
    assert len(rows) == MAX_ROWS


def test_allocation_never_repeats_a_row():
    rows = allocate_kis_rows(_cands(60))
    assert len(set(rows)) == len(rows)


def test_allocation_covers_several_videos_inside_the_top_five():
    """Rank 2-5 is worth 0.8; spending it all on one video risks the whole bucket."""
    rows = allocate_kis_rows(_cands(30))
    assert len({v for v, _ in rows[:5]}) >= 2


def test_allocation_gives_the_top_video_a_real_ladder_by_rank_twenty():
    rows = allocate_kis_rows(_cands(30))
    top_frames = sorted({f for v, f in rows[:20] if v == "V000"})
    assert len(top_frames) >= 3, f"top video only got {top_frames} by rank 20"
    assert max(top_frames) - min(top_frames) >= 20


def test_allocation_handles_a_single_candidate():
    rows = allocate_kis_rows(_cands(1))
    assert rows[0] == ("V000", 1000)
    assert all(v == "V000" for v, _ in rows)
    assert len(set(rows)) == len(rows)


def test_allocation_respects_a_smaller_budget():
    rows = allocate_kis_rows(_cands(30), AllocationPlan(budget=17))
    assert len(rows) == 17


def test_allocation_keeps_ladders_inside_short_videos():
    c = [Candidate("V", 30, video_last_frame=60)]
    rows = allocate_kis_rows(c, AllocationPlan(budget=20))
    assert all(0 <= f <= 60 for _, f in rows)


# ------------------------------------------------------- TRAKE allocation


def test_trake_rows_all_use_one_video_and_start_from_the_base_guess():
    rows = allocate_trake_rows("L10_V010", [100, 150, 200], budget=40)
    assert all(v == "L10_V010" for v, _ in rows)
    assert rows[0][1] == [100, 150, 200]


def test_trake_rows_are_ordered_by_total_displacement():
    rows = allocate_trake_rows("V", [100, 200], budget=30, step=10)
    disp = [sum(abs(f - b) for f, b in zip(frames, [100, 200])) for _, frames in rows]
    assert disp == sorted(disp)


def test_trake_perturbation_recovers_a_near_miss():
    """The point of the product ladder: one event slightly off must be fixable."""
    spans = [(95, 105), (145, 155), (195, 205)]
    base = [100, 170, 200]  # event 2 is 15 frames late
    rows = allocate_trake_rows("V", base, budget=MAX_ROWS, step=10)
    best = max(r_score_trake(v, f, "V", spans) for v, f in rows)
    assert best == pytest.approx(1.0)


def test_trake_respects_budget_and_dedupes():
    rows = allocate_trake_rows("V", [100, 200, 300, 400], budget=MAX_ROWS)
    assert len(rows) <= MAX_ROWS
    assert len({tuple(f) for _, f in rows}) == len(rows)


# ------------------------------------------------------------- packaging


def test_csv_name_follows_the_query_file_stem():
    assert csv_name_for_query("query-1-kis.txt") == "query-1-kis.csv"
    assert csv_name_for_query("/a/b/query-12-qa.txt") == "query-12-qa.csv"


def test_zip_contains_the_mandatory_submission_folder(tmp_path):
    d = tmp_path / "csvs"
    # 100 rows, because a shorter file now (correctly) draws a truncation warning
    write_query_csv(d / "query-1-kis.csv", [("L01_V001", 505 + 10 * i) for i in range(100)])
    z = package_submission(d, tmp_path / "sub.zip")
    with zipfile.ZipFile(z) as zf:
        assert zf.namelist() == ["submission/query-1-kis.csv"]
    assert verify_submission_zip(z) == []


def test_verifier_rejects_a_header_row(tmp_path):
    d = tmp_path / "csvs"
    write_query_csv(d / "query-1-kis.csv", [("video_id", "frame_id"), ("L01_V001", 505)])
    z = package_submission(d, tmp_path / "sub.zip")
    problems = verify_submission_zip(z)
    assert any("header" in p for p in problems)


def test_verifier_rejects_mp4_extension(tmp_path):
    d = tmp_path / "csvs"
    write_query_csv(d / "query-1-kis.csv", [("L01_V001.mp4", 505)])
    z = package_submission(d, tmp_path / "sub.zip")
    assert any("mp4" in p for p in verify_submission_zip(z))


def test_verifier_rejects_more_than_100_rows(tmp_path):
    d = tmp_path / "csvs"
    write_query_csv(d / "query-1-kis.csv", [("V", i) for i in range(101)])
    z = package_submission(d, tmp_path / "sub.zip")
    assert any("exceeds" in p for p in verify_submission_zip(z))


def test_csv_has_no_bom_and_no_crlf(tmp_path):
    p = tmp_path / "query-1-kis.csv"
    write_query_csv(p, [("L01_V001", 505), ("L01_V001", 515)])
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert raw.decode("utf-8") == "L01_V001,505\nL01_V001,515\n"


def test_trake_csv_is_variable_width(tmp_path):
    p = tmp_path / "query-3-trake.csv"
    write_query_csv(p, [("V", 10, 20, 30), ("V", 11, 21, 31)])
    assert p.read_text(encoding="utf-8") == "V,10,20,30\nV,11,21,31\n"


def test_packaging_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError):
        package_submission(tmp_path / "nope", tmp_path / "x.zip")


def test_rank_thresholds_are_the_published_ones():
    assert RANK_THRESHOLDS == (1, 5, 20, 50, 100)


def test_a_truncated_csv_is_flagged_even_though_it_is_legal(tmp_path):
    """Round 2, live: a by-hand edit deleted rows by their answer text and left a
    4-row file. The rules allow it (the limit is a maximum), so the verifier said
    "format check passed" — and 96 free R@100 chances went in the bin. This
    pipeline always fills 100 rows, so a short file can only mean truncation.
    """
    import zipfile

    from src.core.submission import verify_submission_zip

    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    (csv_dir / "query-1-kis.csv").write_bytes(b"L01_V001,10\nL01_V001,20\n")
    zp = tmp_path / "submission.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.write(csv_dir / "query-1-kis.csv", "submission/query-1-kis.csv")
    problems = verify_submission_zip(zp)
    assert any("only 2 rows" in p for p in problems), problems
