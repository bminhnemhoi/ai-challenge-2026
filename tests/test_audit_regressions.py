"""Regression tests for defects an adversarial audit found in this pipeline.

Every test here corresponds to a bug that would have cost real points in a
contest round, several of them silently.
"""

import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_submission import (  # noqa: E402
    detect_task,
    read_query_text,
    split_events,
)
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    allocate_trake_rows,
    package_submission,
    r_score_trake,
    verify_submission_zip,
    write_query_csv,
)


# ---------------------------------------------- query parsing / decoding


def test_rulebook_numbered_trake_phrasing_yields_the_right_event_count():
    """The event count sets the CSV column count; getting it wrong scores 0.

    This is the rulebook's own example phrasing, which the first parser
    collapsed into a single event.
    """
    text = (
        "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: "
        "(1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy."
    )
    events = split_events(text)
    assert len(events) == 4, events
    assert "giậm nhảy" in events[0]
    # the shared stem must be carried into each event or retrieval has no context
    assert "vận động viên" in events[0]


def test_e_marker_and_semicolon_forms_still_work():
    assert len(split_events("bối cảnh: E1: mở cửa; E2: bước vào; E3: đóng cửa")) == 3
    assert len(split_events("nấu ăn: dầu ăn; thịt bò; hành lá")) == 3


def test_a_scene_setting_first_line_is_not_counted_as_an_event():
    """Round 1's real TRAKE prompt, verbatim in shape.

    The markers carry no punctuation — "E1 Khoảnh khắc", not "E1: Khoảnh khắc" —
    so the marker split missed and the line fallback counted the opening
    description as a fourth event. That writes five columns into a file the key
    scores with four, which loses the whole query rather than one event.
    """
    text = (
        "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ, "
        "bên cạnh lá cờ trắng viền đỏ.\r\n"
        "E1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay vòng.\r\n"
        "E2 Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên các thanh trụ.\r\n"
        "E3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân."
    )
    events = split_events(text)
    assert len(events) == 3, events
    assert events[0].startswith("Khoảnh khắc đầu tiên xuất hiện")
    assert "rồng vàng" in events[0]
    assert "kẻng đồng" in events[2]
    # the opening description is context, not a moment to locate
    assert not any("bắt đầu bằng ảnh cận đầu" in e for e in events)


def test_an_e_marker_inside_a_sentence_does_not_split_unpunctuated():
    """Only a marker that OPENS a line may go without punctuation."""
    text = "Người mặc áo E1 đứng bên cạnh xe. E1: một E2: hai E3: ba"
    assert len(split_events(text)) == 3


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_query_files_decode_in_any_common_encoding(tmp_path, encoding):
    """A UTF-16 query file used to abort the whole run and produce no zip."""
    p = tmp_path / "query-1-kis.txt"
    p.write_text("Con mèo vàng trên sân", encoding=encoding)
    got = read_query_text(p)
    assert got == "Con mèo vàng trên sân"
    assert not got.startswith("﻿")


def test_undecodable_file_returns_none_rather_than_raising(tmp_path):
    p = tmp_path / "query-1-kis.txt"
    p.write_bytes(b"\xff\xfe\x00\x00\xd8\x00")
    read_query_text(p)  # must not raise; None or garbage is fine, a crash is not


# ------------------------------------------------------ TRAKE allocation


@pytest.mark.parametrize("n_events", [1, 2, 3, 4])
def test_trake_fills_the_budget_for_any_event_count(n_events):
    """A fixed 5-value offset ladder gave a 2-event query only 25 of 100 rows."""
    base = [1000 + 500 * i for i in range(n_events)]
    rows = allocate_trake_rows("V", base, budget=MAX_ROWS, step=10, video_last_frame=90_000)
    assert len(rows) == MAX_ROWS, f"{n_events} events produced only {len(rows)} rows"
    assert len({tuple(f) for _, f in rows}) == len(rows)


def test_trake_perturbs_every_event_not_just_the_last():
    """Depth-first truncation froze the leading coordinates for large N."""
    n = 6
    base = [1000 + 500 * i for i in range(n)]
    rows = allocate_trake_rows("V", base, budget=MAX_ROWS, step=10, video_last_frame=90_000)
    for i in range(n):
        moved = {f[i] for _, f in rows}
        assert len(moved) > 1, f"event {i} was never perturbed across {len(rows)} rows"


def test_trake_is_ordered_by_total_displacement():
    base = [1000, 2000, 3000]
    rows = allocate_trake_rows("V", base, budget=60, step=10, video_last_frame=90_000)
    disp = [sum(abs(f - b) for f, b in zip(fr, base)) for _, fr in rows]
    assert disp == sorted(disp)
    assert rows[0][1] == base


def test_trake_recovers_a_near_miss_on_a_middle_event():
    spans = [(995, 1005), (2495, 2505), (2995, 3005)]
    base = [1000, 2480, 3000]  # event 2 is 20 frames early
    rows = allocate_trake_rows("V", base, budget=MAX_ROWS, step=10, video_last_frame=90_000)
    assert max(r_score_trake(v, f, "V", spans) for v, f in rows) == pytest.approx(1.0)


def test_trake_large_n_is_fast():
    """The old full-product enumeration was 5**n; n=8 must still be instant."""
    import time

    t0 = time.time()
    rows = allocate_trake_rows("V", list(range(1000, 1000 + 8 * 400, 400)), budget=MAX_ROWS)
    assert len(rows) == MAX_ROWS
    assert time.time() - t0 < 2.0


# ---------------------------------------------------- submission verifier


def _zip_with(tmp_path, name, rows):
    d = tmp_path / "csv"
    write_query_csv(d / name, rows)
    return package_submission(d, tmp_path / "s.zip")


def test_verifier_catches_a_header_hidden_below_row_one(tmp_path):
    """Only row 1 used to be parsed."""
    rows = [("V", i) for i in range(30)] + [("video_id", "frame_id")] + [("V", 99)]
    z = _zip_with(tmp_path, "query-1-kis.csv", rows)
    assert any("row 31" in p for p in verify_submission_zip(z))


def test_verifier_catches_blank_qa_answers(tmp_path):
    """Blank answers score 0 under rules 2.1.2 no matter how good the frame is."""
    z = _zip_with(tmp_path, "query-9-qa.csv", [("V", 100, ""), ("V", 110, "")])
    problems = verify_submission_zip(z)
    assert any("blank answer" in p for p in problems)


def test_verifier_accepts_a_filled_qa_file(tmp_path):
    # 100 rows: a shorter file now (correctly) draws the truncation warning
    z = _zip_with(tmp_path, "query-9-qa.csv",
                  [("V", 100 + 10 * i, "Màu đỏ") for i in range(100)])
    assert verify_submission_zip(z) == []


def test_verifier_catches_a_missing_query_csv(tmp_path):
    """A query that crashed leaves no file, and the archive still looks valid."""
    z = _zip_with(tmp_path, "query-1-kis.csv", [("V", 100)])
    problems = verify_submission_zip(z, expect_names={"query-1-kis.csv", "query-2-kis.csv"})
    assert any("query-2-kis.csv" in p for p in problems)


def test_verifier_catches_a_utf8_bom(tmp_path):
    d = tmp_path / "csv"
    d.mkdir(parents=True)
    (d / "query-1-kis.csv").write_bytes(b"\xef\xbb\xbfL01_V001,505\n")
    z = package_submission(d, tmp_path / "s.zip")
    assert any("BOM" in p for p in verify_submission_zip(z))


def test_verifier_reports_rather_than_crashes_on_non_utf8(tmp_path):
    d = tmp_path / "csv"
    d.mkdir(parents=True)
    (d / "query-1-kis.csv").write_bytes(b"L01_V001,505\n\xff\xfe\x00bad\n")
    z = package_submission(d, tmp_path / "s.zip")
    problems = verify_submission_zip(z)  # must not raise
    assert problems


def test_qa_detection_matches_the_submission_tool():
    """The verifier and the builder must agree on which files are Q&A."""
    from src.core.submission import _is_qa_name

    for name in ("query-9-qa.csv", "query-12-vqa.csv"):
        assert _is_qa_name(name)
        assert detect_task(name.replace(".csv", ".txt")) == "qa"
    for name in ("query-1-kis.csv", "query-3-trake.csv"):
        assert not _is_qa_name(name)
