"""Tests for the submission formatter (risk R4 isolation)."""

import pytest

from src.task3_trake.formatter import format_csv, format_submission

EVENTS = [
    {"frame_idx": 1450, "ts_ms": 58000},
    {"frame_idx": 1802, "ts_ms": 72080},
    {"frame_idx": 2110, "ts_ms": 84400},
]


def test_frame_idx_line():
    assert format_submission("L21_V001", EVENTS) == "L21_V001,1450,1802,2110"


def test_ts_ms_line():
    assert format_submission("L21_V001", EVENTS, value_field="ts_ms") == (
        "L21_V001,58000,72080,84400"
    )


def test_answer_column():
    assert format_submission("L21_V001", EVENTS[:1], answer="42") == "L21_V001,1450,42"


def test_bad_field():
    with pytest.raises(ValueError):
        format_submission("V", EVENTS, value_field="gid")


def test_csv_multi():
    results = [
        {"video_id": "A", "events": EVENTS[:2]},
        {"video_id": "B", "events": EVENTS[1:]},
    ]
    lines = format_csv(results).splitlines()
    assert lines == ["A,1450,1802", "B,1802,2110"]


def test_csv_reuses_the_records_own_line():
    """The engine decides the column order; format_csv must not re-derive it."""
    results = [{"video_id": "A", "events": EVENTS[:2], "submission_line": "A,1802,1450"}]
    assert format_csv(results) == "A,1802,1450"


def test_csv_honours_value_field_over_the_cached_line():
    """A cached line is frame_idx-based; asking for ts_ms must re-derive it."""
    results = [{"video_id": "A", "events": EVENTS[:2], "submission_line": "A,1450,1802"}]
    assert format_csv(results, value_field="ts_ms") == "A,58000,72080"


def test_csv_skips_degraded_records():
    """A None submission_line means the frame numbers are not frame numbers.

    Regression: format_csv re-derived the line from events, defeating the
    guard and emitting a confidently wrong answer.
    """
    results = [
        {"video_id": "A", "events": EVENTS[:2], "submission_line": None, "warning": "no table"},
        {"video_id": "B", "events": EVENTS[1:], "submission_line": "B,1802,2110"},
    ]
    assert format_csv(results) == "B,1802,2110"
    with pytest.raises(ValueError, match="A"):
        format_csv(results, strict=True)
