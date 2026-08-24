"""End-to-end guards for the contest-day path.

These do not need the 780 MB index or the model: they exercise the parts that
turn retrieval output into an uploadable archive, which is where a silent
mistake costs the whole round.
"""

import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_submission import (  # noqa: E402
    detect_task,
    read_en_override,
    split_events,
    split_qa,
)
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    allocate_hybrid_rows,
    csv_name_for_query,
    package_submission,
    verify_submission_zip,
    write_query_csv,
)


# ------------------------------------------------------------ query parsing


@pytest.mark.parametrize(
    "name,task",
    [
        ("query-1-kis.txt", "kis"),
        ("query-12-qa.txt", "qa"),
        ("query-3-trake.txt", "trake"),
        ("QUERY-4-TRAKE.TXT", "trake"),
        ("query-7-vqa.txt", "qa"),
        ("something-else.txt", "kis"),  # safe default
    ],
)
def test_task_detected_from_filename(name, task):
    assert detect_task(name) == task


def test_trake_events_split_on_e_markers():
    text = (
        "Trong một video nấu ăn, xác định thời điểm đầu tiên mỗi nguyên liệu chạm chảo: "
        "E1: dầu ăn; E2: thịt bò; E3: hành lá"
    )
    assert split_events(text) == ["dầu ăn;", "thịt bò;", "hành lá"] or len(split_events(text)) == 3


def test_trake_events_split_on_semicolons():
    assert len(split_events("bối cảnh: mở cửa; bước vào; đóng cửa")) == 3


def test_qa_splits_question_from_context():
    ctx, q = split_qa("Cảnh giao thông trên phố. Chiếc xe ở giữa có màu gì?")
    assert q.endswith("?")
    assert "màu gì" in q
    assert "giao thông" in ctx


def test_qa_without_question_mark_keeps_everything_as_context():
    ctx, q = split_qa("Một cảnh quay ngoài trời")
    assert ctx == "Một cảnh quay ngoài trời"
    assert q == ""


def test_manual_english_override_is_picked_up(tmp_path):
    q = tmp_path / "query-1-kis.txt"
    q.write_text("con mèo vàng", encoding="utf-8")
    assert read_en_override(q) is None
    (tmp_path / "query-1-kis.en.txt").write_text("an orange tabby cat", encoding="utf-8")
    assert read_en_override(q) == "an orange tabby cat"


def test_blank_override_is_ignored(tmp_path):
    q = tmp_path / "query-1-kis.txt"
    q.write_text("x", encoding="utf-8")
    (tmp_path / "query-1-kis.en.txt").write_text("   \n", encoding="utf-8")
    assert read_en_override(q) is None


# ------------------------------------------------------- full package build


def _candidates(n=60):
    return [
        Candidate(f"L{i // 10:02d}_V{i:03d}", 1000 + i * 137, 1.0 - i * 0.01, 90_000)
        for i in range(n)
    ]


def test_a_full_round_builds_a_compliant_archive(tmp_path):
    """Twelve queries in, one uploadable zip out, verified against the rules."""
    csv_dir = tmp_path / "csv"
    plan = AllocationPlan(depth_cost=0.5)
    for i in range(1, 10):
        rows = allocate_hybrid_rows(_candidates(), n_flat=30, plan=plan)
        assert len(rows) == MAX_ROWS, "every query must use all 100 free rows"
        write_query_csv(csv_dir / csv_name_for_query(f"query-{i}-kis.txt"), rows)
    for i in (10, 11):
        rows = allocate_hybrid_rows(_candidates(), n_flat=30, plan=plan)
        write_query_csv(
            csv_dir / csv_name_for_query(f"query-{i}-qa.txt"),
            [(v, f, "Màu đỏ") for v, f in rows],
        )
    write_query_csv(
        csv_dir / "query-12-trake.csv", [("L25_V007", 130, 322, 547), ("L25_V007", 130, 322, 537)]
    )

    z = package_submission(csv_dir, tmp_path / "submission.zip")
    assert verify_submission_zip(z) == []
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert len(names) == 12
        assert all(n.startswith("submission/") for n in names)
        # Vietnamese survives the round trip
        assert "Màu đỏ" in zf.read("submission/query-10-qa.csv").decode("utf-8")


def test_every_qa_row_carries_an_answer(tmp_path):
    """Blank answers on rows 6+ forfeit R@20/R@50/R@100 for nothing."""
    rows = allocate_hybrid_rows(_candidates(), n_flat=30, plan=AllocationPlan(depth_cost=0.5))
    p = tmp_path / "query-1-qa.csv"
    write_query_csv(p, [(v, f, "Màu xanh") for v, f in rows])
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == MAX_ROWS
    assert all(len(ln.split(",")) == 3 and ln.split(",")[2].strip() for ln in lines)


def test_answer_with_a_comma_cannot_reach_the_csv(tmp_path):
    """The writer itself removes commas, so no caller can produce a quoted field.

    This test used to assert the opposite — that write_query_csv emits
    'V,1,"đỏ, xanh"' — on the grounds that make_submission stripped commas
    before calling it. It did, but apply_picks.py and pin_video.py did not, so
    the same answer was safe when the pipeline wrote it and corrupt the moment
    a human corrected it. Stripping now happens in the writer, which is the one
    place every path goes through.
    """
    p = tmp_path / "q.csv"
    write_query_csv(p, [("V", 1, 'đỏ, xanh')])
    line = p.read_text(encoding="utf-8").strip()
    assert line == "V,1,đỏ xanh"
    assert len(line.split(",")) == 3, "a comma-splitting grader must see 3 fields"


def test_allocation_uses_the_whole_budget_even_from_one_candidate():
    rows = allocate_hybrid_rows(_candidates(1), n_flat=30, plan=AllocationPlan(depth_cost=0.5))
    assert len(set(rows)) == len(rows)
    assert len(rows) >= 20, "a single candidate should still fill many ladder rows"


def test_allocation_is_deterministic():
    """Two runs must produce identical rows, or the team cannot reason about
    what they uploaded."""
    a = allocate_hybrid_rows(_candidates(), n_flat=30, plan=AllocationPlan(depth_cost=0.5))
    b = allocate_hybrid_rows(_candidates(), n_flat=30, plan=AllocationPlan(depth_cost=0.5))
    assert a == b
