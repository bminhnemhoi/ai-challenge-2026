"""The human-in-the-loop path: review page -> pick string -> corrected CSVs.

Every one of these guards a mistake that would silently cost points on contest
day rather than raise an error.
"""

from __future__ import annotations

import re

import pytest

from scripts.apply_picks import parse_picks


def test_single_frame_pick():
    assert parse_picks("query-p1-1-kis=L21_V015:25605") == [
        ("query-p1-1-kis", "L21_V015", 25605, None)
    ]


def test_trake_pick_keeps_the_whole_chain():
    """The operator approves a sequence, not a frame.

    Only event 1 used to survive and the rest were re-derived, so the chain that
    got submitted was not the chain that was looked at.
    """
    (q, v, f, a), = parse_picks("query-p1-18-trake=L26_V214:2315|2432|2546|2809")
    assert (q, v, a) == ("query-p1-18-trake", "L26_V214", None)
    assert f == [2315, 2432, 2546, 2809]


def test_qa_answer_survives_diacritics():
    """A Vietnamese answer must come through byte-identical or it is marked wrong."""
    (_, _, frame, answer), = parse_picks("query-p1-15-qa=L30_V072:5376:Xã Vạn Thắng")
    assert frame == 5376
    assert answer == "Xã Vạn Thắng"


def test_answer_may_contain_colons():
    """split(':', 2) keeps everything after the frame as one answer."""
    (_, _, _, answer), = parse_picks("q=V:1:Chương trình: Cùng em đến trường")
    assert answer == "Chương trình: Cùng em đến trường"


def test_several_picks_and_newlines():
    picks = parse_picks("a=V1:1;b=V2:2\n\nc=V3:3|4  ; ")
    assert [p[0] for p in picks] == ["a", "b", "c"]
    assert picks[2][2] == [3, 4]


def test_missing_frame_means_let_the_engine_choose():
    (_, video, frame, _), = parse_picks("q=L21_V001:")
    assert (video, frame) == ("L21_V001", None)


def test_malformed_pick_is_rejected_loudly():
    with pytest.raises(ValueError):
        parse_picks("query-p1-1-kis L21_V015:25605")


def _render_page():
    """The page exactly as build_review_page writes it, with an empty round."""
    from pathlib import Path

    from scripts.build_review_page import PAGE

    alloc = (Path(__file__).resolve().parents[1] / "scripts" / "review_export.js").read_text(
        encoding="utf-8"
    )
    return PAGE.format(
        body="", qdir="q", outdir="o", nq=0, warnlist="[]", tag="t",
        alloc_js=alloc, data_json="{}", vid_json="{}", cdn="https://example.invalid",
        local_json='"file:///C:/mirror"',
        plan_json='{"breadthCost":1.0,"depthCost":0.5,"step":10,'
                  '"budget":100,"maxDepth":24,"nFlat":30}',
    )


def test_review_page_template_has_no_unfilled_placeholders():
    """PAGE.format() must consume every {name}; a stray one ships broken HTML."""
    filled = _render_page()
    leftover = [m.group(0) for m in re.finditer(r"(?<!\$)\{[a-z_]+\}", filled)]
    assert leftover == [], f"unfilled placeholders: {leftover}"
    # every {{ }} written for str.format must have collapsed to a single brace
    assert "{{" not in filled and "}}" not in filled


def test_review_page_falls_back_to_the_local_mirror_when_the_cdn_dies():
    """The page is the round's most valuable tool and it used to be 100% CDN.

    A contest-room network blip must not kill the 55 minutes of eyeball review,
    so every image that fails to load off the CDN retries once from the on-disk
    mirror (data/frames, same <video>/NNN.jpg layout). The handler has to be on
    the CAPTURE phase — img error events do not bubble — and must mark the
    element so a missing mirror file cannot loop forever.
    """
    filled = _render_page()
    assert "addEventListener('error'" in filled
    assert "}, true);" in filled, "img error events do not bubble; capture phase required"
    assert "LOCAL + t.src.slice(CDN.length)" in filled
    assert "dataset.fbk" in filled, "must not retry the same image forever"


def test_review_page_offers_an_answer_box_for_qa_only():
    from scripts.build_review_page import PAGE

    assert "ansbox" in PAGE
    assert "apply_picks.py" in PAGE


def test_every_operator_tool_ranks_the_way_the_submission_does():
    """review.html, inspect_run, apply_picks and the sweep must all use ranked_hits.

    make_submission builds its CSVs from ranked_hits(), which folds a
    hand-written .en.txt into the ranking.  A tool that calls engine.search()
    instead ranks differently the moment such a file exists — and the runbook
    tells the team to write them.  The operator would then approve a frame that
    is not the one at row 1, which is the worst possible failure: silent, and
    it corrupts exactly the human judgement the whole loop exists to capture.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    must_agree = [
        "build_review_page.py",
        "inspect_run.py",
        "apply_picks.py",
        "pin_video.py",
        "rerank_vlm.py",
        "evaluate_official.py",
        "experiment_allocation.py",
    ]
    offenders = []
    for name in must_agree:
        src = (root / "scripts" / name).read_text(encoding="utf-8")
        # a bare engine.search(...) outside a comment is the bug
        for m in re.finditer(r"^\s*[^#\n]*\b(?:eng|engine)\.search\(", src, re.M):
            offenders.append(f"{name}: {m.group(0).strip()}")
        assert "ranked_hits" in src, f"{name} does not use ranked_hits at all"
    assert offenders == [], (
        "these bypass ranked_hits, so what the operator sees is not what gets "
        f"submitted: {offenders}"
    )


def test_generated_javascript_actually_parses():
    """The review page's whole behaviour lives in one inline <script>.

    A browser discards an inline script with a parse error IN FULL, so a single
    bad character makes every click, every keyboard shortcut and every button
    silently do nothing — while the page still renders and looks fine. That
    already happened once: PAGE is a non-raw triple-quoted Python string, so a
    `\n` written for JavaScript was expanded by Python into a real newline
    inside a single-quoted JS literal, and the entire review loop was inert.

    Checked with node when it is available, and with a quote-balance scan
    otherwise, so this holds on a machine without node too.
    """
    import re
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    html = _render_page()
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(blocks) >= 2, "expected the allocator block and the page block"

    node = shutil.which("node")
    if node:
        # the real parser, so no heuristic can return a false verdict, and each
        # block is checked separately because a browser parses them separately
        for i, block in enumerate(blocks):
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / f"block{i}.js"
                f.write_text(block, encoding="utf-8")
                r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
            assert r.returncode == 0, f"node rejects <script> block {i}:\n{r.stderr}"
        return

    js = "\n".join(blocks)

    # No node available: catch the one failure mode that actually occurred, a
    # raw line break inside a '...' or "..." literal. Regex and template
    # literals legally carry unpaired quotes, so strip those first.
    unbalanced = []
    for i, line in enumerate(js.splitlines(), 1):
        s = re.sub(r"\\.", "", line)                  # an escape is not a delimiter
        s = re.sub(r"`[^`]*`", "", s)                 # template literals span lines
        s = re.sub(r"/\[[^\]]*\]/[gimsuy]*", "", s)   # /[;,"]/g and friends
        if s.count("'") % 2 or s.count('"') % 2:
            unbalanced.append(f"line {i}: {line.strip()[:70]}")
    assert unbalanced == [], f"unterminated JS string literal: {unbalanced}"


def test_answer_with_a_comma_cannot_corrupt_the_csv(tmp_path):
    """Round-1 asks "Hai câu thơ đó là gì?" — the answer will contain commas.

    csv.writer would quote such a field. That is valid RFC 4180 and still
    scores wrong, because a grader that splits on commas reads it as several
    fields with a stray quote, and apply_picks reading it back with
    split(',')[2] truncates it a second time.
    """
    from src.core.submission import verify_submission_zip  # noqa: F401
    from src.core.submission import package_submission, write_query_csv

    csv_dir = tmp_path / "csv"
    answer = 'Quê hương là chùm khế ngọt, cho con trèo hái mỗi ngày'
    write_query_csv(csv_dir / "query-p1-19-qa.csv", [("L28_V012", 20790, answer)])

    raw = (csv_dir / "query-p1-19-qa.csv").read_text(encoding="utf-8")
    assert '"' not in raw, f"a field got RFC-quoted: {raw!r}"
    fields = raw.splitlines()[0].split(",")
    assert len(fields) == 3, f"naive comma split must see exactly 3 fields, got {fields}"
    assert "Quê hương là chùm khế ngọt" in fields[2]
    assert "trèo hái mỗi ngày" in fields[2]

    zp = tmp_path / "s.zip"
    package_submission(csv_dir, zp)
    assert verify_submission_zip(zp) == []


def test_verifier_rejects_a_quoted_field_written_by_something_else():
    """If a quote ever reaches a CSV, the upload must be blocked, not passed."""
    import zipfile

    from src.core.submission import verify_submission_zip

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        zp = Path(d) / "bad.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("submission/query-p1-19-qa.csv", 'L28_V012,20790,"a, b"\n')
        problems = verify_submission_zip(zp)
    assert any("double quote" in p for p in problems), problems


def test_review_page_strips_commas_from_typed_answers():
    from scripts.build_review_page import PAGE

    assert '/[;,"' in PAGE, "the answer box must strip , and \" as well as ;"


def test_a_confirmed_frame_gets_its_ladder_immediately():
    """Ranks 2-5 after a human pick must be F±step, not other keyframes.

    Measured on the ground truth for exactly this scenario, laddering straight
    off the confirmed frame scores 0.810 against 0.654 for n_flat=30 — the
    other keyframes of the same video sit 55+ frames away and cannot land in a
    ~10 frame answer window, while F±10 and F±20 can.
    """
    from src.core.submission import AllocationPlan, Candidate, allocate_hybrid_rows

    confirmed = 10_000
    cands = [Candidate("V", confirmed, 1e9, 50_000)] + [
        Candidate("V", confirmed + 55 * k, 1.0 - k / 100, 50_000) for k in range(1, 40)
    ]
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=0.5, step=10, budget=50)
    rows = allocate_hybrid_rows(cands, n_flat=1, plan=plan)

    assert rows[0] == ("V", confirmed), "the confirmed frame must be row 1"
    # rows 2 and 3 are the immediate neighbours; the sweep then interleaves the
    # next keyframe, which is correct — it hedges a slightly-off confirmation
    assert {f - confirmed for _v, f in rows[1:3]} == {-10, 10}
    # what matters for R@5/R@20 is that the ladder dominates the early ranks
    close = sum(1 for _v, f in rows[:8] if abs(f - confirmed) <= 30)
    assert close >= 5, f"only {close}/8 early rows are near the confirmed frame: {rows[:8]}"

    # and the old behaviour must not come back: with n_flat=30 the ladder is
    # pushed out of R@20 entirely
    wide = allocate_hybrid_rows(cands, n_flat=30, plan=plan)
    assert sum(1 for _v, f in wide[:8] if abs(f - confirmed) <= 30) < close


def test_a_picks_file_may_carry_comments():
    """The reason a correction is right is the part worth keeping.

    A picks FILE exists so that reasoning can sit beside the pick; without
    comment support it is no better than the command line.
    """
    picks = parse_picks(
        "# OCR proved the submitted frame is a different stele\n"
        "query-p1-19-qa=L27_V010:3275:Hỏa hồng Nhựt Tảo oanh thiên địa\n"
        "\n"
        "  # transcript: 'lớp học làm bánh miễn phí'\n"
        "query-p1-22-qa=L30_V078:1623:Nhân bánh cuốn\n"
    )
    assert [p[0] for p in picks] == ["query-p1-19-qa", "query-p1-22-qa"]
    assert picks[1][1:] == ("L30_V078", 1623, "Nhân bánh cuốn")


def test_a_slash_in_an_answer_survives():
    """Two lines of verse are naturally written with a separator."""
    (_, _, _, a), = parse_picks("q=V:1:Hỏa hồng Nhựt Tảo oanh thiên địa / Kiếm bạc Kiên Giang")
    assert a == "Hỏa hồng Nhựt Tảo oanh thiên địa / Kiếm bạc Kiên Giang"
