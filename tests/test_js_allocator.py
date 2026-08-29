"""The browser allocator must produce the same rows as the Python one.

review.html builds the finished upload itself, so scripts/review_export.js is a
second implementation of src/core/submission.py. Two implementations of a
scoring rule are only safe while something forces them to agree — otherwise the
page shows one ranking and the zip contains another, which is precisely the
failure this whole review loop exists to prevent.

Skipped when node is unavailable, so the suite still runs on a bare machine.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.core.submission import (
    AllocationPlan,
    Candidate,
    allocate_coverage_rows,
    allocate_hybrid_rows,
    frame_ladder,
)

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "scripts" / "review_export.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _run_js(snippet: str):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "run.js"
        f.write_text(
            f'const A = require({json.dumps(str(JS))});\n{snippet}\n', encoding="utf-8"
        )
        r = subprocess.run(
            [shutil.which("node"), str(f)], capture_output=True, text=True, encoding="utf-8"
        )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _random_candidates(seed: int, n: int):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        vid = f"L{rng.randint(21, 30)}_V{rng.randint(1, 400):03d}"
        last = rng.randint(2000, 60000)
        out.append(Candidate(vid, rng.randint(0, last), rng.random(), last))
    return out


def test_frame_ladder_matches():
    cases = [(1000, 24, 10, 0, 50000), (5, 24, 10, 0, 100), (49995, 24, 14, 0, 50000)]
    got = _run_js(
        "console.log(JSON.stringify("
        + json.dumps(cases)
        + ".map(c => A.frameLadder(c[0], c[1], c[2], c[3], c[4]))))"
    )
    want = [frame_ladder(c, n, s, lo=lo, hi=hi) for c, n, s, lo, hi in cases]
    assert got == want


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11])
def test_hybrid_allocation_matches_row_for_row(seed):
    """Fuzzed: the same candidates must yield the same 100 rows, in the same order."""
    cands = _random_candidates(seed, 60)
    payload = [{"v": c.video_id, "f": c.frame_idx, "last": c.video_last_frame} for c in cands]
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=0.5, step=10, budget=100, max_depth=24)

    got = _run_js(
        f"const plan = {{breadthCost:1.0, depthCost:0.5, step:10, budget:100, maxDepth:24}};\n"
        f"console.log(JSON.stringify(A.allocateHybridRows({json.dumps(payload)}, 30, plan)))"
    )
    want = [list(r) for r in allocate_hybrid_rows(cands, n_flat=30, plan=plan)]
    assert len(got) == len(want) == 100
    assert got == want


def test_csv_text_is_byte_identical_to_write_query_csv(tmp_path):
    from src.core.submission import write_query_csv

    rows = [("L21_V001", 100), ("L21_V001", 110), ("L22_V002", 55)]
    got = _run_js("console.log(JSON.stringify(A.rowsToCsv(" + json.dumps([list(r) for r in rows]) + ")))")
    p = tmp_path / "q.csv"
    write_query_csv(p, rows)
    assert got == p.read_text(encoding="utf-8")


def test_qa_answer_is_sanitised_the_same_way(tmp_path):
    from src.core.submission import write_query_csv

    rows = [("L28_V012", 20790, 'Quê hương là chùm khế ngọt, cho con trèo hái mỗi ngày')]
    got = _run_js("console.log(JSON.stringify(A.rowsToCsv(" + json.dumps([list(r) for r in rows]) + ")))")
    p = tmp_path / "q.csv"
    write_query_csv(p, rows)
    assert got == p.read_text(encoding="utf-8")
    assert '"' not in got


def test_browser_zip_passes_the_real_verifier(tmp_path):
    """The zip the page hands the operator must satisfy verify_submission_zip."""
    from src.core.submission import verify_submission_zip

    # full 100-row files, matching what the page really exports — shorter files
    # now (correctly) trip the verifier's truncation guard
    files = [
        {"name": "submission/query-p1-1-kis.csv",
         "text": "".join(f"L21_V001,{100 + 10 * i}\n" for i in range(100))},
        {"name": "submission/query-p1-15-qa.csv",
         "text": "".join(f"L30_V072,{1745 + 10 * i},Xã Vạn Thắng\n" for i in range(100))},
        {"name": "submission/query-p1-4-trake.csv",
         "text": "".join(f"L26_V208,{10 + i},{20 + i},{30 + i},{40 + i}\n" for i in range(100))},
    ]
    data = _run_js(
        "console.log(JSON.stringify(Array.from(A.buildZip(" + json.dumps(files) + "))))"
    )
    zp = tmp_path / "submission.zip"
    zp.write_bytes(bytes(data))

    import zipfile

    with zipfile.ZipFile(zp) as zf:
        assert zf.testzip() is None
        assert sorted(zf.namelist()) == sorted(f["name"] for f in files)
        qa = zf.read("submission/query-p1-15-qa.csv").decode("utf-8")
        assert qa.startswith("L30_V072,1745,Xã Vạn Thắng\n")
        assert len(qa.strip().split("\n")) == 100

    expect = {Path(f["name"]).name for f in files}
    assert verify_submission_zip(zp, expect_names=expect) == []


# ---------------------------------------------------------------- coverage
#
# The probability-coverage allocator (+15.3% on TEST, docs/SHIP_PHU_XAC_SUAT.md)
# exists twice: allocate_coverage_rows in Python and allocateCoverageRows in
# review_export.js. The determinism contract (scores pre-rounded to 4 decimals,
# mass quantised to 1e-9, insertion-order video visits, first-maximum argmax)
# is exactly what these tests enforce ROW FOR ROW — never approximately.

# mirrors the Python defaults: CoveragePlan() + tail_n_flat=20 + AllocationPlan()
_COVERAGE_PLAN_JS = ("{nhiet:0.02, sigma:30, nuaCuaSo:6, luoi:5, budget:100, "
                     "nFlat:20, breadthCost:1.0, depthCost:0.75, step:10, maxDepth:24}")


def _coverage_payload(cands):
    # scores are rounded to 4 decimals HERE, in Python, before either side sees
    # them — the page embeds round(score, 4) and the JS uses the value verbatim
    return [
        {"v": c.video_id, "f": c.frame_idx, "last": c.video_last_frame,
         "s": round(float(c.score), 4)}
        for c in cands
    ]


def _js_coverage(cands):
    return _run_js(
        f"const plan = {_COVERAGE_PLAN_JS};\n"
        "console.log(JSON.stringify(A.allocateCoverageRows("
        + json.dumps(_coverage_payload(cands))
        + ", plan)))"
    )


def _clustered_candidates(seed: int, n_videos: int = 8, per_video: int = 8):
    """Candidates CLUSTERED per video: many frames of the same video close
    together with near-equal scores, so neighbouring grid windows carry
    near-identical mass — the regime where one mis-ordered addition or a
    wrong argmax tie-break flips a row."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_videos):
        vid = f"L{rng.randint(21, 30)}_V{rng.randint(1, 400):03d}"
        last = rng.randint(3_000, 60_000)
        centre = rng.randint(200, last - 200)
        base = 0.28 + rng.random() * 0.06
        for _ in range(per_video):
            f = min(last, max(0, centre + rng.randint(-120, 120)))
            out.append(Candidate(vid, f, base + rng.random() * 0.004, last))
    rng.shuffle(out)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 11])
def test_coverage_matches_row_for_row(seed):
    """Fuzzed: identical (pre-rounded) inputs must yield identical 100 rows."""
    pytest.importorskip("numpy")
    cands = _clustered_candidates(seed)
    got = _js_coverage(cands)
    want = [list(r) for r in allocate_coverage_rows(cands)]
    assert len(got) == len(want) == 100
    assert got == want


def test_coverage_tail_fill_matches():
    """A tiny pool on a short video runs out of coverable mass long before 100
    rows; the mandatory hybrid tail must top up IDENTICALLY on both sides —
    same row count, same rows, same order."""
    pytest.importorskip("numpy")
    cands = [Candidate("L21_V001", 100, 0.9, 300)]
    got = _js_coverage(cands)
    want = [list(r) for r in allocate_coverage_rows(cands)]
    assert got == want
    assert len(want) < 100, "one candidate on a 300-frame video cannot fill 100"
    # the greedy alone covers ~15 grid cells; anything beyond that IS the tail
    assert len(want) > 20, "the hybrid tail fill never engaged"
    assert len({tuple(r) for r in want}) == len(want), "tail fill duplicated a row"


def test_coverage_tail_fill_reaches_the_full_budget_identically():
    """A tight one-video cluster stalls the greedy around ~20 rows; with 20
    flat candidates the hybrid tail can and must reach exactly 100 — on both
    sides, row for row."""
    pytest.importorskip("numpy")
    rng = random.Random(5)
    cands = [
        Candidate("L23_V007", 10_000 + rng.randint(-30, 30), 0.4 - 0.001 * i, 60_000)
        for i in range(20)
    ]
    got = _js_coverage(cands)
    want = [list(r) for r in allocate_coverage_rows(cands)]
    assert got == want
    assert len(got) == 100


def test_coverage_ties_resolve_identically():
    """A perfectly symmetric pool — equal scores, equal frames, different
    videos — must resolve toward the video inserted FIRST on both sides:
    insertion-order visits plus strict-> argmax are part of the contract."""
    pytest.importorskip("numpy")
    cands = [
        Candidate("L21_V010", 1_000, 0.5, 50_000),
        Candidate("L21_V011", 1_000, 0.5, 50_000),
    ]
    got = _js_coverage(cands)
    want = [list(r) for r in allocate_coverage_rows(cands)]
    assert got == want
    assert got[0][0] == "L21_V010", "the tie must go to the earlier candidate"
    assert got[1][0] == "L21_V011", "after V010's window is zeroed, V011's full peak wins"


@pytest.mark.parametrize(
    "frames,last",
    [
        ([1426, 1477, 1558, 1712], 50000),
        ([500], 50000),
        ([500, 1500], 50000),
        ([500, 1500, 2500], 50000),
        ([500, 1500, 2500, 3500, 4500], 9000),
        ([100, 200, 300, 400], 450),      # clamped hard against the video end
        ([5, 15], 30),                     # clamped at both ends
    ],
)
def test_trake_lattice_matches_python(frames, last):
    """The browser walks the offset lattice by sorting; Python uses a heap.

    Those agree only because the heap's pop order is exactly a sort by
    (total displacement, lexicographic) — verified here rather than assumed,
    since a TRAKE row on the wrong frames scores zero with no other symptom.
    """
    from src.core.submission import allocate_trake_rows

    got = _run_js(
        f"console.log(JSON.stringify(A.allocateTrakeRows('V', {json.dumps(frames)}, 100, 10, {last})))"
    )
    want = [[v] + list(fr) for v, fr in allocate_trake_rows("V", frames, video_last_frame=last)]
    assert got == want
