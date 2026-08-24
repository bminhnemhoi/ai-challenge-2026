"""The zip review.html builds must equal the zip make_submission writes.

The page now produces the finished upload without Python, using the candidate
pool embedded in it. If that ever diverges from the pipeline, an operator who
changed nothing would still upload something different from what they reviewed
— and would have no way to notice.

So this drives the real generated page: it pulls DATA, PLAN and the allocator
out of round_p1/review.html, runs them under node with every query left in its
default order, and requires the result to match the CSVs on disk row for row.

Skipped unless a built page and its matching run are both present, so the suite
still passes on a fresh clone with no data.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "round_p1" / "review.html"
RUN = ROOT / "round_p1" / "run1" / "csv"

pytestmark = [
    pytest.mark.skipif(shutil.which("node") is None, reason="node not installed"),
    pytest.mark.skipif(
        not PAGE.is_file() or not RUN.is_dir(),
        reason="no built review page + run to compare (build one with build_review_page.py)",
    ),
]


@pytest.fixture(scope="module")
def exported():
    """{query stem: csv text} exactly as the page would download it."""
    html = PAGE.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) >= 2, "expected the allocator and the page script"
    alloc = scripts[0]
    data = re.search(r"^const DATA = (\{.*?\});$", scripts[1], re.M).group(1)
    plan = re.search(r"^const PLAN = (\{.*?\});$", scripts[1], re.M).group(1)

    driver = f"""
{alloc}
const DATA = {data};
const PLAN = {plan};
const out = {{}};
for (const qid of Object.keys(DATA)) {{
  const d = DATA[qid];
  let rows;
  if (d.task === 'trake') {{
    rows = d.chainRows[0] || [];              // default order: chain 0 at #1
  }} else {{
    rows = allocateHybridRows(d.cands.map(c => ({{v: c[0], f: c[1], last: c[2]}})),
                              PLAN.nFlat, PLAN);
    if (d.task === 'qa') rows = rows.map(r => [r[0], r[1], '']);
  }}
  out[qid] = rowsToCsv(rows);
}}
console.log(JSON.stringify(out));
"""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "drive.js"
        f.write_text(driver, encoding="utf-8")
        r = subprocess.run(
            [shutil.which("node"), str(f)], capture_output=True, text=True, encoding="utf-8"
        )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_every_query_is_present(exported):
    on_disk = {p.stem for p in RUN.glob("*.csv")}
    assert set(exported) == on_disk


@pytest.mark.parametrize("kind", ["kis", "qa", "trake"])
def test_exported_csv_matches_the_pipeline(exported, kind):
    """Row for row, including order — not merely the same set of rows."""
    checked = 0
    for stem, text in sorted(exported.items()):
        if not stem.endswith(kind):
            continue
        want = (RUN / f"{stem}.csv").read_text(encoding="utf-8")
        if kind == "qa":
            # the page cannot know an answer nobody typed; compare the frames
            strip = lambda s: "\n".join(",".join(l.split(",")[:2]) for l in s.strip().splitlines())
            assert strip(text) == strip(want), stem
        else:
            assert text == want, stem
        checked += 1
    assert checked, f"no {kind} queries to compare"


def test_default_export_uses_the_whole_budget(exported):
    for stem, text in exported.items():
        n = len([ln for ln in text.splitlines() if ln.strip()])
        assert n == 100, f"{stem} exported {n} rows; extra rows are free and never hurt"


def _drive(body: str):
    """Run `body` against the real page's allocator, DATA and PLAN."""
    html = PAGE.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    data = re.search(r"^const DATA = (\{.*?\});$", scripts[1], re.M).group(1)
    plan = re.search(r"^const PLAN = (\{.*?\});$", scripts[1], re.M).group(1)
    src = scripts[0] + "\nconst DATA = " + data + ";\nconst PLAN = " + plan + ";\n" + body
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "drive.js"
        f.write_text(src, encoding="utf-8")
        r = subprocess.run(
            [shutil.which("node"), str(f)], capture_output=True, text=True, encoding="utf-8"
        )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_dragging_a_candidate_to_the_top_moves_it_to_row_one():
    """The whole point of the drag: what sits at #1 on screen is CSV row 1.

    And nothing else may be lost — the rows the operator did not touch keep
    their relative order, and the budget is still fully spent, because a
    reorder that quietly dropped rows would cost score without showing it.
    """
    out = _drive(
        """
const qid = Object.keys(DATA).find(k => DATA[k].task === 'kis');
const d = DATA[qid];
const mk = cs => cs.map(c => ({v: c[0], f: c[1], last: c[2]}));
const base = allocateHybridRows(mk(d.cands), PLAN.nFlat, PLAN);
const order = [...Array(d.shown).keys()];
order.splice(order.indexOf(1), 1); order.unshift(1);
const moved = allocateHybridRows(
  mk(order.map(i => d.cands[i]).concat(d.cands.slice(d.shown))), PLAN.nFlat, PLAN);
console.log(JSON.stringify({
  qid, promoted: d.cands[1], base0: base[0], moved: moved,
  nBase: base.length, nMoved: moved.length,
}));
"""
    )
    promoted = out["promoted"]
    assert out["moved"][0] == [promoted[0], promoted[1]], "the dragged candidate is not row 1"
    assert out["moved"][1] == out["base0"], "the old row 1 should slide to row 2"
    assert out["nMoved"] == out["nBase"] == 100, "a reorder must not cost rows"
    assert len({tuple(r) for r in out["moved"]}) == 100, "duplicate rows waste rank slots"


def test_reordering_never_drops_a_candidate_the_operator_can_see():
    """Every visible thumbnail must still be reachable in the exported rows."""
    out = _drive(
        """
const qid = Object.keys(DATA).find(k => DATA[k].task === 'kis');
const d = DATA[qid];
const rows = allocateHybridRows(
  d.cands.map(c => ({v: c[0], f: c[1], last: c[2]})), PLAN.nFlat, PLAN);
const keys = new Set(rows.map(r => r[0] + ':' + r[1]));
const shown = d.cands.slice(0, d.shown).map(c => c[0] + ':' + c[1]);
console.log(JSON.stringify({missing: shown.filter(k => !keys.has(k))}));
"""
    )
    assert out["missing"] == [], f"visible candidates absent from the upload: {out['missing']}"
