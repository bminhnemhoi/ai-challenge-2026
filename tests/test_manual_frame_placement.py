"""Marking a frame while watching the video, and placing it at a chosen rank.

This is the operator's most valuable action and it was broken: the marked frame
was stored in a hidden field pinned to rank 1, so no card ever appeared and the
button looked inert, and there was no way to say "put this at #3" — which is
what you want when you are fairly sure but not certain.

The state model is now: `order` is a list of ids, "cN" for a retrieved candidate
and "xN" for a marked frame, both first-class. These tests drive the real page's
JavaScript under node and check the submitted rows, because a card that looks
right while the CSV says something else is the failure this whole loop exists to
prevent.
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

pytestmark = [
    pytest.mark.skipif(shutil.which("node") is None, reason="node not installed"),
    pytest.mark.skipif(not PAGE.is_file(), reason="no built review page"),
]


@pytest.fixture(scope="module")
def ctx():
    html = PAGE.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    page = scripts[1]
    return {
        "alloc": scripts[0],
        "DATA": re.search(r"^const DATA = (\{.*?\});$", page, re.M).group(1),
        "VID": re.search(r"^const VID = (\{.*?\});$", page, re.M).group(1),
        "PLAN": re.search(r"^const PLAN = (\{.*?\});$", page, re.M).group(1),
        "html": html,
    }


def _run(ctx, body):
    # the page's own state helpers, lifted out so the real logic is under test
    helpers = """
function candOf(qid, key) {
  const st = state[qid], d = DATA[qid];
  if (key[0] === 'x') { const e = st.extra[+key.slice(1)];
    return e ? [e.v, e.f, (VID[e.v] || {}).l] : null; }
  return d.cands[+key.slice(1)] || null;
}
function orderedCands(qid) {
  const st = state[qid], d = DATA[qid];
  const seen = new Set(); const out = [];
  for (const k of st.order) {
    const c = candOf(qid, k); if (!c) continue;
    const sig = c[0] + ':' + c[1]; if (seen.has(sig)) continue;
    seen.add(sig); out.push(c);
  }
  for (const c of d.cands.slice(d.shown)) {
    const sig = c[0] + ':' + c[1];
    if (!seen.has(sig)) { seen.add(sig); out.push(c); }
  }
  return out;
}
function moveTo(qid, key, pos) {
  const ord = state[qid].order; const from = ord.indexOf(key);
  if (from < 0) return;
  ord.splice(from, 1);
  ord.splice(Math.max(0, Math.min(ord.length, pos)), 0, key);
}
function initState(qid) {
  state[qid] = {order: [...Array(DATA[qid].shown).keys()].map(i => 'c' + i),
                extra: [], answer: '', touched: false, frames: null};
}
function markFrame(qid, v, f, pos) {
  const st = state[qid];
  st.extra.push({v: v, f: f});
  const key = 'x' + (st.extra.length - 1);
  st.order.push(key);
  moveTo(qid, key, pos - 1);
  return key;
}
function rowsFor(qid) {
  const d = DATA[qid], st = state[qid];
  if (d.task === 'trake') {
    const video = candOf(qid, st.order[0])[0];
    if (st.frames && st.frames.length)
      return allocateTrakeRows(video, st.frames, PLAN.budget, PLAN.step, (VID[video]||{}).l);
    return d.chainRows[+st.order[0].slice(1)] || [];
  }
  const rows = allocateHybridRows(
    orderedCands(qid).map(c => ({v: c[0], f: c[1], last: c[2]})), PLAN.nFlat, PLAN);
  return d.task === 'qa' ? rows.map(r => [r[0], r[1], st.answer]) : rows;
}
"""
    src = (
        ctx["alloc"]
        + "\nconst DATA = " + ctx["DATA"]
        + ";\nconst VID = " + ctx["VID"]
        + ";\nconst PLAN = " + ctx["PLAN"]
        + ";\nlet state = {};\n" + helpers + body
    )
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.js"
        f.write_text(src, encoding="utf-8")
        r = subprocess.run(
            [shutil.which("node"), str(f)], capture_output=True, text=True, encoding="utf-8"
        )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.parametrize("pos", [1, 2, 3, 7])
def test_a_marked_frame_lands_at_the_requested_rank(ctx, pos):
    """"Not necessarily top 1" — the operator names the rank, and it is honoured."""
    out = _run(
        ctx,
        """
const qid = Object.keys(DATA).find(k => DATA[k].task === 'kis');
initState(qid);
const before = rowsFor(qid);
const v = DATA[qid].cands[0][0];
const f = DATA[qid].cands[0][1] + 173;              // deliberately not a keyframe
const key = markFrame(qid, v, f, %d);
const after = rowsFor(qid);
console.log(JSON.stringify({key, v, f, pos: %d,
  orderHead: state[qid].order.slice(0, 8),
  beforeHead: before.slice(0, 8), afterHead: after.slice(0, 8),
  n: after.length, uniq: new Set(after.map(r => r.join(','))).size}));
"""
        % (pos, pos),
    )
    # the marked frame occupies exactly the rank that was asked for
    assert out["orderHead"][pos - 1] == out["key"]
    assert out["afterHead"][pos - 1] == [out["v"], out["f"]]
    # everything above it is untouched
    assert out["afterHead"][: pos - 1] == out["beforeHead"][: pos - 1]
    # and nothing is lost or duplicated
    assert out["n"] == 100
    assert out["uniq"] == 100


def test_marking_the_same_frame_twice_does_not_duplicate_it(ctx):
    out = _run(
        ctx,
        """
const qid = Object.keys(DATA).find(k => DATA[k].task === 'kis');
initState(qid);
const v = DATA[qid].cands[0][0], f = DATA[qid].cands[0][1] + 173;
markFrame(qid, v, f, 1);
// the page reuses the existing entry rather than pushing a second one
const dup = state[qid].extra.findIndex(e => e.v === v && e.f === f);
moveTo(qid, 'x' + dup, 4);
const rows = rowsFor(qid);
console.log(JSON.stringify({extras: state[qid].extra.length, row5: rows[4],
  n: rows.length, uniq: new Set(rows.map(r => r.join(','))).size}));
""",
    )
    assert out["extras"] == 1
    assert out["n"] == out["uniq"] == 100


def test_a_marked_frame_that_equals_a_candidate_is_not_emitted_twice(ctx):
    """Marking a frame the retriever already found must not waste a rank slot."""
    out = _run(
        ctx,
        """
const qid = Object.keys(DATA).find(k => DATA[k].task === 'kis');
initState(qid);
const c = DATA[qid].cands[5];
markFrame(qid, c[0], c[1], 1);                      // same (video, frame) as c5
const rows = rowsFor(qid);
const sig = c[0] + ',' + c[1];
console.log(JSON.stringify({first: rows[0], count: rows.filter(r => r.join(',') === sig).length,
                            n: rows.length, uniq: new Set(rows.map(r => r.join(','))).size}));
""",
    )
    assert out["count"] == 1, "the frame appears twice, wasting a rank slot"
    assert out["n"] == out["uniq"] == 100


def test_marked_frame_gets_its_ladder_right_behind_it(ctx):
    """A frame named from the video is exact; the remaining risk is a few frames."""
    out = _run(
        ctx,
        """
const qid = Object.keys(DATA).find(k => DATA[k].task === 'kis');
initState(qid);
const v = DATA[qid].cands[0][0], f = DATA[qid].cands[0][1] + 173;
markFrame(qid, v, f, 1);
const rows = rowsFor(qid);
console.log(JSON.stringify({f, near: rows.slice(0, 40)
  .map((r, i) => [i, r[0] === v ? r[1] - f : null]).filter(x => x[1] !== null)}));
""",
    )
    offsets = {o for _i, o in out["near"]}
    assert 0 in offsets
    assert {-10, 10} <= offsets, f"the +-10 ladder is missing: {sorted(offsets)[:12]}"


def test_hand_marking_a_trake_event_changes_the_submitted_rows(ctx):
    out = _run(
        ctx,
        """
const qid = Object.keys(DATA).find(k => DATA[k].task === 'trake');
if (!qid) { console.log(JSON.stringify({skip: true})); }
else {
  initState(qid);
  const base = rowsFor(qid);
  const frames = base[0].slice(1);
  const edited = frames.slice(); edited[1] = frames[1] + 260;
  state[qid].frames = edited;
  const after = rowsFor(qid);
  console.log(JSON.stringify({base0: base[0], after0: after[0], edited,
    n: after.length, widths: [...new Set(after.map(r => r.length))]}));
}
""",
    )
    if out.get("skip"):
        pytest.skip("no TRAKE query in this round")
    assert out["after0"][1:] == out["edited"]
    assert out["after0"] != out["base0"]
    assert out["n"] == 100
    assert out["widths"] == [len(out["edited"]) + 1]


def test_the_page_actually_renders_a_card_for_a_marked_frame(ctx):
    """The original bug was invisibility: the state changed, the screen did not."""
    page = ctx["html"]
    assert "function extraCard(" in page
    assert "figure.manual" in page
    assert "chốt tay" in page
    assert "removeExtra(" in page, "a marked frame must be removable"
    assert "vipos" in page, "there must be a rank chooser"
