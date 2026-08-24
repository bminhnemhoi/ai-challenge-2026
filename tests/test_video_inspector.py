"""The video inspector: watch the real moment, then commit an exact frame.

A 158-pixel thumbnail cannot settle a TRAKE chain, cannot be read for a Q&A
answer, and cannot break a tie the retriever is unsure about. The organisers'
media-info gives a YouTube watch_url for all 873 videos and metadata.json gives
pts_time and fps per keyframe, so the page can seek the real video to the exact
instant — and let the operator commit a frame the retriever never proposed.

That last part is the point. The answer window is about 10 frames wide and
keyframes sit ~55 apart, so a frame named by a human watching the video beats
anything picked off the candidate list.
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
DATA_DIR = ROOT / "data"

pytestmark = [
    pytest.mark.skipif(shutil.which("node") is None, reason="node not installed"),
    pytest.mark.skipif(not PAGE.is_file(), reason="no built review page"),
]


@pytest.fixture(scope="module")
def page():
    html = PAGE.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    return {
        "alloc": scripts[0],
        "DATA": json.loads(re.search(r"^const DATA = (\{.*?\});$", scripts[1], re.M).group(1)),
        "VID": json.loads(re.search(r"^const VID = (\{.*?\});$", scripts[1], re.M).group(1)),
        "PLAN": json.loads(re.search(r"^const PLAN = (\{.*?\});$", scripts[1], re.M).group(1)),
        "raw": scripts[1],
    }


def test_every_shown_candidate_has_a_playable_video(page):
    """A ▶ button that opens nothing is worse than no button."""
    shown = {c[0] for d in page["DATA"].values() for c in d["cands"][: d["shown"]]}
    missing = sorted(v for v in shown if v not in page["VID"])
    assert missing == [], f"no inspector data for {missing[:5]}"
    no_link = sorted(v for v in shown if not page["VID"][v].get("y"))
    assert no_link == [], f"no YouTube id for {no_link[:5]}"


def test_youtube_ids_look_like_youtube_ids(page):
    bad = {v: d["y"] for v, d in page["VID"].items() if not re.fullmatch(r"[A-Za-z0-9_-]{11}", d["y"])}
    assert bad == {}, bad


def test_keyframe_timelines_match_the_corpus(page):
    """The embedded timeline is a compression of metadata.json; verify it decompresses."""
    md = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    want: dict = {}
    fps: dict = {}
    for m in md:
        v = m["video_id"]
        if v in page["VID"]:
            want.setdefault(v, []).append(int(m["frame_idx"]))
            fps[v] = float(m["fps"])
    for v, d in page["VID"].items():
        assert d["k"] == sorted(want[v]), f"{v}: keyframe list does not match metadata.json"
        assert d["f"] == fps[v], f"{v}: wrong fps"
        assert d["l"] >= max(d["k"]), f"{v}: last_frame is before the final keyframe"


def test_seconds_to_frame_round_trips(page):
    """The inspector converts a YouTube timestamp to a frame with frame = sec * fps.

    metadata.json satisfies pts_time * fps == frame_idx across the whole corpus,
    so a timestamp read off the player maps back to the frame the grader scores.
    """
    md = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    checked = 0
    for m in md:
        if m["video_id"] not in page["VID"]:
            continue
        assert abs(m["pts_time"] * m["fps"] - m["frame_idx"]) <= 1.01
        checked += 1
        if checked > 20000:
            break
    assert checked > 1000


def _drive(page, body: str):
    src = (
        page["alloc"]
        + "\nconst DATA = " + json.dumps(page["DATA"])
        + ";\nconst VID = " + json.dumps(page["VID"])
        + ";\nconst PLAN = " + json.dumps(page["PLAN"]) + ";\n"
        + body
    )
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "drive.js"
        f.write_text(src, encoding="utf-8")
        r = subprocess.run(
            [shutil.which("node"), str(f)], capture_output=True, text=True, encoding="utf-8"
        )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_a_hand_marked_frame_becomes_row_one_with_a_ladder(page):
    """The operator watched the video and named an instant no candidate covers.

    That frame must land at row 1 with the +-step ladder immediately behind it,
    because the only remaining risk is being a few frames off — the video is
    settled.
    """
    out = _drive(
        page,
        """
const qid = Object.keys(DATA).find(k => DATA[k].task === 'kis');
const d = DATA[qid];
const v = d.cands[0][0];
const custom = {v: v, f: d.cands[0][1] + 137, last: VID[v].l};   // not a keyframe
let ordered = [[custom.v, custom.f, custom.last]].concat(
  d.cands.filter(c => !(c[0] === custom.v && c[1] === custom.f)));
const rows = allocateHybridRows(
  ordered.map(c => ({v: c[0], f: c[1], last: c[2]})), PLAN.nFlat, PLAN);
console.log(JSON.stringify({custom, rows: rows.slice(0, 6), n: rows.length,
                            keyframes: VID[v].k.includes(custom.f)}));
""",
    )
    assert out["keyframes"] is False, "pick a frame that is deliberately not a keyframe"
    assert out["rows"][0] == [out["custom"]["v"], out["custom"]["f"]]
    assert out["n"] == 100
    assert len({tuple(r) for r in out["rows"]}) == len(out["rows"])


def test_hand_marking_one_trake_event_rebuilds_the_whole_lattice(page):
    """Marking E2 by eye must keep the other events and re-centre the lattice."""
    trake = [k for k, d in page["DATA"].items() if d["task"] == "trake"]
    if not trake:
        pytest.skip("no TRAKE query in this round")
    out = _drive(
        page,
        """
const qid = %s;
const d = DATA[qid];
const base = d.chainRows[0];
const video = base[0][0];
const frames = base[0].slice(1);
const edited = frames.slice(); edited[1] = frames[1] + 260;   // operator moves E2
const rows = allocateTrakeRows(video, edited, PLAN.budget, PLAN.step, VID[video].l);
console.log(JSON.stringify({video, frames, edited, row0: rows[0], n: rows.length,
                            width: new Set(rows.map(r => r.length)).size}));
"""
        % json.dumps(trake[0]),
    )
    assert out["row0"] == [out["video"]] + out["edited"], "row 1 must be exactly what was marked"
    assert out["n"] == 100
    assert out["width"] == 1, "every TRAKE row needs the same number of frame columns"
    assert out["row0"][1] == out["frames"][0], "untouched events must not move"


def test_inspector_refuses_a_frame_from_the_wrong_video(page):
    """A TRAKE row whose events span two videos scores zero; the page must say so."""
    assert "Mọi sự kiện phải nằm trong CÙNG một video" in page["raw"]
    assert "chainVideo" in page["raw"]


def test_inspector_uses_the_iframe_api_not_a_static_embed(page):
    """A plain <iframe src=...?start=N> can only jump to a whole second and can
    never be asked where it is. Capturing the exact instant needs seekTo() with
    fractional seconds and getCurrentTime() — which is precisely how the two
    AIC 2025 systems that scored Outstanding on TRAKE work.
    """
    raw = page["raw"]
    assert "youtube.com/iframe_api" in raw
    assert "onYouTubeIframeAPIReady" in raw
    assert "getCurrentTime()" in raw
    assert "seekTo(" in raw
    # and the frame maths must use the video's own fps, not a constant
    assert "VID[vi.video].f" in raw


def test_frame_stepping_moves_exactly_one_source_frame(page):
    """Arrow keys step 1/fps, not YouTube's undocumented ','/'.' shortcuts.

    Those step the rendered stream, which is not the frame numbering the grader
    scores against, and they are not documented for embeds at all.
    """
    raw = page["raw"]
    assert "function stepFrame(" in raw
    assert "Math.round(t * fps) + n" in raw
    assert "',' / '.'" not in raw or "not relied on" in raw


def test_capture_writes_the_playhead_verbatim(page):
    raw = page["raw"]
    assert "function captureNow(" in raw
    assert "Math.round(t * VID[vi.video].f)" in raw


def test_the_timeline_the_page_assumes_matches_the_organisers_table():
    """frame_id = round(pts_time * fps) must hold, or every captured frame is wrong.

    The page converts a playhead time into a frame index with exactly this
    formula. If the organisers' own table disagreed anywhere, a captured frame
    would be silently off and score zero.
    """
    md = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
    worst = max(abs(m["pts_time"] * m["fps"] - m["frame_idx"]) for m in md)
    assert worst <= 1.01, f"pts_time*fps deviates from frame_idx by up to {worst}"
