"""Official AIC 2026 scoring, row allocation, and submission packaging.

Everything in this module follows the BTC rules document
("Thong tin vong So tuyen AIC2026.pdf", sections 2.1 and 2.2) literally.

Scoring, in the organisers' own terms
-------------------------------------
Textual KIS   R-Score(r) = I(v = GT_v  and  id in [s, e])
Q&A           R-Score(r) = I(v = GT_v  and  id in [s, e]  and  answer == GT_a)
TRAKE         R-Score(r) = 0 if v != GT_v, else (1/N) * sum_j I(id_j in [s_j, e_j])

              R@k         = max over the FIRST k submitted rows of R-Score
              Final Score = (1/5) * sum of R@k for k in {1, 5, 20, 50, 100}

Two consequences drive every design decision here, and neither is obvious:

**Extra rows are free.**  R@k is a MAXIMUM over a prefix, never a sum and never
an average.  A row that is wrong cannot lower any R@k.  So the only cost of an
extra candidate is the rank slot it occupies — and rows 51..100 occupy slots
that are otherwise worth nothing at all.  Always submit all 100.

**A keyframe index is usually the wrong thing to submit.**  ``frame_id`` is an
arbitrary integer in the original video; it is NOT required to be one of the
extracted keyframes.  The organisers' KIS example uses the window [500, 510],
eleven frames wide, and the rules say a TRAKE event window is "thường rất ngắn,
thông thường là dưới 10 frame".  Consecutive keyframes in this corpus sit a
median of 55 frames apart.  Submitting only keyframe indices therefore caps the
achievable score at about 18% for an 11-frame window — however good the
retrieval is.  Spending a few rank slots on a ladder of plain integers around
the chosen keyframe lifts that ceiling to ~90% and costs nothing else.
"""

from __future__ import annotations

import csv
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: rank thresholds the Final Score averages over (rules section 2.2)
RANK_THRESHOLDS: Tuple[int, ...] = (1, 5, 20, 50, 100)
#: hard cap on submitted answers per query (rules section 2)
MAX_ROWS = 100

#: The organisers' worked example uses an 11-frame window and TRAKE events are
#: documented as "usually under 10 frames".  A ladder step of 10 guarantees that
#: any window at least this wide, lying anywhere inside the ladder's span,
#: contains one of our submitted ids.
ASSUMED_WINDOW_FRAMES = 10


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def r_score_kis(video_id: str, frame_id: int, gt_video: str, gt_span: Tuple[int, int]) -> float:
    """Rules 2.1.1 — right video AND frame inside the answer interval."""
    if video_id != gt_video:
        return 0.0
    s, e = gt_span
    return 1.0 if s <= int(frame_id) <= e else 0.0


def r_score_qa(
    video_id: str,
    frame_id: int,
    answer: str,
    gt_video: str,
    gt_span: Tuple[int, int],
    gt_answer: str,
    answer_matcher=None,
) -> float:
    """Rules 2.1.2 — KIS conditions plus a semantic answer match.

    The organisers match answers by meaning and accept Vietnamese or English,
    so ``answer_matcher`` is injectable; the default is a lenient normalised
    comparison used only for offline estimation.
    """
    if r_score_kis(video_id, frame_id, gt_video, gt_span) == 0.0:
        return 0.0
    match = answer_matcher or _default_answer_match
    return 1.0 if match(answer, gt_answer) else 0.0


def r_score_trake(
    video_id: str,
    frame_ids: Sequence[int],
    gt_video: str,
    gt_spans: Sequence[Tuple[int, int]],
) -> float:
    """Rules 2.1.3 — wrong video scores 0; otherwise the fraction of events hit.

    Note the asymmetry the rules impose: the video is all-or-nothing but the
    events earn partial credit, so a candidate video is worth submitting many
    times with different frame combinations, and worth submitting for no other
    video at all.
    """
    if video_id != gt_video:
        return 0.0
    if not gt_spans:
        return 0.0
    hit = 0
    for j, (s, e) in enumerate(gt_spans):
        if j < len(frame_ids) and s <= int(frame_ids[j]) <= e:
            hit += 1
    return hit / len(gt_spans)


def r_at_k(row_scores: Sequence[float], k: int) -> float:
    """Best R-Score among the first k rows (rules 2.2)."""
    prefix = row_scores[:k]
    return max(prefix) if prefix else 0.0


def final_score(row_scores: Sequence[float]) -> float:
    """Final Score = mean of R@k over k in {1, 5, 20, 50, 100} (rules 2.2)."""
    return sum(r_at_k(row_scores, k) for k in RANK_THRESHOLDS) / len(RANK_THRESHOLDS)


def score_bucket(first_hit_rank: Optional[int]) -> float:
    """Final Score for a BINARY task whose first correct row is at this rank.

    Useful for reasoning about where effort pays: the score is a step function
    with breakpoints exactly at the rank thresholds, so moving a hit from rank
    15 to rank 10 is worth nothing while moving it from 6 to 5 is worth 0.2.
    """
    if first_hit_rank is None:
        return 0.0
    return sum(1.0 for k in RANK_THRESHOLDS if first_hit_rank <= k) / len(RANK_THRESHOLDS)


def _normalise_answer(a: str) -> str:
    a = (a or "").strip().lower()
    a = re.sub(r"[.,;:!?\"']", " ", a)
    return re.sub(r"\s+", " ", a).strip()


def _default_answer_match(pred: str, gold: str) -> bool:
    """Lenient offline stand-in for the organisers' semantic matcher.

    Deliberately generous (substring either way) so that offline estimates do
    not flatter the system: it can only over-count, never under-count, and an
    over-counted estimate is the safer direction to be wrong in when deciding
    whether a change helped.
    """
    p, g = _normalise_answer(pred), _normalise_answer(gold)
    if not p or not g:
        return False
    return p == g or p in g or g in p


# ---------------------------------------------------------------------------
# Frame ladders — turning rank slots into window coverage
# ---------------------------------------------------------------------------


def frame_ladder(
    center: int,
    n_ids: int,
    step: int = ASSUMED_WINDOW_FRAMES,
    lo: int = 0,
    hi: Optional[int] = None,
) -> List[int]:
    """Integer frame ids around ``center``, nearest first.

    Ordered by distance so that truncating the list keeps the most probable
    ids: [c, c-step, c+step, c-2*step, c+2*step, ...].  ``step`` should not
    exceed the assumed answer-window width, or the ladder leaves gaps a window
    can fall into.
    """
    out: List[int] = []
    seen = set()

    def push(x: int) -> None:
        x = int(x)
        if x < lo or (hi is not None and x > hi) or x in seen:
            return
        seen.add(x)
        out.append(x)

    push(center)
    k = 1
    while len(out) < n_ids and k <= 4 * n_ids:
        push(center - k * step)
        if len(out) < n_ids:
            push(center + k * step)
        k += 1
    return out[:n_ids]


# ---------------------------------------------------------------------------
# Row allocation
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One retrieved keyframe, as the retriever ranked it."""

    video_id: str
    frame_idx: int
    score: float = 0.0
    #: last frame index of the containing video, so ladders stay in range
    video_last_frame: Optional[int] = None


@dataclass
class AllocationPlan:
    """How rank slots are split between *more videos* and *more frame ids*.

    Every submitted row is a pair (candidate index ``i``, ladder depth ``d``).
    Walking outward from (0, 0) by the linear cost

        cost(i, d) = breadth_cost * i + depth_cost * d

    gives a single dial for the only real trade-off in the task: a wrong video
    can never be rescued by any number of frame ids, but a right video with a
    near-miss frame can — and the second case is far more common than it looks,
    because keyframes sit ~55 frames apart while an answer window is ~10 wide.

    Lower ``depth_cost`` relative to ``breadth_cost`` buys frame coverage
    first; raising it buys more candidate videos first.  The defaults were
    chosen by sweeping both against the ground-truth set (see
    ``scripts/tune_allocation.py``) rather than by intuition.
    """

    breadth_cost: float = 1.0
    depth_cost: float = 0.75
    step: int = ASSUMED_WINDOW_FRAMES
    budget: int = MAX_ROWS
    #: never spend more than this many ids on one candidate
    max_depth: int = 24


def allocate_kis_rows(
    candidates: Sequence[Candidate],
    plan: Optional[AllocationPlan] = None,
) -> List[Tuple[str, int]]:
    """Turn ranked keyframes into up to ``budget`` ``(video_id, frame_id)`` rows.

    Rank 1 always goes to the single best keyframe — it is the only slot worth
    a full 1.0.  After that the order follows :class:`AllocationPlan`.
    """
    plan = plan or AllocationPlan()
    if not candidates:
        return []

    ladders = [
        frame_ladder(c.frame_idx, plan.max_depth, plan.step, lo=0, hi=c.video_last_frame)
        for c in candidates
    ]

    slots = [
        (plan.breadth_cost * i + plan.depth_cost * d, i, d)
        for i, ids in enumerate(ladders)
        for d in range(len(ids))
    ]
    slots.sort(key=lambda t: (t[0], t[1], t[2]))

    rows: List[Tuple[str, int]] = []
    seen: set = set()
    for _cost, i, d in slots:
        key = (candidates[i].video_id, int(ladders[i][d]))
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
        if len(rows) >= plan.budget:
            break
    return rows


def allocate_hybrid_rows(
    candidates: Sequence[Candidate],
    n_flat: int = 20,
    plan: Optional[AllocationPlan] = None,
) -> List[Tuple[str, int]]:
    """Hedge against the unknown answer-window width.

    The organisers never publish how wide [s, e] is, and the best strategy
    flips depending on it:

      * a WIDE window (a whole scene, hundreds of frames) is best served by
        spending every row on a different high-similarity keyframe, since any
        keyframe inside the scene scores;
      * a NARROW window (the rulebook's 11-frame example) makes keyframes
        almost unusable on their own — they sit ~55 frames apart — and is best
        served by a dense ladder of plain integers around the best few.

    Because R@k is a maximum over a PREFIX, the two strategies compose almost
    for free: give the expensive early ranks (1, 2-5, 6-20 — worth 1.0, 0.8,
    0.6) to distinct keyframes, and spend the cheap tail (21-100, worth only
    0.4 and 0.2) on ladders.  A wide window then scores exactly as well as the
    pure-keyframe strategy, while a narrow window still scores instead of
    scoring nothing.
    """
    plan = plan or AllocationPlan()
    rows: List[Tuple[str, int]] = []
    seen: set = set()

    for c in candidates[:n_flat]:
        key = (c.video_id, int(c.frame_idx))
        if key not in seen:
            seen.add(key)
            rows.append(key)
        if len(rows) >= plan.budget:
            return rows

    tail = AllocationPlan(
        breadth_cost=plan.breadth_cost,
        depth_cost=plan.depth_cost,
        step=plan.step,
        budget=plan.budget * 3,
        max_depth=plan.max_depth,
    )
    for key in allocate_kis_rows(candidates, tail):
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
        if len(rows) >= plan.budget:
            break
    return rows


def allocate_trake_rows(
    video_id: str,
    event_frames: Sequence[int],
    budget: int = MAX_ROWS,
    step: int = ASSUMED_WINDOW_FRAMES,
    video_last_frame: Optional[int] = None,
) -> List[Tuple[str, List[int]]]:
    """Rows for one TRAKE query: the same video, perturbed frame tuples.

    A TRAKE row scores 0 outright if the video is wrong, so every row uses the
    single best video.  Within that video the score is the FRACTION of events
    whose frame lands in its window, and R@k takes the maximum over rows, so we
    want one row where as many events as possible hit simultaneously.

    Rows are the Cartesian product of a small per-event offset ladder, emitted
    in order of increasing total displacement so the most probable combinations
    occupy the earliest ranks.
    """
    n = len(event_frames)
    if n == 0:
        return []

    # Grow the offset ladder until the lattice can actually fill the budget.
    # A fixed five-value alphabet gives 5**n combinations, which is only 5 rows
    # for a 1-event query and 25 for a 2-event one — throwing away most of the
    # 100 free rank slots on exactly the queries where frames are hardest to
    # hit.
    reach = 2
    while (2 * reach + 1) ** n < budget and reach < budget:
        reach += 1
    offsets = [0] + [s * k * step for k in range(1, reach + 1) for s in (-1, 1)]

    # Best-first walk of the offset lattice, cheapest total displacement first.
    # Enumerating the whole product and sorting afterwards is both explosive
    # (25**6 for a 6-event query) and, once truncated, biased: a depth-first
    # cutoff freezes the leading coordinates, so the first events would never
    # get perturbed at all.
    import heapq

    start = tuple(0 for _ in range(n))
    heap: List[Tuple[int, Tuple[int, ...]]] = [(0, start)]
    queued = {start}
    rows: List[Tuple[str, List[int]]] = []
    seen: set = set()

    while heap and len(rows) < budget:
        cost, offs = heapq.heappop(heap)
        frames = []
        for f, o in zip(event_frames, offs):
            v = max(0, int(f) + o)
            if video_last_frame is not None:
                v = min(v, video_last_frame)
            frames.append(v)
        key = tuple(frames)
        if key not in seen:
            seen.add(key)
            rows.append((video_id, frames))

        # neighbours: move any one event one rung further out
        for i in range(n):
            for delta in (-step, step):
                nxt = list(offs)
                nxt[i] += delta
                if abs(nxt[i]) > reach * step:
                    continue
                t = tuple(nxt)
                if t not in queued:
                    queued.add(t)
                    heapq.heappush(heap, (sum(abs(x) for x in t), t))
    return rows


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------


def csv_name_for_query(query_filename: str) -> str:
    """``query-1-kis.txt`` -> ``query-1-kis.csv`` (rules: same stem)."""
    return Path(query_filename).with_suffix(".csv").name


#: characters that cannot appear in a field of a comma-split CSV
_UNSAFE_FIELD = re.compile(r'[,\r\n"]+')


def sanitise_field(value: object) -> str:
    """A field that survives a naive ``line.split(',')`` parse.

    A Q&A answer is typed by a human, and the round-1 set asks things like
    "Hai câu thơ đó là gì?" — nobody writes two lines of verse without a comma.
    ``csv.writer`` would quote such a field correctly per RFC 4180, but a grader
    that splits on commas then sees four fields with a stray quote glued on, and
    the answer is marked wrong. Reading it back with ``split(',')[2]`` truncates
    it a second time.

    Stripping is done HERE, at the single point where any CSV is written, rather
    than in each caller: make_submission stripped commas but the two operator
    paths (apply_picks, pin_video) did not, so the same answer was safe when the
    pipeline wrote it and corrupt when a human corrected it.
    """
    return re.sub(r"\s+", " ", _UNSAFE_FIELD.sub(" ", str(value))).strip()


def write_query_csv(path: Path, rows: Iterable[Sequence]) -> int:
    """One query's answers. No header, UTF-8, comma separated, LF endings.

    ``newline=""`` plus an explicit ``\\n`` terminator keeps the file free of
    CRLF, which some graders split on incorrectly on Windows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        for row in rows:
            w.writerow([sanitise_field(x) for x in row])
            n += 1
    return n


def package_submission(csv_dir: Path, zip_path: Path) -> Path:
    """Zip the CSVs with the mandatory ``submission/`` folder INSIDE the archive.

    The rules require the archive to contain a directory literally named
    ``submission``; zipping the loose CSV files is rejected. Getting this wrong
    still consumes one of the three allowed attempts, so the check belongs in
    code rather than in a checklist.
    """
    csv_dir = Path(csv_dir)
    zip_path = Path(zip_path)
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"no .csv files to package in {csv_dir}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f"submission/{f.name}")
    return zip_path


def _is_qa_name(name: str) -> bool:
    return bool(re.search(r"(?:^|[-_])(qa|q&a|vqa)(?:[-_.]|$)", Path(name).stem, re.IGNORECASE))


def verify_submission_zip(
    zip_path: Path,
    expect_max_rows: int = MAX_ROWS,
    expect_names: Optional[Iterable[str]] = None,
    allow_blank_answers: bool = False,
) -> List[str]:
    """Re-open the finished archive and check it against the rules.

    Returns a list of problems; empty means the package looks compliant.  This
    is the last gate before an upload and a rejection costs one of only three
    attempts, so it checks the whole file rather than sampling it, and it
    checks the things that are silently fatal as well as the things the
    organisers list:

    * every row, not just the first — a header hidden at row 40 is still a header;
    * a UTF-8 BOM, which glues invisible bytes onto the first ``video_id``;
    * blank answers in a Q&A file, which score 0 under rules 2.1.2 no matter
      how good the frame is;
    * a missing CSV, when ``expect_names`` says which queries the round had —
      a query that crashed leaves no file and the archive still looks valid.
    """
    problems: List[str] = []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if not names:
            return ["archive is empty"]
        if not all(n.startswith("submission/") for n in names):
            problems.append(
                "every entry must live under a 'submission/' folder inside the zip; "
                f"found {sorted({n.split('/')[0] for n in names})}"
            )
        if expect_names is not None:
            have = {Path(n).name for n in names}
            missing = sorted(set(expect_names) - have)
            if missing:
                problems.append(
                    f"{len(missing)} querie(s) produced no CSV: {', '.join(missing[:6])}"
                    + (" ..." if len(missing) > 6 else "")
                )
            # A leftover CSV from a previous round sits in the same csv/ folder
            # and gets packaged too. The grader matches CSVs to queries by name,
            # so at best it is ignored and at worst the upload is rejected for a
            # file that answers a question nobody asked.
            extra = sorted(have - set(expect_names))
            if extra:
                problems.append(
                    f"{len(extra)} CSV(s) match no query in this round — delete them or "
                    f"use a fresh --out directory: {', '.join(extra[:6])}"
                    + (" ..." if len(extra) > 6 else "")
                )

        for n in names:
            if not n.endswith(".csv"):
                problems.append(f"{n}: not a .csv")
                continue
            raw = z.read(n)
            if raw.startswith(b"\xef\xbb\xbf"):
                problems.append(f"{n}: starts with a UTF-8 BOM; the first video_id will not match")
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                problems.append(f"{n}: not valid UTF-8 ({exc})")
                continue
            if b"\r\n" in raw:
                # Every check below runs on splitlines(), which discards \r
                # before anything can notice it — so a CSV hand-edited in
                # Notepad or Excel passed the last gate and was declared "safe
                # to upload" while carrying a \r on the end of every answer.
                problems.append(
                    f"{n}: has Windows CRLF line endings; a trailing carriage return "
                    "becomes part of the last field. Rebuild it with repackage.py "
                    "instead of saving from Notepad or Excel"
                )

            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                problems.append(f"{n}: empty file")
                continue
            if len(lines) > expect_max_rows:
                problems.append(f"{n}: {len(lines)} rows exceeds the {expect_max_rows} limit")
            if ".mp4" in text:
                problems.append(f"{n}: video ids must not carry the .mp4 extension")
            if '"' in text:
                # csv.writer quotes a field that contains a comma. That is valid
                # RFC 4180 and still wrong here: a grader splitting on commas
                # reads one answer as several fields with a quote attached. The
                # writer strips commas so this cannot happen — if a quote is
                # present, something bypassed write_query_csv.
                problems.append(
                    f"{n}: contains a double quote, so a field was RFC-quoted — "
                    "a comma-splitting grader will mis-read that row"
                )

            qa = _is_qa_name(n)
            blank_answers = 0
            widths = {len(ln.split(",")) for ln in lines}
            if "trake" in Path(n).name.lower():
                # A row is `video_id,frame_1,...,frame_n`, so the width is the
                # event count PLUS ONE for the video column — 4 events means 5
                # columns, and `widths == {2}` below means ONE event, not two.
                # A splitter that fell back to a single event emits 2 columns
                # for a 4-event query and scores 0 — invisibly, because every
                # other check passes on a 2-column row.
                if len(widths) > 1:
                    problems.append(
                        f"{n}: rows disagree on column count {sorted(widths)}; every TRAKE "
                        "row must have one frame column per event"
                    )
                elif widths == {2}:
                    problems.append(
                        f"{n}: only 1 frame column, so the query was parsed as a single "
                        "event. Check that it splits on 'E1:' / '(1)' / ';'"
                    )

            for i, ln in enumerate(lines, 1):
                fields = ln.split(",")
                if len(fields) < 2:
                    problems.append(f"{n}: row {i} has fewer than 2 fields: {ln!r}")
                    break
                if not fields[1].strip().lstrip("-").isdigit():
                    problems.append(
                        f"{n}: row {i} field 2 is not an integer frame id ({fields[1]!r})"
                        + (" — a header row is not allowed" if i == 1 else "")
                    )
                    break
                if qa and (len(fields) < 3 or not fields[2].strip()):
                    blank_answers += 1
            if qa and blank_answers and not allow_blank_answers:
                problems.append(
                    f"{n}: {blank_answers}/{len(lines)} rows have a blank answer — "
                    "rules 2.1.2 score those 0 regardless of the frame "
                    "(pass allow_blank_answers=True only for a deliberate format test)"
                )
    return problems
