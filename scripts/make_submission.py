"""Contest-day tool: BTC query files in, a compliant submission.zip out.

    python scripts/make_submission.py --queries path/to/queries --out submission_run1

Each round is a three-hour window and only three uploads are allowed, of which
only the LAST counts.  So this script does the whole thing in one command and
verifies its own output against the rules before it finishes: correct CSV name
per query, no header row, <=100 rows, no .mp4 suffix, and everything inside a
folder literally named ``submission/`` within the archive.

Query files are read from a directory.  The task is taken from the filename,
following the organisers' own convention:

    query-1-kis.txt    -> Textual KIS   -> "<video_id>,<frame_id>"
    query-2-qa.txt     -> Q&A           -> "<video_id>,<frame_id>,<answer>"
    query-3-trake.txt  -> TRAKE         -> "<video_id>,<frame_1>,...,<frame_n>"

Anything not matching is treated as KIS, which is the safe default: a KIS row
is a prefix of the other two formats.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    CoveragePlan,
    allocate_coverage_rows,
    allocate_hybrid_rows,
    allocate_trake_rows,
    csv_name_for_query,
    package_submission,
    verify_submission_zip,
    write_query_csv,
)

#: Measured on the 60-sample ground truth across answer-window widths from 10 to
#: 200 frames (scripts/experiment_strategies.py).  Beats the previous
#: keyframes-only strategy at EVERY width tested, not just on average.
DEFAULT_N_FLAT = 30
DEFAULT_DEPTH_COST = 0.5

#: How many candidates every tool retrieves before allocating rows.
#: This has to be ONE number, not a per-caller choice: the object boost is a
#: max over the candidates of each video, so a bigger pool can find a stronger
#: match and reorder the videos differently. When make_submission used 200 and
#: the review page used 400, the page ranked a different video first than the
#: CSV it was supposed to be showing.
RETRIEVE_TOP_N = 400


# --------------------------------------------------------------- query files


def detect_task(name: str) -> str:
    low = name.lower()
    if "trake" in low:
        return "trake"
    if "qa" in low or "q&a" in low or "vqa" in low:
        return "qa"
    return "kis"


def split_events(text: str) -> List[str]:
    """Pull ordered event descriptions out of a TRAKE prompt.

    The number of events decides the number of frame columns in the CSV, and
    the grader compares column j against window j — so getting the count wrong
    scores 0 on the whole query, not just on one event.  Three shapes are
    handled, in order of how explicit they are:

      1. ``E1: ... E2: ...``            explicit markers
      2. ``(1) ... (2) ...`` / ``1. ``  numbered lists — this is the phrasing
                                        the rulebook's own example uses
      3. ``context: a; b; c``           a separated list after a colon

    The shared stem before the list is prepended to each event, because
    "giậm nhảy" alone retrieves nothing while "vận động viên nhảy cao, giậm
    nhảy" retrieves the right shot.
    """
    text = (text or "").strip()

    marks = re.split(r"\bE\s*\d+\s*[:.\-–—]\s*", text, flags=re.IGNORECASE)
    if len(marks) >= 3:
        return [m.strip().strip(".,;") for m in marks[1:] if m.strip()]

    # Same markers, but written with nothing after the number:
    #
    #     Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng ...
    #     E1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng ...
    #     E2 ...
    #
    # Round 1 was phrased exactly this way. Without punctuation after "E1" the
    # split above misses, the prompt falls through to the line fallback, and the
    # scene-setting first line is counted as a fourth event — which writes five
    # columns where the key has four and throws the whole query away. Anchoring
    # to the line start keeps this from firing on an "E1" inside a sentence.
    marks = re.split(r"(?:^|\r?\n)[ \t]*E\s*\d+[ \t]+", text, flags=re.IGNORECASE)
    if len(marks) >= 3:
        return [m.strip().strip(".,;") for m in marks[1:] if m.strip()]

    # numbered list: "(1) giậm nhảy, (2) bay qua xà, ..."
    num = re.split(r"(?:^|[\s,;])\(?\d{1,2}[\).:]\s+", text)
    if len(num) >= 3:
        stem = num[0].strip().rstrip(":,. ")
        stem = re.sub(r"^.*?:\s*", "", stem) if ":" in stem else stem
        out = []
        for p in num[1:]:
            p = p.strip().strip(".,;")
            if p:
                out.append(f"{stem}, {p}" if stem else p)
        return out

    if ":" in text:
        head, tail = text.rsplit(":", 1)
        parts = [p.strip().strip(".,;") for p in re.split(r"[;\n]", tail) if p.strip()]
        if len(parts) >= 2:
            stem = head.strip().strip(".,;")
            return [f"{stem}, {p}" if stem else p for p in parts]

    return [p.strip() for p in re.split(r"[;\n]", text) if p.strip()]


def read_query_text(path: Path) -> Optional[str]:
    """Decode a query file without assuming the organisers used UTF-8.

    A single UTF-16 file (Notepad's "Unicode" default) used to raise out of the
    per-query try block and abort the whole run, leaving no zip at all after
    the index had already been loaded and half the queries retrieved.
    ``utf-8-sig`` comes first so a BOM is dropped rather than prepended to the
    retrieval prompt.
    """
    return decode_text(path.read_bytes())


#: Vietnamese is dense in these, so a decode that produces none of them from a
#: Vietnamese source is the wrong decode
_VN_MARKS = "ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"


def decode_text(raw: bytes) -> Optional[str]:
    """Best-effort decode of a query file, preferring the reading that looks Vietnamese.

    Trying codecs in order and returning the first that does not raise is not
    enough: Python's ``utf-16`` codec accepts almost ANY even-length byte string
    (with no BOM it assumes little-endian and only rejects unpaired surrogates),
    so a cp1258 or latin-1 file silently decodes to CJK garbage and cp1258 is
    never reached. The query then retrieves on nonsense and scores 0 with no
    error anywhere. So every candidate decode is scored, and the one that
    actually contains Vietnamese wins.
    """
    best = None
    for enc in ("utf-8-sig", "utf-16", "cp1258", "latin-1"):
        try:
            s = raw.decode(enc).strip()
        except (UnicodeDecodeError, UnicodeError):
            continue
        if not s:
            continue
        if "\x00" in s:
            continue  # a NUL means we read UTF-16 bytes as single-byte, or vice versa
        low = s.lower()
        score = (
            sum(low.count(c) for c in _VN_MARKS) * 3
            + sum(ch.isascii() and (ch.isalnum() or ch.isspace()) for ch in s)
            - sum(ord(ch) > 0x2000 for ch in s) * 5  # CJK/odd blocks: wrong decode
        )
        if best is None or score > best[0]:
            best = (score, s)
    return best[1] if best else None


def split_qa(text: str) -> Tuple[str, str]:
    """Separate the scene description from the question.

    The question is whatever sentence ends in '?'; everything before it is the
    scene to retrieve on.  Falling back to the whole text keeps retrieval
    working even when the split fails.
    """
    m = list(re.finditer(r"[^.?!\n]*\?", text))
    if m:
        question = m[-1].group(0).strip()
        context = text[: m[-1].start()].strip() or text.strip()
        return context, question
    return text.strip(), ""


# ------------------------------------------------------------------ builders


def read_en_override(qfile: Path) -> Optional[str]:
    """A hand-written English rendering placed next to the query file.

    ``query-1-kis.txt`` -> ``query-1-kis.en.txt``.  Automatic translation is a
    scraped endpoint that rate-limits under load, and a good English prompt is
    worth about 8 points of video R@1, so a teammate typing translations during
    the three-hour window is a real and cheap fallback — not a workaround.
    """
    side = qfile.with_suffix("")
    for cand in (qfile.parent / f"{side.name}.en.txt", qfile.parent / f"{qfile.stem}.en"):
        if cand.exists():
            # Same tolerant decode as the query itself. Reading this with strict
            # UTF-8 meant a teammate saving the sidecar from Notepad as
            # "Unicode" raised inside the per-query try, so the query fell to
            # the placeholder row — a guaranteed 0 caused by the very file that
            # was meant to improve it. A BOM is dropped here too, or it becomes
            # the first character of the English prompt.
            t = (decode_text(cand.read_bytes()) or "").strip()
            if t:
                return t
    return None


def ranked_hits(engine, query_text: str, query_en: Optional[str], top_n: int = RETRIEVE_TOP_N):
    """The one ranking every tool must agree on: original text, human English if given.

    This used to merge two candidate lists — one from the automatic translation,
    one from the hand-written English — taking the better score per frame, on
    the argument that R@k is a maximum over a prefix so an extra candidate can
    only help.  That argument is wrong, because merging also *reorders*: a frame
    the rewrite likes is promoted past one the automatic reading found, and the
    top 30 slots are finite.

    Measured on all 60 ground-truth queries (which carry a human English
    rendering), scored with the official formula against a non-snapped answer
    key, 24 re-draws:

        automatic translation only      0.292
        human English only              0.321
        both in one vector  (this)      0.337   <- best, video R@1 26/60
        merged candidate lists          0.305

    So the two readings belong in one query vector, where the 4-prompt ensemble
    already weighs them, not in two lists competing for rank slots.
    """
    hits = (
        engine.search(query_text, query_en=query_en, top_n=top_n)
        if query_en
        else engine.search(query_text, top_n=top_n)
    )
    hits = _peak_preference(engine, hits)
    return _object_boost(engine, hits, query_en or engine.translate(query_text))


#: the old name, kept so nothing breaks mid-round; it no longer merges
merged_hits = ranked_hits

#: how much a keyframe gains for standing out from the ones beside it in time.
#: Measured on the ground truth: +2.2% at 0.01, +1.0% at 0.002, so the curve is
#: gentle and this sits on it rather than on a spike.
PEAK_WEIGHT = 0.01


def _peak_preference(engine, hits):
    """Prefer a keyframe that is a local maximum over its own video's timeline.

    A moment is a peak; a scene is a plateau. When a query describes an instant,
    the frames on either side of it are the same scene a second earlier and a
    second later, and they score almost as well — so the embedding's ordering
    inside a video is close to arbitrary among them. Measured: the keyframe
    NEAREST the true instant is rank 1 inside the right video only 48% of the
    time, though it is in that video's top 5 76% of the time.

    That matters because of how the row budget is spent: cost(i, d) = i + 0.5*d,
    so a candidate at global rank 1 gets a ladder reaching +-120 frames while one
    at rank 25 gets a single flat row and no ladder at all. Moving the right
    keyframe a few places up the list is worth far more than the score change
    suggests.

    Only neighbours already among the candidates are considered; a keyframe whose
    neighbours scored too low to be retrieved is a peak by definition.
    """
    if not hits or PEAK_WEIGHT <= 0:
        return hits
    try:
        timeline = getattr(engine, "_kf_timeline", None)
        if timeline is None:
            import numpy as _np

            timeline = {}
            for m in engine.metadata:
                timeline.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
            timeline = {v: _np.array(sorted(f)) for v, f in timeline.items()}
            engine._kf_timeline = timeline

        import numpy as np

        seen: dict = {}
        for h in hits:
            seen.setdefault(h.video_id, {})[int(h.frame_idx)] = h.score

        scored = []
        for i, h in enumerate(hits):
            arr = timeline.get(h.video_id)
            bump = 0.0
            if arr is not None and len(arr) > 1:
                j = int(np.searchsorted(arr, h.frame_idx))
                near = [
                    seen[h.video_id].get(int(arr[k]))
                    for k in (j - 1, j + 1)
                    if 0 <= k < len(arr)
                ]
                near = [x for x in near if x is not None]
                if near:
                    bump = max(h.score - max(near), 0.0)
            scored.append((h.score + PEAK_WEIGHT * bump / max(abs(h.score), 1e-6), i, h))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [t[2] for t in scored]
    except Exception as exc:  # noqa: BLE001 - a tie-break must never break a run
        print(f"    ! peak preference skipped ({type(exc).__name__}: {exc})")
        return hits


#: built lazily on first use and reused; None once we know there is no data
_OBJECTS = "unset"
#: set False by --no-objects
USE_OBJECTS = True


def _object_boost(engine, hits, query_en):
    """Reorder the candidate VIDEOS by detector agreement (+3.3%, measured).

    See src/core/objects.py for the measurement and, more importantly, for why
    the same bonus applied per FRAME instead of per video makes the score worse
    while making video accuracy look better.
    """
    global _OBJECTS
    if not USE_OBJECTS or not hits:
        return hits
    try:
        from src.core.objects import ObjectIndex

        if _OBJECTS == "unset":
            _OBJECTS = ObjectIndex(getattr(engine, "data_dir", "data"))
        if _OBJECTS is None:
            return hits

        meta = getattr(engine, "_meta_by_key", None)
        if meta is None:
            meta = {(m["video_id"], m["frame_idx"]): m for m in engine.metadata}
            engine._meta_by_key = meta

        def stem_of(h):
            m = meta.get((h.video_id, h.frame_idx))
            return Path(m["frame_filename"]).stem if m else ""

        _OBJECTS.load_for({(h.video_id, stem_of(h)) for h in hits})
        if not _OBJECTS.available:
            _OBJECTS = None  # no objects zip on this machine; stop retrying
            return hits
        return _OBJECTS.rerank(hits, query_en, stem_of)
    except Exception as exc:  # noqa: BLE001 - a nicety must never break a run
        print(f"    ! object reranking skipped ({type(exc).__name__}: {exc})")
        _OBJECTS = None
        return hits


def them_ung_vien_canh_b(engine, cands, query_text: str, khoa: str, m: int):
    """Truy xuất THÊM bằng cảnh B cho câu mô tả hai cảnh nối tiếp.

    Cơ chế hỏng đã đo được trên bộ đo khớp phân bố đề thật
    (docs/UNG_VIEN_CANH_B.md): với câu hai cảnh, keyframe đáp án **không hề nằm
    trong 400 ứng viên** ở 31/66 câu, trong khi video vẫn được tìm đúng ngang
    câu một cảnh. Truy vấn nén cả hai cảnh vào một vector nên nó khớp cảnh MỞ
    ĐẦU; keyframe của cảnh B — chính là khoảnh khắc phải nộp — bị bỏ lại ngoài
    danh sách. Chấm lại ứng viên không cứu được thứ chưa bao giờ được truy xuất.

    Đây KHÔNG phải phép trộn điểm hai kênh (bằng chứng nội bộ đã đóng cửa đó);
    nó là **hợp hai lần truy xuất của cùng một encoder**, cùng thang cosine, nên
    không có hệ số pha trộn nào phải chọn. Ứng viên mới nối vào CUỐI danh sách.

    Đo được: keyframe đáp án có mặt trong pool tăng **53% → 76%** (đếm tất định,
    không phải ước lượng); điểm câu qua cổng +23,3% trên TEST.
    """
    if m <= 0 or not query_text.strip():
        return cands, None, None
    try:
        from scripts.gan_nhan_hai_canh import nhan_mot_cau
    except Exception:  # noqa: BLE001
        return cands, None, None
    rec = nhan_mot_cau(query_text, khoa)
    if not rec or not rec.get("co_2_canh"):
        return cands, None, None
    b_vi = (rec.get("canh_B_vi") or "").strip()
    if not b_vi:
        return cands, None, None
    try:
        import numpy as np

        sims = engine.query_similarities(b_vi, rec.get("canh_B_en") or None)
        k = min(m, len(sims) - 1)
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        co = {(c.video_id, int(c.frame_idx)) for c in cands}
        them = []
        for j in top:
            md = engine.metadata[int(j)]
            key = (md["video_id"], int(md["frame_idx"]))
            if key in co:
                continue
            co.add(key)
            them.append(Candidate(key[0], key[1], float(sims[int(j)]),
                                  engine.last_frame.get(key[0])))
        return list(cands) + them, f"{len(them)} ung vien tu canh B", sims
    except Exception as exc:  # noqa: BLE001
        print(f"    ! canh B bo qua ({type(exc).__name__}: {str(exc)[:60]})")
        return cands, None, None


_HANG_CACHE: dict = {}


def _hang_of(engine):
    """{(video_id, frame_idx): hang trong ma tran} — dung MOT lan cho ca vong."""
    k = id(engine)
    if k not in _HANG_CACHE:
        _HANG_CACHE[k] = {(m["video_id"], int(m["frame_idx"])): i
                          for i, m in enumerate(engine.metadata)}
    return _HANG_CACHE[k]


def hoan_vi_theo_canh_b(cands, simsB, hang_of, w: float = 1.0, alpha: float = 0.5,
                        so_video: int = 3, so_khung: int = 12):
    """Hoán vị ĐIỂM trong từng video theo độ tương đồng với cảnh B.

    Không thêm, không bớt, không đổi video nào: với mỗi video, **đa tập điểm giữ
    nguyên**, chỉ đổi xem điểm nào thuộc khung nào. Vì tổng khối lượng softmax
    của mỗi video không đổi theo xây dựng, bề rộng phủ video **không thể** bị
    phá — chỉ hình dạng khối lượng *bên trong* video dịch đi. Đó đúng là thứ cần:
    chẩn đoán cho thấy sai số của câu hai cảnh là sai số **một ô keyframe**
    (~56 frame), không phải sai số chọn video.

    Điểm định vị, không phải điểm phân loại:

        loc(f) = B(f) · (1 − α · B(khung được chấm liền trước cùng video))

    α = 0 là "khung này có phải cảnh B không"; α = 0,5 nghiêng về "khung ĐẦU
    TIÊN của cảnh B" — mà với câu hai cảnh, khoảnh khắc phải nộp *được định
    nghĩa* là chỗ cảnh B bắt đầu. Trục thời gian lấy từ ``frame_idx``, không hỏi
    model, nên không có chỗ nào để nhầm số thứ tự ảnh.

    Vector ``simsB`` đã được ``them_ung_vien_canh_b`` tính sẵn và trước đây bị
    vứt sau khi lấy top-M — lever này không thêm một phép nhân ma trận nào.

    Đo được (docs/KE_HOACH_DINH_VI.md §1): **+57,6%** trên 66 câu hai cảnh của bộ
    đo khớp phân bố, KTC theo câu tách khỏi 0, và giữ nguyên dấu lẫn độ lớn dưới
    **bốn** mô hình bốc khoảnh khắc khác nhau (+50,6% → +62,8%) — chữ ký của phép
    chọn Ô, ngược hẳn trục sigma của allocator vốn đổi chiều theo giả định bốc.
    Bốn đối chứng đều đi đúng chiều: khoá ngẫu nhiên −0,5→−8,1%, đảo dấu cảnh B
    −32→−37%, dùng cảnh A −11→−35%, và VLM trả phí hỏi bằng cảnh B *thua* tín
    hiệu 0 đồng này.
    """
    if simsB is None or w <= 0 or len(cands) < 2:
        return cands

    # --- chọn khung được chấm: top-N video theo thứ tự hạng, mỗi video top-K
    # ứng viên ĐIỂM CAO NHẤT (hoán vị chỉ dịch được khối lượng nó chạm tới)
    thu_tu, theo_video = [], {}
    for i, c in enumerate(cands):
        if c.video_id not in theo_video:
            thu_tu.append(c.video_id)
            theo_video[c.video_id] = []
        theo_video[c.video_id].append(i)

    key_of: dict = {}
    for vid in thu_tu[:so_video]:
        pos = sorted(theo_video[vid], key=lambda i: -float(cands[i].score))[:so_khung]
        if len(pos) < 2:
            continue
        gia = []
        for i in pos:
            r = hang_of.get((cands[i].video_id, int(cands[i].frame_idx)))
            gia.append(float(simsB[r]) if r is not None else 0.0)
        lo, hi = min(gia), max(gia)
        chuan = [(g - lo) / (hi - lo) for g in gia] if hi > lo else [0.0] * len(gia)

        # loc theo trục thời gian: sắp theo frame_idx rồi trừ phần "khung trước
        # đã là cảnh B rồi", để thưởng cho chỗ BẮT ĐẦU chứ không thưởng cả vùng
        thu = sorted(range(len(pos)), key=lambda k: int(cands[pos[k]].frame_idx))
        truoc = 0.0
        for k in thu:
            b = chuan[k]
            # khoá đánh theo CHỈ SỐ ứng viên, không theo cặp (video, khung):
            # pool sản xuất có khung TRÙNG (cùng keyframe, hai điểm khác nhau),
            # đánh khoá theo cặp làm vỡ bất biến w=0
            key_of[pos[k]] = float(cands[pos[k]].score) + w * b * (1.0 - alpha * truoc)
            truoc = b

    if len(key_of) < 2:
        return cands

    # --- hoán vị điểm trong từng video, giữ nguyên đa tập điểm
    diem_moi = [float(c.score) for c in cands]
    for _vid, pos in theo_video.items():
        co = [i for i in pos if i in key_of]
        if len(co) < 2:
            continue
        cac_diem = sorted((float(cands[i].score) for i in co), reverse=True)
        for i, d in zip(sorted(co, key=lambda i: (-key_of[i], i)), cac_diem):
            diem_moi[i] = d
    return [Candidate(c.video_id, c.frame_idx, diem_moi[i], c.video_last_frame)
            for i, c in enumerate(cands)]


def allocate_rows(cands, allocator: str, n_flat: int, plan: AllocationPlan):
    """One dispatch point so KIS and Q&A can never disagree on the allocator.

    ``coverage`` is the probability-coverage allocator, the default since the
    ship gate passed on this exact code path (+15.3%/+16.0% over hybrid on the
    two TEST halves, >2 sigma — docs/SHIP_PHU_XAC_SUAT.md §3b); ``hybrid`` is
    the previous baseline, kept as the one-flag rollback.  The hybrid plan is
    passed through as the coverage tail-fill plan, so the rows past the point
    where coverage runs out of mass are exactly the hybrid rows.
    """
    if allocator == "coverage":
        return allocate_coverage_rows(
            cands,
            plan=CoveragePlan(budget=plan.budget),
            tail_n_flat=n_flat,
            tail_plan=plan,
        )
    return allocate_hybrid_rows(cands, n_flat=n_flat, plan=plan)


def build_kis_rows(engine, query_text: str, n_flat: int, depth_cost: float, step: int,
                   query_en: Optional[str] = None, allocator: str = "hybrid",
                   canh_b: int = 0, khoa: str = "", hoan_vi: bool = False):
    hits = merged_hits(engine, query_text, query_en)
    cands = [
        Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits
    ]
    cands, ghi_chu, simsB = them_ung_vien_canh_b(engine, cands, query_text, khoa, canh_b)
    if ghi_chu:
        print(f"    + {ghi_chu}")
    if hoan_vi and simsB is not None:
        cands = hoan_vi_theo_canh_b(cands, simsB, _hang_of(engine))
        print("    + hoan vi diem noi-video theo canh B")
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=depth_cost, step=step)
    return allocate_rows(cands, allocator, n_flat, plan)


def build_qa_rows(engine, query_text: str, answerer, n_flat: int, depth_cost: float, step: int,
                  query_en: Optional[str] = None, allocator: str = "hybrid",
                  canh_b: int = 0, khoa: str = "", hoan_vi: bool = False):
    """Q&A rows: the same frames as KIS, every one carrying an answer.

    The answer column is the same string on every row.  Leaving later rows
    blank (or "Không xác định") forfeits R@20/R@50/R@100 — three of the five
    terms in the Final Score — for no reason, because a row only scores when
    its frame is right anyway, and a wrong row costs nothing.

    The answerer is given the retrieval Hit objects, not the ladder rows: it
    needs the keyframe ordinal ``n`` to fetch the actual image, and a ladder
    row's frame id is a synthetic integer that is not a keyframe at all.
    """
    context, question = split_qa(query_text)
    hits = merged_hits(engine, context or query_text, query_en)
    cands = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits]
    cands, ghi_chu, simsB = them_ung_vien_canh_b(engine, cands, context or query_text,
                                                 khoa, canh_b)
    if ghi_chu:
        print(f"    + {ghi_chu}")
    if hoan_vi and simsB is not None:
        cands = hoan_vi_theo_canh_b(cands, simsB, _hang_of(engine))
        print("    + hoan vi diem noi-video theo canh B")
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=depth_cost, step=step)
    frame_rows = allocate_rows(cands, allocator, n_flat, plan)

    answer = ""
    if answerer is not None and cands:
        try:
            # ĐƯỜNG ĐÃ ĐO, không phải đường riêng của file này.
            #
            # Trước đây chỗ này gọi `answerer(hits[:5], ...)` — một đường sinh
            # đáp án RIÊNG (gemini_engine.answer_single_frame + biểu quyết đa số)
            # tách hẳn khỏi answer_qa.py, nên MỌI cải tiến đã đo ở đó chưa bao
            # giờ tới được công cụ ngày thi. Đo trên 8 câu Q&A đề thật đã người
            # kiểm chứng: đường cũ cho **1/8 đúng, 4/8 RỖNG**, và 2 trong 3 câu
            # "sai" thực chất là model từ chối trả lời.
            #
            # Ba lỗi của đường cũ: max_output_tokens=25 (model trả rỗng thay vì
            # câu ngắn); prompt cho phép nói "không xác định"; và biểu quyết đa
            # số trên 5 khung **chủ động loại bỏ** phiếu của khung đúng khi 4/5
            # khung là cảnh A — đúng hồ sơ của câu hai cảnh.
            #
            # Dùng `cands` chứ không phải `hits`: lever cảnh B và lever hoán vị
            # đều sửa `cands`, nên đọc `hits` là đọc thứ tự TRƯỚC mọi cải tiến.
            from scripts.answer_qa import tra_loi_tu_ung_vien

            answer, ghi = tra_loi_tu_ung_vien(
                answerer.judge, answerer.model, cands, answerer.meta,
                answerer.by_n, answerer.caps, query_text)
            if ghi:
                print(f"    doc dap an: {ghi}")
        except Exception as exc:  # noqa: BLE001 - never lose the frames over this
            print(f"    ! answering failed ({type(exc).__name__}: {exc})")
    answer = re.sub(r"[,\r\n\"]+", " ", str(answer)).strip()
    if not answer:
        print("    ! WARNING: blank answer — these rows score 0 under rules 2.1.2.")
        print("      Type the answer into column 3 of the CSV by hand before uploading.")
    return [(v, f, answer) for v, f in frame_rows]


def build_trake_rows(engine, query_text: str, step: int, query_en: Optional[str] = None):
    """TRAKE rows: one video, many frame-tuple variants.

    A row on the wrong video scores exactly 0 (rules 2.1.3), so every row uses
    the single best candidate video; the budget is spent perturbing frames
    instead, because the per-event windows are "usually under 10 frames" and a
    raw keyframe index rarely lands inside one.
    """
    from src.task3_trake import TRAKEEngine

    # a hand-written English rendering helps TRAKE at least as much as KIS,
    # since every event description is encoded separately
    events = split_events(query_en or query_text)

    # The event count IS the column count, and a TRAKE CSV with the wrong number
    # of columns scores 0 with nothing to show for it. A hand-written .en.txt is
    # prose more often than a numbered list, and prose falls through every
    # splitter to a single event — so the sidecar, meant to help, would quietly
    # turn a 4-event query into a 2-column file. Cross-check the two readings
    # and keep whichever yields more events.
    if query_en:
        vi_events = split_events(query_text)
        if len(vi_events) > len(events):
            print(
                f"    ! the .en.txt splits into {len(events)} event(s) but the original "
                f"has {len(vi_events)} — using the original, so the column count is right"
            )
            events = vi_events
    if len(events) < 2:
        print(
            f"    ! WARNING: only {len(events)} event parsed — a TRAKE answer needs one "
            "column per event. Check the query splits on 'E1:'/'(1)'/';' or write an "
            ".en.txt using those markers."
        )

    trake = TRAKEEngine(engine=engine).load_index()
    first = bool(re.search(r"đầu tiên|lần đầu|first", query_text, re.IGNORECASE))
    results = trake.align_sequence(events, first_occurrence=first, top_k=1)
    if not results:
        return []
    best = results[0]
    last = engine.last_frame.get(best["video_id"])
    return allocate_trake_rows(
        best["video_id"], best["sequence_frames"], budget=MAX_ROWS, step=step,
        video_last_frame=last,
    )


# ---------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True, help="directory of BTC query .txt files")
    ap.add_argument("--out", default="submission_out", help="working directory for the CSVs")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--zip", default=None, help="path of the .zip (default: <out>/submission.zip)")
    ap.add_argument("--n-flat", type=int, default=DEFAULT_N_FLAT,
                    help="rows spent on distinct keyframes before the frame ladder starts")
    ap.add_argument("--depth-cost", type=float, default=DEFAULT_DEPTH_COST)
    ap.add_argument("--step", type=int, default=10,
                    help="ladder spacing; must not exceed the assumed answer-window width")
    ap.add_argument(
        "--allocator",
        choices=("hybrid", "coverage"),
        default="coverage",
        help="how KIS/Q&A rows are spent. 'coverage' (probability coverage) is the "
        "default since the ship gate passed on this exact build: +15.3%%/+16.0%% "
        "over hybrid on the two TEST halves, >2 sigma (docs/SHIP_PHU_XAC_SUAT.md "
        "s3b). 'hybrid' is the previous baseline and the one-flag rollback.",
    )
    ap.add_argument(
        "--canh-b", type=int, default=100,
        help="voi cau mo ta HAI CANH noi tiep: truy xuat them top-M keyframe theo "
        "rieng canh B roi gop vao pool ung vien (0 = tat). Do duoc: keyframe dap an "
        "co mat trong pool 53%% -> 76%%, diem cau qua cong +23,3%% TEST "
        "(docs/UNG_VIEN_CANH_B.md). Ton 1 lan goi LLM/cau de gan nhan, co cache.",
    )
    ap.add_argument(
        "--hoan-vi-canh-b", type=int, choices=(0, 1), default=1,
        help="hoan vi DIEM noi-video theo sim(canh B) cho cau qua cong hai canh "
        "(0 = tat, ra dong y het duong san xuat truoc do). Do duoc +57,6%% tren "
        "66 cau hai canh, giu dau va do lon duoi 4 mo hinh boc — "
        "docs/KE_HOACH_DINH_VI.md §1. Doc lap voi --canh-b, nhung --canh-b 0 "
        "cung tu tat lever nay.",
    )
    ap.add_argument("--no-answer", action="store_true", help="skip the VQA model, emit frames only")
    ap.add_argument(
        "--no-objects",
        action="store_true",
        help="skip the object-detection video boost (worth +3.3%%, measured; "
        "turn it off only if data/objects-aic25-b1.zip is making the run slow)",
    )
    ap.add_argument(
        "--allow-blank-answers",
        action="store_true",
        help="let the package pass verification with empty Q&A answers. Use ONLY for a "
        "deliberate format smoke-test upload — those rows score 0 under rules 2.1.2.",
    )
    args = ap.parse_args()

    global USE_OBJECTS
    USE_OBJECTS = not args.no_objects

    qdir = Path(args.queries)
    if not qdir.is_dir():
        print(f"ERROR: --queries {qdir} is not a directory")
        return 2
    # `.en.txt` / `.vi.txt` are translation sidecars, not queries — globbing
    # them would ship an extra, meaningless CSV inside submission/
    qfiles = sorted(
        p for p in qdir.glob("*.txt") if not p.name.lower().endswith((".en.txt", ".vi.txt"))
    )
    if not qfiles:
        print(f"ERROR: no .txt query files in {qdir}")
        return 2

    out_dir = Path(args.out)
    csv_dir = out_dir / "csv"
    if csv_dir.exists():
        for f in csv_dir.glob("*.csv"):
            f.unlink()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(qfiles)} query files in {qdir}")
    print(f"allocator: {args.allocator}")
    # the review page and apply_picks read this to stay on the same allocator;
    # package_submission only globs *.csv so it never enters the upload
    (csv_dir / "allocator.txt").write_text(args.allocator + "\n", encoding="utf-8")
    print("loading the SigLIP-2 index (this is the slow part; it happens once) ...", flush=True)
    t0 = time.time()
    from src.core.kis_engine import KISEngine

    engine = KISEngine(args.data).load()
    print(f"  ready in {time.time() - t0:.1f}s\n")

    answerer = None
    if not args.no_answer:
        answerer = _make_answerer(engine, args.data)

    counts = {"kis": 0, "qa": 0, "trake": 0}
    failed: List[str] = []
    for qf in qfiles:
        task = detect_task(qf.name)
        t1 = time.time()
        try:
            text = read_query_text(qf)
            if text is None:
                raise ValueError("could not decode the file in any known encoding")
            en = read_en_override(qf)
            if task == "trake":
                rows = build_trake_rows(engine, text, args.step, query_en=en)
            elif task == "qa":
                rows = build_qa_rows(
                    engine, text, answerer, args.n_flat, args.depth_cost, args.step,
                    query_en=en, allocator=args.allocator,
                    canh_b=args.canh_b, khoa=qf.stem,
                    hoan_vi=bool(args.hoan_vi_canh_b),
                )
            else:
                rows = build_kis_rows(
                    engine, text, args.n_flat, args.depth_cost, args.step,
                    query_en=en, allocator=args.allocator,
                    canh_b=args.canh_b, khoa=qf.stem,
                    hoan_vi=bool(args.hoan_vi_canh_b),
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  {qf.name:26s} FAILED: {type(exc).__name__}: {exc}")
            rows, en = [], None

        if task == "trake":
            flat = [(v, *f) for v, f in rows]
        else:
            flat = rows
        flat = flat[:MAX_ROWS]

        if not flat:
            # An empty CSV would make the verifier block the WHOLE upload, so a
            # failed query must still produce a well-formed (if useless) file.
            # A wrong row can never lower any R@k.
            fallback = engine.metadata[0]
            flat = [(fallback["video_id"], int(fallback["frame_idx"]))]
            failed.append(qf.name)
            print(f"  {qf.name:26s} {task:5s}  placeholder row written — ANSWER THIS ONE BY HAND")
        else:
            counts[task] += 1

        n = write_query_csv(csv_dir / csv_name_for_query(qf.name), flat)
        if flat and qf.name not in failed:
            tag = " [manual EN]" if en else ""
            print(f"  {qf.name:26s} {task:5s}  {n:3d} rows  {time.time() - t1:5.1f}s{tag}")

    zip_path = Path(args.zip) if args.zip else out_dir / "submission.zip"
    package_submission(csv_dir, zip_path)
    expected = {csv_name_for_query(q.name) for q in qfiles}
    problems = verify_submission_zip(
        zip_path, expect_names=expected, allow_blank_answers=args.allow_blank_answers
    )

    print(f"\nwrote {zip_path}  ({zip_path.stat().st_size / 1024:.0f} KB)")
    print(f"  kis={counts['kis']}  qa={counts['qa']}  trake={counts['trake']}  of {len(qfiles)} queries")
    if failed:
        print(f"\n  {len(failed)} querie(s) produced only a placeholder row — fix these by hand:")
        for name in failed:
            print("    -", name)
    if problems:
        print("\nFORMAT PROBLEMS — do not upload this:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nformat check passed: submission/ folder, every row parsed, <=100 rows, no BOM, no .mp4")
    if failed:
        print("(the placeholder rows above are valid but will score 0 — replace them if you can)")
    print("Upload this zip. Remember: only 3 attempts per round, and the LAST one counts.")
    return 0


class _DocDapAn:
    """Gói mọi thứ đường sinh đáp án ĐÃ ĐO cần, dựng một lần cho cả vòng.

    Thay cho ``_make_answerer`` cũ (gemini_engine + biểu quyết đa số). Lý do
    thay nằm ở comment tại chỗ gọi trong ``build_qa_rows``: đường cũ đo được
    **1/8 đúng, 4/8 rỗng** trên câu Q&A đề thật đã người kiểm chứng.
    """

    def __init__(self, engine, data_dir: str, model: str, cua_so_loi: float = 30.0):
        from src.core.vlm import VLMJudge

        self.judge = VLMJudge(data_dir, model=model)
        self.model = model
        self.meta = {(m["video_id"], m["frame_idx"]): m for m in engine.metadata}
        self.by_n = {(m["video_id"], int(m["n"])): m for m in engine.metadata}
        from scripts.answer_qa import nap_loi_thoai

        self.caps = nap_loi_thoai(Path(data_dir)) if cua_so_loi > 0 else {}

    @property
    def ready(self) -> bool:
        return bool(self.judge.ready)


def _make_answerer(engine=None, data_dir: str = "data", model: str = None):
    """Đường sinh đáp án, hoặc None kèm cảnh báo to nếu không dùng được.

    Dòng Q&A không có đáp án là vô giá trị, nên thiếu key phải lộ ra NGAY chứ
    không phải phát hiện trên bảng điểm.
    """
    from src.core.vlm import DEFAULT_MODEL

    try:
        d = _DocDapAn(engine, data_dir, model or DEFAULT_MODEL)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! VQA tat: khong dung duoc duong sinh dap an ({exc})")
        return None
    if not d.ready:
        print("  ! VQA tat: khong co GEMINI_API_KEY. Dong Q&A se mang dap an rong.")
        print("    Dat key, hoac tra loi tay roi chay lai voi --no-answer.")
        return None
    return d


if __name__ == "__main__":
    raise SystemExit(main())
