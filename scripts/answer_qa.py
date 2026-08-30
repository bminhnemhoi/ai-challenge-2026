"""Answer the Q&A queries by looking at the frames.

A Q&A row scores zero without an answer string, however good the frame is, so
three blank answers is three guaranteed zeros out of twenty-four queries — the
largest single loss in the round and the cheapest to fix.

All three of round-1's Q&A questions are text-reading tasks: the name of a
commune on a banner, two lines of verse on a page, the title on a recipe card.
That is exactly what a VLM is good at and exactly what a similarity score is
blind to.

The model is shown several candidate frames at once and asked to answer only
from what it can see, and to say so when it cannot. A confident wrong answer and
a blank both score zero, but a blank is honest and tells the operator to go and
look.

    python scripts/answer_qa.py --queries round_p1/queries --out round_p1/final

Nothing is overwritten silently: an answer already in the CSV is kept unless
--overwrite is given.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import (  # noqa: E402
    detect_task,
    ranked_hits,
    read_en_override,
    read_query_text,
    split_qa,
)
from src.core.submission import (  # noqa: E402
    csv_name_for_query,
    package_submission,
    sanitise_field,
    verify_submission_zip,
    write_query_csv,
)
from src.core.vlm import DEFAULT_MODEL, VLMJudge  # noqa: E402


def full_frame(video_id: str, filename: str, max_side: int):
    """Khung hình như CDN công bố (~1280px), chỉ thu nhỏ nếu vượt ``max_side``.

    Cùng một hàm với scripts/read_answer.py — bản mirror 512px đủ để nói "đúng
    cảnh không" nhưng không đủ để đọc "biển ghi số mấy".
    """
    import io
    import urllib.request

    from PIL import Image

    from src.core.vlm import CDN, UA

    try:
        raw = urllib.request.urlopen(
            urllib.request.Request(f"{CDN}/{video_id}/{filename}", headers=UA), timeout=60
        ).read()
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=95)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


def nap_loi_thoai(data_dir: Path) -> dict:
    """{video_id: [(giây, câu)]} từ data/captions — kênh thứ hai, 873/873 video."""
    d = Path(data_dir) / "captions"
    out = {}
    if not d.is_dir():
        return out
    for p in d.glob("*.json"):
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    return out


def loi_thoai_quanh(caps: dict, video_id: str, giay: float, cua_so: float = 30.0) -> str:
    seg = caps.get(video_id) or []
    lay = [str(t[1]) for t in seg if isinstance(t, (list, tuple)) and len(t) >= 2
           and abs(float(t[0]) - giay) <= cua_so]
    return " ".join(lay).strip()[:1500]

#: Đo trên 60 câu ground truth, chia TUNE/TEST 30/30 (scripts/experiment_qa_answer.py,
#: 30/08/2026): prompt này cộng ảnh gốc + lời thoại + 2 keyframe lân cận đưa độ
#: chính xác đáp án từ 63,3% lên 86,7% trên TEST (>2 sd). Ba điều nó sửa, mỗi
#: điều đều đo được riêng — chi tiết ở docs/NGHIEN_CUU_SOTA.md §1①:
#:
#:  * bản cũ dặn "không thấy thì trả chuỗi rỗng" và model nghe lời ở 11/60 câu.
#:    Đáp án rỗng CHẮC CHẮN 0 điểm còn đáp án đoán sai cũng 0 — nên đoán là trội
#:    tuyệt đối. Cấm bỏ trống xoá sạch cả 11 câu đó.
#:  * ảnh gốc MỘT MÌNH làm tệ đi (71,7% → 65,0%): độ phân giải cao cho model
#:    thấy thêm chữ nền và nó chép băng rôn thay vì trả lời. Phải cấm rõ.
#:  * lời thoại quanh khoảnh khắc là kênh thứ hai — thứ ảnh không có.
PROMPT = """Bạn đang trả lời một câu hỏi về một đoạn video tiếng Việt.

{question}
{loi_thoai}
Dưới đây là {n} khung hình lấy từ những video ứng viên hàng đầu, đánh số 1..{n}.

Quy tắc:
- TRẢ LỜI ĐÚNG CÂU HỎI ĐƯỢC HỎI. Chữ trên màn hình chỉ dùng khi nó trả lời câu
  hỏi đó; đừng chép tiêu đề/băng rôn nếu câu hỏi hỏi về màu sắc, vật thể hay
  hành động.
- Trả DANH TỪ CỤ THỂ NHẤT thấy được: tên riêng, tên loài, nhãn hiệu, con số
  chính xác (đọc kỹ dấu thập phân), màu cụ thể. TRÁNH từ hạng mục chung.
- Ngắn gọn: một cụm từ, giữ nguyên dấu tiếng Việt. Không giải thích.
- **LUÔN LUÔN đưa ra đáp án.** Nếu không chắc, vẫn đoán phương án hợp lý nhất
  và hạ "confidence" xuống. Bỏ trống chắc chắn bị 0 điểm, còn đoán thì vẫn có
  cơ hội đúng — không bao giờ trả chuỗi rỗng.

Trả về DUY NHẤT JSON:
{{"answer": "...", "nguon": "nhìn thấy|nghe thấy|đọc thấy", "frame": <số khung hình chứa câu trả lời, hoặc 0>, "confidence": 0-100, "seen": "chữ/chi tiết bạn căn cứ"}}"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--frames", type=int, default=12,
                    help="tổng số khung hình đưa cho model (đường cũ, khi --neo 0)")
    ap.add_argument("--neo", type=int, default=2,
                    help="số keyframe LÂN CẬN mỗi bên quanh khung neo (0 = tắt, "
                    "quay về đường cũ). Cấu hình đã đo: 2 (tức 5 khung), "
                    "86,7%% TEST so 63,3%% của đường cũ — docs/NGHIEN_CUU_SOTA.md")
    ap.add_argument("--them-video", type=int, default=2,
                    help="số video xếp sau được đưa thêm 1 khung, để phòng khi "
                    "video hạng 1 sai (sản xuất không biết trước khung nào đúng)")
    ap.add_argument("--max-side", type=int, default=1900,
                    help="cạnh dài tối đa của ảnh gửi đi; ảnh gốc CDN ~1280px")
    ap.add_argument("--cua-so-loi", type=float, default=30.0,
                    help="số giây lời thoại lấy quanh khoảnh khắc (0 = không gửi)")
    ap.add_argument("--overwrite", action="store_true", help="replace answers already in the CSV")
    ap.add_argument("--repackage", action="store_true", help="rebuild and verify the zip afterwards")
    args = ap.parse_args()

    judge = VLMJudge(args.data, model=args.model)
    if not judge.ready:
        print("Khong co GEMINI_API_KEY. Dat vao .env:  GEMINI_API_KEY=...")
        return 2

    from google.genai import types

    qdir = Path(args.queries)
    csv_dir = Path(args.out) / "csv"
    if not csv_dir.is_dir():
        print(f"ERROR: {csv_dir} khong ton tai — chay make_submission.py truoc")
        return 2

    qfiles = [
        p
        for p in sorted(qdir.glob("*.txt"))
        if not p.name.lower().endswith((".en.txt", ".vi.txt")) and detect_task(p.name) == "qa"
    ]
    if not qfiles:
        print("Khong co cau Q&A nao trong thu muc nay.")
        return 0

    from src.core.kis_engine import KISEngine

    print(f"model: {args.model}\nloading index ...", flush=True)
    eng = KISEngine(args.data).load()
    meta = {(m["video_id"], m["frame_idx"]): m for m in eng.metadata}
    by_n = {(m["video_id"], int(m["n"])): m for m in eng.metadata}
    caps = nap_loi_thoai(Path(args.data)) if args.cua_so_loi > 0 else {}
    if args.neo > 0:
        print(f"che do da do: anh goc <={args.max_side}px, {args.neo} keyframe lan can, "
              f"+{args.them_video} video, loi thoai +-{args.cua_so_loi:.0f}s "
              f"({len(caps)} video co loi thoai)")
    client = judge._get_client()

    filled = 0
    for qf in qfiles:
        text = read_query_text(qf) or ""
        context, question = split_qa(text)
        csv_path = csv_dir / csv_name_for_query(qf.name)

        existing = ""
        rows: list = []
        try:
            for ln in csv_path.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                parts = ln.split(",")
                rows.append(parts)
            if rows and len(rows[0]) > 2:
                existing = rows[0][2].strip()
        except Exception:  # noqa: BLE001
            pass
        if existing and not args.overwrite:
            print(f"  {qf.stem:22s} da co dap an: {existing!r} (dung --overwrite de thay)")
            continue

        # Frames from the CSV itself when it has them, so the answer is read off
        # the shots that are actually being submitted, not a fresh ranking.
        cands = []
        seen = set()
        for parts in rows:
            if len(parts) < 2:
                continue
            v, f = parts[0], int(parts[1])
            m = meta.get((v, f))
            if m and (v, f) not in seen:
                seen.add((v, f))
                cands.append((v, f, m["frame_filename"]))
            if len(cands) >= max(args.frames, 24):
                break
        if not cands:
            for h in ranked_hits(eng, context or text, read_en_override(qf)):
                key = (h.video_id, h.frame_idx)
                m = meta.get(key)
                if m and key not in seen:
                    seen.add(key)
                    cands.append((h.video_id, h.frame_idx, m["frame_filename"]))
                if len(cands) >= 24:
                    break

        # Đường ĐÃ ĐO (mặc định): neo vào khung hạng 1 — khung ta tin nhất —
        # kèm các keyframe lân cận của chính video đó, tất cả ở ảnh gốc; rồi
        # thêm 1 khung của vài video xếp sau để phòng khi hạng 1 sai video.
        # Đo cho thấy chùm lân cận đáng +5% so với chỉ một khung: khoảnh khắc
        # thật hay rơi vào khe giữa hai keyframe (4/60 câu ground truth có
        # frame_idx không phải keyframe nào), và khung kề bên thường thấy rõ
        # thứ mà khung neo che mất.
        if args.neo > 0 and cands:
            neo_v, neo_f, neo_fn = cands[0]
            neo_n = int(meta[(neo_v, neo_f)]["n"])
            chon = [(neo_v, neo_f, neo_fn)]
            for d in range(1, args.neo + 1):
                for nn in (neo_n - d, neo_n + d):
                    m = by_n.get((neo_v, nn))
                    if m:
                        chon.append((neo_v, int(m["frame_idx"]), m["frame_filename"]))
            da_co = {neo_v}
            for v, f, fn in cands[1:]:
                if len(da_co) > args.them_video:
                    break
                if v not in da_co:
                    da_co.add(v)
                    chon.append((v, f, fn))
            cands = chon
        else:
            cands = cands[: args.frames]

        blobs = []
        kept = []
        for v, f, fn in cands:
            b = (full_frame(v, fn, args.max_side) if args.neo > 0
                 else judge._fetch(v, fn))
            if b:
                blobs.append(b)
                kept.append((v, f))
        if not blobs:
            print(f"  {qf.stem:22s} khong tai duoc khung hinh nao")
            continue

        lt = ""
        if args.cua_so_loi > 0 and kept:
            m0 = meta.get(kept[0])
            if m0:
                lt = loi_thoai_quanh(caps, kept[0][0], float(m0.get("pts_time") or 0.0),
                                     args.cua_so_loi)
        prompt = PROMPT.format(
            question=text.strip()[:1200], n=len(blobs),
            loi_thoai=(f'\nLời thoại quanh khoảnh khắc này: "{lt}"\n' if lt else ""),
        )
        answer, conf, seen_text, frame_no = "", 0, "", 0
        try:
            r = client.models.generate_content(
                model=args.model,
                contents=[*[types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in blobs],
                          prompt],
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=1200),
            )
            u = getattr(r, "usage_metadata", None)
            if u:
                judge.calls += 1
                judge.tokens_in += u.prompt_token_count or 0
                judge.tokens_out += u.candidates_token_count or 0
            m = re.search(r"\{.*\}", r.text or "", re.S)
            if m:
                j = json.loads(m.group(0))
                answer = sanitise_field(j.get("answer", ""))
                conf = int(j.get("confidence", 0) or 0)
                seen_text = str(j.get("seen", ""))[:110]
                frame_no = int(j.get("frame", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            print(f"  {qf.stem:22s} LOI: {type(exc).__name__}: {str(exc)[:90]}")
            continue

        if not answer:
            print(f"  {qf.stem:22s} model khong doc duoc dap an tu {len(blobs)} khung hinh")
            print(f"      da thay: {seen_text}")
            continue

        src = ""
        if 1 <= frame_no <= len(kept):
            v, f = kept[frame_no - 1]
            src = f"  (tu {v} frame {f})"
        print(f"  {qf.stem:22s} {answer!r}  tin cay {conf}%{src}")
        if seen_text:
            print(f"      doc duoc: {seen_text}")

        out_rows = [(p[0], int(p[1]), answer) for p in rows if len(p) >= 2]
        if out_rows:
            write_query_csv(csv_path, out_rows)
            filled += 1

    print(f"\n{filled} cau da dien dap an. {judge.cost_note()}")

    if args.repackage and filled:
        zip_path = Path(args.out) / "submission.zip"
        package_submission(csv_dir, zip_path)
        expect = {
            csv_name_for_query(p.name)
            for p in qdir.glob("*.txt")
            if not p.name.lower().endswith((".en.txt", ".vi.txt"))
        }
        problems = verify_submission_zip(zip_path, expect_names=expect)
        print(f"-> {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")
        if problems:
            print("\nFORMAT PROBLEMS — dung nop:")
            for p in problems:
                print("  -", p)
            return 1
        print("format check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
