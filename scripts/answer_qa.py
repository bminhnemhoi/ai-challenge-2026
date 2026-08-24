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

PROMPT = """Bạn đang trả lời một câu hỏi về một đoạn video tiếng Việt.

{question}

Dưới đây là {n} khung hình lấy từ những video ứng viên hàng đầu, đánh số 1..{n}.
Hãy ĐỌC KỸ mọi chữ hiện trên các khung hình (băng rôn, phụ đề, bảng, tờ giấy,
tiêu đề chương trình) và trả lời câu hỏi.

Quy tắc:
- Chỉ trả lời từ những gì THẤY ĐƯỢC trong ảnh. Không suy đoán, không dùng kiến
  thức bên ngoài.
- Đáp án phải NGẮN GỌN, đúng như chữ hiện trên hình (giữ nguyên dấu tiếng Việt).
- Nếu không khung hình nào cho thấy câu trả lời, trả về chuỗi rỗng cho "answer".

Trả về DUY NHẤT JSON:
{{"answer": "...", "frame": <số khung hình chứa câu trả lời, hoặc 0>, "confidence": 0-100, "seen": "chữ bạn đọc được"}}"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--frames", type=int, default=12, help="candidate frames shown to the model")
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
            if len(cands) >= args.frames:
                break
        if len(cands) < args.frames:
            for h in ranked_hits(eng, context or text, read_en_override(qf)):
                key = (h.video_id, h.frame_idx)
                m = meta.get(key)
                if m and key not in seen:
                    seen.add(key)
                    cands.append((h.video_id, h.frame_idx, m["frame_filename"]))
                if len(cands) >= args.frames:
                    break

        blobs = []
        kept = []
        for v, f, fn in cands:
            b = judge._fetch(v, fn)
            if b:
                blobs.append(b)
                kept.append((v, f))
        if not blobs:
            print(f"  {qf.stem:22s} khong tai duoc khung hinh nao")
            continue

        prompt = PROMPT.format(question=text.strip()[:1200], n=len(blobs))
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
