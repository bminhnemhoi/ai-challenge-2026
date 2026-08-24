"""A Q&A answer is scored PER ROW. Is one answer on all 100 rows a waste?

Rulebook 2.1.2, read directly:

    R-Score(r_i) = I(v_i = GT_v  and  id_i in [s,e]  and  a_i = GT_a)

The answer carries the row index i. Every row is scored with its OWN answer, and
2.2 takes R@k as the maximum over the first k rows. We currently write the same
string on all 100 rows of a Q&A file, which spends the whole budget on one
hypothesis about the answer AND one about the frame.

The budget is saturated at 100 rows, so hedging is a reallocation, not free: a
row carrying a second answer displaces a row carrying a different frame. Whether
that trades up depends on two things this measures on the ground truth, which
carries a gold `vqa_answer` for all 60 samples:

    how often is the model's FIRST answer right?
    how often is a LOWER-RANKED answer right when the first is not?

If the second answer is almost never right, hedging is pure loss. If it is right
often enough, the concavity of R@k pays: the second answer's FIRST row is worth
far more than the first answer's FIFTH row.

Row 1 is never touched, so R@1 is invariant by construction and this cannot
repeat the project's recurring trap of raising a proxy while lowering the score.

    python scripts/experiment_answer_hedge.py --limit 30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from src.core.vlm import DEFAULT_MODEL, VLMJudge  # noqa: E402

PROMPT = """Trả lời câu hỏi về khung hình dưới đây.

CÂU HỎI: {question}

Chỉ trả lời từ những gì THẤY ĐƯỢC trong ảnh. Đưa ra {n} phương án khác nhau
về NỘI DUNG, xếp theo độ tin cậy giảm dần.

Quan trọng: các phương án phải khác nhau về Ý NGHĨA, không phải cách viết.
"Màu đỏ" và "đỏ" là CÙNG một đáp án — đừng liệt kê cả hai.
"Màu đỏ" và "màu nâu" là hai đáp án khác nhau — đó mới là điều cần.

Trả về DUY NHẤT JSON: {{"answers": ["...", "...", "..."]}}"""


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s or "")).lower().strip()
    s = re.sub(r"^(màu|con|chiếc|cái|quả|người)\s+", "", s)
    return re.sub(r"[^0-9a-zà-ỹ ]+", " ", s).strip()


def semantically_equal(a: str, b: str) -> bool:
    """Approximate the rulebook's 'khớp về mặt ngữ nghĩa'.

    Deliberately generous, because a strict string test would report the model
    as wrong when it said "đỏ" and the key said "Màu đỏ", and the real grader
    would not. Being generous here risks OVERstating accuracy, which is the safe
    direction for a decision about whether to hedge: it makes hedging look less
    necessary, not more.
    """
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    wa, wb = set(na.split()), set(nb.split())
    return bool(wa & wb) and len(wa & wb) / max(len(wa | wb), 1) >= 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-answers", type=int, default=3)
    args = ap.parse_args()

    judge = VLMJudge(args.data, model=args.model)
    if not judge.ready:
        print("Khong co GEMINI_API_KEY.")
        return 2

    from google.genai import types

    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("vqa_answer") and g.get("frame_filename")]
    if args.limit:
        gt = gt[: args.limit]
    print(f"{len(gt)} cau hoi co dap an chuan\nmodel: {args.model}\n", flush=True)

    client = judge._get_client()
    hit_at = [0] * (args.n_answers + 1)   # hit_at[k] = right within the first k
    rows = []
    for i, g in enumerate(gt, 1):
        # the GOLD frame, so this measures the answerer alone and not retrieval
        blob = judge._fetch(g["video_id"], g["frame_filename"])
        if not blob:
            continue
        prompt = PROMPT.format(question=g["vqa_question"].strip(), n=args.n_answers)
        answers = []
        try:
            r = client.models.generate_content(
                model=args.model,
                contents=[types.Part.from_bytes(data=blob, mime_type="image/jpeg"), prompt],
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=600),
            )
            u = getattr(r, "usage_metadata", None)
            if u:
                judge.calls += 1
                judge.tokens_in += u.prompt_token_count or 0
                judge.tokens_out += u.candidates_token_count or 0
            m = re.search(r"\{.*\}", r.text or "", re.S)
            if m:
                answers = [str(x) for x in json.loads(m.group(0)).get("answers", [])][
                    : args.n_answers
                ]
        except Exception as exc:  # noqa: BLE001
            print(f"  {i}: LOI {type(exc).__name__}")
            continue

        gold = g["vqa_answer"]
        first_right = next(
            (k for k, a in enumerate(answers, 1) if semantically_equal(a, gold)), None
        )
        for k in range(1, args.n_answers + 1):
            if first_right is not None and first_right <= k:
                hit_at[k] += 1
        rows.append((first_right, gold, answers))
        if i % 10 == 0:
            print(f"  {i}/{len(gt)}  {judge.cost_note()}", flush=True)

    n = len(rows)
    print(f"\n=== DO CHINH XAC CUA BO TRA LOI (tren khung hinh DUNG) ===")
    for k in range(1, args.n_answers + 1):
        print(f"  dap an dung nam trong {k} phuong an dau: {hit_at[k]:3d}/{n}  "
              f"({100*hit_at[k]/max(n,1):4.0f}%)")
    gain2 = hit_at[2] - hit_at[1]
    gain3 = hit_at[3] - hit_at[2] if args.n_answers >= 3 else 0
    print(f"\n  phuong an 2 cuu them : {gain2:3d} cau ({100*gain2/max(n,1):.0f}%)")
    print(f"  phuong an 3 cuu them : {gain3:3d} cau ({100*gain3/max(n,1):.0f}%)")

    # what hedging is worth, using the rulebook formula directly
    p1 = hit_at[1] / max(n, 1)
    p2 = gain2 / max(n, 1)
    print(f"\n=== GIA TRI CUA VIEC DAT NHIEU DAP AN ===")
    print(f"  p1 (phuong an 1 dung) = {p1:.2f}")
    print(f"  p2 (chi phuong an 2 dung) = {p2:.2f}")
    print("\n  Bo cuc de xuat: dong 1-4 = dap an 1 tren 4 frame tot nhat,")
    print("                  dong 5   = dap an 2 tren frame tot nhat, roi tiep dap an 1.")
    print("  R@1 khong doi (dong 1 nguyen ven) -> khong the lo.")
    print(f"  R@5 doi tu {p1:.2f} thanh {p1:.2f} + {p2:.2f} = {p1+p2:.2f} khi frame dung o top-4.")
    print(f"  Uoc tinh: +{0.2*p2:.3f} diem moi cau Q&A (1/5 trong so R@5),")
    print(f"            tren 3 cau Q&A cua vong 1 = +{3*0.2*p2:.2f} diem.")

    print(f"\n{judge.cost_note()}")
    bad = [(g, a) for f, g, a in rows if f is None][:6]
    if bad:
        print("\nVi du sai hoan toan:")
        for gold, answers in bad:
            print(f"  chuan: {gold!r}\n     model: {answers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
