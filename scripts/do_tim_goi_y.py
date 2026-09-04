"""Cheat-sheet vòng thi — dò BM25 lời thoại + OCR cho TỪNG câu đề, một lệnh.

Đấu pháp #1 đã kiểm chứng (vòng 2: 5/9 đáp án Q&A lộ nguyên văn trong thoại)
nhưng đang làm TAY từng câu — mất 10–15 phút và dễ sót. Script này chạy ngay
sau make_submission: với mỗi câu, dò (a) chỉ mục lời thoại (`TranscriptIndex`),
(b) chữ OCR đã quét (toàn kho đang phủ dần + ứng viên vòng), in top video kèm
dòng khớp + link YouTube cho người soát bấm thẳng.

CHỈ LÀ GỢI Ý cho người — không đụng CSV, không đụng đáp án tự động.

    python scripts/do_tim_goi_y.py --queries round3/queries
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

from scripts.make_submission import detect_task, read_query_text, split_qa  # noqa: E402
from src.core.transcripts import TranscriptIndex, tokenise  # noqa: E402


def nap_ocr_docs(data: Path):
    """Mỗi video một 'tài liệu' chữ OCR (những video đã quét)."""
    docs = {}
    for p in (data / "ocr").glob("L*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        mau = []
        for _f, items in d.items():
            for x in items or []:
                t = x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else str(x)
                if t:
                    mau.append(str(t))
        if mau:
            docs[p.stem] = " ".join(mau)
    return docs


def diem_ocr(docs, query: str, top: int = 5):
    """BM25 giản lược trên tài liệu OCR (idf xấp xỉ log(N/df))."""
    import math
    qt = [t for t in tokenise(query) if len(t) > 1]
    if not qt or not docs:
        return []
    df = {}
    tok_docs = {}
    for v, txt in docs.items():
        ts = set(tokenise(txt))
        tok_docs[v] = ts
        for t in set(qt) & ts:
            df[t] = df.get(t, 0) + 1
    N = len(docs)
    ra = []
    for v, ts in tok_docs.items():
        s = sum(math.log(1 + N / df[t]) for t in set(qt) if t in ts and t in df)
        if s > 0:
            ra.append((s, v))
    ra.sort(reverse=True)
    return ra[:top]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--ra", default=None, help="ghi thêm ra file text")
    args = ap.parse_args()

    data = Path(args.data)
    ti = TranscriptIndex().load_dir(ROOT.parent / "transcripts_full",
                                    data / "captions")
    import zipfile
    urls = {}
    z = data / "media-info-aic25-b1.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith(".json"):
                    urls[Path(n).stem] = json.loads(zf.read(n)).get("watch_url", "")
    ocr_docs = nap_ocr_docs(data)
    print(f"chỉ mục: lời thoại {ti.n_videos} video | OCR {len(ocr_docs)} video\n")

    dong_ra = []

    def ghi(s=""):
        print(s)
        dong_ra.append(s)

    qfiles = sorted(Path(args.queries).glob("*.txt"))
    qfiles = [q for q in qfiles if not q.name.lower().endswith((".en.txt", ".vi.txt"))]
    for qf in qfiles:
        text = read_query_text(qf) or ""
        task = detect_task(qf.name)
        ghi("=" * 78)
        ghi(f"{qf.name}  [{task}]")
        ghi(f"  đề: {text[:160]}")
        cau_do = text
        if task == "qa":
            bc, ch = split_qa(text)
            cau_do = f"{bc} {ch}"
        # (a) lời thoại
        try:
            sc = ti.score_videos(cau_do)
            top_t = sorted(sc.items(), key=lambda kv: -kv[1])[: args.top]
        except Exception as exc:  # noqa: BLE001
            top_t = []
            ghi(f"  (lời thoại lỗi: {type(exc).__name__})")
        if top_t:
            ghi("  — LỜI THOẠI (bấm link để soát):")
            for v, s in top_t:
                u = urls.get(v, "")
                yt = ("  " + u.replace("https://youtube.com/watch?v=",
                                       "https://youtu.be/")) if u else ""
                ghi(f"    {v}  (bm25 {s:.1f}){yt}")
        # (b) OCR
        top_o = diem_ocr(ocr_docs, cau_do, args.top)
        if top_o:
            ghi("  — CHỮ TRÊN HÌNH (OCR):")
            for s, v in top_o:
                ghi(f"    {v}  (ocr {s:.1f})")
        # từ khoá đáng dò tay thêm
        tu = [t for t in tokenise(cau_do) if len(t) >= 4][:8]
        ghi(f"  — thử dò tay: python scripts/search_transcripts.py \"{' '.join(tu[:4])}\" -n 5")
    if args.ra:
        Path(args.ra).write_text("\n".join(dong_ra), encoding="utf-8")
        ghi(f"\nđã ghi: {args.ra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
