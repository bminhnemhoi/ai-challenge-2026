"""R7 — BẢO LÃNH suất kênh chữ trong 100 dòng KIS (VISIONE 5.0), đo đủ cổng.

Tiền đề đã đếm (10h40 04/09): 6/78 câu TUNE có video đúng trong top-5 kênh
chữ (BM25 lời thoại+OCR ghép) trong khi vision trượt top-5 — đúng cơ chế
"model phụ đôi khi nắm đáp án mà model chính mù" của VISIONE (vô địch VBS'24).

CƠ CHẾ (đăng ký trước, không tham số quét):
  * kênh chữ = BM25 trên tài liệu ghép (transcript + OCR đã quét);
  * lấy top-3 VIDEO kênh chữ chưa có mặt trong 17 dòng đầu sản xuất;
  * FRAME cho dòng chèn: (a) ứng viên pool tốt nhất của video đó nếu có;
    (b) không có → frame từ ĐOẠN THOẠI khớp nhất (timestamp × fps, chốt về
    keyframe gần nhất) — đúng cách kênh thoại đã cứu p1-19/p1-22;
  * chèn vào hạng 18/19/20; dòng 1..17 BẤT BIẾN (assert) ⇒ R@1, R@5 nguyên;
    các dòng sau tụt tối đa 3 hạng (rủi ro chặn cứng trong BUCKET liền kề).

CỔNG: TUNE trước; dương → TEST đọc MỘT lần; bootstrap theo câu; chấm y sản
xuất (cửa sổ {6,10,20} × 4 họ × 48 bốc, allocate_rows thật).

    python -u scripts/dem_bao_lanh_top5.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.do_bo_do_moi import GOC_HAT  # noqa: E402
from scripts.do_cap_thoi_gian_moi import canh_cua  # noqa: E402
from scripts.experiment_cap_thoi_gian import _plan  # noqa: E402
from scripts.experiment_phu_quet_luoi import (  # noqa: E402
    cac_lan_boc,
    cham_nhanh,
    ma_tran_dong,
)
from scripts.make_submission import DEFAULT_N_FLAT, allocate_rows  # noqa: E402
from src.core.submission import MAX_ROWS, Candidate  # noqa: E402
from src.core.transcripts import TranscriptIndex, tokenise  # noqa: E402

KHOA = 17
VI_TRI = (17, 18, 19)  # chỉ số 0-based -> hạng 18/19/20


def bm25_diem(docs_tok, q):
    qt = [t for t in tokenise(q) if len(t) > 1]
    if not qt:
        return {}
    df = {}
    for _v, ts in docs_tok.items():
        for t in set(qt) & set(ts):
            df[t] = df.get(t, 0) + 1
    N = max(1, len(docs_tok))
    avg = sum(len(ts) for ts in docs_tok.values()) / N
    ra = {}
    for v, ts in docs_tok.items():
        tf = {}
        for t in ts:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in set(qt):
            if t in tf and t in df:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(ts) / avg))
        if s > 0:
            ra[v] = s
    return ra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache-uv", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    args = ap.parse_args()

    data = Path(args.data)
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache_uv) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    gt_sub = [moi[i] for i in giu]

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    kf, fps_of = {}, {}
    for m in meta:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        pt = float(m.get("pts_time") or 0)
        if pt > 1:
            fps_of[m["video_id"]] = int(m["frame_idx"]) / pt
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf.items()}
    del meta

    ti = TranscriptIndex().load_dir(ROOT.parent / "transcripts_full",
                                    data / "captions")
    ocr_tok = {}
    for p in (data / "ocr").glob("L*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        mau = [str(x[1]) for _f, it in d.items() for x in (it or [])
               if isinstance(x, (list, tuple)) and len(x) > 1 and x[1]]
        if mau:
            ocr_tok[p.stem] = tokenise(" ".join(mau))
    ghep = {v: list(ti.docs.get(v, [])) + list(ocr_tok.get(v, []))
            for v in set(ocr_tok) | set(ti.docs)}
    print(f"kênh chữ: {len(ghep)} video có tài liệu")

    def frame_cho(v, q, pool_map):
        if v in pool_map:
            return pool_map[v]
        segs = ti.segments.get(v) or []
        qt = set(t for t in tokenise(q) if len(t) > 1)
        tot, at = 0, None
        for t0, txt in segs:
            h = len(qt & set(tokenise(txt)))
            if h > tot:
                tot, at = h, t0
        if at is None or v not in kf:
            return None
        f = int(at * fps_of.get(v, 25.0))
        a = kf[v]
        return int(a[int(np.argmin(np.abs(a - f)))])

    rows_nen, rows_bl = [], []
    for k, i in enumerate(giu):
        g = moi[i]
        cands = [Candidate(v, f, s, lf) for v, f, s, lf in uv[i]]
        rows = allocate_rows(cands, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
        rows_nen.append(rows)
        d_g = bm25_diem(ghep, g["kis_query_vi"])
        top_chu = [v for v, _s in sorted(d_g.items(), key=lambda kv: -kv[1])[:5]]
        co_dau = {v for v, _f in rows[:KHOA]}
        chen_v = [v for v in top_chu if v not in co_dau][:3]
        pool_map = {}
        for v, f, s, _lf in sorted(uv[i], key=lambda t: -float(t[2])):
            pool_map.setdefault(v, int(f))
        chen = []
        for v in chen_v:
            f = frame_cho(v, g["kis_query_vi"], pool_map)
            if f is not None:
                chen.append((v, int(f)))
        moi_r = list(rows[:KHOA])
        for x in chen:
            if x not in moi_r:
                moi_r.append(x)
        for r in rows[KHOA:]:
            if len(moi_r) >= MAX_ROWS:
                break
            if r not in moi_r:
                moi_r.append(r)
        assert moi_r[:KHOA] == rows[:KHOA], "17 dòng đầu phải bất biến"
        rows_bl.append(moi_r[:MAX_ROWS])

    ho = cac_lan_boc(GOC_HAT, args.seeds, args.draws, gt_sub, kf)

    def diem_cau(rows_of):
        mats = ma_tran_dong(rows_of, gt_sub)
        d = np.zeros(len(gt_sub))
        for qi in range(len(gt_sub)):
            d[qi] = float(np.mean([cham_nhanh([mats[qi]], [b[qi]], [6, 10, 20])
                                   for b in ho]))
        return d

    dn, db = diem_cau(rows_nen), diem_cau(rows_bl)

    rng = np.random.default_rng(20260903)
    nhom = {}
    for q in range(len(gt_sub)):
        nhom.setdefault(bool(canh_cua(gt_sub[q])), []).append(q)
    tune = set()
    for _c, idxs in sorted(nhom.items()):
        x = rng.permutation(len(idxs))
        tune |= {idxs[j] for j in x[: len(x) // 2]}
    test = [q for q in range(len(gt_sub)) if q not in tune]
    tune = sorted(tune)

    def bao(ten, chi_so):
        a, b = dn[chi_so], db[chi_so]
        ch = b.mean() - a.mean()
        rng2 = np.random.default_rng(4242)
        lay = rng2.integers(0, len(chi_so), size=(4000, len(chi_so)))
        dd = b[lay].mean(axis=1) - a[lay].mean(axis=1)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        print(f"\n=== {ten} (n={len(chi_so)}) ===")
        print(f"  nền {a.mean():.4f} -> bảo lãnh {b.mean():.4f} ({ch:+.4f}, "
              f"{100 * ch / a.mean() if a.mean() else 0:+.1f}%)")
        print(f"  bootstrap câu: KTC [{lo:+.4f}, {hi:+.4f}]; P(<=0)={(dd <= 0).mean():.1%}")
        return ch, lo

    ch_t, _ = bao("TUNE", tune)
    if ch_t <= 0:
        print("\n=== ÂM/hoà trên TUNE — DỪNG, không đọc TEST. Ghi cửa đóng. ===")
        return 0
    ch, lo = bao("TEST (đọc MỘT lần)", test)
    print("\n=== KẾT LUẬN ===")
    if ch > 0 and lo > -1e-9:
        print("  DƯƠNG, KTC không âm — đủ điều kiện ship (cờ --bao-lanh-chu,")
        print("  còn cổng smoke + pytest trước đông cứng).")
    else:
        print("  Chưa đủ bằng chứng ship hôm nay — ghi tín hiệu, cân hậu giải.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
