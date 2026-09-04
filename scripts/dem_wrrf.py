"""R6 — WRRF gộp chữ (OCR+lời thoại) với hình để TÌM VIDEO, bậc đếm TUNE.

`docs/NGHIEN_CUU_DA_NGUON_0309.md` R6 (MMMORRF SIGIR'25: trên kho video TIN TỨC
đa ngữ, chỉ mục ghép OCR+ASR nDCG 0,551 đè vision-only 0,375; WRRF k=0 0,586).

KHUNG ĐỌC SỐ — ghi trước khi chạy:
  * KHÔNG SHIP HÔM NAY dù số đẹp: OCR mới phủ ~43% kho; fusion nửa kho ở giờ
    thi thiên vị video-được-phủ. Phép đếm này là MÁY PHÁT HIỆN NO-GO + xếp độ
    ưu tiên hậu giải.
  * Thiên vị đã biết: 143/143 video ĐÚNG của bộ đo đã có OCR, distractor chỉ
    ~43% → kênh chữ được nịnh. Nếu fusion vẫn KHÔNG thắng vision-only dưới
    thiên vị này ⇒ đóng chắc tay. Nếu thắng ⇒ "GO có điều kiện, đo lại khi
    OCR phủ đủ 873".
  * Chỉ chạy nửa TUNE (seed 20260903) — TEST giữ trinh cho phép đo hậu giải.

Cấu hình (đăng ký): vision / ASR / OCR / GHÉP(OCR+ASR) / RRF k=60 / RRF k=0 /
WRRF k=0 với α_d = clip(số_ký_tự_OCR(d)/2000, 0,2, 0,8).
Ngưỡng (từ doc): GHÉP ≥ ASR-đơn +3 điểm R@10; WRRF ≥ max(RRF-thường, nguồn
đơn tốt nhất) +2 điểm R@10.

    python -u scripts/dem_wrrf.py
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

from scripts.do_cap_thoi_gian_moi import canh_cua  # noqa: E402
from src.core.transcripts import TranscriptIndex, tokenise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache-uv", default=str(ROOT / "data" / "cache_bo_do_moi"))
    args = ap.parse_args()

    data = Path(args.data)
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache_uv) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]

    # nửa TUNE — cùng seed với mọi phép đếm R
    rng = np.random.default_rng(20260903)
    nhom = {}
    for k, i in enumerate(giu):
        nhom.setdefault(bool(canh_cua(moi[i])), []).append(k)
    tune = set()
    for _c, idxs in sorted(nhom.items()):
        x = rng.permutation(len(idxs))
        tune |= {idxs[j] for j in x[: len(x) // 2]}

    # --- tài liệu chữ: ASR (TranscriptIndex) + OCR
    ti = TranscriptIndex().load_dir(ROOT.parent / "transcripts_full",
                                    data / "captions")
    ocr_tok, ocr_len = {}, {}
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
        txt = " ".join(mau)
        if txt.strip():
            ocr_tok[p.stem] = tokenise(txt)
            ocr_len[p.stem] = len(txt)
    print(f"tài liệu: ASR {ti.n_videos} video | OCR {len(ocr_tok)} video")

    def bm25_diem(docs_tok, query):
        qt = [t for t in tokenise(query) if len(t) > 1]
        if not qt:
            return {}
        df = {}
        for _v, ts in docs_tok.items():
            st = set(ts)
            for t in set(qt) & st:
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
                if t not in tf or t not in df:
                    continue
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                f = tf[t]
                s += idf * f * 2.2 / (f + 1.2 * (1 - 0.75 + 0.75 * len(ts) / avg))
            if s > 0:
                ra[v] = s
        return ra

    ghep_tok = {}
    for v in set(ocr_tok) | set(ti.docs):
        ghep_tok[v] = list(ti.docs.get(v, [])) + list(ocr_tok.get(v, []))

    def hang_video(diem_map, gt_v):
        thu = sorted(diem_map.items(), key=lambda kv: -kv[1])
        for r, (v, _s) in enumerate(thu, 1):
            if v == gt_v:
                return r
        return 10**6

    ket = {c: [] for c in ("vision", "asr", "ocr", "ghep", "rrf60", "rrf0", "wrrf")}
    for k, i in enumerate(giu):
        if k not in tune:
            continue
        g = moi[i]
        q = g["kis_query_vi"]
        gt_v = g["video_id"]
        # vision: điểm max theo video từ pool sản xuất
        vis = {}
        for v, _f, s, _lf in uv[i]:
            vis[v] = max(vis.get(v, -1e9), float(s))
        d_asr = ti.score_videos(q)
        d_ocr = bm25_diem(ocr_tok, q)
        d_ghep = bm25_diem(ghep_tok, q)
        r_vis = hang_video(vis, gt_v)
        r_text = hang_video(d_ghep, gt_v)

        def rrf(k0):
            xh_t = {v: r for r, (v, _s) in enumerate(
                sorted(d_ghep.items(), key=lambda kv: -kv[1]), 1)}
            xh_v = {v: r for r, (v, _s) in enumerate(
                sorted(vis.items(), key=lambda kv: -kv[1]), 1)}
            tat = {}
            for v in set(xh_t) | set(xh_v):
                tat[v] = (1.0 / (k0 + xh_t.get(v, 10**6))
                          + 1.0 / (k0 + xh_v.get(v, 10**6)))
            return tat

        def wrrf():
            xh_t = {v: r for r, (v, _s) in enumerate(
                sorted(d_ghep.items(), key=lambda kv: -kv[1]), 1)}
            xh_v = {v: r for r, (v, _s) in enumerate(
                sorted(vis.items(), key=lambda kv: -kv[1]), 1)}
            tat = {}
            for v in set(xh_t) | set(xh_v):
                al = min(0.8, max(0.2, ocr_len.get(v, 0) / 2000.0))
                tat[v] = (al / (1 + xh_t.get(v, 10**6))
                          + (1 - al) / (1 + xh_v.get(v, 10**6)))
            return tat

        ket["vision"].append(r_vis)
        ket["asr"].append(hang_video(d_asr, gt_v))
        ket["ocr"].append(hang_video(d_ocr, gt_v))
        ket["ghep"].append(r_text)
        ket["rrf60"].append(hang_video(rrf(60), gt_v))
        ket["rrf0"].append(hang_video(rrf(1), gt_v))
        ket["wrrf"].append(hang_video(wrrf(), gt_v))

    n = len(ket["vision"])
    print(f"\nnửa TUNE: {n} câu | R@k-video (%)")
    print(f"{'cấu hình':<10}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'R@100':>8}{'trung vị hạng':>15}")
    for c, rs in ket.items():
        a = np.array(rs)
        print(f"{c:<10}{100*(a<=1).mean():>6.0f}%{100*(a<=5).mean():>6.0f}%"
              f"{100*(a<=10).mean():>6.0f}%{100*(a<=100).mean():>7.0f}%"
              f"{np.median(np.minimum(a, 999)):>15.0f}")

    a10 = {c: 100 * (np.array(rs) <= 10).mean() for c, rs in ket.items()}
    print("\n=== NGƯỠNG (đăng ký, nhớ thiên vị phủ nịnh kênh chữ) ===")
    print(f"  GHÉP ≥ ASR + 3 điểm R@10? {a10['ghep']:.0f} vs {a10['asr']:.0f} -> "
          + ("ĐẠT" if a10["ghep"] >= a10["asr"] + 3 else "KHÔNG"))
    nen_tot = max(a10["vision"], a10["ghep"], a10["asr"], a10["ocr"], a10["rrf60"])
    print(f"  WRRF ≥ max(RRF60, nguồn tốt nhất) + 2? {a10['wrrf']:.0f} vs {nen_tot:.0f} -> "
          + ("ĐẠT" if a10["wrrf"] >= nen_tot + 2 else "KHÔNG"))
    print(f"  fusion tốt nhất vs vision-only: "
          f"{max(a10['rrf0'], a10['rrf60'], a10['wrrf']):.0f} vs {a10['vision']:.0f}")
    print("\nNhắc: KHÔNG ship hôm nay bất kể kết quả — quyết định là độ ưu tiên"
          " HẬU GIẢI; số dương phải đo lại khi OCR phủ đủ 873 video, trên TEST trinh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
