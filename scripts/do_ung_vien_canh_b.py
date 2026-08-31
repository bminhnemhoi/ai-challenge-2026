"""Truy xuất THÊM bằng cảnh B — chữa đúng cơ chế hỏng của câu hai cảnh.

Chẩn đoán trên bộ đo khớp phân bố (132 mục) nói rất rõ:

    câu MỘT cảnh: 58/65 có keyframe đáp án trong 400 ứng viên, hạng nội-video 2
    câu HAI cảnh: **35/65** — 30 câu keyframe đáp án KHÔNG HỀ được truy xuất

Video vẫn tìm đúng ngang nhau (53/66 vs 51/66), nhưng qua keyframe của **cảnh A**:
truy vấn nén cả hai cảnh vào một vector nên nó khớp cảnh mở đầu, còn keyframe của
cảnh B — chính là đáp án — không lọt nổi vào 400 ứng viên.

Đó là lý do lever ③ chỉ mua được +6,7%: nó **chấm lại** ứng viên sẵn có, mà thứ
cần cứu thì chưa bao giờ nằm trong danh sách. Vấn đề ở **khâu sinh ứng viên**,
không phải khâu xếp hạng.

Cách chữa ở đây không trộn điểm hai kênh — nó là **hợp hai lần truy xuất của
cùng một encoder**: lấy thêm top-M keyframe theo độ tương đồng với riêng cảnh B
rồi gộp vào danh sách ứng viên. Cùng model, cùng thang đo cosine, nên không có
công thức pha trộn nào phải chọn.

Câu KHÔNG qua cổng giữ nguyên 100% đường cũ — kiểm bằng assert, không bằng mắt.

    python -u scripts/do_ung_vien_canh_b.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.do_cap_thoi_gian_moi import GOC_TEST_MOI, GOC_TUNE_MOI, canh_cua  # noqa: E402
from scripts.experiment_cap_thoi_gian import KhoSims, _plan  # noqa: E402
from scripts.experiment_phu_quet_luoi import cac_lan_boc, cham_nhanh, ma_tran_dong  # noqa: E402
from scripts.make_submission import DEFAULT_N_FLAT, allocate_rows  # noqa: E402
from src.core.submission import MAX_ROWS, Candidate  # noqa: E402

LUOI_M = (25, 50, 100, 200)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--allocator", default="coverage")
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--tune-seeds", type=int, default=3)
    ap.add_argument("--tune-draws", type=int, default=32)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--test-draws", type=int, default=48)
    args = ap.parse_args()

    data = Path(args.data)
    windows = [int(w) for w in args.windows.split(",")]
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    sach = [moi[i] for i in giu]
    cands0 = [[Candidate(v, f, s, lf) for v, f, s, lf in uv[i]] for i in giu]

    nhan = []
    for m in sach:
        c = canh_cua(m)
        nhan.append({"co_2_canh": bool(c), "canh_B_vi": c[2] if c else "",
                     "canh_B_en": c[3] if c else ""})
    bat = [i for i, d in enumerate(nhan) if d["co_2_canh"]]
    print(f"bo sach {len(sach)} muc, cong BAT {len(bat)}")

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    kf_list, last_of, vid_of, frm_of = {}, {}, [], []
    for m in meta:
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        last_of[m["video_id"]] = max(last_of.get(m["video_id"], 0), int(m["frame_idx"]))
        vid_of.append(m["video_id"])
        frm_of.append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}
    vid_of = np.array(vid_of)
    frm_of = np.array(frm_of)
    del meta, kf_list

    print("nap chi muc (mot lan, qua KhoSims) ...", flush=True)
    kho = KhoSims(args.data, False)
    simsB = {i: kho.lay(nhan[i]["canh_B_vi"], nhan[i]["canh_B_en"]) for i in bat}

    rows_nen = [allocate_rows(c, args.allocator, DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
                for c in cands0]

    def them_B(M):
        ra = []
        for i, c0 in enumerate(cands0):
            if i not in simsB:
                ra.append(c0)
                continue
            s = simsB[i]
            top = np.argpartition(-s, M)[:M]
            top = top[np.argsort(-s[top])]
            co = {(c.video_id, int(c.frame_idx)) for c in c0}
            them = []
            for j in top:
                key = (str(vid_of[j]), int(frm_of[j]))
                if key in co:
                    continue
                co.add(key)
                them.append(Candidate(key[0], key[1], float(s[j]),
                                      last_of.get(key[0], key[1] + 1000)))
            ra.append(list(c0) + them)
        return ra

    tat = [i for i in range(len(sach)) if i not in bat]
    i_tune = sorted(bat[0::2] + tat[0::2])
    i_test = sorted(bat[1::2] + tat[1::2])
    ho_tune = cac_lan_boc(GOC_TUNE_MOI, args.tune_seeds, args.tune_draws,
                          [sach[i] for i in i_tune], kf)
    ho_test = cac_lan_boc(GOC_TEST_MOI, args.test_seeds, args.test_draws,
                          [sach[i] for i in i_test], kf)
    print(f"TUNE {len(i_tune)} ({sum(1 for i in i_tune if i in bat)} qua cong) | "
          f"TEST {len(i_test)} ({sum(1 for i in i_test if i in bat)} qua cong)")

    def cham(idx, rows_all, ho):
        gt_s = [sach[i] for i in idx]
        mats = ma_tran_dong([rows_all[i] for i in idx], gt_s)
        return [cham_nhanh(mats, d, windows) for d in ho]

    print(f"\n{'M':>5}{'diem TUNE':>12}{'+-':>8}{'so nen':>9}")
    print("-" * 34)
    nen_t = cham(i_tune, rows_nen, ho_tune)
    m_nen = float(np.mean(nen_t))
    print(f"{'nen':>5}{m_nen:>12.4f}{np.std(nen_t):>8.4f}")
    ket, rows_cache = {}, {}
    for M in LUOI_M:
        cc = them_B(M)
        rows = [allocate_rows(c, args.allocator, DEFAULT_N_FLAT, _plan())[:MAX_ROWS] for c in cc]
        for i in tat:
            assert rows[i] == rows_nen[i], f"muc {i} doi dong du cong TAT (M={M})"
        rows_cache[M] = rows
        d = cham(i_tune, rows, ho_tune)
        ket[M] = float(np.mean(d))
        print(f"{M:>5}{ket[M]:>12.4f}{np.std(d):>8.4f}{100*(ket[M]/m_nen-1):>+8.1f}%", flush=True)

    chot = max(ket, key=lambda k: ket[k])
    print(f"\nCHOT tren TUNE: M={chot} ({100*(ket[chot]/m_nen-1):+.1f}%)")
    print(f"bat bien: {len(tat)} muc cong TAT ra dong giong het nen (assert, moi M).")

    print("\n=== TEST (doc DUNG MOT LAN) ===")
    a_l = cham(i_test, rows_nen, ho_test)
    b_l = cham(i_test, rows_cache[chot], ho_test)
    a, b = float(np.mean(a_l)), float(np.mean(b_l))
    print(f"  nen : {a:.4f} +-{np.std(a_l):.4f}")
    print(f"  chot: {b:.4f} +-{np.std(b_l):.4f}   ({100*(b/a-1):+.1f}%)")

    gt_t = [sach[i] for i in i_test]
    mn = ma_tran_dong([rows_nen[i] for i in i_test], gt_t)
    mc = ma_tran_dong([rows_cache[chot][i] for i in i_test], gt_t)

    def tung_cau(mats):
        ra = np.zeros(len(gt_t))
        for draws in ho_test:
            for q in range(len(gt_t)):
                ra[q] += cham_nhanh([mats[q]], [draws[q]], windows)
        return ra / len(ho_test)

    dn, dc = tung_cau(mn), tung_cau(mc)
    rng = np.random.default_rng(4242)
    lay = rng.integers(0, len(gt_t), size=(4000, len(gt_t)))
    delta = dc[lay].mean(axis=1) - dn[lay].mean(axis=1)
    lo, hi = np.percentile(delta, [2.5, 97.5])
    print("\n=== bootstrap theo CAU ===")
    print(f"  toan bo TEST ({len(gt_t)} muc): chenh {dc.mean()-dn.mean():+.4f}; "
          f"KTC 95%: [{lo:+.4f}, {hi:+.4f}]; P(<=0) = {(delta<=0).mean():.1%}")

    # Nhom BI TAC DONG moi la dai luong quyet dinh: ta chi ap dung cho cau qua
    # cong, nen 33 cau cong TAT chi lam loang thanh sai so. Chung ra dong y het
    # nen (assert o tren), tuc dong gop delta = 0 dung bang may, khong phai xap xi.
    vt = [k for k, i in enumerate(i_test) if i in bat]
    if vt:
        dn_g, dc_g = dn[vt], dc[vt]
        lay_g = rng.integers(0, len(vt), size=(4000, len(vt)))
        dg = dc_g[lay_g].mean(axis=1) - dn_g[lay_g].mean(axis=1)
        lo_g, hi_g = np.percentile(dg, [2.5, 97.5])
        print(f"  chi cau QUA CONG ({len(vt)} muc): {dn_g.mean():.4f} -> {dc_g.mean():.4f}"
              f" = {100*(dc_g.mean()/dn_g.mean()-1):+.1f}%")
        print(f"    KTC 95%: [{lo_g:+.4f}, {hi_g:+.4f}]; P(<=0) = {(dg<=0).mean():.1%}")
    print("\n=== KET LUAN ===")
    up = 100 * (b / a - 1) if a else 0.0
    if lo > 0 and up >= 5.0:
        print(f"  GIU DUOC: TEST {up:+.1f}%, khoang tin cay khong chua 0.")
    elif lo > 0:
        print(f"  DUONG nhung yeu: {up:+.1f}%.")
    else:
        print(f"  CHUA KET LUAN: {up:+.1f}% nhung khoang tin cay chua 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
