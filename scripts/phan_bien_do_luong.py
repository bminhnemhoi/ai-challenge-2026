"""PHẢN BIỆN kỷ luật đo lường — hiệu chuẩn ĐỘ PHÂN GIẢI của bộ đo 132 mục.

Ba lane trả về ÂM và một lane trả về một con số DƯƠNG lớn, tất cả trên cùng một
bộ 132 mục. Câu hỏi mà không lane nào trả lời bằng số: **bộ đo này phân giải
được hiệu ứng nhỏ cỡ nào?** Cảnh báo "n = 66 là NHỎ" là lời, không phải số.

Script này KHÔNG đề xuất cải tiến nào và KHÔNG đọc TEST của ai. Nó chỉ đo chính
cái thước:

1. Điểm nền TỪNG CÂU trên cả 132 mục với MỘT họ hạt giống chung (mọi câu dùng
   đúng cùng một tập lần bốc), nên chênh lệch giữa hai nửa là chênh lệch của TẬP
   CÂU chứ không lẫn nhiễu bốc thăm.
2. Phép chia ``hai[0::2] + mot[0::2]`` mà HAI lane cùng dùng: độ lệch nền giữa
   hai nửa nằm ở đâu trong phân bố của 4000 phép chia phân tầng NGẪU NHIÊN?
   Nếu nó là một giá trị bình thường thì "hai nửa không hoán đổi được" không
   phải lỗi của phép chia — nó là độ phân giải của bộ đo.
3. MDE — hiệu ứng nhỏ nhất mà n = 66 / n = 33 bắt được ở lực 80%, α = 5% hai
   phía, tính từ ĐỘ LỆCH CHUẨN THẬT của hiệu số theo câu (mô phỏng: dịch một
   phần câu lên một lượng cố định rồi bootstrap theo câu).
4. Đối chiếu thành phần hai nửa theo shard / model sinh câu — phép chia phân
   tầng theo (một cảnh / hai cảnh) có vô tình lệch theo trục khác không.

    python -u scripts/phan_bien_do_luong.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.experiment_phu_quet_luoi import (  # noqa: E402
    cac_lan_boc,
    cham_nhanh,
    ma_tran_dong,
)
from scripts.make_submission import DEFAULT_DEPTH_COST, DEFAULT_N_FLAT, allocate_rows  # noqa: E402
from src.core.submission import MAX_ROWS, AllocationPlan, Candidate  # noqa: E402

GOC_KIEM = 555000  # tách hẳn khỏi mọi gốc đã dùng (61/62k, 77k, 81/82k, 310/320/330k)


def _plan() -> AllocationPlan:
    return AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)


def diem_tung_cau(rows_of, gt, ho, windows):
    mats = ma_tran_dong(rows_of, gt)
    ra = np.zeros(len(gt))
    for draws in ho:
        for q in range(len(gt)):
            ra[q] += cham_nhanh([mats[q]], [draws[q]], windows)
    return ra / len(ho)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--ho", type=int, default=6)
    ap.add_argument("--boc", type=int, default=48)
    ap.add_argument("--lan-chia", type=int, default=4000)
    args = ap.parse_args()

    windows = [int(w) for w in args.windows.split(",")]
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    sach = [moi[i] for i in giu]
    cands = [[Candidate(v, f, s, lf) for v, f, s, lf in uv[i]] for i in giu]
    hai = [i for i, g in enumerate(sach) if g.get("co_2_canh")]
    hai_set = set(hai)
    mot = [i for i in range(len(sach)) if i not in hai_set]
    print(f"bo SACH {len(sach)} | HAI canh {len(hai)} | MOT canh {len(mot)}")

    meta = json.loads((Path(args.data) / "metadata.json").read_text(encoding="utf-8"))
    kfl: dict = {}
    for m in meta:
        kfl.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kfl.items()}
    del meta, kfl

    rows = [allocate_rows(c, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS] for c in cands]
    ho = cac_lan_boc(GOC_KIEM, args.ho, args.boc, sach, kf)
    d = diem_tung_cau(rows, sach, ho, windows)
    print(f"\ndiem nen tung cau: {args.ho} ho x {args.boc} boc = {args.ho*args.boc} lan boc/cau, "
          f"CUNG mot tap boc cho moi cau")
    print(f"  toan bo  n={len(d):3d}  TB {d.mean():.4f}  trung vi {np.median(d):.4f}  "
          f"do lech chuan {d.std(ddof=1):.4f}  so cau = 0: {(d < 1e-9).sum()}")
    for ten, idx in (("HAI canh", hai), ("MOT canh", mot)):
        x = d[idx]
        print(f"  {ten} n={len(x):3d}  TB {x.mean():.4f}  trung vi {np.median(x):.4f}  "
              f"do lech chuan {x.std(ddof=1):.4f}  so cau = 0: {(x < 1e-9).sum()}")

    # ---------------- 2. phep chia hai lane cung dung -----------------------
    i_tune = sorted(hai[0::2] + mot[0::2])
    i_test = sorted(hai[1::2] + mot[1::2])
    g_that = d[i_test].mean() - d[i_tune].mean()
    print("\n" + "=" * 78)
    print("2. PHEP CHIA hai[0::2]+mot[0::2] — dung boi CA HAI lane phan-bo-sau va tin-hieu")
    print("=" * 78)
    print(f"  nen TUNE {d[i_tune].mean():.4f} | nen TEST {d[i_test].mean():.4f} "
          f"| chenh TEST-TUNE {g_that:+.4f} ({100*(d[i_test].mean()/d[i_tune].mean()-1):+.1f}%)")
    hai_t = [i for i in i_tune if i in hai_set]
    hai_s = [i for i in i_test if i in hai_set]
    g_hai = d[hai_s].mean() - d[hai_t].mean()
    print(f"  nhom HAI canh: TUNE {d[hai_t].mean():.4f} | TEST {d[hai_s].mean():.4f} "
          f"| chenh {g_hai:+.4f}")

    rng = np.random.default_rng(20260901)
    for ten, nhom_key in (
        ("phan tang theo (hai canh)", lambda i: (i in hai_set,)),
        ("phan tang theo (hai canh, shard)", lambda i: (i in hai_set, sach[i].get("shard"))),
    ):
        tang: dict = collections.defaultdict(list)
        for i in range(len(sach)):
            tang[nhom_key(i)].append(i)
        gaps = np.empty(args.lan_chia)
        gaps_h = np.empty(args.lan_chia)
        for k in range(args.lan_chia):
            A, B = [], []
            for _key, ids in tang.items():
                p = rng.permutation(ids)
                A.extend(p[: len(ids) // 2])
                B.extend(p[len(ids) // 2:])
            gaps[k] = d[B].mean() - d[A].mean()
            bh = [i for i in B if i in hai_set]
            ah = [i for i in A if i in hai_set]
            gaps_h[k] = d[bh].mean() - d[ah].mean()
        q = float((np.abs(gaps) >= abs(g_that)).mean())
        qh = float((np.abs(gaps_h) >= abs(g_hai)).mean())
        print(f"\n  {args.lan_chia} phep chia NGAU NHIEN, {ten}:")
        print(f"    |chenh| toan bo : do lech chuan {gaps.std(ddof=1):.4f}, "
              f"p95 {np.percentile(np.abs(gaps),95):.4f}, max {np.abs(gaps).max():.4f}")
        print(f"      -> chenh THAT {g_that:+.4f} nam o phan vi |.|: "
              f"{100*(1-q):.1f}%  (P(|ngau nhien| >= that) = {q:.1%})")
        print(f"    |chenh| HAI canh: do lech chuan {gaps_h.std(ddof=1):.4f}, "
              f"p95 {np.percentile(np.abs(gaps_h),95):.4f}")
        print(f"      -> chenh THAT {g_hai:+.4f}: P(|ngau nhien| >= that) = {qh:.1%}")

    # ---------------- 3. MDE ------------------------------------------------
    print("\n" + "=" * 78)
    print("3. MDE — hieu ung nho nhat ma phep do bat duoc (luc 80%, KTC 95% bootstrap cau)")
    print("=" * 78)
    print("  *** BAT BIEN TY LE: hieu so theo cau o cac lane la TAT DINH khi da co tap boc")
    print("  (nen va can thiep dung DUNG cung lan boc), nen nhan ca vector hieu so voi mot")
    print("  hang so duong khong doi dau cua bat ky phan vi bootstrap nao. => KTC bootstrap")
    print("  theo cau o day chi chung thuc DAU cua hieu ung, KHONG chung thuc DO LON.")
    print("  Luc thong ke phu thuoc HINH DANG (bao nhieu cau doi, bao nhieu dung chieu),")
    print("  KHONG phu thuoc hieu ung to hay nho. Bang duoi la luc theo hinh dang:\n")
    print(f"  {'n cau':>7}{'k doi':>7}{'u len':>7}{'d xuong':>9}"
          f"{'P(KTC 95% tach khoi 0)':>26}")
    print("  " + "-" * 56)
    for n in (33, 66, 132):
        for k, u in ((2, 2), (3, 3), (5, 5), (5, 4), (8, 7), (11, 10), (11, 8),
                     (20, 14), (33, 22)):
            if k > n:
                continue
            ok, nrep = 0, 600
            for _ in range(nrep):
                hs = np.zeros(n)
                cho = rng.choice(n, size=k, replace=False)
                hs[cho[:u]] = 1.0
                hs[cho[u:]] = -1.0
                lay = rng.integers(0, n, size=(600, n))
                ok += np.percentile(hs[lay].mean(axis=1), 2.5) > 0
            print(f"  {n:>7}{k:>7}{u:>7}{k-u:>9}{ok/nrep:>25.0%}")
    print()
    print("  Phep so o moi lane la GHEP CAP (cung cau, nen vs can thiep) va bootstrap")
    print("  cung ghep cap, nen do nhay KHONG phu thuoc muc diem — no phu thuoc HIEU SO")
    print("  theo cau trai rong the nao. Mo phong dung hinh dang hieu ung THAT quan sat")
    print("  duoc: k cau doi, ty le p di DUNG chieu, moi cau doi mot luong |delta| bang")
    print("  nhau. MDE = muc tang TONG nho nhat co >= 80% lan cho KTC 95% khong chua 0.")
    print(f"\n  {'nhom':<22}{'k cau doi':>10}{'p dung chieu':>14}{'MDE (% tong)':>14}")
    print("  " + "-" * 60)
    for ten, idx in (("TEST HAI canh (n=33)", hai_s),
                     ("TEST toan bo (n=66)", i_test),
                     ("ca 132 muc", list(range(len(sach))))):
        base = d[idx]
        n = len(base)
        for k, p_up in ((3, 1.0), (11, 10 / 11), (n // 3, 0.8), (n, 0.7), (n, 1.0)):
            found = None
            for r in (0.001, 0.01, 0.05, 0.12, 0.30, 0.65, 1.30, 2.0):
                tong_them = r * base.mean() * n  # tong luong diem phai them
                # so cau di dung chieu = round(k*p_up); nguoc chieu k - do
                n_up = max(1, int(round(k * p_up)))
                n_dn = k - n_up
                if n_up - n_dn <= 0:
                    continue
                delta = tong_them / (n_up - n_dn)
                ok, nrep = 0, 300
                for _ in range(nrep):
                    hs = np.zeros(n)
                    cho = rng.choice(n, size=k, replace=False)
                    hs[cho[:n_up]] = delta
                    hs[cho[n_up:]] = -delta
                    lay = rng.integers(0, n, size=(600, n))
                    dd = hs[lay].mean(axis=1)
                    ok += np.percentile(dd, 2.5) > 0
                if ok / nrep >= 0.80:
                    found = r
                    break
            txt = (f"{found:+.1%}" if found else "> +200%")
            print(f"  {ten:<22}{k:>10}{p_up:>14.2f}{txt:>14}")

    # ---------------- 4. thanh phan hai nua --------------------------------
    print("\n" + "=" * 78)
    print("4. THANH PHAN hai nua theo truc KHONG duoc phan tang")
    print("=" * 78)
    for truc in ("shard", "model"):
        ct = collections.Counter((sach[i].get(truc), "TUNE") for i in i_tune)
        ct.update((sach[i].get(truc), "TEST") for i in i_test)
        keys = sorted({k for k, _ in ct})
        print(f"\n  {truc}:")
        print(f"    {'gia tri':<26}{'TUNE':>6}{'TEST':>6}{'nen TUNE':>10}{'nen TEST':>10}")
        for k in keys:
            a = [i for i in i_tune if sach[i].get(truc) == k]
            b = [i for i in i_test if sach[i].get(truc) == k]
            print(f"    {str(k):<26}{len(a):>6}{len(b):>6}"
                  f"{(d[a].mean() if a else float('nan')):>10.4f}"
                  f"{(d[b].mean() if b else float('nan')):>10.4f}")
        # nen TB theo gia tri, gop ca hai nua
        print(f"    {'--- gop ca hai nua ---':<26}")
        for k in keys:
            ids = [i for i in range(len(sach)) if sach[i].get(truc) == k]
            nh = [i for i in ids if i in hai_set]
            print(f"    {str(k):<26}{len(ids):>6}{'':>6}{d[ids].mean():>10.4f}"
                  f"   (hai canh {len(nh)}, nen {d[nh].mean() if nh else float('nan'):.4f})")

    # ---------------- 4b. chenh sau khi HAU-PHAN-TANG ----------------------
    print("\n  4b. Chenh TEST-TUNE sau khi hau-phan-tang (chuan hoa ve phan bo gop)")
    for truc in ("shard", "model", "co_2_canh"):
        gia = {}
        for i in range(len(sach)):
            k = (i in hai_set) if truc == "co_2_canh" else sach[i].get(truc)
            gia.setdefault(k, []).append(i)
        tong, tw = 0.0, 0.0
        for k, ids in gia.items():
            a = [i for i in i_tune if i in set(ids)]
            b = [i for i in i_test if i in set(ids)]
            if not a or not b:
                continue
            w = len(ids) / len(sach)
            tong += w * (d[b].mean() - d[a].mean())
            tw += w
        print(f"    chuan theo {truc:<12}: chenh {tong/max(tw,1e-9):+.4f} "
              f"(tho {g_that:+.4f}, giai thich duoc "
              f"{100*(1-abs(tong/max(tw,1e-9))/abs(g_that)):.0f}%)")

    print("\n  4c. Vi sao [0::2] khong phai boc ngau nhien: chu ky trong THU TU FILE")
    seq = [sach[i].get("model", "?")[-14:] for i in range(len(sach))]
    print(f"    model theo vi tri (16 muc dau): {seq[:16]}")
    for m in sorted(set(seq)):
        vt = [i for i in range(len(sach)) if seq[i] == m]
        chan = sum(1 for i in vt if i % 2 == 0)
        print(f"    {m:<16} n={len(vt):>3}  o chi so CHAN {chan:>3} / LE {len(vt)-chan:>3}")

    # ---------------- 5. van ban canh B vs cau hoi -------------------------
    print("\n" + "=" * 78)
    print("5. VAN BAN canh_B co suy ra duoc TU CAU HOI khong? (nguon cua +80,5%)")
    print("=" * 78)
    print("  Duong san xuat lay canh_B tu gan_nhan_hai_canh.py — chi doc CAU HOI.")
    print("  Bo do lay canh_B tu buoc SINH, von dang NHIN ANH khi viet no.")
    print("  Do: ty le tu NOI DUNG cua canh_B khong xuat hien trong cau hoi.")

    def tu(s):
        import re
        return [w for w in re.findall(r"\w+", (s or "").lower(), flags=re.UNICODE)
                if len(w) > 2]

    dung = [
        "canh", "cang", "hinh", "khung", "mot", "cac", "nhung", "dang", "voi", "cua",
        "trong", "tren", "duoi", "cho", "khi", "nguoi", "được", "đang", "với", "của",
        "trong", "trên", "dưới", "một", "các", "những", "cảnh", "hình", "khung",
    ]
    ty = []
    for i in hai:
        g = sach[i]
        b = g.get("canh_B") or g.get("canh_B_vi") or ""
        q = set(tu(g.get("kis_query_vi", "")))
        tb = [w for w in tu(b) if w not in dung]
        if not tb:
            continue
        moi_tu = [w for w in tb if w not in q]
        ty.append(len(moi_tu) / len(tb))
    ty = np.array(ty)
    print(f"\n  n = {len(ty)} cau hai canh co van ban canh_B")
    print(f"  ty le TU trong canh_B KHONG co trong cau hoi: TB {ty.mean():.1%}, "
          f"trung vi {np.median(ty):.1%}, p90 {np.percentile(ty,90):.1%}")
    print(f"  so cau co >= 30% tu moi: {(ty >= 0.30).sum()}/{len(ty)}; "
          f">= 50%: {(ty >= 0.50).sum()}/{len(ty)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
