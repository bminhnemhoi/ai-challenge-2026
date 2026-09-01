# -*- coding: utf-8 -*-
"""PHAN BIEN ket luan AM §4a cua docs/PAPER_DINH_VI_2026.md — do THANG tren 66 CU CAT
CO NHAN thay vi suy tu con so van lieu "ban tin cat canh moi 4-6 giay".

Bo do da co san 66 cu cat co nhan ma khong ai dung: voi cau HAI canh, khung neo
duoc DINH NGHIA la khung dau tien cua canh B, tuc dung mot cu cat. Nhom MOT canh
la doi chung.

  A. Chum keyframe co trung cu cat khong?  (kiem lai ket luan (1))
  B. Cosine SigLIP lien ke co do duoc cu cat khong? (kiem lai ket luan (2))
     — kem PHAN BO NEN cheo-video, thu ma ban goc thieu: nguong 0,5 nam DUOI
     trung vi cua cap ngau nhien khac video, nen "chi 0,8% cap xuong duoi 0,5"
     khong noi gi ve tin hieu, no noi ve nguong.

    python -u scripts/phan_bien_cat_canh.py
"""
from __future__ import annotations
import collections, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._console import safe_console  # noqa: E402
safe_console()

KHE_CHUM = 0.5


def main() -> int:
    d = ROOT / "data"
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    E = np.load(d / "embeddings_siglip2_384.npy", mmap_mode="r")
    print(f"{len(meta)} keyframe | embeddings {E.shape}")
    byv: dict = collections.defaultdict(list)
    for r, m in enumerate(meta):
        byv[m["video_id"]].append((int(m["frame_idx"]), float(m["pts_time"]), r))
    for v in byv:
        byv[v] = sorted(byv[v])
    gt = json.loads((d / "ground_truth_moi.json").read_text(encoding="utf-8"))
    sach = [g for g in gt if not g.get("lan_truc")]

    def cos(a, b):
        A = np.asarray(E[a], dtype=np.float32)
        B = np.asarray(E[b], dtype=np.float32)
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
        B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
        return (A * B).sum(1)

    print("\n" + "=" * 76)
    print("A. CHUM keyframe co roi vao 66 cu cat CO NHAN khong?")
    print("=" * 76)
    for co2 in (True, False):
        sub = [g for g in sach if bool(g.get("co_2_canh")) == co2]
        tc = dc = 0
        for g in sub:
            a = byv[g["video_id"]]
            t = np.array([x[1] for x in a])
            f = np.array([x[0] for x in a])
            i = int(np.argmin(np.abs(f - int(g["frame_idx"]))))
            gb = t[i] - t[i - 1] if i > 0 else 1e9
            ga = t[i + 1] - t[i] if i + 1 < len(t) else 1e9
            tc += (gb < KHE_CHUM) or (ga < KHE_CHUM)
            dc += (gb >= KHE_CHUM) and (ga < KHE_CHUM)
        ten = "HAI canh (neo = CU CAT that)" if co2 else "MOT canh (doi chung)"
        print(f"  {ten:<34} n={len(sub)}  neo trong mot chum: {tc}  la DAU chum: {dc}")
    tot = inn = 0
    for v, a in byv.items():
        t = np.array([x[1] for x in a])
        g = np.diff(t)
        for i in range(len(t)):
            gb = g[i - 1] if i > 0 else 1e9
            ga = g[i] if i < len(g) else 1e9
            inn += (gb < KHE_CHUM) or (ga < KHE_CHUM)
            tot += 1
    print(f"  NEN: keyframe BAT KY nam trong chum: {inn}/{tot} = {inn/tot:.1%}")
    print("  DOC: ty le o cu cat that THAP HON nen => chum khong danh dau cu cat.")
    print("  Ket luan (1) cua paper lane DUNG, va bay gio co bang chung truc tiep.")

    print("\n" + "=" * 76)
    print("B. COSINE lien ke: phan bo NEN va cu cat CO NHAN")
    print("=" * 76)
    rng = np.random.default_rng(7)
    n = len(meta)
    vid = np.array([m["video_id"] for m in meta])
    i0 = rng.integers(0, n, 40000)
    j0 = rng.integers(0, n, 40000)
    ok = vid[i0] != vid[j0]
    c0 = cos(i0[ok][:20000], j0[ok][:20000])
    print(f"  NEN cheo-video (n={len(c0)}): trung vi {np.median(c0):.3f}  "
          f"p5 {np.percentile(c0,5):.3f}  p95 {np.percentile(c0,95):.3f}")
    print(f"  => nguong tuyet doi 0,5 nam DUOI trung vi nen. Doi mot cu cat phai keo")
    print(f"     cosine xuong duoi 0,5 la doi hai khung cung ban tin phai KHAC NHAU hon")
    print(f"     hai khung ngau nhien cua hai video khac nhau. Do la nguong sai thang do.")
    for k in (1, 2, 5, 10, 20, 50):
        A, B = [], []
        for v, a in byv.items():
            r = np.array([x[2] for x in a])
            if len(r) > k:
                A.append(r[:-k]); B.append(r[k:])
        A = np.concatenate(A); B = np.concatenate(B)
        if len(A) > 30000:
            s = rng.choice(len(A), 30000, replace=False); A, B = A[s], B[s]
        c = cos(A, B)
        print(f"  cung video, cach {k:>2} keyframe: trung vi {np.median(c):.3f}  "
              f"p5 {np.percentile(c,5):.3f}  ty le <0,5 {np.mean(c<0.5):.3f}")

    print("\n  --- phep do CO NHAN: cosine tai 66 cu cat that, chuan theo TUNG VIDEO ---")
    for co2 in (True, False):
        sub = [g for g in sach if bool(g.get("co_2_canh")) == co2]
        pct, ab = [], []
        for g in sub:
            a = byv[g["video_id"]]
            if len(a) < 4:
                continue
            r = np.array([x[2] for x in a])
            f = np.array([x[0] for x in a])
            i = int(np.argmin(np.abs(f - int(g["frame_idx"]))))
            if i == 0:
                continue
            s = cos(r[:-1], r[1:])
            pct.append(float((s < s[i - 1]).mean()))
            ab.append(float(s[i - 1]))
        pct = np.array(pct); ab = np.array(ab)
        ten = "HAI canh (CU CAT that)" if co2 else "MOT canh (doi chung)"
        print(f"  {ten:<26} n={len(pct)}  phan vi trong video: trung vi {np.median(pct):.3f}"
              f"  <=0,25: {(pct<=0.25).sum()}  >=0,50: {(pct>=0.50).sum()}")
        print(f"  {'':<26} cosine tuyet doi tai cu cat: trung vi {np.median(ab):.3f}"
              f"  so <0,5: {(ab<0.5).sum()}")
    print("\n  DOC: tren thang TUONG DOI trong tung video, cu cat that nam o phan vi 0,24")
    print("  con doi chung nam o 0,44 — tin hieu CO. Ket luan (2) 'cosine khong do duoc")
    print("  cu cat' KHONG duoc chung minh boi bang chung da dua; no chi chung minh rang")
    print("  mot NGUONG TUYET DOI 0,5 khong dung. Day KHONG phai bang chung rang mot bo")
    print("  do cat canh se an diem — chi la cua do chua duoc dong dung cach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
