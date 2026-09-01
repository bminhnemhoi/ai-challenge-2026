"""Lưới bù trừ TRAKE: quả cầu L1 (hiện tại) vs TÍCH ĐẦY ĐỦ — và vì sao khác nhau.

`docs/GT_TRAKE.md` đo được: biết đúng video **và** đúng cả N mốc vẫn chỉ đạt
**0,5350**, không phải 1,0. Đó là trần của **cách phân bổ dòng**, không phải của
việc nhận dạng sự kiện. Script này tấn công đúng trần ấy.

## Quan sát cấu trúc

Điểm một dòng = `(1/N)·Σ_j I(t_j ∈ f_j ± w)`, và `R@k` là **max trên k dòng đầu**.
Nếu tập dòng là **tích Descartes đầy đủ** `S_1 × … × S_N` thì trong tập ấy **có
sẵn** dòng chọn offset tốt nhất cho *mọi* trục cùng lúc, nên

    max_dòng (1/N)·Σ_j I(...)  =  (1/N)·Σ_j I(∃ s ∈ S_j : t_j ∈ s ± w)

tức bài toán **phân rã thành N bài toán phủ độc lập theo từng trục**. Với tập
KHÔNG đầy đủ thì đẳng thức thành bất đẳng thức `≤` — mất mát nằm ở chỗ đó.

Allocator hiện tại (`allocate_trake_rows`) sắp theo **tổng độ dời**, tức lấy một
**quả cầu L1**. Đếm được: với N=3, budget 100, step 10 nó phủ offset {−2..2} trên
cả ba trục nhưng chỉ chứa **100/125** điểm tích — thiếu đúng 25 điểm góc, và
những điểm góc ấy là các dòng cần khi cả ba sự kiện đều lệch nhiều cùng chiều.

## Cái giá phải trả

Tích đầy đủ với 100 dòng và N=3 chỉ cho 4×5×5, tức một trục **hẹp hơn** (4 offset
thay vì 5). Nên đây là đánh đổi thật, không phải bữa trưa miễn phí: **đầy đủ hơn
nhưng ngắn hơn**. Chỉ phép đo mới nói được bên nào thắng.

Và đây chính là chỗ **lưới phi-đều** của `CHAN_DOAN_TRAKE.md` có nghĩa: khi đã ở
dạng tích, chọn `m_j` cho từng trục là một bài toán tối ưu rõ ràng
(`Π m_j ≤ budget`), và trục nào bất định hơn thì đáng được nhiều offset hơn.

    python -u scripts/do_luoi_trake.py
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.do_trake_bo_moi import GOC, boc_moc, cham  # noqa: E402
from src.core.submission import MAX_ROWS, allocate_trake_rows  # noqa: E402


def offsets_doi_xung(m: int):
    """m offset quanh 0, gần giữa trước: 0, −1, +1, −2, +2, …"""
    ra = [0]
    k = 1
    while len(ra) < m:
        ra.append(-k)
        if len(ra) < m:
            ra.append(k)
        k += 1
    return ra[:m]


def chia_m(n_truc: int, budget: int):
    """m_j cân bằng nhất với Π m_j ≤ budget (bản đều — bản phi-đều ở dưới)."""
    m = [1] * n_truc
    while True:
        j = int(np.argmin(m))
        thu = list(m)
        thu[j] += 1
        if int(np.prod(thu)) > budget:
            return m
        m = thu


def lattice_tich(video, frames, m_list, step, last):
    """Tích Descartes đầy đủ, sắp theo tổng độ dời (dòng 1 = tâm)."""
    tap = [[int(np.clip(f + step * o, 0, last if last else 10**9))
            for o in offsets_doi_xung(m)] for f, m in zip(frames, m_list)]
    dong = []
    for combo in product(*[list(enumerate(t)) for t in tap]):
        doi = sum(abs(offsets_doi_xung(m)[i]) for (i, _), m in zip(combo, m_list))
        dong.append((doi, [v for _, v in combo]))
    dong.sort(key=lambda t: (t[0], t[1]))
    return [(video, fr) for _, fr in dong]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--gt", default=str(ROOT / "data" / "gt_trake.json"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    args = ap.parse_args()

    data = Path(args.data)
    windows = [int(w) for w in args.windows.split(",")]
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    kf_list, last_of = {}, {}
    for m in meta:
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        last_of[m["video_id"]] = max(last_of.get(m["video_id"], 0), int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}
    del meta, kf_list

    n_truc = len(gt[0]["frames"])
    m_deu = chia_m(n_truc, MAX_ROWS)
    print(f"{len(gt)} muc | N = {n_truc} su kien | tich deu: {m_deu} "
          f"= {int(np.prod(m_deu))} dong")

    ho = []
    for s in range(args.seeds):
        bocs = [boc_moc(GOC + s * 1000 + t, gt, kf) for t in range(args.draws)]
        ho.append([[b[i] for b in bocs] for i in range(len(gt))])

    def diem(rows_of):
        return float(np.mean([cham(rows_of, gt, b, windows) for b in ho]))

    def tung_cau(rows_of):
        """Điểm từng mục — cho bootstrap theo câu."""
        ra = np.zeros(len(gt))
        for b in ho:
            for i in range(len(gt)):
                ra[i] += cham([rows_of[i]], [gt[i]], [b[i]], windows)
        return ra / len(ho)

    cau_hinh = {}
    for step in (5, 10, 15, 20):
        cau_hinh[f"L1 (hien tai) step={step}"] = [
            allocate_trake_rows(m["video_id"], m["frames"], budget=MAX_ROWS, step=step,
                                video_last_frame=last_of.get(m["video_id"]))
            for m in gt]
        cau_hinh[f"TICH day du   step={step}"] = [
            lattice_tich(m["video_id"], m["frames"], m_deu, step,
                         last_of.get(m["video_id"]))[:MAX_ROWS]
            for m in gt]

    print(f"\n{'cau hinh':<28}{'diem':>9}{'so L1 step=10':>15}")
    print("-" * 54)
    nen = None
    ket = {}
    for ten, rows in cau_hinh.items():
        d = diem(rows)
        ket[ten] = (d, rows)
        if ten == "L1 (hien tai) step=10":
            nen = d
    for ten in cau_hinh:
        d = ket[ten][0]
        print(f"{ten:<28}{d:>9.4f}{100*(d/nen-1):>+14.1f}%")

    chot = max(ket, key=lambda k: ket[k][0])
    print(f"\nTOT NHAT: {chot} -> {ket[chot][0]:.4f} ({100*(ket[chot][0]/nen-1):+.1f}%)")

    # bootstrap theo câu — n = 12, phải nói thẳng về lực thống kê
    a = tung_cau(ket["L1 (hien tai) step=10"][1])
    b = tung_cau(ket[chot][1])
    rng = np.random.default_rng(4242)
    lay = rng.integers(0, len(gt), size=(4000, len(gt)))
    d = b[lay].mean(axis=1) - a[lay].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\nbootstrap theo CAU (n={len(gt)}): chenh {b.mean()-a.mean():+.4f}; "
          f"KTC 95% [{lo:+.4f}, {hi:+.4f}]; P(<=0) = {(d<=0).mean():.1%}")
    print("\n=== KET LUAN ===")
    if lo > 0:
        print(f"  DUONG: KTC khong chua 0. Nhung n={len(gt)} rat nho — day la TIN HIEU,")
        print("  chua phai cong chot. Sinh them muc TRAKE roi do lai truoc khi ship.")
    else:
        print(f"  CHUA KET LUAN: KTC [{lo:+.4f}, {hi:+.4f}] chua 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
