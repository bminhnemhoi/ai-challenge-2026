"""Chấm nhánh TRAKE trên bộ đo mới — lần đầu tiên TRAKE có bộ đo.

TRAKE chiếm 8,5% đề thật nhưng có 0 câu trong mọi bộ đo cũ, nên nhánh này chưa
bao giờ được đo. `data/gt_trake.json` (sinh bởi `sinh_gt_trake.py`) mở khoá nó.

Luật chấm (mục 2.1.3) có hai đặc điểm mà không nhánh nào khác có, và cả hai đều
đổi cách nên tiêu 100 dòng:

  * **0 tuyệt đối nếu sai video** — nên mọi dòng phải dồn vào MỘT video;
  * **điểm TỪNG PHẦN, KHÔNG ràng buộc thứ tự**: `(1/N)·Σ_j I(id_j ∈ [s_j,e_j])`.
    Một câu 4 sự kiện đoán trúng 2 vẫn được nửa điểm. Rộng lượng hơn KIS nhiều.

Ba mức đo, mỗi mức tách một nguồn mất điểm:

    NEN          đường sản xuất thật (build_trake_rows → allocate_trake_rows)
    ORACLE-MOC   giữ nguyên video của nền, nhưng các mốc = mốc THẬT
                 -> đo riêng phần mất do định vị sự kiện
    ORACLE-VIDEO video đúng + mốc thật -> trần tuyệt đối

Khoảnh khắc thật bốc như bên KIS: đều trong khe giữa hai keyframe (không snap),
cửa sổ chấm {6, 10, 20} — BTC ghi cửa sổ TRAKE "thường dưới 10 frame" nên dải
này đã bao trùm.

    python -u scripts/do_trake_bo_moi.py
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

from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    allocate_trake_rows,
    r_score_trake,
)

GOC = 991000


def boc_moc(seed, muc_list, kf):
    """Khoảnh khắc thật của TỪNG sự kiện, bốc đều trong khe keyframe của nó."""
    rng = np.random.default_rng(seed)
    ra = []
    for m in muc_list:
        a = kf[m["video_id"]]
        moc = []
        for f in m["frames"]:
            i = int(np.argmin(np.abs(a - int(f))))
            lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
            hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
            moc.append(int(rng.integers(lo, max(lo + 1, hi))))
        ra.append(moc)
    return ra


def cham(rows_of, muc_list, boc_list, windows):
    """Final Score trung bình (cửa sổ × mục × lần bốc), dùng r_score_trake THẬT."""
    from src.core.submission import final_score

    tong, n = 0.0, 0
    for w in windows:
        for rows, m, mocs in zip(rows_of, muc_list, boc_list):
            for moc in mocs:
                spans = [(x - w, x + w) for x in moc]
                tong += final_score(
                    [r_score_trake(v, fr, m["video_id"], spans) for v, fr in rows])
                n += 1
    return tong / max(1, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--gt", default=str(ROOT / "data" / "gt_trake.json"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()

    data = Path(args.data)
    windows = [int(w) for w in args.windows.split(",")]
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    if not gt:
        print("bo do TRAKE rong")
        return 2

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    kf_list, last_of = {}, {}
    for m in meta:
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        last_of[m["video_id"]] = max(last_of.get(m["video_id"], 0), int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}
    del meta, kf_list

    n_sk = [len(m["frames"]) for m in gt]
    print(f"{len(gt)} muc TRAKE | so su kien: {min(n_sk)}-{max(n_sk)} "
          f"(trung vi {int(np.median(n_sk))}) | {len({m['video_id'] for m in gt})} video")

    # --- ba muc do -----------------------------------------------------------
    # ORACLE-VIDEO: video dung + moc that
    r_oracle = [allocate_trake_rows(m["video_id"], m["frames"], budget=MAX_ROWS,
                                    step=args.step,
                                    video_last_frame=last_of.get(m["video_id"]))
                for m in gt]

    # NEN: duong san xuat that. Nap chi muc mot lan.
    print("nap chi muc + chay duong san xuat TRAKE ...", flush=True)
    from scripts.make_submission import build_trake_rows
    from src.core.kis_engine import KISEngine

    eng = KISEngine(args.data).load()
    r_nen, video_nen = [], []
    for i, m in enumerate(gt, 1):
        de = (m.get("boi_canh", "") + "\n"
              + "\n".join(f"E{j+1}: {s}" for j, s in enumerate(m["su_kien"])))
        try:
            rows = build_trake_rows(eng, de, args.step)
            rows = [(v, list(f)) for v, f in rows][:MAX_ROWS]
        except Exception as exc:  # noqa: BLE001
            print(f"  muc {i}: LOI {type(exc).__name__}: {str(exc)[:60]}")
            rows = []
        r_nen.append(rows)
        video_nen.append(rows[0][0] if rows else "?")
        if i % 5 == 0:
            print(f"  {i}/{len(gt)}", flush=True)

    # ORACLE-MOC: giu VIDEO cua nen, moc = moc that
    r_moc = []
    for m, v in zip(gt, video_nen):
        r_moc.append(allocate_trake_rows(v, m["frames"], budget=MAX_ROWS, step=args.step,
                                         video_last_frame=last_of.get(v, 90_000))
                     if v != "?" else [])

    # ho[s][i] = danh sach `draws` bo moc cua muc i, o ho hat giong s
    ho = []
    for s in range(args.seeds):
        bocs = [boc_moc(GOC + s * 1000 + t, gt, kf) for t in range(args.draws)]
        ho.append([[b[i] for b in bocs] for i in range(len(gt))])

    def diem(rows_of):
        return float(np.mean([cham(rows_of, gt, b, windows) for b in ho]))

    d_nen, d_moc, d_or = diem(r_nen), diem(r_moc), diem(r_oracle)
    dung_video = sum(1 for m, v in zip(gt, video_nen) if v == m["video_id"])

    print(f"\n{'muc do':<34}{'diem':>9}{'so nen':>10}")
    print("-" * 55)
    print(f"{'NEN (duong san xuat)':<34}{d_nen:>9.4f}{'—':>10}")
    print(f"{'ORACLE-MOC (video cua nen, moc that)':<34}{d_moc:>9.4f}"
          f"{100*(d_moc/d_nen-1) if d_nen else float('inf'):>+9.1f}%")
    print(f"{'ORACLE-VIDEO (video + moc that)':<34}{d_or:>9.4f}"
          f"{100*(d_or/d_nen-1) if d_nen else float('inf'):>+9.1f}%")
    print(f"\nvideo dong 1 DUNG: {dung_video}/{len(gt)} muc")
    print("\n=== DOC KET QUA ===")
    print("  ORACLE-MOC - NEN   = phan mat do DINH VI SU KIEN (video da dung/sai giu nguyen)")
    print("  ORACLE-VIDEO - ORACLE-MOC = phan mat do CHON SAI VIDEO")
    print("  Luat cham TRAKE cho diem TUNG PHAN va KHONG rang buoc thu tu, nen mot cau")
    print("  4 su kien trung 2 van duoc nua diem — rong luong hon KIS nhieu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
