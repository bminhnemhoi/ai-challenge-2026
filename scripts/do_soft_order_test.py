"""Phép đo TEST tiền-đăng-ký cho soft_order — đọc TEST đúng một lần.

Bối cảnh (docs/GT_TRAKE.md §7): trên 12 mục TUNE, `soft_order/gap=2` cho +3,7%
so với sản xuất (`ordered/gap=2`) ở cột quyết định ±6, dương ở cả ba cửa sổ,
không bao giờ âm trong mẫu — nhưng n=12 quá nhỏ để chốt. Giả thuyết được CHỐT
TRƯỚC trên TUNE; script này đọc bộ TEST (12 mục mới, `data/gt_trake_test_moi.json`)
đúng một lần cho DUY NHẤT cặp so sánh đã đăng ký:

    ordered/gap=2  (sản xuất)   vs   soft_order/gap=2  (ứng viên)

Không quét, không chọn trên TEST. Báo cáo TỪNG cửa sổ {6,10,20}, cột quyết định
±6, bootstrap theo CÂU, kèm chênh lệch từng mục để thấy hiệu ứng nằm ở đâu.

    python -u scripts/do_soft_order_test.py --gt data/gt_trake_test_moi.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.do_trake_bo_moi import GOC, boc_moc, cham  # noqa: E402
from src.core.submission import MAX_ROWS, allocate_trake_rows  # noqa: E402

CAP = (("ordered", 2), ("soft_order", 2))  # cặp duy nhất, chốt trước trên TUNE


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--gt", default=str(ROOT / "data" / "gt_trake_test_moi.json"))
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    args = ap.parse_args()

    data = Path(args.data)
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    print(f"{len(gt)} muc tu {args.gt}")

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    kf_list, last_of = {}, {}
    for m in meta:
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        last_of[m["video_id"]] = max(last_of.get(m["video_id"], 0), int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}
    del meta, kf_list

    print("nap chi muc ...", flush=True)
    from scripts.make_submission import split_events
    from src.core.kis_engine import KISEngine
    from src.task3_trake import TRAKEEngine

    eng = KISEngine(args.data).load()
    trake = TRAKEEngine(engine=eng).load_index()

    # đầu vào Y HỆT sản xuất (luật: dựng khác đã từng cho 5/12 thay vì 10/12)
    ev_list, first_list = [], []
    for m in gt:
        de = (m.get("boi_canh", "") + "\n"
              + "\n".join(f"E{j+1}: {s}" for j, s in enumerate(m["su_kien"])))
        ev_list.append(split_events(de))
        first_list.append(bool(re.search(r"đầu tiên|lần đầu|first", de, re.IGNORECASE)))

    ho = []
    for s in range(args.seeds):
        bocs = [boc_moc(GOC + s * 1000 + t, gt, kf) for t in range(args.draws)]
        ho.append([[b[i] for b in bocs] for i in range(len(gt))])

    ket = {}
    for mode, gap in CAP:
        rows_of, dung = [], 0
        for m, ev, fi in zip(gt, ev_list, first_list):
            try:
                res = trake.align_sequence(ev, first_occurrence=fi, top_k=1,
                                           min_gap=gap, align_mode=mode) or []
            except Exception as exc:  # noqa: BLE001
                print(f"  {mode}/{gap}: LOI {type(exc).__name__}: {str(exc)[:50]}")
                res = []
            if not res:
                rows_of.append([])
                continue
            v = res[0]["video_id"]
            dung += v == m["video_id"]
            rows_of.append(allocate_trake_rows(
                v, res[0]["sequence_frames"], budget=MAX_ROWS, step=args.step,
                video_last_frame=last_of.get(v)))
        d = {w: float(np.mean([cham(rows_of, gt, b, [w]) for b in ho]))
             for w in (6, 10, 20)}
        ket[(mode, gap)] = (d, rows_of, dung)

    print(f"\n{'che do':<16}{'video dung':>12}{'±6 (QUYET DINH)':>18}{'±10':>9}{'±20':>9}")
    print("-" * 66)
    for (mode, gap), (d, _, dung) in ket.items():
        print(f"{mode}/gap={gap:<3}{dung:>9}/{len(gt):<2}{d[6]:>18.4f}"
              f"{d[10]:>9.4f}{d[20]:>9.4f}")

    a_key, b_key = CAP
    for w in (6, 10, 20):
        da, db = ket[a_key][0][w], ket[b_key][0][w]
        print(f"  ±{w}: chenh {db-da:+.4f} ({100*(db/da-1) if da else 0:+.1f}%)")

    # điểm từng mục ở ±6 + bootstrap theo CÂU
    def tung_cau(rows):
        r = np.zeros(len(gt))
        for b in ho:
            for i in range(len(gt)):
                r[i] += cham([rows[i]], [gt[i]], [b[i]], [6])
        return r / len(ho)

    a, b = tung_cau(ket[a_key][1]), tung_cau(ket[b_key][1])
    print("\nchenh tung muc o ±6 (soft_order - ordered):")
    n_doi = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if abs(y - x) > 1e-12:
            n_doi += 1
            print(f"  muc {i+1:>2}: {x:.4f} -> {y:.4f} ({y-x:+.4f})")
    print(f"  {n_doi}/{len(gt)} muc doi diem; con lai giong het (bat bien).")

    rng = np.random.default_rng(4242)
    lay = rng.integers(0, len(gt), size=(4000, len(gt)))
    d = b[lay].mean(axis=1) - a[lay].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\nbootstrap theo CAU (n={len(gt)}, ±6): chenh {b.mean()-a.mean():+.4f}; "
          f"KTC [{lo:+.4f}, {hi:+.4f}]; P(<=0) = {(d<=0).mean():.1%}; "
          f"P(<0) = {(d<0).mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
