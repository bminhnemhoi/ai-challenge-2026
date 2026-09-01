"""Chế độ căn chỉnh TRAKE — bộ căn chỉnh có đang giải bài toán khó hơn cần thiết?

Luật chấm TRAKE (mục 2.1.3): `(1/N)·Σ_j I(id_j ∈ [s_j, e_j])` — chấm **từng sự
kiện độc lập**, **KHÔNG** có ràng buộc thứ tự, **KHÔNG** đòi khoảng cách tối
thiểu giữa các mốc.

Nhưng `align_sequence` mặc định chạy `align_mode="ordered"` và `min_gap=2`: nó
tìm một chuỗi mốc **tăng dần** và **cách nhau ít nhất 2 keyframe**. Tức nó áp hai
ràng buộc mà bộ chấm **không hề đòi**.

Nếu mô tả sự kiện bị viết lệch thứ tự, hoặc hai sự kiện thật sự xảy ra sát nhau,
thì ràng buộc ấy **ép** bộ căn chỉnh bỏ mốc tốt nhất của một sự kiện để giữ tính
đơn điệu — mất điểm ở một trục để bảo toàn một tính chất không được chấm.

`docs/GT_TRAKE.md` đo được **67% khoảng cách TRAKE nằm ở định vị sự kiện**. Đây
là giả thuyết rẻ nhất cho phần đó, và nó thuần CPU.

Bốn chế độ có sẵn: `ordered` (hiện tại), `unordered`, `soft_order`, `strict_window`.
Cùng quét `min_gap`.

**Cột quyết định là cửa sổ ±6** (docs/GT_TRAKE.md §5.3).

    python -u scripts/do_che_do_can_chinh.py
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

CHE_DO = ("ordered", "unordered", "soft_order")
GAP = (0, 1, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--gt", default=str(ROOT / "data" / "gt_trake.json"))
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    args = ap.parse_args()

    data = Path(args.data)
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))

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

    # dung dau vao Y HET san xuat (luat ii: dung khac da cho 5/12 thay vi 10/12)
    de_list, ev_list, first_list = [], [], []
    for m in gt:
        de = (m.get("boi_canh", "") + "\n"
              + "\n".join(f"E{j+1}: {s}" for j, s in enumerate(m["su_kien"])))
        de_list.append(de)
        ev_list.append(split_events(de))
        first_list.append(bool(re.search(r"đầu tiên|lần đầu|first", de, re.IGNORECASE)))

    ho = []
    for s in range(args.seeds):
        bocs = [boc_moc(GOC + s * 1000 + t, gt, kf) for t in range(args.draws)]
        ho.append([[b[i] for b in bocs] for i in range(len(gt))])

    print(f"\n{'che do':<14}{'gap':>4}{'video dung':>12}{'±6 (QUYET DINH)':>18}"
          f"{'±10':>9}{'±20':>9}")
    print("-" * 68)
    ket = {}
    for mode in CHE_DO:
        for gap in GAP:
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
            print(f"{mode:<14}{gap:>4}{dung:>9}/{len(gt):<2}{d[6]:>18.4f}"
                  f"{d[10]:>9.4f}{d[20]:>9.4f}", flush=True)

    nen_key = ("ordered", 2)
    nen = ket[nen_key][0][6]
    chot = max(ket, key=lambda k: ket[k][0][6])
    print(f"\nNEN (san xuat: ordered, gap=2): {nen:.4f}")
    print(f"TOT NHAT o ±6: {chot[0]}/gap={chot[1]} -> {ket[chot][0][6]:.4f} "
          f"({100*(ket[chot][0][6]/nen-1) if nen else 0:+.1f}%)")

    if chot != nen_key:
        def tung_cau(rows):
            r = np.zeros(len(gt))
            for b in ho:
                for i in range(len(gt)):
                    r[i] += cham([rows[i]], [gt[i]], [b[i]], [6])
            return r / len(ho)

        a, b = tung_cau(ket[nen_key][1]), tung_cau(ket[chot][1])
        rng = np.random.default_rng(4242)
        lay = rng.integers(0, len(gt), size=(4000, len(gt)))
        d = b[lay].mean(axis=1) - a[lay].mean(axis=1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"bootstrap theo CAU (n={len(gt)}, ±6): chenh {b.mean()-a.mean():+.4f}; "
              f"KTC [{lo:+.4f}, {hi:+.4f}]; P(<=0) = {(d<=0).mean():.1%}")
        print(f"\nCANH BAO: n={len(gt)} rat nho. Day la TIN HIEU, chua phai cong chot.")
    else:
        print("Che do san xuat van tot nhat o cot quyet dinh — cua nay DONG.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
