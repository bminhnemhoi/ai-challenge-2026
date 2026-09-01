"""Hedge VIDEO cho TRAKE — chia 100 dòng cho nhiều video, đáng hay không?

`docs/GT_TRAKE.md` đo được: 33% khoảng cách của TRAKE nằm ở **chọn sai video**
(10/12 mục đã đúng, nhưng 2 mục sai thì mất trắng vì luật chấm cho **0 tuyệt đối**
khi sai video).

Đường sản xuất hiện tại dồn **cả 100 dòng vào MỘT video** — hợp lý theo luật, vì
dòng ở video sai đáng 0 đồng. Nhưng chính vì luật khắc nghiệt như vậy mà câu hỏi
"chia bớt dòng cho video hạng 2" là một **canh bạc thật**, không phải tinh chỉnh:

  * mất: video đúng có ít dòng hơn ⇒ thang bù trừ thưa hơn ⇒ giảm ở 10/12 mục;
  * được: 2/12 mục đang 0 điểm có cơ hội khác 0.

Với `R@k = max trên tiền tố`, dòng của video hạng 2 chỉ có giá trị nếu **video
hạng 1 sai**. Nên đây là bài toán đánh đổi thuần tuý, và chỉ phép đo mới trả lời.

**Cột quyết định là cửa sổ ±6** (`docs/GT_TRAKE.md` §5.3): quy định BTC ghi cửa
sổ TRAKE "thường dưới 10 frame". Lấy trung bình trên {6,10,20} từng suýt đẩy một
thay đổi âm vào sản xuất.

    python -u scripts/do_hedge_video_trake.py
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

from scripts.do_trake_bo_moi import GOC, boc_moc, cham  # noqa: E402
from src.core.submission import MAX_ROWS, allocate_trake_rows  # noqa: E402

# ty le dong cho video hang 1 / 2 / 3
CHIA = {
    "khong hedge (100/0/0)": (100, 0, 0),
    "85/15/0": (85, 15, 0),
    "75/25/0": (75, 25, 0),
    "60/40/0": (60, 40, 0),
    "70/20/10": (70, 20, 10),
    "50/30/20": (50, 30, 20),
}


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

    print("nap chi muc + can chinh TRAKE top-3 ...", flush=True)
    from src.core.kis_engine import KISEngine
    from src.task3_trake import TRAKEEngine

    eng = KISEngine(args.data).load()
    trake = TRAKEEngine(engine=eng).load_index()

    # DUNG cach dung van ban su kien Y HET duong san xuat: dung de bai day du roi
    # cho split_events tach ra. Ban dau toi ghep "boi_canh + su_kien" cho tung su
    # kien va chi duoc 5/12 video dung, trong khi duong san xuat duoc 10/12 —
    # phep do khi ay khong so duoc voi san xuat.
    import re as _re

    from scripts.make_submission import split_events

    chuoi = []
    for i, m in enumerate(gt, 1):
        de = (m.get("boi_canh", "") + "\n"
              + "\n".join(f"E{j+1}: {s}" for j, s in enumerate(m["su_kien"])))
        events = split_events(de)
        first = bool(_re.search(r"đầu tiên|lần đầu|first", de, _re.IGNORECASE))
        try:
            res = trake.align_sequence(events, first_occurrence=first, top_k=3) or []
        except Exception as exc:  # noqa: BLE001
            print(f"  muc {i}: LOI {type(exc).__name__}")
            res = []
        chuoi.append(res)
        if i % 4 == 0:
            print(f"  {i}/{len(gt)}", flush=True)

    hang1_dung = sum(1 for m, r in zip(gt, chuoi) if r and r[0]["video_id"] == m["video_id"])
    trong_top3 = sum(1 for m, r in zip(gt, chuoi)
                     if any(x["video_id"] == m["video_id"] for x in r[:3]))
    print(f"\nvideo dung o hang 1: {hang1_dung}/{len(gt)} | trong top-3: {trong_top3}/{len(gt)}")
    if trong_top3 == hang1_dung:
        print("  => hedge KHONG THE giup: video dung khong bao gio nam o hang 2-3.")

    def rows_chia(ty_le):
        ra = []
        for res in chuoi:
            dong = []
            for k, phan in enumerate(ty_le):
                if phan <= 0 or k >= len(res):
                    continue
                v = res[k]["video_id"]
                dong += allocate_trake_rows(v, res[k]["sequence_frames"], budget=phan,
                                            step=args.step,
                                            video_last_frame=last_of.get(v))[:phan]
            ra.append(dong[:MAX_ROWS])
        return ra

    ho = []
    for s in range(args.seeds):
        bocs = [boc_moc(GOC + s * 1000 + t, gt, kf) for t in range(args.draws)]
        ho.append([[b[i] for b in bocs] for i in range(len(gt))])

    print(f"\n{'chia dong':<24}{'±6 (QUYET DINH)':>18}{'±10':>10}{'±20':>10}")
    print("-" * 64)
    ket = {}
    for ten, ty in CHIA.items():
        rows = rows_chia(ty)
        d = {w: float(np.mean([cham(rows, gt, b, [w]) for b in ho])) for w in (6, 10, 20)}
        ket[ten] = (d, rows)
        print(f"{ten:<24}{d[6]:>18.4f}{d[10]:>10.4f}{d[20]:>10.4f}")

    nen = ket["khong hedge (100/0/0)"][0][6]
    chot = max(ket, key=lambda k: ket[k][0][6])
    print(f"\nTOT NHAT o cot quyet dinh ±6: {chot} -> {ket[chot][0][6]:.4f} "
          f"({100*(ket[chot][0][6]/nen-1) if nen else 0:+.1f}%)")

    if chot != "khong hedge (100/0/0)":
        def tung_cau(rows):
            r = np.zeros(len(gt))
            for b in ho:
                for i in range(len(gt)):
                    r[i] += cham([rows[i]], [gt[i]], [b[i]], [6])
            return r / len(ho)

        a = tung_cau(ket["khong hedge (100/0/0)"][1])
        b = tung_cau(ket[chot][1])
        rng = np.random.default_rng(4242)
        lay = rng.integers(0, len(gt), size=(4000, len(gt)))
        d = b[lay].mean(axis=1) - a[lay].mean(axis=1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"bootstrap theo CAU (n={len(gt)}, cua so ±6): chenh {b.mean()-a.mean():+.4f}; "
              f"KTC [{lo:+.4f}, {hi:+.4f}]; P(<=0) = {(d<=0).mean():.1%}")
    else:
        print("Khong hedge van tot nhat o cot quyet dinh — cua nay DONG.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
