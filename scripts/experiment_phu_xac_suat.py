"""Chia 100 dòng bằng PHỦ XÁC SUẤT thay vì đi theo chi phí tuyến tính.

Bộ phân bổ hiện tại đi theo `cost(i, d) = i + 0.5·d` — một quy tắc hợp lý nhưng
được chọn bằng cách quét tham số, không phải suy ra từ luật chấm. Ba lần quét
đã cho thấy nó là điểm tối ưu *trong họ đó*; điều đó không nói gì về các họ khác.

Suy thẳng từ luật chấm thì bài toán có tên: **phủ cực đại có trọng số**.

    Final = 1/5 · Σ R@k  với k ∈ {1, 5, 20, 50, 100},  R@k = max trên k dòng đầu.

Nên một dòng ĐÚNG ở hạng r đáng giá đúng bằng số ngưỡng k ≥ r, chia 5:

    hạng 1      → 5/5 = 1,00        hạng 21-50  → 2/5 = 0,40
    hạng 2-5    → 4/5 = 0,80        hạng 51-100 → 1/5 = 0,20
    hạng 6-20   → 3/5 = 0,60

Và vì R@k là *max*, một câu chỉ cần **một** dòng trúng: mọi dòng trúng thêm sau
đó đều vô giá trị. Đó chính là cấu trúc của bài toán phủ — không phải bài toán
xếp hạng.

Vậy: đặt một phân bố tiên nghiệm p(v, f) cho vị trí khoảnh khắc thật, rồi tham
lam chọn dòng nào phủ được nhiều phần khối lượng CHƯA ĐƯỢC PHỦ nhất. Trọng số
giảm dần theo hạng khiến phép tham lam theo "khối lượng mới" là đúng thứ tự.

Điều này khác hẳn `experiment_per_video_depth.py` (đã đo −15% đến −30%): cái đó
vẫn đi theo chi phí tuyến tính, chỉ đổi hệ số. Cái này bỏ hẳn khái niệm chi phí.

    python scripts/experiment_phu_xac_suat.py --limit 20
    python scripts/experiment_phu_xac_suat.py               # cả 60 câu
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import DEFAULT_DEPTH_COST, DEFAULT_N_FLAT, ranked_hits  # noqa: E402
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    RANK_THRESHOLDS,
    AllocationPlan,
    Candidate,
    allocate_hybrid_rows,
    final_score,
    r_score_kis,
)


def trong_so_hang(r: int) -> float:
    """Giá trị của một dòng ĐÚNG ở hạng r (1-based), theo đúng công thức chấm."""
    return sum(1 for k in RANK_THRESHOLDS if k >= r) / len(RANK_THRESHOLDS)


def phu_xac_suat(
    candidates,
    nhiet: float = 0.02,
    sigma: float = 20.0,
    nua_cua_so: int = 10,
    budget: int = MAX_ROWS,
    luoi: int = 5,
):
    """Chọn 100 dòng bằng phép tham lam cực đại hoá khối lượng xác suất chưa phủ.

    ``nhiet``      nhiệt độ softmax trên điểm tương đồng — quyết định tiên nghiệm
                   dồn vào vài ứng viên đầu hay trải rộng.
    ``sigma``      độ rộng nhân Gauss quanh mỗi keyframe. Đo trên ground truth:
                   khoảng cách từ khoảnh khắc thật tới keyframe gần nhất có trung
                   vị 14 frame, Q3 = 31 — nên ~20 là hợp lý.
    ``nua_cua_so`` nửa bề rộng cửa sổ chấm mà ta GIẢ ĐỊNH khi tính phủ.
    ``luoi``       bước rời rạc hoá trục thời gian, cho rẻ.
    """
    if not candidates:
        return []

    # ---- tiên nghiệm: mỗi ứng viên rải khối lượng quanh frame của nó ----------
    diem = np.array([c.score for c in candidates], dtype=np.float64)
    w = np.exp((diem - diem.max()) / max(nhiet, 1e-9))
    w /= w.sum()

    # gom theo video, dựng lưới thời gian cho từng video
    theo_video: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    for c, wi in zip(candidates, w):
        theo_video[c.video_id].append((int(c.frame_idx), float(wi), int(c.video_last_frame)))

    khoi: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for vid, items in theo_video.items():
        last = max(x[2] for x in items)
        lo = max(0, min(x[0] for x in items) - 4 * int(sigma))
        hi = min(last, max(x[0] for x in items) + 4 * int(sigma))
        truc = np.arange(lo, hi + 1, luoi, dtype=np.int64)
        if truc.size == 0:
            continue
        mass = np.zeros(truc.size, dtype=np.float64)
        for f, wi, _ in items:
            mass += wi * np.exp(-0.5 * ((truc - f) / sigma) ** 2)
        khoi[vid] = (truc, mass)

    # ---- tham lam: mỗi bước lấy dòng phủ được nhiều khối lượng mới nhất -------
    chua_phu = {v: m.copy() for v, (_t, m) in khoi.items()}
    nua = max(1, nua_cua_so // luoi)
    rows: list[tuple[str, int]] = []
    da_dung = set()

    while len(rows) < budget:
        tot_v, tot_i, tot_gia = None, -1, 0.0
        for vid, (truc, _m) in khoi.items():
            con = chua_phu[vid]
            if con.size == 0:
                continue
            # tổng trượt: khối lượng chưa phủ trong cửa sổ quanh mỗi vị trí
            tich = np.cumsum(np.concatenate(([0.0], con)))
            lo = np.maximum(0, np.arange(con.size) - nua)
            hi = np.minimum(con.size, np.arange(con.size) + nua + 1)
            gia = tich[hi] - tich[lo]
            j = int(np.argmax(gia))
            if gia[j] > tot_gia and (vid, int(truc[j])) not in da_dung:
                tot_v, tot_i, tot_gia = vid, j, float(gia[j])
        if tot_v is None or tot_gia <= 0:
            break
        truc, _ = khoi[tot_v]
        f = int(truc[tot_i])
        rows.append((tot_v, f))
        da_dung.add((tot_v, f))
        lo = max(0, tot_i - nua)
        chua_phu[tot_v][lo : tot_i + nua + 1] = 0.0

    return rows[:budget]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="số họ hạt giống độc lập, để có sai số thật")
    args = ap.parse_args()

    from src.core.kis_engine import KISEngine

    print("nạp chỉ mục ...", flush=True)
    eng = KISEngine(args.data).load()
    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("video_id") in eng.last_frame]
    if args.limit:
        gt = gt[: args.limit]

    print(f"{len(gt)} câu, truy xuất bằng ĐƯỜNG SẢN XUẤT (ranked_hits) ...", flush=True)
    cands_of = []
    for g in gt:
        hits = ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en"))
        cands_of.append([Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits])

    kf: dict = {}
    for m in eng.metadata:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    for v in kf:
        kf[v] = np.array(sorted(kf[v]))

    def draw(seed):
        rng = np.random.default_rng(seed)
        out = []
        for g in gt:
            a = kf[g["video_id"]]
            i = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
            lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
            hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
            out.append(int(rng.integers(lo, max(lo + 1, hi))))
        return out

    windows = [int(w) for w in args.windows.split(",")]

    def cham(rows_of, draws):
        per_w = []
        for half in windows:
            tot = 0.0
            for qi, g in enumerate(gt):
                for truth in draws:
                    span = (truth[qi] - half, truth[qi] + half)
                    tot += final_score([r_score_kis(v, f, g["video_id"], span) for v, f in rows_of[qi]])
            per_w.append(tot / (len(gt) * len(draws)))
        return sum(per_w) / len(per_w)

    plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)
    rows_nen = [allocate_hybrid_rows(c, n_flat=DEFAULT_N_FLAT, plan=plan)[:MAX_ROWS] for c in cands_of]

    print("\nchạy trên nhiều họ hạt giống độc lập để có sai số thật ...", flush=True)
    ho_nen, ho_moi = [], []
    cauhinh = [(0.02, 20.0, 10), (0.02, 30.0, 10), (0.05, 20.0, 10), (0.01, 20.0, 10)]
    rows_moi = {c: [phu_xac_suat(cs, nhiet=c[0], sigma=c[1], nua_cua_so=c[2]) for cs in cands_of]
                for c in cauhinh}

    for s in range(args.seeds):
        draws = [draw(30000 + s * 1000 + t) for t in range(args.draws)]
        ho_nen.append(cham(rows_nen, draws))
        ho_moi.append({c: cham(rows_moi[c], draws) for c in cauhinh})
        print(f"  họ {s+1}/{args.seeds}: nền {ho_nen[-1]:.4f}", flush=True)

    nen = float(np.mean(ho_nen))
    sd_nen = float(np.std(ho_nen))
    print(f"\nNỀN (bộ phân bổ đang nộp): {nen:.4f}  ±{sd_nen:.4f}")
    print(f"\n{'nhiệt':>7}{'sigma':>8}{'nửa cửa sổ':>12}{'điểm':>10}{'±':>9}{'so nền':>10}")
    print("-" * 58)
    tot = (nen, "nền")
    for c in cauhinh:
        vals = [h[c] for h in ho_moi]
        m, sd = float(np.mean(vals)), float(np.std(vals))
        print(f"{c[0]:7.3f}{c[1]:8.1f}{c[2]:12d}{m:10.4f}{sd:9.4f}{100*(m/nen-1):+9.1f}%")
        if m > tot[0]:
            tot = (m, f"nhiệt={c[0]} sigma={c[1]}")

    print(f"\nTốt nhất: {tot[1]} -> {tot[0]:.4f}  ({100*(tot[0]/nen-1):+.1f}%)")
    bien = max(sd_nen, 0.0005)
    if tot[1] != "nền" and (tot[0] - nen) < 2 * bien:
        print(f"CẢNH BÁO: chênh lệch {tot[0]-nen:.4f} chưa bằng 2 lần sai số ({bien:.4f}) — coi như hoà.")

    # bao nhiêu câu có video đúng xuất hiện trong 100 dòng
    def co_video(rows_of):
        return sum(1 for qi, g in enumerate(gt) if any(v == g["video_id"] for v, _f in rows_of[qi]))

    print(f"\nSố câu có VIDEO ĐÚNG trong 100 dòng:")
    print(f"  nền                : {co_video(rows_nen)}/{len(gt)}")
    for c in cauhinh:
        print(f"  phủ {c}: {co_video(rows_moi[c])}/{len(gt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
