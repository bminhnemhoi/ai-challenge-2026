"""R5 — DP TRAKE gộp (DANTE-λ ≡ Drop-DTW) + k-best 100 chuỗi, bậc đếm trên TUNE.

`docs/NGHIEN_CUU_DA_NGUON_0309.md` R5. Cơ chế:
  * S (N sự kiện × T keyframe của video ĐÚNG) — sims SigLIP-2 từ cache KhoSims
    (đã nhúng bằng `nap_sims_trake.py`, chạy lại 0 đồng);
  * DP đơn điệu nghiêm t_1 < … < t_N với phạt tuyến tính λ·(chênh CHỈ SỐ lưới)
    (DANTE, đội Outstanding TRAKE AIC 2025): DP[i,t] = S[i,t] + max_{τ<t}(DP[i−1,τ] − λ(t−τ));
  * k-best: mỗi trạng thái giữ top-B đường; 100 chuỗi PHÂN BIỆT tốt nhất = 100
    dòng nộp (luật TRAKE không phạt thứ tự nộp ⇒ coverage@100 là cận trên ăn được).

Ba phép so trên TUNE (12 mục cũ; 12 mục mới để dành đọc-một-lần nếu TUNE qua):
  A. argmax độc lập + thang bù trừ sản xuất (`allocate_trake_rows`, step 10)
     — đứng thay đường hiện hành trên CÙNG tín hiệu S;
  B. DP top-1 + thang bù trừ — tách riêng "DP có giúp ĐỊNH VỊ không";
  C. DP k-best-100 chuỗi — tách riêng "trải 100 dòng bằng chuỗi thay vì thang".

Chế độ phạt drop-cost (Drop-DTW percentile) CHƯA cài ở bậc đếm này — ghi rõ để
không ai tưởng harness đã đủ hai chế độ; cài nếu λ-mode cho thấy sự sống.

NGƯỠNG TIỀN-ĐĂNG-KÝ (doc R5): điểm TRAKE ≥ +5 điểm % so A ở cột ±6 VÀ
coverage/điểm k-best ≥ +5 điểm % so thang; bootstrap theo câu; n=12 là TÍN HIỆU.

    python -u scripts/do_dp_trake.py
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
from scripts.experiment_cap_thoi_gian import KhoSims  # noqa: E402
from scripts.make_submission import split_events  # noqa: E402
from src.core.submission import MAX_ROWS, allocate_trake_rows  # noqa: E402

LAM = (0.001, 0.003, 0.01)
B_KBEST = 8


def dp_kbest(S: np.ndarray, lam: float, k_out: int = 100):
    """DP đơn điệu nghiêm với phạt λ·gap; trả (chuỗi tốt nhất, k chuỗi phân biệt)."""
    N, T = S.shape
    # trạng thái [i][t] = list (điểm, đường) top-B, đường là tuple chỉ số lưới
    cur = [[(float(S[0, t]), (t,))] for t in range(T)]
    for i in range(1, N):
        moi = [[] for _ in range(T)]
        # chạy τ tăng dần, giữ "bể" ứng viên tốt nhất đã trừ phạt tới τ
        be = []  # list (điểm − λ·(t−τ) quy về gốc τ=0: điểm + λ·τ, đường)
        for t in range(T):
            if t >= 1:
                for d, p in cur[t - 1]:
                    be.append((d + lam * (t - 1), p))
                if len(be) > 4 * B_KBEST:
                    be.sort(key=lambda x: -x[0])
                    del be[4 * B_KBEST:]
            if not be:
                continue
            be.sort(key=lambda x: -x[0])
            them = []
            thay = set()
            for d, p in be:
                if p[-1] in thay:
                    continue
                thay.add(p[-1])
                them.append((float(S[i, t]) + d - lam * t, p + (t,)))
                if len(them) >= B_KBEST:
                    break
            moi[t] = them
        cur = moi
    tat = [x for o in cur for x in o]
    tat.sort(key=lambda x: -x[0])
    phan_biet, thay = [], set()
    for d, p in tat:
        if p in thay:
            continue
        thay.add(p)
        phan_biet.append(p)
        if len(phan_biet) >= k_out:
            break
    return (phan_biet[0] if phan_biet else None), phan_biet


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--gt", default=str(ROOT / "data" / "gt_trake.json"))
    ap.add_argument("--tune", type=int, default=12, help="số mục đầu = TUNE cũ")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))[: args.tune]

    meta = json.loads((Path(args.data) / "metadata.json").read_text(encoding="utf-8"))
    hang_of, kf_list, last_of = {}, {}, {}
    for r, m in enumerate(meta):
        hang_of[(m["video_id"], int(m["frame_idx"]))] = r
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        last_of[m["video_id"]] = max(last_of.get(m["video_id"], 0), int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}
    del meta, kf_list

    kho = KhoSims(args.data, False)

    # S cho từng mục (sims đã cache — không nạp model)
    S_of, luoi_of = [], []
    for m in gt:
        de = (m.get("boi_canh", "") + "\n"
              + "\n".join(f"E{j+1}: {s}" for j, s in enumerate(m["su_kien"])))
        evs = split_events(de)
        a = kf[m["video_id"]]
        rows = np.array([hang_of[(m["video_id"], int(f))] for f in a])
        S = np.stack([kho.lay(ev, "")[rows].astype(np.float64) for ev in evs])
        S_of.append(S)
        luoi_of.append(a)

    ho = []
    for s in range(args.seeds):
        bocs = [boc_moc(GOC + s * 1000 + t, gt, kf) for t in range(args.draws)]
        ho.append([[b[i] for b in bocs] for i in range(len(gt))])

    def diem(rows_of):
        return {w: float(np.mean([cham(rows_of, gt, b, [w]) for b in ho]))
                for w in (6, 10, 20)}

    def diem_cau(rows_of, w=6):
        d = np.zeros(len(gt))
        for b in ho:
            for i in range(len(gt)):
                d[i] += cham([rows_of[i]], [gt[i]], [b[i]], [w])
        return d / len(ho)

    # A — argmax độc lập + thang sản xuất
    rows_A, chain_A = [], []
    for m, S, a in zip(gt, S_of, luoi_of):
        c = [int(a[int(np.argmax(S[j]))]) for j in range(S.shape[0])]
        chain_A.append(c)
        rows_A.append(allocate_trake_rows(m["video_id"], c, budget=MAX_ROWS,
                                          step=args.step,
                                          video_last_frame=last_of.get(m["video_id"])))
    dA = diem(rows_A)
    print(f"{'cau hinh':<26}{'±6 (QUYET DINH)':>16}{'±10':>9}{'±20':>9}")
    print("-" * 62)
    print(f"{'A argmax + thang':<26}{dA[6]:>16.4f}{dA[10]:>9.4f}{dA[20]:>9.4f}")

    ket = {}
    for lam in LAM:
        rows_B, rows_C = [], []
        for m, S, a in zip(gt, S_of, luoi_of):
            top1, kbest = dp_kbest(S, lam, k_out=MAX_ROWS)
            c = [int(a[t]) for t in top1]
            rows_B.append(allocate_trake_rows(m["video_id"], c, budget=MAX_ROWS,
                                              step=args.step,
                                              video_last_frame=last_of.get(m["video_id"])))
            rows_C.append([(m["video_id"], [int(a[t]) for t in p]) for p in kbest])
        dB, dC = diem(rows_B), diem(rows_C)
        ket[lam] = (dB, dC, rows_B, rows_C)
        print(f"{'B DP(λ=' + str(lam) + ') + thang':<26}{dB[6]:>16.4f}"
              f"{dB[10]:>9.4f}{dB[20]:>9.4f}")
        print(f"{'C DP(λ=' + str(lam) + ') k-best100':<26}{dC[6]:>16.4f}"
              f"{dC[10]:>9.4f}{dC[20]:>9.4f}", flush=True)

    tot = max(ket, key=lambda k: max(ket[k][0][6], ket[k][1][6]))
    dB, dC, rows_B, rows_C = ket[tot]
    print(f"\nλ tốt nhất ở ±6: {tot}")
    for ten, rows, d in (("B (DP định vị)", rows_B, dB), ("C (k-best trải)", rows_C, dC)):
        ch = d[6] - dA[6]
        a_c, b_c = diem_cau(rows_A), diem_cau(rows)
        rng = np.random.default_rng(4242)
        lay = rng.integers(0, len(gt), size=(4000, len(gt)))
        dd = b_c[lay].mean(axis=1) - a_c[lay].mean(axis=1)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        print(f"  {ten}: {dA[6]:.4f} -> {d[6]:.4f} ({ch:+.4f} = "
              f"{100 * ch / dA[6] if dA[6] else 0:+.1f}%); bootstrap câu "
              f"KTC [{lo:+.4f}, {hi:+.4f}], P(<=0)={(dd <= 0).mean():.1%}")

    print(f"\nNGƯỠNG tiền-đăng-ký: ≥ +0,05 tuyệt đối (+5 điểm %) ở cột ±6.")
    print("n=12 — đây là TÍN HIỆU; qua ngưỡng mới đọc 12 mục mới (một lần) và")
    print("mới cài chế độ drop-cost + so với đường sản xuất thật.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
