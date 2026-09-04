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

LAM = (0.001, 0.003, 0.01, 0.02, 0.05)
B_KBEST = 8


def dp_kbest(S: np.ndarray, lam, k_out: int = 100):
    """DP đơn điệu nghiêm; lam là số HOẶC list theo khe (λ của bước vào sự kiện i)."""
    N, T = S.shape
    lam_i = [float(lam)] * N if np.isscalar(lam) else [float(x) for x in lam]
    # trạng thái [i][t] = list (điểm, đường) top-B, đường là tuple chỉ số lưới
    cur = [[(float(S[0, t]), (t,))] for t in range(T)]
    for i in range(1, N):
        li = lam_i[i]
        moi = [[] for _ in range(T)]
        # chạy τ tăng dần, giữ "bể" ứng viên tốt nhất đã trừ phạt tới τ
        be = []  # list (điểm − λ·(t−τ) quy về gốc τ=0: điểm + λ·τ, đường)
        for t in range(T):
            if t >= 1:
                for d, p in cur[t - 1]:
                    be.append((d + li * (t - 1), p))
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
                them.append((float(S[i, t]) + d - li * t, p + (t,)))
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

    # --- biến thể đăng ký trước, chạy cùng bảng ---
    # DROP (Drop-DTW thích nghi): sự kiện có max(S_i) dưới phân vị 30 của toàn S
    # là "bước không khớp được" — làm phẳng hàng đó về 0 để DP đặt nó theo
    # khoảng cách tối ưu giữa hai láng giềng thay vì theo nhiễu.
    # LAM_I (entropy khe): λ_i = λ · (1 − H(softmax(S_i))/H_max) — sự kiện càng
    # bất định (entropy cao) càng được phạt gap NHẸ hơn.
    def bien_the(S, kieu, lam):
        if kieu == "goc":
            return S, [lam] * S.shape[0]
        if kieu == "drop":
            S2 = S.copy()
            nguong = np.percentile(S, 30)
            for j in range(S.shape[0]):
                if S2[j].max() < nguong:
                    S2[j] = 0.0
            return S2, [lam] * S.shape[0]
        if kieu == "lam_i":
            hs = []
            for j in range(S.shape[0]):
                p_ = np.exp(S[j] * 50)  # nhiệt ~ dải sim SigLIP
                p_ /= p_.sum()
                H = -(p_ * np.log(p_ + 1e-12)).sum() / np.log(len(p_))
                hs.append(lam * max(0.1, 1.0 - H))
            return S, hs
        raise ValueError(kieu)

    # C (k-best trải dòng) đã ÂM dứt khoát ở lần chạy 1 (−28,6%, P=100%) — bỏ,
    # chỉ quét B (DP làm định vị) qua ba kiểu phạt.
    ket = {}
    for kieu in ("goc", "drop", "lam_i"):
        for lam in LAM:
            rows_B = []
            for m, S, a in zip(gt, S_of, luoi_of):
                S2, lam_list = bien_the(S, kieu, lam)
                top1, _kb = dp_kbest(S2, lam_list, k_out=1)
                c = [int(a[t]) for t in top1]
                rows_B.append(allocate_trake_rows(m["video_id"], c, budget=MAX_ROWS,
                                                  step=args.step,
                                                  video_last_frame=last_of.get(m["video_id"])))
            dB = diem(rows_B)
            ket[(kieu, lam)] = (dB, rows_B)
            print(f"{'B ' + kieu + '(λ=' + str(lam) + ')':<26}{dB[6]:>16.4f}"
                  f"{dB[10]:>9.4f}{dB[20]:>9.4f}", flush=True)

    tot = max(ket, key=lambda k: ket[k][0][6])
    dB, rows_B = ket[tot]
    print(f"\ncấu hình tốt nhất ở ±6: {tot[0]}/λ={tot[1]}")
    for ten, rows, d in (("B (DP định vị)", rows_B, dB),):
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


if __name__ == "__main__" and "--chung-ket" not in sys.argv:
    raise SystemExit(main())


def chung_ket() -> int:
    """Bậc CUỐI R5: so với đường sản xuất THẬT + đọc 12 mục TEST đúng MỘT lần.

    Cấu hình DP đóng băng TRƯỚC khi nhìn TEST: lam_i / λ=0.01 (chọn trên TUNE).
    Sản xuất thật = align_sequence (ordered, gap=2) + allocate_trake_rows —
    dựng ĐÚNG như do_che_do_can_chinh.py (luật ii: input y hệt sản xuất).
    Ngưỡng ship (đăng ký từ đêm 03/09): TEST ≥ +5 điểm tuyệt đối ở ±6.

        python -u scripts/chay_gon_ram.py scripts/do_dp_trake.py --chung-ket
    """
    import re as _re

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--gt", default=str(ROOT / "data" / "gt_trake.json"))
    ap.add_argument("--chung-ket", action="store_true")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()
    KIEU, LAM_CHOT = "lam_i", 0.01

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    meta = json.loads((Path(args.data) / "metadata.json").read_text(encoding="utf-8"))
    hang_of, kf_list, last_of = {}, {}, {}
    for r, m in enumerate(meta):
        hang_of[(m["video_id"], int(m["frame_idx"]))] = r
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        last_of[m["video_id"]] = max(last_of.get(m["video_id"], 0), int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}
    del meta, kf_list

    print("nap chi muc + engine san xuat ...", flush=True)
    from src.core.kis_engine import KISEngine
    from src.task3_trake import TRAKEEngine

    eng = KISEngine(args.data).load()
    trake = TRAKEEngine(engine=eng).load_index()
    kho = KhoSims(args.data, False)

    rows_sx, rows_dp = [], []
    for m in gt:
        de = (m.get("boi_canh", "") + "\n"
              + "\n".join(f"E{j+1}: {s}" for j, s in enumerate(m["su_kien"])))
        evs = split_events(de)
        first = bool(_re.search(r"đầu tiên|lần đầu|first", de, _re.IGNORECASE))
        try:
            res = trake.align_sequence(evs, first_occurrence=first, top_k=1) or []
        except Exception:  # noqa: BLE001
            res = []
        if res:
            v = res[0]["video_id"]
            rows_sx.append(allocate_trake_rows(v, res[0]["sequence_frames"],
                                               budget=MAX_ROWS, step=args.step,
                                               video_last_frame=last_of.get(v)))
        else:
            rows_sx.append([])
        # DP trên video sản xuất CHỌN (không phải video GT) — đúng điều kiện thi;
        # nếu sản xuất chọn sai video thì cả hai cùng 0 điểm (luật TRAKE).
        v_sx = res[0]["video_id"] if res else m["video_id"]
        a = kf[v_sx]
        rws = np.array([hang_of[(v_sx, int(f))] for f in a])
        S = np.stack([kho.lay(ev, "")[rws].astype(np.float64) for ev in evs])
        # bien the lam_i (sao chep tu main de chay doc lap)
        hs = []
        for j in range(S.shape[0]):
            p_ = np.exp(S[j] * 50)
            p_ /= p_.sum()
            H = -(p_ * np.log(p_ + 1e-12)).sum() / np.log(len(p_))
            hs.append(LAM_CHOT * max(0.1, 1.0 - H))
        top1, _kb = dp_kbest(S, hs, k_out=1)
        c = [int(a[t]) for t in top1]
        rows_dp.append(allocate_trake_rows(v_sx, c, budget=MAX_ROWS, step=args.step,
                                           video_last_frame=last_of.get(v_sx)))

    ho = []
    for s in range(args.seeds):
        bocs = [boc_moc(GOC + s * 1000 + t, gt, kf) for t in range(args.draws)]
        ho.append([[b[i] for b in bocs] for i in range(len(gt))])

    def diem_cau(rows_of, chi_so, w=6):
        d = np.zeros(len(chi_so))
        for b in ho:
            for k, i in enumerate(chi_so):
                d[k] += cham([rows_of[i]], [gt[i]], [b[i]], [w])
        return d / len(ho)

    for ten, chi_so, quyet in (("TUNE (12 cũ)", list(range(12)), False),
                               ("TEST (12 mới, ĐỌC MỘT LẦN)", list(range(12, len(gt))), True)):
        a_c = diem_cau(rows_sx, chi_so)
        b_c = diem_cau(rows_dp, chi_so)
        ch = b_c.mean() - a_c.mean()
        rng = np.random.default_rng(4242)
        lay = rng.integers(0, len(chi_so), size=(4000, len(chi_so)))
        dd = b_c[lay].mean(axis=1) - a_c[lay].mean(axis=1)
        lo, hi = np.percentile(dd, [2.5, 97.5])
        print(f"\n=== {ten} (±6) ===")
        print(f"  SẢN XUẤT {a_c.mean():.4f} -> DP lam_i/0.01 {b_c.mean():.4f} "
              f"({ch:+.4f}); KTC [{lo:+.4f}, {hi:+.4f}]; P(<=0)={(dd <= 0).mean():.1%}")
        if quyet:
            print(f"\nNGƯỠNG SHIP (đăng ký 03/09 đêm): TEST ≥ +0,05 tuyệt đối ở ±6.")
            print("  " + ("ĐỦ ĐIỀU KIỆN SHIP qua cờ --dp-trake."
                          if ch >= 0.05 else
                          "KHÔNG đủ — ghi 'tín hiệu treo, cần n lớn hơn', không ship."))
    return 0


if "--chung-ket" in sys.argv:
    raise SystemExit(chung_ket())
