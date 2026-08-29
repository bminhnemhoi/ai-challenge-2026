"""Nạp ứng viên NỘI-VIDEO vào tiên nghiệm của bộ phủ xác suất.

Chẩn đoán 24/08: với các câu thất bại, keyframe đúng thường KHÔNG nằm trong
400 ứng viên toàn cục — 9/15 câu thất bại cách ứng viên hạng-1 hơn 1.000 frame,
trong khi VIDEO đúng lại thường có mặt. Bộ phủ chỉ nhìn thấy 400 ứng viên, nên
khối lượng tiên nghiệm không thể rơi vào vùng đúng dù video đúng dẫn đầu.

Giả thuyết: với các video dẫn đầu (top-K theo thứ tự ranked_hits), đưa TOÀN BỘ
keyframe của chúng vào tiên nghiệm — điểm lấy từ ``eng.query_similarities``
(SigLIP có sẵn, không API) — thì bộ phủ tự rải thêm dòng vào trong các video
đó, KHÔNG đổi thứ tự video, không đổi công thức phủ.

Biến thể: K ∈ {3, 5, 10} × trọng số ứng viên mới ∈ {1.0 (như SigLIP thuần),
0.5}. Trọng số nhân vào e^{s/T} trong softmax tiên nghiệm.

Chống overfit: 60 câu chia 30 TUNE (chỉ số chẵn) / 30 TEST (chỉ số lẻ);
chọn biến thể trên TUNE, chỉ sau đó mới đọc TEST. Chênh < 2 sigma = HOÀ.

Harness chuẩn: ứng viên từ ĐƯỜNG SẢN XUẤT ranked_hits (cache 400 hit/câu),
khoảnh khắc thật KHÔNG bám keyframe (bốc đều trong ô Voronoi), 3 họ hạt giống,
chấm đúng công thức final_score / r_score_kis (có kiểm đối chiếu trong script).

    python scripts/experiment_phu_noi_video.py                # dùng/cất cache rồi đo
    python scripts/experiment_phu_noi_video.py --limit 8      # chạy thử, không cất cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    RANK_THRESHOLDS,
    final_score,
    r_score_kis,
)

# giá trị của một dòng ĐÚNG ở hạng r (1-based): sum(k >= r)/5; chỉ số MAX_ROWS = trượt
BUCKET = np.array(
    [sum(1 for k in RANK_THRESHOLDS if k >= r) / len(RANK_THRESHOLDS) for r in range(1, MAX_ROWS + 1)]
    + [0.0]
)


# ------------------------------------------------------------------ dữ liệu
def nap_hoac_dung_cache(args, gt):
    """Trả về (hits_of, sims, valid). Xây bằng engine nếu cache thiếu.

    ``hits_of[qi]`` = danh sách [video_id, frame_idx, score, video_last_frame]
    đúng thứ tự ranked_hits sản xuất; ``sims`` = (Q, T) điểm SigLIP của MỌI
    keyframe qua ``eng.query_similarities``; ``valid`` = mặt nạ keyframe hợp lệ
    (bỏ frame đầu video và frame trắng) — đúng mặt nạ ``engine.search`` dùng.
    """
    hits_path = Path(args.hits_cache)
    sims_path = Path(args.sims_cache)
    if hits_path.exists() and sims_path.exists():
        cache = json.loads(hits_path.read_text(encoding="utf-8"))
        z = np.load(sims_path)
        sims, valid = z["sims"], z["valid"]
        if len(cache) >= len(gt) and sims.shape[0] >= len(gt):
            for qi, g in enumerate(gt):
                assert cache[qi]["video_id"] == g["video_id"], "cache lệch thứ tự câu"
            return [q["hits"] for q in cache[: len(gt)]], sims[: len(gt)], valid

    print("cache thiếu — nạp engine SigLIP và tính (một lần) ...", flush=True)
    from scripts.make_submission import ranked_hits
    from src.core.kis_engine import KISEngine

    eng = KISEngine(args.data).load()
    hits_of, rows = [], []
    for qi, g in enumerate(gt):
        t0 = time.time()
        sims_q = eng.query_similarities(g["kis_query_vi"], g.get("kis_query_en"))
        hs = ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en"))
        hits_of.append([[h.video_id, int(h.frame_idx), float(h.score), int(h.video_last_frame)] for h in hs])
        rows.append(np.asarray(sims_q, dtype=np.float32))
        print(f"  câu {qi + 1}/{len(gt)}  ({time.time() - t0:.1f}s)", flush=True)
    sims = np.stack(rows)
    valid = np.asarray(eng.valid, dtype=bool)
    if not args.limit:  # chỉ cất cache khi chạy đủ 60 câu
        hits_path.write_text(
            json.dumps([
                {"video_id": g["video_id"], "frame_idx": int(g["frame_idx"]), "hits": h}
                for g, h in zip(gt, hits_of)
            ]),
            encoding="utf-8",
        )
        np.savez_compressed(sims_path, sims=sims, valid=valid)
        print(f"đã cất cache: {hits_path.name}, {sims_path.name}", flush=True)
    return hits_of, sims, valid


# ------------------------------------------------------------- bộ phủ (nhanh)
def phu_xac_suat(vids, frames, lasts, w, sigma=30.0, nua_cua_so=10, luoi=5, budget=MAX_ROWS):
    """Tham lam cực đại hoá khối lượng chưa phủ; ``w`` là tiên nghiệm đã chuẩn hoá.

    Cùng thuật toán với experiment_phu_xac_suat.py, thêm cache best-per-video
    để mỗi bước chỉ quét lại video vừa bị phủ (nhanh gấp ~50 lần, cùng kết quả).
    """
    theo_video: dict[str, list[int]] = defaultdict(list)
    for i, v in enumerate(vids):
        theo_video[v].append(i)

    khoi: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for vid, idxs in theo_video.items():
        f = frames[idxs]
        last = int(lasts[idxs].max())
        lo = max(0, int(f.min()) - 4 * int(sigma))
        hi = min(last, int(f.max()) + 4 * int(sigma))
        truc = np.arange(lo, hi + 1, luoi, dtype=np.int64)
        if truc.size == 0:
            continue
        mass = np.zeros(truc.size, dtype=np.float64)
        for i in idxs:
            wi = w[i]
            if wi <= 0:
                continue
            mass += wi * np.exp(-0.5 * ((truc - int(frames[i])) / sigma) ** 2)
        khoi[vid] = (truc, mass)

    nua = max(1, int(nua_cua_so) // luoi)

    def best_cua(con):
        tich = np.cumsum(np.concatenate(([0.0], con)))
        idx = np.arange(con.size)
        gia = tich[np.minimum(con.size, idx + nua + 1)] - tich[np.maximum(0, idx - nua)]
        j = int(np.argmax(gia))
        return float(gia[j]), j

    chua_phu = {v: m.copy() for v, (_t, m) in khoi.items()}
    best = {v: best_cua(chua_phu[v]) for v in khoi}
    rows, da_dung = [], set()
    while len(rows) < budget and best:
        vid = max(best, key=lambda v: best[v][0])
        gia_v, j = best[vid]
        if gia_v <= 0:
            break
        truc, _ = khoi[vid]
        f = int(truc[j])
        if (vid, f) in da_dung:
            best[vid] = (0.0, j)
            continue
        rows.append((vid, f))
        da_dung.add((vid, f))
        con = chua_phu[vid]
        con[max(0, j - nua): j + nua + 1] = 0.0
        best[vid] = best_cua(con)
    return rows


# ---------------------------------------------------------------- chấm điểm
def make_draws(gt, kf, seed, n_draws):
    """(Q, D) frame đáp án: bốc ĐỀU trong ô Voronoi quanh keyframe gần GT nhất."""
    rng = np.random.default_rng(seed)
    out = np.empty((len(gt), n_draws), dtype=np.int64)
    for qi, g in enumerate(gt):
        a = kf[g["video_id"]]
        i = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
        lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
        hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
        out[qi] = rng.integers(lo, max(lo + 1, hi), size=n_draws)
    return out


def cham_vec(rows_of, gt, draws, windows):
    """Điểm từng câu, trung bình trên bốc x cửa sổ — trùng khớp final_score."""
    per_q = np.zeros(len(gt))
    for qi, g in enumerate(gt):
        rows = rows_of[qi]
        if not rows:
            continue
        ok = np.array([v == g["video_id"] for v, _f in rows], dtype=bool)
        f = np.array([fi for _v, fi in rows], dtype=np.int64)
        s = 0.0
        for half in windows:
            hit = ok[:, None] & (np.abs(f[:, None] - draws[qi][None, :]) <= half)
            any_hit = hit.any(axis=0)
            first = np.where(any_hit, hit.argmax(axis=0), MAX_ROWS)
            s += float(BUCKET[first].mean())
        per_q[qi] = s / len(windows)
    return per_q


def kiem_doi_chieu(rows_of, gt, draws, windows):
    """Khẳng định bộ chấm vec-tơ = final_score/r_score_kis chính thức."""
    for qi in range(min(3, len(gt))):
        g = gt[qi]
        for t in draws[qi][:2]:
            for half in windows:
                span = (int(t) - half, int(t) + half)
                chinh_thuc = final_score([r_score_kis(v, f, g["video_id"], span) for v, f in rows_of[qi]])
                ok = np.array([v == g["video_id"] for v, _f in rows_of[qi]], dtype=bool)
                f = np.array([fi for _v, fi in rows_of[qi]], dtype=np.int64)
                hit = ok & (np.abs(f - int(t)) <= half)
                vec = float(BUCKET[int(hit.argmax()) if hit.any() else MAX_ROWS])
                assert abs(chinh_thuc - vec) < 1e-12, f"bộ chấm lệch: {chinh_thuc} vs {vec}"


# -------------------------------------------------------------------- chính
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--hits-cache", default=str(ROOT / "data" / "_prodhits60.json"))
    ap.add_argument("--sims-cache", default=str(ROOT / "data" / "_sims_siglip60.npz"))
    ap.add_argument("--nhiet", type=float, default=0.02, help="tham số tốt nhất đã biết (quét 28/08)")
    ap.add_argument("--sigma", type=float, default=30.0)
    ap.add_argument("--nua", type=int, default=10)
    ap.add_argument("--luoi", type=int, default=5)
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads((Path(args.data) / "metadata.json").read_text(encoding="utf-8"))
    m_vid = np.array([m["video_id"] for m in meta], dtype=object)
    m_frame = np.array([int(m["frame_idx"]) for m in meta], dtype=np.int64)
    last_frame: dict[str, int] = {}
    kf_list: dict[str, list[int]] = defaultdict(list)
    for v, f in zip(m_vid, m_frame):
        kf_list[v].append(int(f))
        if f > last_frame.get(v, -1):
            last_frame[v] = int(f)
    kf = {v: np.array(sorted(a)) for v, a in kf_list.items()}
    dong_cua_video: dict[str, np.ndarray] = {}  # video -> chỉ số hàng metadata
    thu_tu = defaultdict(list)
    for i, v in enumerate(m_vid):
        thu_tu[v].append(i)
    dong_cua_video = {v: np.array(a, dtype=np.int64) for v, a in thu_tu.items()}

    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("video_id") in last_frame]
    if args.limit:
        gt = gt[: args.limit]
    hits_of, sims, valid = nap_hoac_dung_cache(args, gt)

    # ---------------- ứng viên gốc + video dẫn đầu theo ranked_hits ----------
    Q = len(gt)
    goc, video_dan_dau, bo_goc = [], [], []
    for qi in range(Q):
        hs = hits_of[qi]
        vids = [h[0] for h in hs]
        frames = np.array([h[1] for h in hs], dtype=np.int64)
        scores = np.array([h[2] for h in hs], dtype=np.float64)
        lasts = np.array([h[3] for h in hs], dtype=np.int64)
        goc.append((vids, frames, scores, lasts))
        thay, order = set(), []
        for v in vids:  # thứ tự video = lần xuất hiện đầu trong danh sách xếp hạng
            if v not in thay:
                thay.add(v)
                order.append(v)
        video_dan_dau.append(order)
        bo_goc.append(set(zip(vids, frames.tolist())))

    # ---------------- ứng viên tiêm theo K ----------------------------------
    K_LIST = [3, 5, 10]
    tiem: dict[int, list[tuple[list, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for K in K_LIST:
        per_q = []
        for qi in range(Q):
            iv, ifr, isc = [], [], []
            for v in video_dan_dau[qi][:K]:
                for i in dong_cua_video[v]:
                    if not valid[i]:
                        continue
                    f = int(m_frame[i])
                    if (v, f) in bo_goc[qi]:
                        continue
                    iv.append(v)
                    ifr.append(f)
                    isc.append(float(sims[qi, i]))
            per_q.append((
                iv,
                np.array(ifr, dtype=np.int64),
                np.array(isc, dtype=np.float64),
                np.array([last_frame[v] for v in iv], dtype=np.int64),
            ))
        tiem[K] = per_q

    # -------- keyframe gần đáp án nhất có mặt trong tập ứng viên? -----------
    def gan_nhat(qi):
        g = gt[qi]
        a = kf[g["video_id"]]
        return g["video_id"], int(a[np.argmin(np.abs(a - int(g["frame_idx"])))])

    truoc = sum(1 for qi in range(Q) if gan_nhat(qi) in bo_goc[qi])
    print(f"\nKeyframe GẦN ĐÁP ÁN NHẤT có mặt trong tập ứng viên:")
    print(f"  trước (400 toàn cục)           : {truoc}/{Q}")
    for K in K_LIST:
        sau = 0
        for qi in range(Q):
            cap = gan_nhat(qi)
            them = set(zip(tiem[K][qi][0], tiem[K][qi][1].tolist()))
            sau += 1 if (cap in bo_goc[qi] or cap in them) else 0
        trong_topk = sum(1 for qi in range(Q) if gt[qi]["video_id"] in video_dan_dau[qi][:K])
        print(f"  sau, K={K:>2} (video GT trong top-K: {trong_topk}/{Q}): {sau}/{Q}")

    # ---------------- phân bổ ------------------------------------------------
    def tien_nghiem(scores, mult):
        w = np.asarray(mult, dtype=np.float64) * np.exp((scores - scores.max()) / max(args.nhiet, 1e-9))
        return w / w.sum()

    cau_hinh = [("nền (400 toàn cục)", None, None)]
    for K in K_LIST:
        for m in (1.0, 0.5):
            cau_hinh.append((f"tiêm K={K:>2} m={m}", K, m))

    rows_cfg, khoi_luong_moi = {}, {}
    for ten, K, m in cau_hinh:
        t0 = time.time()
        rows_of, share = [], []
        for qi in range(Q):
            vids, frames, scores, lasts = goc[qi]
            if K is None:
                av, af, asc, al = list(vids), frames, scores, lasts
                mult = np.ones(len(av))
            else:
                iv, ifr, isc, il = tiem[K][qi]
                av = list(vids) + iv
                af = np.concatenate([frames, ifr])
                asc = np.concatenate([scores, isc])
                al = np.concatenate([lasts, il])
                mult = np.concatenate([np.ones(len(vids)), np.full(len(iv), m)])
            w = tien_nghiem(asc, mult)
            share.append(float(w[len(vids):].sum()) if K is not None else 0.0)
            rows_of.append(phu_xac_suat(av, af, al, w, sigma=args.sigma, nua_cua_so=args.nua, luoi=args.luoi))
        rows_cfg[ten] = rows_of
        khoi_luong_moi[ten] = float(np.mean(share))
        print(f"  phân bổ {ten}: {time.time() - t0:.1f}s  (khối lượng tiên nghiệm vào ứng viên mới: {100 * khoi_luong_moi[ten]:.1f}%)", flush=True)

    # ---------------- chấm: TUNE (chẵn) trước, TEST (lẻ) sau ----------------
    windows = [int(w) for w in args.windows.split(",")]
    tune = np.arange(Q) % 2 == 0
    per_cfg = {ten: [] for ten, _k, _m in cau_hinh}
    for s in range(args.seeds):
        draws = make_draws(gt, kf, 30000 + s * 1000, args.draws)
        if s == 0:
            kiem_doi_chieu(rows_cfg[cau_hinh[0][0]], gt, draws, windows)
        for ten, _k, _m in cau_hinh:
            per_cfg[ten].append(cham_vec(rows_cfg[ten], gt, draws, windows))
    per_cfg = {t: np.stack(v) for t, v in per_cfg.items()}  # (S, Q)

    nen_ten = cau_hinh[0][0]

    def bang(mask, nhan):
        nen_v = per_cfg[nen_ten][:, mask].mean(axis=1)
        print(f"\n--- {nhan} ({int(mask.sum())} câu) ---")
        print(f"{'cấu hình':<22}{'điểm':>9}{'±':>9}{'so nền':>9}{'%mass mới':>11}")
        print("-" * 60)
        out = {}
        for ten, _k, _m in cau_hinh:
            v = per_cfg[ten][:, mask].mean(axis=1)
            out[ten] = (float(v.mean()), float(v.std()))
            so = f"{100 * (v.mean() / nen_v.mean() - 1):+8.1f}%" if ten != nen_ten else "        "
            print(f"{ten:<22}{v.mean():9.4f}{v.std():9.4f}{so:>9}{100 * khoi_luong_moi[ten]:10.1f}%")
        return out

    tune_kq = bang(tune, "TUNE — chỉ số chẵn, dùng để CHỌN")
    nen_tune, sd_tune = tune_kq[nen_ten]
    chon = max((t for t in tune_kq if t != nen_ten), key=lambda t: tune_kq[t][0])
    print(f"\nChọn trên TUNE: {chon}  ({tune_kq[chon][0]:.4f} vs nền {nen_tune:.4f})")

    test_kq = bang(~tune, "TEST — chỉ số lẻ, chỉ đọc SAU khi chốt")
    nen_test, sd_test = test_kq[nen_ten]
    diem, _sd = test_kq[chon]
    bien = max(sd_test, 0.0005)
    print(f"\nKẾT LUẬN TEST: {chon} {diem:.4f} vs nền {nen_test:.4f} ({100 * (diem / nen_test - 1):+.1f}%)", end="")
    if abs(diem - nen_test) < 2 * bien:
        print(f" — chênh {diem - nen_test:+.4f} < 2×{bien:.4f} = HOÀ, không kết luận.")
    else:
        print(f" — chênh {diem - nen_test:+.4f} ≥ 2×{bien:.4f}, {'ĂN THẬT' if diem > nen_test else 'ÂM THẬT'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
