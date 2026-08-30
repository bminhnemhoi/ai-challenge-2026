"""Truy vấn CẶP THỜI GIAN (lever ③) — s_temp(i) = HM(s_A(i), max_{i<j<=i+W} s_B(j)).

Kỹ thuật chung của mọi hệ thống thắng Video Browser Showdown (vibro, vitrivr):
một câu mô tả HAI cảnh nối tiếp bị hệ hiện tại nén thành MỘT vector, nên mất
hoàn toàn cấu trúc thời gian.  Chấm theo cặp nâng video có đủ cả hai cảnh và
kéo keyframe của cảnh A lên hạng nội-video.

**Cổng:** chỉ câu được ``gan_nhan_hai_canh.py`` gắn ``co_2_canh=true`` mới đổi
điểm.  Câu một cảnh phải ra 100 dòng GIỐNG HỆT đường cũ — kiểm bằng assert
trong ``cmd_gt``, không bằng mắt.

Ba chế độ:

    python scripts/experiment_cap_thoi_gian.py            # cổng TUNE/TEST trên 60 câu GT
    python scripts/experiment_cap_thoi_gian.py --de round1/queries round2/queries
    python scripts/experiment_cap_thoi_gian.py --nghen    # chẩn đoán 6 câu nghẽn

Chế độ ``--de`` là DIFF CẤU TRÚC trên đề thật (không có ground truth): nó nói
s_temp đổi những gì, KHÔNG nói bên nào đúng.  Chế độ ``--nghen`` là chẩn đoán
thăm dò trên đúng 6 câu nghẽn (chỉ số 5, 9, 12, 15, 40, 41) — n=6, không có
kết luận thống kê nào được rút ra từ nó.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.experiment_phu_quet_luoi import (  # noqa: E402
    BUCKET,
    GOC_TEST,
    GOC_TUNE,
    CacheDiem,
    _doi_chieu_bo_cham,
    cac_lan_boc,
    cham_nhanh,
    ma_tran_dong,
    nap_ung_vien,
)
from scripts.gan_nhan_hai_canh import nap_nhan  # noqa: E402
from scripts.make_submission import (  # noqa: E402
    DEFAULT_DEPTH_COST,
    DEFAULT_N_FLAT,
    allocate_rows,
)
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
)

CACHE = ROOT / "data" / "cache_cap_thoi_gian"

#: cửa sổ tính bằng SỐ KEYFRAME đi sau, theo đề xuất trong docs/NGHIEN_CUU_SOTA.md
LUOI_W = (2, 3, 5, 8)
#: cách gộp hai điểm thành một.  ``chiA`` là ĐỐI CHỨNG, không phải một biến thể
#: của lever: nó bỏ hẳn cảnh B, nên tách được "truy vấn con ngắn thì nhiễu" khỏi
#: "ghép cặp thời gian sai".  Nếu chiA cũng âm đúng bằng hm thì cái âm không
#: thuộc về việc ghép cặp.
LUOI_GOP = ("hm", "tich", "chiA")
#: pha trộn với điểm gốc: 1.0 = thay hẳn (đúng công thức), 0.5 = nửa nọ nửa kia
LUOI_LAMBDA = (1.0, 0.5)

#: 6 câu mà keyframe đáp án nằm hạng nội-video 95-276 (docs/SHIP_PHU_XAC_SUAT.md §4.5)
CAU_NGHEN = (5, 9, 12, 15, 40, 41)


def _plan() -> AllocationPlan:
    return AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)


# ---------------------------------------------------------------------------
# Dòng thời gian keyframe: video -> (frame_idx tăng dần, hàng trong chỉ mục)
# ---------------------------------------------------------------------------


def nap_truc_video(data_dir: Path, refresh: bool = False):
    """{video: (frames[n], rows[n])} — rows là chỉ số hàng trong embeddings/sims.

    Đây là thứ biến "keyframe kế tiếp trong CÙNG video" thành một phép cắt mảng.
    Đọc thẳng từ metadata.json nên không cần nạp mô hình.
    """
    f = CACHE / "truc_video.npz"
    if f.exists() and not refresh:
        z = np.load(f, allow_pickle=True)
        return {str(v): (a, b) for v, a, b in zip(z["vids"], z["frames"], z["rows"])}

    meta = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    tam: dict[str, list[tuple[int, int]]] = {}
    for r, m in enumerate(meta):
        tam.setdefault(m["video_id"], []).append((int(m["frame_idx"]), r))
    truc = {}
    for v, items in tam.items():
        items.sort()
        truc[v] = (np.array([x[0] for x in items], dtype=np.int64),
                   np.array([x[1] for x in items], dtype=np.int64))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        f,
        vids=np.array(list(truc), dtype=object),
        frames=np.array([truc[v][0] for v in truc], dtype=object),
        rows=np.array([truc[v][1] for v in truc], dtype=object),
    )
    return truc


# ---------------------------------------------------------------------------
# Công thức cặp thời gian
# ---------------------------------------------------------------------------


def _chuan_01(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Đưa điểm tương đồng về [0,1] trên ĐÚNG giá đỡ đang xét.

    HM và tích chỉ có nghĩa với số dương, mà cosine SigLIP chạy từ -0,21 tới
    +0,29 trên kho này.  Chuẩn hoá trên giá đỡ (các keyframe của những video
    đang có ứng viên) chứ không trên toàn kho: toàn kho bị đuôi âm kéo, làm mọi
    ứng viên dồn vào một dải hẹp và HM suy biến thành trung bình cộng.
    """
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def diem_cap_thoi_gian(cands, simsA, simsB, truc, W: int, gop: str,
                       doi_xung: bool = False):
    """s_temp cho từng ứng viên, cùng thứ tự với ``cands``.

    ``doi_xung=False`` là công thức của lane: cảnh B phải ở SAU cảnh A, trong
    ``W`` keyframe kế tiếp cùng video.  ``doi_xung=True`` bỏ ràng buộc thứ tự
    (|i-j| <= W) — dùng cho chẩn đoán câu KHÔNG có trình tự thời gian, nơi hai
    mệnh đề chỉ cần ở gần nhau, và được ghi rõ là một giả thuyết KHÁC.

    Trả về (s_temp, so_cau_khong_co_hau_ke).
    """
    vids = {c.video_id for c in cands}
    ho = np.concatenate([truc[v][1] for v in vids if v in truc]) if vids else np.array([], np.int64)
    loA, hiA = (float(simsA[ho].min()), float(simsA[ho].max())) if ho.size else (0.0, 1.0)
    loB, hiB = (float(simsB[ho].min()), float(simsB[ho].max())) if ho.size else (0.0, 1.0)

    # trượt max của B trên từng video, tính một lần cho mỗi video
    bestB: dict[str, np.ndarray] = {}
    for v in vids:
        if v not in truc:
            continue
        rows = truc[v][1]
        b = _chuan_01(simsB[rows], loB, hiB)
        n = b.size
        out = np.full(n, -1.0)
        for p in range(n):
            lo = max(0, p - W) if doi_xung else p + 1
            hi = min(n, p + W + 1)
            if lo < hi:
                out[p] = b[lo:hi].max()
        bestB[v] = out

    s = np.zeros(len(cands))
    thieu = 0
    for i, c in enumerate(cands):
        t = truc.get(c.video_id)
        if t is None:
            thieu += 1
            continue
        p = int(np.searchsorted(t[0], int(c.frame_idx)))
        if p >= t[0].size or t[0][p] != int(c.frame_idx):
            thieu += 1
            continue
        a = float(_chuan_01(simsA[t[1][p] : t[1][p] + 1], loA, hiA)[0])
        if gop == "chiA":  # đối chứng: bỏ hẳn cảnh B
            s[i] = a
            continue
        b = bestB[c.video_id][p]
        if b < 0.0:  # không có keyframe hậu kế trong cửa sổ -> cặp không thành
            thieu += 1
            s[i] = 0.0
            continue
        if gop == "hm":
            s[i] = 0.0 if (a + b) <= 1e-12 else 2 * a * b / (a + b)
        elif gop == "tich":
            s[i] = a * b
        else:
            raise ValueError(gop)
    return s, thieu


def ve_thang_diem_goc(s_moi: np.ndarray, goc: np.ndarray, lam: float) -> np.ndarray:
    """Đưa s_temp về ĐÚNG thang của điểm gốc, rồi pha theo lambda.

    Không phải một tham số để quét — là một điều kiện bắt buộc của phép so sánh.
    Bộ phân bổ phủ xác suất lấy softmax nhiệt 0,02 trên điểm; điểm gốc trải rộng
    0,02-0,11 còn s_temp trải [0,1].  Ném thẳng s_temp vào allocator là đo ĐỘ
    TRẢI chứ không đo cấu trúc thời gian: softmax sẽ suy biến về argmax và
    coverage sụp còn một video.  Chuẩn hoá về cùng trung bình/độ lệch chuẩn giữ
    nhiệt hiệu dụng như cũ, nên phần chênh còn lại đúng là do THỨ TỰ và KHOẢNG
    CÁCH tương đối giữa các ứng viên.
    """
    gm, gs = float(goc.mean()), float(goc.std())
    nm, ns = float(s_moi.mean()), float(s_moi.std())
    z = (s_moi - nm) / ns if ns > 1e-12 else np.zeros_like(s_moi)
    quy = gm + z * gs
    if lam >= 1.0:
        return quy
    zg = (goc - gm) / gs if gs > 1e-12 else np.zeros_like(goc)
    return gm + gs * ((1 - lam) * zg + lam * z)


# ---------------------------------------------------------------------------
# Similarity cho từng cảnh, cache xuống đĩa
# ---------------------------------------------------------------------------


class KhoSims:
    """simsA/simsB theo văn bản cảnh, cache theo băm nội dung (177k float16/cảnh)."""

    def __init__(self, data_dir: str, refresh: bool = False):
        self.data_dir = data_dir
        self.refresh = refresh
        self.dir = CACHE / "sims"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._eng = None

    def _engine(self):
        if self._eng is None:
            from src.core.kis_engine import KISEngine

            print("  nạp chỉ mục SigLIP (chỉ lần đầu) ...", flush=True)
            self._eng = KISEngine(self.data_dir).load()
        return self._eng

    def lay(self, vi: str, en: str) -> np.ndarray:
        import hashlib

        key = hashlib.sha1(f"{vi}\n{en}".encode("utf-8")).hexdigest()[:16]
        f = self.dir / f"{key}.npy"
        if f.exists() and not self.refresh:
            return np.load(f).astype(np.float32)
        eng = self._engine()
        s = eng.similarities(eng.query_vector(vi, en or None))
        np.save(f, s.astype(np.float16))
        return s.astype(np.float32)


# ---------------------------------------------------------------------------
# Chế độ 1: cổng TUNE/TEST trên 60 câu ground truth
# ---------------------------------------------------------------------------


def cmd_gt(args) -> int:
    cache_dir = Path(args.cache_pxs)
    windows = [int(w) for w in args.windows.split(",")]
    t0 = time.time()

    print("=== 1) Nhãn hai cảnh (cổng) ===", flush=True)
    _gt_full, nhan = nap_nhan(Path(args.data))
    chua = [i for i, d in enumerate(nhan) if d is None]
    if chua:
        print(f"  ! {len(chua)} câu chưa gắn nhãn — chạy scripts/gan_nhan_hai_canh.py trước")
        return 2
    bat = [i for i, d in enumerate(nhan) if d["co_2_canh"]]
    print(f"  {len(bat)}/{len(nhan)} câu có cổng BẬT: {bat if bat else '(không câu nào)'}")

    print("=== 2) Ứng viên đường sản xuất (cache dùng chung) ===", flush=True)
    gt, cands_of, kf = nap_ung_vien(args.data, cache_dir, False)

    print("=== 3) Nền: allocate_rows đúng đường sản xuất ===", flush=True)
    rows_nen = [allocate_rows(c, args.allocator, DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
                for c in cands_of]

    if not bat:
        # Bất biến vẫn phải kiểm: với cổng tắt ở mọi câu, MỌI cấu hình phải cho
        # ra đúng 100 dòng của nền.  Không có nhãn nào bật nên đây là toàn bộ
        # tập — phép kiểm chạy trên 60/60 câu.
        print("\n=== 4) Bất biến: cổng tắt => dòng không đổi ===", flush=True)
        for W in LUOI_W:
            for gop in LUOI_GOP:
                for lam in LUOI_LAMBDA:
                    rows = ap_dung(cands_of, nhan, None, None, W, gop, lam, args.allocator)
                    for i, (a, b) in enumerate(zip(rows_nen, rows)):
                        assert a == b, f"câu {i} đổi dòng dù cổng TẮT (W={W},{gop},λ={lam})"
        print(f"  OK: {len(LUOI_W)*len(LUOI_GOP)*len(LUOI_LAMBDA)} cấu hình × {len(gt)} câu "
              f"ra dòng GIỐNG HỆT nền (assert, không phải nhìn bằng mắt).")

        print("\n=== KẾT LUẬN ===")
        print("  KHÔNG ĐO ĐƯỢC trên bộ 60 câu ground truth: 0 câu qua cổng.")
        print("  Mọi cấu hình (W, gộp, lambda) đều là phép ĐỒNG NHẤT trên tập này,")
        print("  nên không có TUNE, không có TEST, không có luật 2 sigma để áp.")
        print("  Chi tiết + số liệu đề thật: docs/CAP_THOI_GIAN.md")
        print(f"\nXong sau {time.time()-t0:.0f}s.")
        return 0

    # ---- có câu qua cổng: chạy đủ TUNE/TEST -------------------------------
    kho = KhoSims(args.data, args.refresh)
    truc = nap_truc_video(Path(args.data), args.refresh)
    simsA = {i: kho.lay(nhan[i]["canh_A_vi"], nhan[i]["canh_A_en"]) for i in bat}
    simsB = {i: kho.lay(nhan[i]["canh_B_vi"], nhan[i]["canh_B_en"]) for i in bat}

    du_tune = 0 if args.tune_phia == "chan" else 1
    pt, px = ("chan", "le") if du_tune == 0 else ("le", "chan")
    i_tune = [i for i in range(len(gt)) if i % 2 == du_tune]
    i_test = [i for i in range(len(gt)) if i % 2 != du_tune]
    n_bat_tune = sum(1 for i in i_tune if i in bat)
    n_bat_test = sum(1 for i in i_test if i in bat)
    print(f"  TUNE {pt}: {n_bat_tune} câu qua cổng / {len(i_tune)}; "
          f"TEST {px}: {n_bat_test}/{len(i_test)}")
    if min(n_bat_tune, n_bat_test) < 4:
        print("  !! quá ít câu qua cổng ở một nửa — kết quả tổng KHÔNG có lực thống kê.")
        print("     Bảng per-query dưới đây là thứ duy nhất đọc được.")

    def cham(idx_sub, cfg, ho):
        gt_s = [gt[i] for i in idx_sub]
        rows = ap_dung([cands_of[i] for i in idx_sub], [nhan[i] for i in idx_sub],
                       {k: simsA[i] for k, i in enumerate(idx_sub) if i in simsA},
                       {k: simsB[i] for k, i in enumerate(idx_sub) if i in simsB},
                       *cfg, args.allocator, truc=truc)
        return [cham_nhanh(ma_tran_dong(rows, gt_s), d, windows) for d in ho], rows

    ho_tune = cac_lan_boc(GOC_TUNE, args.tune_seeds, args.tune_draws,
                          [gt[i] for i in i_tune], kf)
    ho_test = cac_lan_boc(GOC_TEST, args.test_seeds, args.test_draws,
                          [gt[i] for i in i_test], kf)
    _doi_chieu_bo_cham([rows_nen[i] for i in i_tune][:5], [gt[i] for i in i_tune][:5],
                       [t[:3] for t in ho_tune[0][:5]], windows)
    print("  bộ chấm vector hoá khớp tuyệt đối bản sản xuất.")

    hn_tune = {"windows": windows, "seeds": args.tune_seeds, "draws": args.tune_draws,
               "goc": GOC_TUNE, "phia": pt, "n_bat": n_bat_tune, "alloc": args.allocator}
    hn_test = {"windows": windows, "seeds": args.test_seeds, "draws": args.test_draws,
               "goc": GOC_TEST, "phia": px, "n_bat": n_bat_test, "alloc": args.allocator}
    c_tune = CacheDiem(CACHE / f"diem_tune_{pt}.json", hn_tune, args.refresh)
    c_test = CacheDiem(CACHE / f"diem_test_{px}.json", hn_test, args.refresh)

    nen_tune = c_tune.lay("nen", lambda: [
        cham_nhanh(ma_tran_dong([rows_nen[i] for i in i_tune], [gt[i] for i in i_tune]),
                   d, windows) for d in ho_tune])
    nen_m = float(np.mean(nen_tune))

    print(f"\n=== 4) TUNE ({pt}): {len(LUOI_W)*len(LUOI_GOP)*len(LUOI_LAMBDA)} cấu hình ===")
    print(f"nền: {nen_m:.4f} ±{float(np.std(nen_tune)):.4f}")
    ket = {}
    for W in LUOI_W:
        for gop in LUOI_GOP:
            for lam in LUOI_LAMBDA:
                tag = f"W{W}_{gop}_l{lam:g}"
                d = c_tune.lay(tag, lambda W=W, gop=gop, lam=lam: cham(i_tune, (W, gop, lam), ho_tune)[0])
                ket[(W, gop, lam)] = (float(np.mean(d)), float(np.std(d)))
                print(f"  W={W:<2} {gop:<5} λ={lam:<4g} {ket[(W,gop,lam)][0]:.4f} "
                      f"({100*(ket[(W,gop,lam)][0]/nen_m-1):+.1f}%)")

    chot = max(ket, key=lambda k: ket[k][0])
    print(f"\nCHỐT trên TUNE: W={chot[0]} gộp={chot[1]} λ={chot[2]}")

    print(f"\n=== 5) TEST ({px}) — đọc ĐÚNG MỘT LẦN ===")
    nen_te = c_test.lay("nen", lambda: [
        cham_nhanh(ma_tran_dong([rows_nen[i] for i in i_test], [gt[i] for i in i_test]),
                   d, windows) for d in ho_test])
    chot_te = c_test.lay(f"W{chot[0]}_{chot[1]}_l{chot[2]:g}",
                         lambda: cham(i_test, chot, ho_test)[0])
    nm, nsd = float(np.mean(nen_te)), float(np.std(nen_te))
    cm, csd = float(np.mean(chot_te)), float(np.std(chot_te))
    bien = max(nsd, 0.0005)
    print(f"  nền : {nm:.4f} ±{nsd:.4f}")
    print(f"  chốt: {cm:.4f} ±{csd:.4f}  ({100*(cm/nm-1):+.1f}%)")
    print("\n=== KẾT LUẬN ===")
    if (cm - nm) < 2 * bien:
        print(f"  HOÀ: chênh {cm-nm:+.4f} < 2σ = {2*bien:.4f}.")
    else:
        print(f"  GIỮ ĐƯỢC: {100*(cm/nm-1):+.1f}%, vượt 2σ.")
    print(f"\nXong sau {time.time()-t0:.0f}s.")
    return 0


def ap_dung(cands_of, nhan_of, simsA, simsB, W, gop, lam, allocator, truc=None):
    """100 dòng/câu: câu có cổng bật dùng s_temp, câu còn lại đi nguyên đường cũ."""
    out = []
    for k, (cands, d) in enumerate(zip(cands_of, nhan_of)):
        if d and d["co_2_canh"] and simsA is not None and k in simsA:
            s, _ = diem_cap_thoi_gian(cands, simsA[k], simsB[k], truc, W, gop)
            goc = np.array([c.score for c in cands])
            moi = ve_thang_diem_goc(s, goc, lam)
            cands = [Candidate(c.video_id, c.frame_idx, float(x), c.video_last_frame)
                     for c, x in zip(cands, moi)]
        out.append(allocate_rows(cands, allocator, DEFAULT_N_FLAT, _plan())[:MAX_ROWS])
    return out


# ---------------------------------------------------------------------------
# Chế độ 2: diff cấu trúc trên đề thật của BTC
# ---------------------------------------------------------------------------


def doc_picks(path: Path) -> dict:
    """``query-p1-4-kis = L22_V021:20019|19560`` -> {"query-p1-4-kis": ("L22_V021", [...])}.

    ĐÂY KHÔNG PHẢI GROUND TRUTH.  Là lựa chọn của người soát trong trận, có bằng
    chứng ba kênh nhưng chưa bao giờ được BTC xác nhận đúng.  Vòng 2 chỉ được
    10,0/30 nên một phần đáng kể các lựa chọn cùng dạng này là SAI.  Dùng làm
    proxy chỉ-để-chẩn-đoán, không bao giờ làm căn cứ kết luận.
    """
    import re as _re

    out = {}
    for line in path.read_bytes().decode("utf-8-sig", errors="replace").splitlines():
        line = line.split("#")[0].strip()
        if "=" not in line or not line.startswith("query"):
            continue
        k, v = (x.strip() for x in line.split("=", 1))
        vid = v.split(":")[0]
        phan = v.split(":")
        frames = [int(x) for x in _re.findall(r"\d+", phan[1])] if len(phan) > 1 else []
        out[k] = (vid, frames)
    return out


def doc_gt_de_that(path: Path) -> dict:
    """Các mục ``nguoi_kiem_chung`` của bộ đo đề thật (lane harness) -> cùng dạng picks.

    Khoá là ĐƯỜNG DẪN có tiền tố vòng (``round1/query-p1-15-qa``), không phải
    chỉ tên file: ``round_p1/query-p1-15-qa`` và ``round1/query-p1-15-qa`` là
    HAI câu khác hẳn nhau (cảnh báo trong README §3).  Ghép bằng tên trần sẽ
    lẫn hai vòng.
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for r in d["muc"]:
        if r.get("do_tin") != "nguoi_kiem_chung" or not r.get("video_id"):
            continue
        out[r["ma"]] = (r["video_id"], [int(r["frame_idx"])] if r.get("frame_idx") is not None else [])
    return out


def cmd_de(args) -> int:
    """Không có ground truth: chỉ nói s_temp ĐỔI GÌ, tuyệt đối không nói ai đúng."""
    from scripts.make_submission import (
        decode_text,
        detect_task,
        ranked_hits,
        read_en_override,
        split_qa,
    )

    nhan_dir = CACHE / "nhan_de"
    kho = KhoSims(args.data, args.refresh)
    truc = nap_truc_video(Path(args.data), args.refresh)

    files = []
    for dpath in args.de:
        for p in sorted(Path(dpath).glob("*.txt")):
            if not p.name.lower().endswith((".en.txt", ".vi.txt")):
                files.append(p)

    nhan = {}
    for p in files:
        f = nhan_dir / f"{p.stem}.json"
        nhan[p.stem] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    thieu = [s for s, d in nhan.items() if d is None]
    if thieu:
        print(f"  ! {len(thieu)} câu chưa gắn nhãn — chạy:")
        print(f"    python scripts/gan_nhan_hai_canh.py --de {' '.join(args.de)}")
        return 2
    bat = [p for p in files if nhan[p.stem]["co_2_canh"]]
    print(f"=== {len(bat)}/{len(files)} câu đề thật có cổng BẬT ===", flush=True)

    picks = doc_picks(Path(args.picks)) if args.picks else {}
    if args.gt_de:
        picks.update(doc_gt_de_that(Path(args.gt_de)))
    eng = kho._engine()

    def khoa(p: Path) -> str:
        """``round1/queries/query-p1-9-qa.txt`` -> ``round1/query-p1-9-qa``."""
        return f"{p.parent.parent.name}/{p.stem}"

    # truy xuất MỘT lần cho mỗi câu, rồi mới quét cấu hình — cùng pool cho mọi ô
    print("truy xuất ứng viên (một lần cho mọi cấu hình) ...", flush=True)
    pool = []
    for p in bat:
        text = (decode_text(p.read_bytes()) or "").strip()
        task = detect_task(p.name)
        probe = split_qa(text)[0] or text if task == "qa" else text
        hits = ranked_hits(eng, probe, read_en_override(p))
        cands = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits]
        d = nhan[p.stem]
        pool.append((p.stem, cands,
                     kho.lay(d["canh_A_vi"], d["canh_A_en"]),
                     kho.lay(d["canh_B_vi"], d["canh_B_en"]),
                     allocate_rows(cands, args.allocator, DEFAULT_N_FLAT, _plan())[:MAX_ROWS],
                     khoa(p)))

    Ws = [int(x) for x in str(args.W).split(",")]
    lams = [float(x) for x in str(args.lam).split(",")]
    tat_ca = {}
    for W in Ws:
        for lam in lams:
            ghi, doi = [], 0
            for stem, cands, sA, sB, r1, kh_day in pool:
                s, _thieu = diem_cap_thoi_gian(cands, sA, sB, truc, W, args.gop)
                goc = np.array([c.score for c in cands])
                moi = ve_thang_diem_goc(s, goc, lam)
                c2 = [Candidate(c.video_id, c.frame_idx, float(x), c.video_last_frame)
                      for c, x in zip(cands, moi)]
                r2 = allocate_rows(c2, args.allocator, DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
                khac = bool(r1 and r2 and r1[0][0] != r2[0][0])
                doi += khac
                rec = {"cau": stem, "doi_video": khac,
                       "nen": f"{r1[0][0]}:{r1[0][1]}" if r1 else "?",
                       "stemp": f"{r2[0][0]}:{r2[0][1]}" if r2 else "?",
                       "video_nen": r1[0][0] if r1 else None,
                       "video_stemp": r2[0][0] if r2 else None,
                       "top10_nen": [f"{v}:{f}" for v, f in r1[:10]],
                       "top10_stemp": [f"{v}:{f}" for v, f in r2[:10]]}
                nguon = kh_day if kh_day in picks else (stem if stem in picks else None)
                if nguon:
                    pv = picks[nguon][0]
                    rec["pick_video"] = pv
                    rec["nguon_su_that"] = nguon
                    rec["hang_nen"] = next((k + 1 for k, (v, _f) in enumerate(r1) if v == pv), 0)
                    rec["hang_stemp"] = next((k + 1 for k, (v, _f) in enumerate(r2) if v == pv), 0)
                ghi.append(rec)
            tat_ca[(W, lam)] = (ghi, doi)
            # tên cache PHẢI mang cả tập câu lẫn nguồn sự thật: chạy cùng
            # (W, gộp, λ) trên hai tập khác nhau mà ghi đè lên nhau thì con số
            # trong tài liệu mất file đứng sau.
            tap = "-".join(Path(d).parent.name for d in args.de)
            nguon_st = "gtde" if args.gt_de else ("picks" if args.picks else "khong")
            out = CACHE / f"diff_{tap}_{nguon_st}_W{W}_{args.gop}_l{lam:g}.json"
            out.write_text(json.dumps(ghi, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- bảng từng câu cho cấu hình đầu tiên ------------------------------
    W0, l0 = Ws[0], lams[0]
    print(f"\n=== Từng câu, W={W0} {args.gop} λ={l0:g} ===")
    print(f"{'câu':<20}{'dòng-1 nền':>18}{'dòng-1 s_temp':>18}  đổi")
    for r in tat_ca[(W0, l0)][0]:
        print(f"{r['cau']:<20}{r['nen']:>18}{r['stemp']:>18}  "
              f"{'ĐỔI VIDEO' if r['doi_video'] else ''}")
    print(f"\nvideo dòng 1 ĐỔI ở {tat_ca[(W0,l0)][1]}/{len(bat)} câu")
    print("đây là DIFF CẤU TRÚC — không có ground truth, không nói bên nào đúng.")

    if not picks:
        print(f"cache: {CACHE}")
        return 0

    # ---- proxy theo lựa chọn người soát -----------------------------------
    if args.gt_de:
        print(f"\n=== HẠNG của video ĐÚNG, mục 'nguoi_kiem_chung' (gộp={args.gop}) ===")
        print("   sự thật đã kiểm chứng nhiều kênh — tin hơn picks thô, nhưng n rất nhỏ.")
    else:
        print(f"\n=== Proxy: HẠNG của video người soát đã chốt (gộp={args.gop}) ===")
        print("   !! KHÔNG PHẢI ground truth — xem docstring doc_picks().")
    print("   Dù nguồn nào: không được chọn tham số trên bảng này rồi gọi đó là kết quả đo.")
    print(f"{'W':>3}{'λ':>6}{'dòng-1 khớp':>13}{'tốt hơn':>9}{'tệ hơn':>8}{'hoà':>6}")
    for W in Ws:
        for lam in lams:
            ghi = tat_ca[(W, lam)][0]
            cp = [r for r in ghi if "pick_video" in r]
            d1s = sum(r["video_stemp"] == r["pick_video"] for r in cp)
            tot = sum((r["hang_stemp"] or 10**6) < (r["hang_nen"] or 10**6) for r in cp)
            xau = sum((r["hang_stemp"] or 10**6) > (r["hang_nen"] or 10**6) for r in cp)
            print(f"{W:>3}{lam:>6g}{d1s:>8}/{len(cp):<4}{tot:>9}{xau:>8}{len(cp)-tot-xau:>6}")
    cp0 = [r for r in tat_ca[(W0, l0)][0] if "pick_video" in r]
    d1n = sum(r["video_nen"] == r["pick_video"] for r in cp0)
    print(f"  nền (không đổi theo W/λ): dòng-1 khớp {d1n}/{len(cp0)}")

    print(f"\n=== Từng câu, W={W0} λ={l0:g} ===")
    print(f"{'câu':<20}{'video chốt':>12}{'hạng nền':>10}{'hạng s_temp':>12}")
    for r in cp0:
        hn, hs = r["hang_nen"], r["hang_stemp"]
        hn2, hs2 = (hn or 10**6), (hs or 10**6)
        dau = "  tốt hơn" if hs2 < hn2 else ("  tệ hơn" if hs2 > hn2 else "")
        print(f"{r['cau']:<20}{r['pick_video']:>12}"
              f"{('∞' if hn == 0 else str(hn)):>10}{('∞' if hs == 0 else str(hs)):>12}{dau}")
    print(f"\ncache: {CACHE}")
    return 0


# ---------------------------------------------------------------------------
# Chế độ 3: chẩn đoán 6 câu nghẽn
# ---------------------------------------------------------------------------


_PROMPT_TACH = """Câu mô tả dưới đây tìm MỘT khung hình trong video (nó KHÔNG mô tả hai
cảnh nối tiếp). Hãy tách nó thành HAI MỆNH ĐỀ hình ảnh tách biệt nhất có thể —
hai thứ mà một mô hình nhận ảnh có thể tìm ĐỘC LẬP với nhau (ví dụ: chủ thể +
hành động, và bối cảnh + chữ trên màn hình). Mỗi mệnh đề viết thành một câu tự
đứng được, kèm bản tiếng Anh.

CÂU MÔ TẢ: {vi}
BẢN TIẾNG ANH: {en}

Trả DUY NHẤT JSON: {{"A_vi":"","A_en":"","B_vi":"","B_en":""}}"""


def cmd_nghen(args) -> int:
    """Thăm dò trên 6 câu nghẽn: n=6, KHÔNG rút kết luận thống kê.

    Sáu câu này không có cấu trúc hai cảnh (nhãn đã nói vậy), nên ở đây ta ép
    tách thành hai MỆNH ĐỀ và dùng cửa sổ ĐỐI XỨNG — tức là đang thử một giả
    thuyết KHÁC ("chấm theo độ phủ sub-query", mục #10 trong bảng 33 đề xuất),
    chỉ tận dụng cùng bộ máy.  Số đo là HẠNG NỘI-VIDEO của keyframe đáp án,
    so trực tiếp với bảng trong docs/SHIP_PHU_XAC_SUAT.md §4.5.
    """
    from scripts.gan_nhan_hai_canh import _goi, _khach

    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    truc = nap_truc_video(Path(args.data), args.refresh)
    kho = KhoSims(args.data, args.refresh)
    tach_dir = CACHE / "tach_menh_de"
    tach_dir.mkdir(parents=True, exist_ok=True)

    kh = None
    tach = {}
    for i in CAU_NGHEN:
        f = tach_dir / f"{i:02d}.json"
        if f.exists() and not args.refresh:
            tach[i] = json.loads(f.read_text(encoding="utf-8"))
            continue
        if kh is None:
            kh = _khach(args)
            if kh is None:
                print("thiếu GEMINI_API_KEY")
                return 2
        g = gt[i]
        raw, model = _goi(kh[0], kh[1], kh[2],
                          _PROMPT_TACH.format(vi=g["kis_query_vi"], en=g.get("kis_query_en") or ""))
        raw["model"] = model
        f.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
        tach[i] = raw

    print("=== Hạng NỘI-VIDEO của keyframe đáp án (nhỏ hơn = tốt hơn) ===")
    print("thăm dò, n=6, cửa sổ ĐỐI XỨNG, giả thuyết 'phủ sub-query' — KHÔNG phải cặp thời gian\n")
    cot = [f"W={w}" for w in LUOI_W]
    print(f"{'câu':>4}{'#kf':>6}{'SigLIP thô':>12}" + "".join(f"{c:>8}" for c in cot))
    ghi = []
    for i in CAU_NGHEN:
        g = gt[i]
        v = g["video_id"]
        frames, rows = truc[v]
        # keyframe gần đáp án nhất (4/60 câu GT có frame_idx không phải keyframe)
        p_dap = int(np.argmin(np.abs(frames - int(g["frame_idx"]))))
        sA = kho.lay(tach[i]["A_vi"], tach[i].get("A_en", ""))
        sB = kho.lay(tach[i]["B_vi"], tach[i].get("B_en", ""))
        # nền so sánh: chính truy vấn gốc, cùng đường mã 4-prompt
        tho = kho.lay(g["kis_query_vi"], g.get("kis_query_en") or "")
        hang_tho = int((tho[rows] > tho[rows][p_dap]).sum()) + 1

        # ứng viên giả: MỌI keyframe của chính video đó (đo định vị nội-video)
        gia = [Candidate(v, int(fr), 0.0, int(frames[-1])) for fr in frames]
        hangs = []
        for W in LUOI_W:
            s, _ = diem_cap_thoi_gian(gia, sA, sB, truc, W, args.gop, doi_xung=True)
            hangs.append(int((s > s[p_dap]).sum()) + 1)
        print(f"{i:>4}{len(frames):>6}{hang_tho:>12}" + "".join(f"{h:>8}" for h in hangs))
        ghi.append({"cau": i, "n_kf": len(frames), "hang_siglip_tho": hang_tho,
                    "hang_theo_W": dict(zip([str(w) for w in LUOI_W], hangs)),
                    "A": tach[i]["A_vi"], "B": tach[i]["B_vi"]})

    f = CACHE / f"nghen_{args.gop}.json"
    f.write_text(json.dumps(ghi, ensure_ascii=False, indent=1), encoding="utf-8")
    tot = sum(1 for r in ghi if min(r["hang_theo_W"].values()) < r["hang_siglip_tho"])
    print(f"\ncải thiện ở {tot}/6 câu (so hạng SigLIP thô, lấy W tốt nhất — đây là CẬN TRÊN")
    print("kiểu oracle vì W được chọn sau khi nhìn kết quả; không phải con số ship được).")
    print(f"cache: {f}")
    return 0


def cmd_tu_kiem(args) -> int:
    """Kiểm bất biến khi cổng THẬT SỰ bật ở một số câu.

    Trên 60 câu GT không câu nào qua cổng, nên phép assert trong ``cmd_gt`` là
    đúng nhưng RỖNG: nó chỉ chứng minh mã không chạy.  Ở đây ta ép bật cổng cho
    đúng 6 câu (dùng bản tách mệnh đề đã cache) và đòi hai điều CÙNG LÚC:

      1. 54 câu cổng TẮT ra 100 dòng giống hệt nền, từng dòng một;
      2. các câu cổng BẬT thật sự có dòng ĐỔI — nếu không thì cổng đang là
         no-op và mọi kết quả "hoà" sau này sẽ là hoà giả.
    """
    cache_dir = Path(args.cache_pxs)
    gt, cands_of, _kf = nap_ung_vien(args.data, cache_dir, False)
    truc = nap_truc_video(Path(args.data), args.refresh)
    kho = KhoSims(args.data, args.refresh)
    tach_dir = CACHE / "tach_menh_de"

    ep = {}
    for i in CAU_NGHEN:
        f = tach_dir / f"{i:02d}.json"
        if not f.exists():
            print("  ! chưa có bản tách mệnh đề — chạy --nghen trước")
            return 2
        ep[i] = json.loads(f.read_text(encoding="utf-8"))

    nhan_ep = [{"co_2_canh": i in ep,
                "canh_A_vi": ep[i]["A_vi"] if i in ep else "",
                "canh_A_en": ep[i].get("A_en", "") if i in ep else "",
                "canh_B_vi": ep[i]["B_vi"] if i in ep else "",
                "canh_B_en": ep[i].get("B_en", "") if i in ep else ""}
               for i in range(len(gt))]
    sA = {i: kho.lay(ep[i]["A_vi"], ep[i].get("A_en", "")) for i in ep}
    sB = {i: kho.lay(ep[i]["B_vi"], ep[i].get("B_en", "")) for i in ep}

    rows_nen = [allocate_rows(c, args.allocator, DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
                for c in cands_of]
    print(f"ép bật cổng ở {len(ep)} câu: {sorted(ep)}")
    print(f"{'W':>3} {'gộp':>5} {'λ':>5}  {'54 câu tắt giống nền':>22}  {'6 câu bật có đổi':>18}")
    ok = True
    for W in LUOI_W:
        for gop in LUOI_GOP:
            for lam in LUOI_LAMBDA:
                rows = ap_dung(cands_of, nhan_ep, sA, sB, W, gop, lam,
                               args.allocator, truc=truc)
                giong_tat = all(rows[i] == rows_nen[i]
                                for i in range(len(gt)) if i not in ep)
                doi_bat = sum(1 for i in ep if rows[i] != rows_nen[i])
                ok &= giong_tat and doi_bat == len(ep)
                print(f"{W:>3} {gop:>5} {lam:>5g}  {'OK' if giong_tat else 'HỎNG':>22}  "
                      f"{f'{doi_bat}/{len(ep)}':>18}")
    print("\n" + ("TỰ KIỂM XANH: cổng bật đổi dòng, cổng tắt bất động."
                  if ok else "TỰ KIỂM ĐỎ — xem dòng HỎNG ở trên."))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--cache-pxs", default=str(ROOT / "data" / "cache_phu_quet_luoi"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--allocator", default="coverage", choices=("coverage", "hybrid"))
    ap.add_argument("--tune-seeds", type=int, default=3)
    ap.add_argument("--tune-draws", type=int, default=32)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--test-draws", type=int, default=48)
    ap.add_argument("--tune-phia", choices=("chan", "le"), default="chan")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--de", nargs="+", default=None, help="thư mục đề thật -> diff cấu trúc")
    ap.add_argument("--picks", default=None,
                    help="file picks của người soát -> thêm bảng proxy (KHÔNG phải ground truth)")
    ap.add_argument("--gt-de", default=None,
                    help="data/ground_truth_de_that.json -> chỉ lấy mục nguoi_kiem_chung")
    ap.add_argument("--nghen", action="store_true", help="chẩn đoán 6 câu nghẽn")
    ap.add_argument("--tu-kiem", action="store_true",
                    help="ép bật cổng ở 6 câu để kiểm bất biến thật sự (không rỗng)")
    ap.add_argument("--W", default="5", help="một giá trị, hoặc danh sách '2,3,5,8' (chế độ --de)")
    ap.add_argument("--gop", default="hm", choices=LUOI_GOP)
    ap.add_argument("--lam", default="1.0", help="một giá trị, hoặc danh sách '1.0,0.5'")
    args = ap.parse_args()

    if args.tu_kiem:
        return cmd_tu_kiem(args)
    if args.nghen:
        return cmd_nghen(args)
    if args.de:
        return cmd_de(args)
    return cmd_gt(args)


if __name__ == "__main__":
    raise SystemExit(main())
