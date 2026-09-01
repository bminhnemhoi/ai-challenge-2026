"""Mở lại họ tín hiệu ĐỊNH VỊ NỘI-VIDEO trên bộ đo khớp phân bố đề thật.

Vì sao phải mở lại
------------------
Bảng tín hiệu trong ``docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md`` đóng cửa cả họ này:
làm mượt theo thời gian −0,023, chuẩn hoá theo video −0,129, ép khoảng cách
frame −1,3%, pha CLIP-B32 "không thêm thông tin". **Tất cả đo trên bộ 60 câu
cũ**, mà bộ đó có đặc điểm chết người: SigLIP đã đặt keyframe gần đáp án ở hạng
nội-video trung vị 1,0 (hạng-1 60%). Không còn headroom thì mọi tín hiệu định vị
nội-video đều phải đo ra ~0 — cánh cửa bị đóng bằng một thước không nhìn thấy
vấn đề.

Trên bộ đo khớp phân bố (132 mục sạch), oracle định vị nội-video cho
**+126%** (`scripts/tran_dinh_vi_noi_video.py`). Headroom là có thật.

Đo cái gì
---------
Bốn họ tín hiệu, **mỗi họ một cửa riêng, không gộp**:

  (a) làm mượt Gauss theo trục thời gian trong CÙNG video — σ ∈ {1,2,3,5}
      keyframe, trọng số trộn w ∈ {0,25; 0,5; 1,0};
  (b) chuẩn hoá điểm theo video — trừ trung bình video / z-score nội video /
      min-max nội video, hệ số trộn λ ∈ {0,25; 0,5; 1,0};
  (c) **quét lại** cường độ ưu tiên đỉnh cục bộ đang dùng trong sản xuất
      (``PEAK_WEIGHT = 0,01``, chưa bao giờ quét lại từ khi đổi bộ phân bổ);
  (d) **biên cảnh** — keyframe cao hơn hẳn keyframe TRƯỚC nó trong cùng video
      (đạo hàm bậc một của chuỗi điểm). Khoảnh khắc phải nộp thường là lúc một
      cảnh BẮT ĐẦU. Chưa ai thử tín hiệu này.

Cộng thêm **HAI họ ĐỐI CHỨNG bắt buộc**, cả hai đều KHÔNG đổi một thứ tự nào và
KHÔNG đụng vào truy xuất — chúng chỉ chỉnh hai núm của bộ phân bổ:

  (e) kéo giãn thang điểm quanh trung bình toàn kho. Bộ phủ biến điểm thành khối
      lượng qua ``exp((s − max)/0,02)``, nên **nới/thu độ tán của điểm chính là
      một phép đổi nhiệt độ trá hình**. Nếu (e) một mình đã làm điểm nhúc nhích
      thì mọi con số của (a),(b),(d) phải trừ đi phần đó trước khi đọc.
  (f) nới ``sigma`` của tiên nghiệm phủ (mặc định 30 frame — **hẹp hơn một khe
      keyframe**, trung vị 55). Làm mượt theo thời gian cũng nới rộng vùng được
      phủ; nếu chỉ vặn núm này đã mua được phần điểm của (a) thì (a) không thêm
      thông tin nào, nó chỉ đang sửa hộ một tham số bộ phân bổ.

Không có hai đối chứng này thì cả bảng vô nghĩa: một "tín hiệu" chỉ làm phẳng
hoặc làm nhọn tiên nghiệm sẽ được ghi công là tri thức mới.

Đo trên ĐÚNG đường sản xuất
---------------------------
sims đầy đủ (``query_similarities``, có cắt khúc câu dài y hệt ``search``)
→ mặt nạ ``valid`` → top-400 → ưu tiên đỉnh → nhận dạng đối tượng
→ ``allocate_rows(coverage)`` → 100 dòng → ``final_score``/``r_score_kis``.

Bất biến kiểm bằng ``assert``, không bằng mắt:
  * nền dựng lại từ sims phải **trùng từng dòng** với ứng viên sản xuất đã cache
    (``data/cache_bo_do_moi/uv_moi.json``) — cả video, cả frame, cả điểm (< 1e-9);
  * cấu hình đơn vị của MỌI họ (w=0 / λ=0 / β=0 / α=1 / σ_phủ=30) phải cho ra
    100 dòng **giống hệt nền** trên cả 132 mục;
  * bản vector hoá của từng tín hiệu phải khớp một bản lặp từng video viết thẳng
    thừng (< 1e-5) — họ (a) và (d) đứng trên đúng một mẹo dịch mảng + mặt nạ biên
    video, mà bất biến thứ hai không chứng minh được mẹo đó đúng.

Kết quả: ``docs/TIN_HIEU_NOI_VIDEO.md``.

    python -u scripts/do_tin_hieu_noi_video.py --sims     # lần đầu: dựng cache sims
    python -u scripts/do_tin_hieu_noi_video.py            # toàn bộ (a)-(f)
    python -u scripts/do_tin_hieu_noi_video.py --ho f     # chỉ một họ
    python -u scripts/do_tin_hieu_noi_video.py --nhanh    # chạy thử, mỗi họ 2 cấu hình
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
    cac_lan_boc,
    cham_nhanh,
    ma_tran_dong,
)
from scripts.make_submission import (  # noqa: E402
    DEFAULT_DEPTH_COST,
    DEFAULT_N_FLAT,
    PEAK_WEIGHT,
    RETRIEVE_TOP_N,
    _object_boost,
    allocate_rows,
)
from src.core.kis_engine import Hit  # noqa: E402
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    CoveragePlan,
    allocate_coverage_rows,
)

#: gốc hạt MỚI, tách khỏi mọi gốc đã dùng (30000/50000/61000/62000/70000/77000/90000/123450)
GOC_TUNE = 81000
GOC_TEST = 82000

CACHE = ROOT / "data" / "cache_tin_hieu_noi_video"


# ---------------------------------------------------------------------------
# Trục thời gian: mọi phép "keyframe lân cận trong cùng video" là một phép dịch
# mảng, nếu ta sắp lại chỉ mục theo (video, frame_idx) một lần duy nhất.
# ---------------------------------------------------------------------------


class Truc:
    """Chỉ mục 177k keyframe sắp theo (video, frame) + mặt nạ dịch tiền tính."""

    def __init__(self, vid: np.ndarray, frm: np.ndarray):
        ma = {v: i for i, v in enumerate(dict.fromkeys(vid.tolist()))}
        code = np.array([ma[v] for v in vid.tolist()], dtype=np.int64)
        self.order = np.lexsort((frm, code))          # chỉ số gốc, theo thứ tự trục
        self.nguoc = np.empty_like(self.order)
        self.nguoc[self.order] = np.arange(self.order.size)
        self.seg = code[self.order]                    # id video ở toạ độ trục
        doi = np.flatnonzero(np.diff(self.seg)) + 1
        self.dau = np.concatenate(([0], doi))          # vị trí bắt đầu mỗi video
        self.dai = np.diff(np.concatenate((self.dau, [self.seg.size]))).astype(np.float64)
        self._hop_le: dict[int, np.ndarray] = {}

    @staticmethod
    def _dich(x: np.ndarray, k: int) -> np.ndarray:
        """y[i] = x[i+k], đệm 0 ở mép."""
        if k == 0:
            return x
        y = np.zeros_like(x)
        if k > 0:
            y[:-k] = x[k:]
        else:
            y[-k:] = x[:k]
        return y

    def hop_le(self, k: int) -> np.ndarray:
        """True ở i nếu i+k còn nằm trong CÙNG video (tiền tính, dùng lại mọi câu)."""
        if k not in self._hop_le:
            if k == 0:
                m = np.ones(self.seg.size, dtype=bool)
            else:
                s = self._dich(self.seg, k)
                m = s == self.seg
                if k > 0:
                    m[-k:] = False
                else:
                    m[:-k] = False
            self._hop_le[k] = m
        return self._hop_le[k]

    def theo_video(self, x: np.ndarray):
        """(mu_v, sd_v, min_v, max_v) trải lại về từng phần tử của trục."""
        n = self.dai
        tong = np.add.reduceat(x, self.dau)
        mu = tong / n
        binh = np.add.reduceat(x * x, self.dau) / n
        sd = np.sqrt(np.maximum(binh - mu * mu, 1e-12))
        lo = np.minimum.reduceat(x, self.dau)
        hi = np.maximum.reduceat(x, self.dau)
        rep = self.dai.astype(np.int64)
        return (np.repeat(mu, rep), np.repeat(sd, rep),
                np.repeat(lo, rep), np.repeat(hi, rep))


# ---------------------------------------------------------------------------
# Bốn họ tín hiệu + họ đối chứng.  Mỗi hàm: sims (toạ độ gốc) -> sims mới.
# Tất cả giữ THANG COSINE, để độ tán của điểm không lén trở thành phép đổi
# nhiệt độ của bộ phân bổ (xem họ (e)).
# ---------------------------------------------------------------------------


def lam_muot(s, truc: Truc, sigma: float, w: float):
    """(a) trung bình có trọng số Gauss trên các keyframe lân cận CÙNG video."""
    if w == 0 or sigma <= 0:
        return s
    x = s[truc.order]
    R = int(max(1, round(2.5 * sigma)))
    acc = np.zeros_like(x)
    can = np.zeros_like(x)
    for k in range(-R, R + 1):
        wk = float(np.exp(-0.5 * (k / sigma) ** 2))
        m = truc.hop_le(k)
        acc += wk * Truc._dich(x, k) * m
        can += wk * m
    muot = acc / np.maximum(can, 1e-12)
    ra = (1.0 - w) * x + w * muot
    return ra[truc.nguoc]


def chuan_hoa(s, truc: Truc, kieu: str, lam: float):
    """(b) chuẩn hoá theo video, ĐƯA VỀ LẠI thang cosine trước khi trộn.

    Cả ba kiểu đều là phép affine **hệ số dương trong từng video**, nên chúng
    KHÔNG đổi được thứ tự nội-video một chút nào — chúng chỉ đổi việc so giữa
    các video và độ tán của điểm trong video.  Ghi rõ ở đây vì đó chính là lý do
    họ này không thể là một bộ định vị, dù tên gọi nghe như vậy.
    """
    if lam == 0:
        return s
    x = s[truc.order]
    mu_v, sd_v, lo_v, hi_v = truc.theo_video(x)
    mu_g = float(x.mean())
    sd_g = float(x.std())
    if kieu == "tru_tb":                      # s - λ(mu_v - mu_toan_kho)
        ra = x - lam * (mu_v - mu_g)
    elif kieu == "zscore":
        t = mu_g + sd_g * (x - mu_v) / sd_v
        ra = (1.0 - lam) * x + lam * t
    elif kieu == "minmax":
        r = (x - lo_v) / np.maximum(hi_v - lo_v, 1e-12)
        t = mu_g + sd_g * (r - float(r.mean())) / max(float(r.std()), 1e-12)
        ra = (1.0 - lam) * x + lam * t
    else:
        raise ValueError(kieu)
    return ra[truc.nguoc]


def bien_canh(s, truc: Truc, k: int, beta: float):
    """(d) keyframe cao hơn hẳn k keyframe TRƯỚC nó => "một cảnh vừa bắt đầu"."""
    if beta == 0 or k <= 0:
        return s
    x = s[truc.order]
    acc = np.zeros_like(x)
    can = np.zeros_like(x)
    for j in range(1, k + 1):
        m = truc.hop_le(-j)
        acc += Truc._dich(x, -j) * m
        can += m
    truoc = acc / np.maximum(can, 1e-12)
    bump = np.where(can > 0, np.maximum(x - truoc, 0.0), 0.0)
    return (x + beta * bump)[truc.nguoc]


def keo_gian(s, alpha: float):
    """(e) ĐỐI CHỨNG: chỉ nới/thu độ tán quanh trung bình, thứ tự không đổi.

    Tương đương đúng một phép đổi nhiệt độ softmax của bộ phủ (0,02 -> 0,02/α).
    """
    if alpha == 1.0:
        return s
    mu = float(s.mean())
    return mu + alpha * (s - mu)


# ---------------------------------------------------------------------------
# Đường sản xuất, dựng lại từ sims (đã đối chiếu trùng khít cache ứng viên)
# ---------------------------------------------------------------------------


class Vo:
    """Vỏ nhẹ thay KISEngine cho hai bước xếp lại — không nạp model, không mmap."""

    def __init__(self, data_dir: Path, meta: list):
        self.data_dir = data_dir
        self.metadata = meta


def uu_tien_dinh(eng, hits, w: float):
    """Bản ``make_submission._peak_preference`` có tham số cường độ.

    Ở w = PEAK_WEIGHT nó PHẢI là bản sản xuất từng chữ một — và điều đó được
    chứng minh chứ không phải tin: nền dựng bằng hàm này trùng khít tới 1e-9 với
    ứng viên do ``ranked_hits`` thật sinh ra, trên cả 132 mục (bước 1 của main).
    """
    if not hits or w <= 0:
        return list(hits)
    timeline = getattr(eng, "_kf_timeline", None)
    if timeline is None:
        timeline = {}
        for m in eng.metadata:
            timeline.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        timeline = {v: np.array(sorted(f)) for v, f in timeline.items()}
        eng._kf_timeline = timeline
    seen: dict = {}
    for h in hits:
        seen.setdefault(h.video_id, {})[int(h.frame_idx)] = h.score
    scored = []
    for i, h in enumerate(hits):
        arr = timeline.get(h.video_id)
        bump = 0.0
        if arr is not None and len(arr) > 1:
            j = int(np.searchsorted(arr, h.frame_idx))
            near = [seen[h.video_id].get(int(arr[k])) for k in (j - 1, j + 1)
                    if 0 <= k < len(arr)]
            near = [x for x in near if x is not None]
            if near:
                bump = max(h.score - max(near), 0.0)
        scored.append((h.score + w * bump / max(abs(h.score), 1e-6), i, h))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [t[2] for t in scored]


class Duong:
    """sims -> 400 ứng viên đường sản xuất -> 100 dòng."""

    def __init__(self, data_dir: Path, meta: list, valid: np.ndarray):
        self.vo = Vo(data_dir, meta)
        self.valid = valid
        self.vid = np.array([m["video_id"] for m in meta], dtype=object)
        self.frm = np.array([m["frame_idx"] for m in meta], dtype=np.int64)
        self.n = np.array([m.get("n", 1) for m in meta], dtype=np.int64)
        self.pts = np.array([m.get("pts_time", 0.0) for m in meta], dtype=np.float32)
        self.cuoi: dict = {}
        for v, f in zip(self.vid, self.frm):
            if f > self.cuoi.get(v, -1):
                self.cuoi[v] = int(f)
        self.plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)

    def ung_vien(self, sims, query_en: str, w_dinh: float = PEAK_WEIGHT,
                 top_n: int = RETRIEVE_TOP_N):
        s = np.where(self.valid, sims, -np.inf)
        n = int(min(top_n, np.isfinite(s).sum()))
        top = np.argpartition(-s, n - 1)[:n]
        top = top[np.argsort(-s[top])]
        hits = [Hit(video_id=str(self.vid[i]), frame_idx=int(self.frm[i]),
                    score=float(s[i]), n=int(self.n[i]), pts_time=float(self.pts[i]),
                    video_last_frame=self.cuoi[str(self.vid[i])]) for i in top]
        hits = uu_tien_dinh(self.vo, hits, w_dinh)
        return _object_boost(self.vo, hits, query_en)

    def dong(self, hits, sigma=None):
        """100 dong, dung ``allocate_rows(coverage)``.

        ``sigma`` chi dung cho ho doi chung (f): noi be rong Gauss ma bo phu rai
        quanh moi keyframe, KHONG dung vao diem va KHONG doi thu tu ung vien nao.
        ``sigma=None`` di dung duong san xuat (CoveragePlan mac dinh, sigma=30).
        """
        c = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits]
        if sigma is None:
            return allocate_rows(c, "coverage", DEFAULT_N_FLAT, self.plan)[:MAX_ROWS]
        return allocate_coverage_rows(
            c, plan=CoveragePlan(budget=self.plan.budget, sigma=float(sigma)),
            tail_n_flat=DEFAULT_N_FLAT, tail_plan=self.plan)[:MAX_ROWS]


# ---------------------------------------------------------------------------
# Bước 1 (đắt, chạy một lần): sims đầy đủ cho 132 câu sạch
# ---------------------------------------------------------------------------


def dung_sims(data_dir: Path, sach: list, ra: Path, khoi: int = 30000) -> None:
    """sims đầy đủ cho từng câu, ĐÚNG ngữ nghĩa ``query_similarities``.

    Khác một điểm duy nhất so với gọi thẳng ``query_similarities`` 132 lần:
    thứ tự hai vòng lặp bị đảo. Gọi thẳng thì mỗi câu quét lại toàn bộ 817 MB
    chỉ mục — 132 lượt đọc đĩa, và trên máy đang chạy nhiều lane song song thì
    trang nhớ bị đẩy ra giữa chừng nên nó **chậm gấp mười**. Ở đây chỉ mục được
    quét **một lượt duy nhất**, mỗi khối nhân với mọi vector truy vấn.

    Từng phần tử kết quả vẫn là đúng phép tích vô hướng của đúng hàng với đúng
    vector, nên phép đối chiếu ở bước 1) của ``main`` (trùng khít từng dòng và
    từng điểm với ứng viên sản xuất đã cache) vẫn là phép kiểm thật.
    """
    from src.core.kis_engine import KISEngine

    print("nap chi muc SigLIP + model (chi lan nay) ...", flush=True)
    eng = KISEngine(str(data_dir)).load()
    T = eng.embeddings.shape[0]
    ra.mkdir(parents=True, exist_ok=True)
    np.save(ra / "valid.npy", np.asarray(eng.valid))

    print("  1/2 vector truy van (can model) ...", flush=True)
    t0 = time.time()
    vecs = []
    for i, g in enumerate(sach):
        vi = g["kis_query_vi"]
        en = g.get("kis_query_en") or eng.translate(vi)
        khuc = eng.chunk_text(en)
        if len(khuc) <= 1:
            vs = [eng.query_vector(vi, en)]
        else:
            vs = [eng.query_vector(vi, khuc[0])] + [eng.query_vector(c, c) for c in khuc[1:]]
        vecs.append(np.stack(vs).astype(np.float32))
        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{len(sach)}  ({time.time()-t0:.0f}s)", flush=True)
    try:  # bỏ model khỏi RAM trước khi quét chỉ mục (máy đang chạy nhiều lane)
        eng.model = None
        eng._model = None
    except Exception:  # noqa: BLE001
        pass
    print(f"  so khuc/cau: min {min(len(v) for v in vecs)}, "
          f"max {max(len(v) for v in vecs)}", flush=True)

    print("  2/2 mot luot duy nhat qua chi muc ...", flush=True)
    mm = np.lib.format.open_memmap(ra / "sims_sach.npy", mode="w+",
                                   dtype=np.float32, shape=(len(sach), T))
    t1 = time.time()
    for a in range(0, T, khoi):
        b = min(a + khoi, T)
        E = np.asarray(eng.embeddings[a:b], dtype=np.float32)
        for i, V in enumerate(vecs):
            acc = E @ V[0]
            for v in V[1:]:
                acc = acc + (E @ v)
            mm[i, a:b] = acc / len(V)
        print(f"    hang {b}/{T}  ({time.time()-t1:.0f}s)", flush=True)
    mm.flush()
    del mm, eng
    print(f"xong sims sau {time.time()-t0:.0f}s -> {ra/'sims_sach.npy'}")


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--uv", default=str(ROOT / "data" / "cache_bo_do_moi" / "uv_moi.json"))
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--sims", action="store_true", help="dung lai cache sims (can model)")
    ap.add_argument("--ho", default="",
                    help="chi chay cac ho co ten bat dau bang chu nay, vd --ho f")
    ap.add_argument("--nhanh", action="store_true",
                    help="chay thu: moi ho 2 cau hinh, chi de bat loi code")
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--tune-seeds", type=int, default=3)
    ap.add_argument("--tune-draws", type=int, default=32)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--test-draws", type=int, default=48)
    args = ap.parse_args()

    data = Path(args.data)
    cache = Path(args.cache)
    windows = [int(w) for w in args.windows.split(",")]

    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads(Path(args.uv).read_text(encoding="utf-8"))
    assert len(uv) == len(moi), "cache ung vien lech so muc"
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    sach = [moi[i] for i in giu]
    uv_sach = [uv[i] for i in giu]
    hai = [i for i, g in enumerate(sach) if g.get("co_2_canh")]
    hai_set = set(hai)
    mot = [i for i in range(len(sach)) if i not in hai_set]
    print(f"bo SACH {len(sach)} muc | HAI canh {len(hai)} | MOT canh {len(mot)}")

    if args.sims or not (cache / "sims_sach.npy").exists():
        dung_sims(data, sach, cache)

    print("nap sims + metadata (khong nap model) ...", flush=True)
    sims = np.load(cache / "sims_sach.npy", mmap_mode="r")
    valid = np.load(cache / "valid.npy")
    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    assert sims.shape == (len(sach), len(meta)), "sims lech kich thuoc"

    duong = Duong(data, meta, valid)
    truc = Truc(duong.vid, duong.frm)
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in
          _gom_kf(meta).items()}

    # ---- 1) Bất biến: nền dựng lại từ sims phải trùng khít cache sản xuất ----
    print("\n=== 1) DOI CHIEU: nen dung lai tu sims vs ung vien san xuat da cache ===")
    hits_nen = []
    for qi, g in enumerate(sach):
        h = duong.ung_vien(np.asarray(sims[qi]), g.get("kis_query_en") or "")
        hits_nen.append(h)
        mine = [(x.video_id, int(x.frame_idx)) for x in h]
        ref = [(v, int(f)) for v, f, _s, _l in uv_sach[qi]]
        assert mine == ref, f"muc {qi}: duong dung lai LECH cache san xuat"
        assert all(abs(float(x.score) - float(r[2])) < 1e-9
                   for x, r in zip(h, uv_sach[qi])), f"muc {qi}: diem lech"
    print(f"  OK: {len(sach)}/{len(sach)} muc trung khit tung dong VA tung diem.")

    rows_nen = [duong.dong(h) for h in hits_nen]

    # ---- 2) Bất biến: cấu hình đơn vị của mọi họ == nền -----------------------
    print("\n=== 2) BAT BIEN: cau hinh don vi cua moi ho phai == nen ===")
    don_vi = [("a lam muot  w=0", lambda s: lam_muot(s, truc, 3, 0.0), PEAK_WEIGHT, None),
              ("b chuan hoa lam=0", lambda s: chuan_hoa(s, truc, "zscore", 0.0),
               PEAK_WEIGHT, None),
              ("c uu tien dinh w=0,01 (san xuat)", lambda s: s, PEAK_WEIGHT, None),
              ("d bien canh beta=0", lambda s: bien_canh(s, truc, 1, 0.0), PEAK_WEIGHT, None),
              ("e keo gian alpha=1", lambda s: keo_gian(s, 1.0), PEAK_WEIGHT, None),
              ("f sigma bo phu = 30 (mac dinh)", lambda s: s, PEAK_WEIGHT, 30.0)]
    for ten, fn, wd, sg in don_vi:
        for qi, g in enumerate(sach):
            r = duong.dong(duong.ung_vien(fn(np.asarray(sims[qi])),
                                          g.get("kis_query_en") or "", wd), sg)
            assert r == rows_nen[qi], f"{ten}: muc {qi} doi dong du la phep dong nhat"
        print(f"  OK  {ten}")

    # ---- 2b) Bản vector hoá của TÍN HIỆU vs bản chậm hiển nhiên đúng ----------
    # Bất biến ở (2) chỉ chứng minh mã ngắn mạch ở tham số 0, không chứng minh
    # phép dịch mảng + mặt nạ biên video là đúng.  Cả họ (a) và (d) đứng trên
    # đúng mẹo đó, nên nó phải được đối chiếu với một bản lặp từng video.
    print("\n=== 2b) DOI CHIEU tin hieu: ban vector hoa vs ban lap tung video ===")
    for qi in (0, len(sach) // 2, len(sach) - 1):
        s0 = np.asarray(sims[qi], dtype=np.float32)
        for ten, nhanh, cham_tay in (
            ("a sigma=2 w=0,5", lam_muot(s0, truc, 2, 0.5), _tc_muot(s0, duong, 2, 0.5)),
            ("a sigma=5 w=1,0", lam_muot(s0, truc, 5, 1.0), _tc_muot(s0, duong, 5, 1.0)),
            ("b zscore lam=1,0", chuan_hoa(s0, truc, "zscore", 1.0),
             _tc_chuan(s0, duong, "zscore", 1.0)),
            ("b minmax lam=0,5", chuan_hoa(s0, truc, "minmax", 0.5),
             _tc_chuan(s0, duong, "minmax", 0.5)),
            ("b tru_tb lam=1,0", chuan_hoa(s0, truc, "tru_tb", 1.0),
             _tc_chuan(s0, duong, "tru_tb", 1.0)),
            ("d k=3 beta=0,5", bien_canh(s0, truc, 3, 0.5), _tc_bien(s0, duong, 3, 0.5)),
        ):
            lech = float(np.max(np.abs(np.asarray(nhanh, dtype=np.float64) - cham_tay)))
            assert lech < 1e-5, f"muc {qi} {ten}: ban vector hoa lech {lech:.2e}"
        print(f"  OK  muc {qi}: 6 tin hieu khop ban cham (lech toi da < 1e-5)")

    # ---- 3) Chia TUNE/TEST PHÂN TẦNG theo nhóm hai cảnh -----------------------
    i_tune = sorted(hai[0::2] + mot[0::2])
    i_test = sorted(hai[1::2] + mot[1::2])
    n_hai_tune = sum(1 for i in i_tune if i in hai_set)
    n_hai_test = sum(1 for i in i_test if i in hai_set)
    print(f"\nTUNE {len(i_tune)} muc ({n_hai_tune} hai canh) | "
          f"TEST {len(i_test)} muc ({n_hai_test} hai canh)  [phan tang, khong chan/le tho]")
    ho_tune = cac_lan_boc(GOC_TUNE, args.tune_seeds, args.tune_draws,
                          [sach[i] for i in i_tune], kf)
    ho_test = cac_lan_boc(GOC_TEST, args.test_seeds, args.test_draws,
                          [sach[i] for i in i_test], kf)

    def dong_cua(cfg, idx):
        fn, wd, sg = cfg
        return {i: duong.dong(duong.ung_vien(fn(np.asarray(sims[i])),
                                             sach[i].get("kis_query_en") or "", wd), sg)
                for i in idx}

    def cham(rows_map, idx, ho):
        gt_s = [sach[i] for i in idx]
        mats = ma_tran_dong([rows_map[i] for i in idx], gt_s)
        return [cham_nhanh(mats, d, windows) for d in ho]

    def tung_cau(rows_map, idx, ho):
        gt_s = [sach[i] for i in idx]
        mats = ma_tran_dong([rows_map[i] for i in idx], gt_s)
        ra = np.zeros(len(idx))
        for draws in ho:
            for q in range(len(idx)):
                ra[q] += cham_nhanh([mats[q]], [draws[q]], windows)
        return ra / len(ho)

    # ---- 4) Quét trên TUNE ----------------------------------------------------
    ho_cau = {
        "a lam muot Gauss": [(f"sigma={sg} w={w}", (lambda s, sg=sg, w=w:
                              lam_muot(s, truc, sg, w)), PEAK_WEIGHT, None)
                             for sg in (1, 2, 3, 5) for w in (0.25, 0.5, 1.0)],
        "b chuan hoa theo video": [(f"{k} lam={l}", (lambda s, k=k, l=l:
                                    chuan_hoa(s, truc, k, l)), PEAK_WEIGHT, None)
                                   for k in ("tru_tb", "zscore", "minmax")
                                   for l in (0.25, 0.5, 1.0)],
        "c uu tien dinh cuc bo": [(f"w={w}", (lambda s: s), w, None)
                                  for w in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)],
        "d bien canh (dao ham bac 1)": [(f"k={k} beta={b}", (lambda s, k=k, b=b:
                                         bien_canh(s, truc, k, b)), PEAK_WEIGHT, None)
                                        for k in (1, 2, 3)
                                        for b in (0.05, 0.1, 0.25, 0.5, 1.0)],
        "e DOI CHUNG keo gian thang diem": [(f"alpha={a}", (lambda s, a=a:
                                             keo_gian(s, a)), PEAK_WEIGHT, None)
                                            for a in (0.5, 0.75, 1.5, 2.0)],
        # (f) doi chung thu hai, do chinh ket qua cua (a) doi: lam muot theo thoi
        # gian noi rong vung ma tien nghiem phu.  Bo phu DA co mot tham so lam
        # dung viec do (sigma=30 frame, hep hon mot khe keyframe - trung vi 55).
        # Neu chi noi sigma da mua duoc phan diem cua (a) thi (a) khong them
        # thong tin nao, no chi chinh sai mot tham so bo phan bo.
        "f DOI CHUNG noi sigma bo phu": [(f"sigma_phu={sg}", (lambda s: s), PEAK_WEIGHT, sg)
                                         for sg in (60.0, 120.0, 240.0, 480.0)],
    }

    if args.ho:
        ho_cau = {k: v for k, v in ho_cau.items() if k[0] in set(args.ho)}
        assert ho_cau, f"--ho {args.ho} khong khop ho nao"
        print(f"\n!! CHI CHAY HO: {list(ho_cau)}")
    if args.nhanh:
        ho_cau = {k: v[:2] for k, v in ho_cau.items()}
        print("\n!! CHE DO --nhanh: moi ho chi 2 cau hinh, so lieu KHONG dung de ket luan")

    rows_nen_tune = {i: rows_nen[i] for i in i_tune}
    nen_t = cham(rows_nen_tune, i_tune, ho_tune)
    m_nen = float(np.mean(nen_t))
    hai_tune = [i for i in i_tune if i in hai_set]
    mot_tune = [i for i in i_tune if i not in hai_set]
    ho_tune_hai = [[dr[k] for k, i in enumerate(i_tune) if i in hai_set] for dr in ho_tune]
    ho_tune_mot = [[dr[k] for k, i in enumerate(i_tune) if i not in hai_set] for dr in ho_tune]
    nen_h = float(np.mean(cham(rows_nen_tune, hai_tune, ho_tune_hai)))
    nen_m = float(np.mean(cham(rows_nen_tune, mot_tune, ho_tune_mot)))
    print(f"\nNEN tren TUNE: {m_nen:.4f} +-{np.std(nen_t):.4f}   "
          f"({len(hai_tune)} cau HAI canh: {nen_h:.4f} | "
          f"{len(mot_tune)} cau MOT canh: {nen_m:.4f})")

    # Hai nhóm đi NGƯỢC chiều nhau ở họ (a) ngay từ lần chạy thử, nên chốt theo
    # điểm TỔNG một mình sẽ luôn chọn cấu hình ít hại nhất cho nhóm dễ và không
    # bao giờ đọc được cấu hình tốt nhất cho nhóm khó — đúng nhóm mà lane này
    # đi tìm.  Vì vậy MỖI HỌ chốt hai lần: theo điểm tổng, và theo điểm nhóm
    # HAI CẢNH.  Cả hai đều chốt trên TUNE, TEST vẫn chỉ đọc một lượt.
    thang = {}
    for ho, cfgs in ho_cau.items():
        print(f"\n--- {ho} ---")
        print(f"{'cau hinh':<22}{'TUNE':>9}{'+-':>8}{'so nen':>9}"
              f"{'HAI canh':>11}{'so nen':>9}{'MOT canh':>11}{'so nen':>9}")
        for ten, fn, wd, sg in cfgs:
            t0 = time.time()
            rm = dong_cua((fn, wd, sg), i_tune)
            d = cham(rm, i_tune, ho_tune)
            m = float(np.mean(d))
            mh = float(np.mean(cham(rm, hai_tune, ho_tune_hai)))
            mm = float(np.mean(cham(rm, mot_tune, ho_tune_mot)))
            thang[(ho, ten)] = (m, mh, fn, wd, sg)
            print(f"{ten:<22}{m:>9.4f}{np.std(d):>8.4f}{100*(m/m_nen-1):>+8.1f}%"
                  f"{mh:>11.4f}{100*(mh/nen_h-1):>+8.1f}%"
                  f"{mm:>11.4f}{100*(mm/nen_m-1):>+8.1f}%   [{time.time()-t0:.0f}s]",
                  flush=True)

    # ---- 5) Chốt trên TUNE, đọc TEST đúng một lần ------------------------------
    tot_tong, tot_hai = {}, {}
    for (ho, ten), (m, mh, fn, wd, sg) in thang.items():
        if ho not in tot_tong or m > tot_tong[ho][0]:
            tot_tong[ho] = (m, ten, fn, wd, sg)
        if ho not in tot_hai or mh > tot_hai[ho][0]:
            tot_hai[ho] = (mh, ten, fn, wd, sg)
    that = [k for k in tot_tong if not k.startswith(("e ", "f "))] or list(tot_tong)
    chinh = max((tot_tong[k][0], k) for k in that)[1]
    chinh_hai = max((tot_hai[k][0], k) for k in that)[1]
    print("\n=== CHOT TREN TUNE (moi ho hai lan: theo TONG va theo HAI CANH) ===")
    for ho in ho_cau:
        m, ten = tot_tong[ho][0], tot_tong[ho][1]
        mh, tenh = tot_hai[ho][0], tot_hai[ho][1]
        s1 = "  <== chot chinh (tong)" if ho == chinh else ""
        s2 = "  <== chot chinh (hai canh)" if ho == chinh_hai else ""
        print(f"  {ho:<34}tong  {ten:<18}{m:>9.4f}{100*(m/m_nen-1):>+8.1f}%{s1}")
        print(f"  {'':<34}hai   {tenh:<18}{mh:>9.4f}{100*(mh/nen_h-1):>+8.1f}%{s2}")

    doc_test = []
    for ho in ho_cau:
        _m, ten, fn, wd, sg = tot_tong[ho]
        doc_test.append((ho, "tong", ten, fn, wd, sg))
        _mh, tenh, fnh, wdh, sgh = tot_hai[ho]
        if tenh != ten:
            doc_test.append((ho, "hai canh", tenh, fnh, wdh, sgh))
    print(f"\nSe doc TEST cho {len(doc_test)} cau hinh, tat ca deu chot tren TUNE.")
    print("Nhieu phep so cung mot TEST => cau hinh THANG trong bang duoi bi thoi phong;")
    print("con so duy nhat duoc bao ve boi giao thuc la 'chot chinh (tong)'.")

    print("\n=== TEST (doc DUNG MOT LAN) ===")
    rows_nen_test = {i: rows_nen[i] for i in i_test}
    a_l = cham(rows_nen_test, i_test, ho_test)
    a = float(np.mean(a_l))
    dn = tung_cau(rows_nen_test, i_test, ho_test)
    print(f"  nen: {a:.4f} +-{np.std(a_l):.4f}")
    rng = np.random.default_rng(9191)
    vt_hai = [k for k, i in enumerate(i_test) if i in hai_set]
    vt_mot = [k for k, i in enumerate(i_test) if i not in hai_set]

    print(f"\n{'ho / chot theo':<40}{'cau hinh':<18}{'TEST':>9}{'so nen':>9}"
          f"{'KTC 95% theo cau':>26}{'P(<=0)':>9}")
    print("-" * 111)
    for ho, tieu_chi, ten, fn, wd, sg in doc_test:
        rm = dong_cua((fn, wd, sg), i_test)
        b = float(np.mean(cham(rm, i_test, ho_test)))
        dc = tung_cau(rm, i_test, ho_test)
        lay = rng.integers(0, len(i_test), size=(4000, len(i_test)))
        de = dc[lay].mean(axis=1) - dn[lay].mean(axis=1)
        lo, hi = np.percentile(de, [2.5, 97.5])
        print(f"{ho + ' [' + tieu_chi + ']':<40}{ten:<18}{b:>9.4f}{100*(b/a-1):>+8.1f}%"
              f"   [{lo:+.4f}, {hi:+.4f}]{(de<=0).mean():>9.1%}", flush=True)
        for nhan, vt in (("  |- HAI canh", vt_hai), ("  |- MOT canh", vt_mot)):
            g = rng.integers(0, len(vt), size=(4000, len(vt)))
            dg = dc[vt][g].mean(axis=1) - dn[vt][g].mean(axis=1)
            l2, h2 = np.percentile(dg, [2.5, 97.5])
            print(f"{nhan:<40}{'n=' + str(len(vt)):<18}{dc[vt].mean():>9.4f}"
                  f"{100*(dc[vt].mean()/dn[vt].mean()-1):>+8.1f}%"
                  f"   [{l2:+.4f}, {h2:+.4f}]{(dg<=0).mean():>9.1%}")
    can = [(tot_tong[chinh][1], tot_tong[chinh][2], tot_tong[chinh][3], f"{chinh} [tong]")]
    if (tot_hai[chinh_hai][1], chinh_hai) != (tot_tong[chinh][1], chinh):
        can.append((tot_hai[chinh_hai][1], tot_hai[chinh_hai][2], tot_hai[chinh_hai][3],
                    f"{chinh_hai} [hai canh]"))
    for ten, fn, wd, nhan in can:
        _chan_doan(duong, sims, sach, kf, fn, wd, hai_set, f"{nhan} {ten}")
    return 0


_TV_MEMO: dict = {}


def _theo_video(duong: Duong):
    """{video: chỉ số hàng, đã sắp theo frame_idx} — dùng cho bản chậm đối chiếu."""
    if "tv" not in _TV_MEMO:
        tam: dict = {}
        frm = duong.frm
        for r, v in enumerate(duong.vid.tolist()):
            tam.setdefault(v, []).append(r)
        _TV_MEMO["tv"] = {v: np.array(sorted(a, key=lambda r: int(frm[r])), dtype=np.int64)
                          for v, a in tam.items()}
    return _TV_MEMO["tv"]


def _tc_muot(s, duong: Duong, sigma, w):
    """Bản CHẬM, hiển nhiên đúng, của (a): lặp từng video bằng vòng lặp Python."""
    x = np.asarray(s, dtype=np.float64)
    ra = x.copy()
    R = int(max(1, round(2.5 * sigma)))
    for _v, ridx in _theo_video(duong).items():
        c = x[ridx]
        L = c.size
        m = np.empty(L)
        for i in range(L):
            tu = de = 0.0
            for k in range(-R, R + 1):
                j = i + k
                if 0 <= j < L:
                    wk = float(np.exp(-0.5 * (k / sigma) ** 2))
                    tu += wk * c[j]
                    de += wk
            m[i] = tu / de
        ra[ridx] = (1.0 - w) * c + w * m
    return ra


def _tc_chuan(s, duong: Duong, kieu, lam):
    """Bản CHẬM của (b)."""
    x = np.asarray(s, dtype=np.float64)
    ra = x.copy()
    mu_g, sd_g = float(x.mean()), float(x.std())
    tv = _theo_video(duong)
    if kieu == "minmax":
        r_all = np.empty_like(x)
        for _v, ridx in tv.items():
            c = x[ridx]
            r_all[ridx] = (c - c.min()) / max(c.max() - c.min(), 1e-12)
        r_mu, r_sd = float(r_all.mean()), float(r_all.std())
    for _v, ridx in tv.items():
        c = x[ridx]
        if kieu == "tru_tb":
            ra[ridx] = c - lam * (c.mean() - mu_g)
        elif kieu == "zscore":
            t = mu_g + sd_g * (c - c.mean()) / max(float(c.std()), 1e-6)
            ra[ridx] = (1 - lam) * c + lam * t
        else:
            t = mu_g + sd_g * (r_all[ridx] - r_mu) / max(r_sd, 1e-12)
            ra[ridx] = (1 - lam) * c + lam * t
    return ra


def _tc_bien(s, duong: Duong, k, beta):
    """Bản CHẬM của (d)."""
    x = np.asarray(s, dtype=np.float64)
    ra = x.copy()
    for _v, ridx in _theo_video(duong).items():
        c = x[ridx]
        for i in range(c.size):
            truoc = [c[i - j] for j in range(1, k + 1) if i - j >= 0]
            if truoc:
                ra[ridx[i]] = c[i] + beta * max(c[i] - float(np.mean(truoc)), 0.0)
    return ra


def _gom_kf(meta):
    kf: dict = {}
    for m in meta:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    return kf


def _chan_doan(duong: Duong, sims, sach, kf, fn, wd, hai_set, ten):
    """Hạng NỘI-VIDEO của keyframe đáp án, trước và sau — độ đo KHÔNG dính bộ phân bổ.

    Bắt buộc: nếu điểm tăng mà hạng không cải thiện thì cơ chế đang được kể là
    SAI, và con số điểm chỉ là may rủi của phép phân bổ.

    Hai thước, vì một mình thước thứ nhất đọc sai được:

    * **hạng trong pool** — trong số ứng viên của video ĐÚNG đã lọt vào 400.
      Thước này lẫn kích cỡ pool: tín hiệu kéo thêm keyframe của video đúng vào
      pool sẽ làm hạng *xấu đi* dù nó đang giúp.
    * **hạng trên TOÀN video** — trong toàn bộ keyframe của video đúng, không
      cần lọt pool. Không phụ thuộc pool, không phụ thuộc bộ phân bổ; đây mới
      là thước sạch của "định vị nội-video".
    """
    print(f"\n=== CHAN DOAN (tren ca {len(sach)} muc sach) — cau hinh {ten} ===")
    hang_video: dict = {}
    for r, v in enumerate(duong.vid.tolist()):
        hang_video.setdefault(v, []).append(r)
    hang_video = {v: np.array(a, dtype=np.int64) for v, a in hang_video.items()}

    ket = {}
    for nhan, loc in (("MOT canh", lambda i: i not in hai_set),
                      ("HAI canh", lambda i: i in hai_set)):
        idx = [i for i in range(len(sach)) if loc(i)]
        for kieu in ("nen", "moi"):
            co, hang, toan = 0, [], []
            for i in idx:
                g = sach[i]
                s = np.asarray(sims[i]) if kieu == "nen" else fn(np.asarray(sims[i]))
                w = PEAK_WEIGHT if kieu == "nen" else wd
                arr = kf[g["video_id"]]
                dung = int(arr[np.argmin(np.abs(arr - int(g["frame_idx"])))])
                ridx = hang_video[g["video_id"]]
                sv = np.asarray(s)[ridx]
                vi = int(np.flatnonzero(duong.frm[ridx] == dung)[0])
                toan.append(1 + int((sv > sv[vi]).sum()))
                hits = duong.ung_vien(s, g.get("kis_query_en") or "", w)
                trong = [h for h in hits if h.video_id == g["video_id"]]
                if not any(h.frame_idx == dung for h in trong):
                    continue
                co += 1
                thu = sorted(trong, key=lambda h: -h.score)
                hang.append(1 + next(k for k, h in enumerate(thu) if h.frame_idx == dung))
            ket[(nhan, kieu)] = (co, len(idx), hang, toan)
    print(f"{'nhom':<10}{'':>5}{'dap an trong pool':>19}{'hang trong pool':>17}"
          f"{'hang-1':>8}{'hang TOAN video':>18}{'hang-1':>8}")
    for nhan in ("MOT canh", "HAI canh"):
        for kieu, hien in (("nen", "nen"), ("moi", "sau")):
            co, n, hang, toan = ket[(nhan, kieu)]
            md = float(np.median(hang)) if hang else float("nan")
            t1 = (sum(1 for r in hang if r == 1) / len(hang)) if hang else 0.0
            mt = float(np.median(toan)) if toan else float("nan")
            t2 = (sum(1 for r in toan if r == 1) / len(toan)) if toan else 0.0
            print(f"{nhan:<10}{hien:>5}{co:>14}/{n:<4}{md:>17.1f}{t1:>8.0%}"
                  f"{mt:>18.1f}{t2:>8.0%}")


if __name__ == "__main__":
    raise SystemExit(main())
