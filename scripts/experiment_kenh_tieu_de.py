"""Kênh TIÊU ĐỀ/METADATA của BTC làm NGUỒN ỨNG VIÊN (không phải nguồn điểm).

Bối cảnh — vì sao cửa này lại đáng mở lại dù đã có một kết luận âm:

``scripts/experiment_metadata.py`` đã đo metadata BTC và ghi ÂM 7,6% vào bảng
tín hiệu (``docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md``).  Nhưng phép đo đó **trộn
điểm**: nó cộng ``weight * tanh(BM25)`` vào điểm SigLIP rồi xếp lại.  Kênh tiêu
đề là tín hiệu **cấp video**, còn 60% điểm mất nằm ở **vị trí frame** — cộng
điểm chỉ có thể xáo lại thứ tự video, và khi xáo sai nó cướp thang phân bổ sâu
của keyframe mà SigLIP đã đặt đúng.  Đó là cách dùng sai, và bảng tín hiệu đã
ghi nó đúng như thế.

U-CESE (chung kết chính giải này 2025, arXiv:2605.23274v1) dùng kênh văn bản
theo cách khác hẳn: *"jointly retrieves timestamps from VisualDB and TextualDB,
and merges them into a single list"* — **hợp danh sách, không cộng điểm**.  Vì
``R@k`` là *max* trên tiền tố, một ứng viên thêm vào **đuôi** danh sách 100 dòng
không bao giờ hại: nó chỉ tiêu một chỗ ở hạng 95..100 (giá trị 0,2) và có thể
đổi lấy cả một câu (giá trị 1,0) nếu không kênh nào khác tìm ra video đó.

Script này KHÔNG cộng điểm.  Nó hỏi đúng một câu trước khi tiêu bất kỳ dòng nào:

  (a) CHẨN ĐOÁN — trên 60 câu GT, BM25 tiêu đề xếp video ĐÚNG ở hạng mấy, và
      **bao nhiêu câu có video đúng trong top-3 tiêu đề mà KHÔNG có trong
      top-24 SigLIP**?  Nếu con số đó là 0 thì kênh này không mở được cửa nào
      trên bộ đo hiện có: KẾT LUẬN ÂM, DỪNG, ghi vào tài liệu.

  (b) Nếu (a) > 0 — cắm ứng viên tiêu đề vào các dòng CUỐI bằng
      ``src.core.submission.reserve_tail_rows``, chấm qua ``allocate_rows``
      thật của make_submission, chia TUNE/TEST 30/30 chẵn/lẻ, luật hoà 2 sigma.
      Tiêu chí nghiệm thu là **KHÔNG ÂM** (không tụt quá 1 sd) — không phải
      "thắng", vì 60 câu GT đều đã có video đúng trong top-400 nên kỳ vọng
      trên chính bộ này là ~0.

  (c) Chỉ số về tính — số câu có VIDEO ĐÚNG trong 100 dòng, trước và sau.

    python scripts/experiment_kenh_tieu_de.py                 # (a), rồi (b)+(c) nếu (a)>0
    python scripts/experiment_kenh_tieu_de.py --chi-chan-doan # chỉ (a)
    python scripts/experiment_kenh_tieu_de.py --du-b          # ép chạy (b) dù (a)=0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.experiment_phu_quet_luoi import (  # noqa: E402
    GOC_TEST,
    GOC_TUNE,
    CacheDiem,
    _doi_chieu_bo_cham,
    cac_lan_boc,
    cham_nhanh,
    ma_tran_dong,
    nap_dong,
    nap_ung_vien,
)
from scripts.make_submission import (  # noqa: E402
    DEFAULT_DEPTH_COST,
    DEFAULT_N_FLAT,
    allocate_rows,
)
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    reserve_tail_rows,
)
from src.core.transcripts import TranscriptIndex  # noqa: E402

#: bao nhiêu ứng viên đầu của SigLIP mà VLM được nhìn trong sản xuất
#: (``vlm_rerank_run.py --judge``, mặc định 24) — định nghĩa của "top-24 SigLIP"
TOP_SIGLIP = 24

#: các biến thể trường metadata đưa vào chỉ mục BM25
BIEN_THE = ("tieu_de", "tieu_de_mo_ta", "tieu_de_mo_ta_tu_khoa")


# ---------------------------------------------------------------------------
# 1) Chỉ mục BM25 trên metadata BTC (873 video, vài giây CPU)
# ---------------------------------------------------------------------------


def nap_media_info(data_dir: Path) -> dict:
    """video_id -> {title, description, keywords, author, publish_date}."""
    z = Path(data_dir) / "media-info-aic25-b1.zip"
    if not z.exists():
        raise FileNotFoundError(f"không thấy {z}")
    out: dict = {}
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if not n.endswith(".json"):
                continue
            try:
                j = json.loads(zf.read(n).decode("utf-8"))
            except Exception:  # noqa: BLE001 - một file hỏng không được giết cả kho
                continue
            kw = j.get("keywords") or []
            out[Path(n).stem] = {
                "title": str(j.get("title") or ""),
                "description": str(j.get("description") or ""),
                "keywords": " ".join(kw) if isinstance(kw, list) else str(kw),
                "author": str(j.get("author") or ""),
                "publish_date": str(j.get("publish_date") or ""),
            }
    return out


def chi_muc_tieu_de(info: dict, bien_the: str) -> TranscriptIndex:
    """BM25 trên metadata, tái dùng nguyên bộ tách từ của kênh lời thoại.

    ``TranscriptIndex`` đã có sẵn đúng thứ cần: BM25 unigram+bigram trên tiếng
    Việt đã chuẩn hoá NFC, chuẩn hoá độ dài tài liệu (quan trọng ở đây vì mô tả
    của một video có thể dài gấp 50 lần tiêu đề của video khác), và tiêu đề
    được lặp 3 lần vì nó là phần DO NGƯỜI VIẾT, khác hẳn phần mô tả toàn chữ ký
    kênh.  Viết lại bộ tách từ ở đây chỉ tạo ra một nhánh thứ hai để lệch nhau.

    Mô tả cắt 1200 ký tự: phần đuôi của mọi mô tả là khối "Đăng ký KÊNH ...",
    hashtag và link — nó có mặt ở mọi video nên idf tự dập, nhưng nó làm độ dài
    tài liệu phồng lên và chuẩn hoá độ dài BM25 thì KHÔNG tự dập được.
    """
    idx = TranscriptIndex()
    for vid, m in info.items():
        than = []
        if bien_the in ("tieu_de_mo_ta", "tieu_de_mo_ta_tu_khoa"):
            than.append(m["description"][:1200])
        if bien_the == "tieu_de_mo_ta_tu_khoa":
            than.append(m["keywords"])
        # thân có thể rỗng (biến thể chỉ-tiêu-đề); ``add`` bỏ qua segment rỗng và
        # sẽ bỏ luôn video, nên luôn cấp cho nó ít nhất chính tiêu đề làm thân
        than = [t for t in than if t.strip()] or [m["title"]]
        idx.add(vid, [(0.0, " ".join(than))], title=m["title"])
    return idx.build()


def hang_video(idx: TranscriptIndex, truy_van: str) -> list:
    """Danh sách video xếp giảm dần theo BM25 (chỉ video có ít nhất 1 từ khớp)."""
    s = idx.score_videos(truy_van)
    return [v for v, _ in sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))]


#: giá trị hạng cho video mà không từ khoá nào của truy vấn chạm tới
KHONG_HANG = 10**6


def hang_cua(idx: TranscriptIndex, truy_van: str, video_id: str) -> tuple:
    """(hạng lạc quan, hạng bi quan, số video hoà điểm) của một video.

    Phải phân biệt hai thứ này, nếu không cả phép chẩn đoán là dối trá.  Kho có
    hàng chục bản tin "60 Giây Sáng - Ngày ..." với tiêu đề khác nhau đúng con
    số ngày tháng; một truy vấn nói "logo 60 Giây" khớp CẢ SÁU MƯƠI video ở
    cùng một điểm BM25.  Xếp theo ``(-điểm, video_id)`` thì L21_V001 đứng đầu —
    không phải vì tiêu đề của nó đúng hơn 59 cái kia mà vì bảng chữ cái.  Nếu
    ground truth tình cờ là L21_V001 thì phép đo sẽ ghi "hạng 1" và ta tưởng
    kênh tiêu đề vừa cứu một câu, trong khi cơ hội thật là 1/60.

    Hạng bi quan (mọi video hoà điểm đều tính là đứng trước) là hạng DUY NHẤT
    một kênh ứng viên thật sự bảo đảm được, nên nó là con số dùng để kết luận.
    """
    s = idx.score_videos(truy_van)
    diem = s.get(video_id)
    if diem is None:
        return KHONG_HANG, KHONG_HANG, 0
    tren = sum(1 for x in s.values() if x > diem + 1e-9)
    hoa = sum(1 for x in s.values() if abs(x - diem) <= 1e-9)
    return tren + 1, tren + hoa, hoa


# ---------------------------------------------------------------------------
# 2) (a) CHẨN ĐOÁN
# ---------------------------------------------------------------------------


def chan_doan(gt, cands_of, info, truy_van_of) -> dict:
    """Bảng (a). Trả về, theo biến thể, số câu tiêu đề CỨU được."""
    # tập video mà SigLIP đã đưa tới trước mắt VLM, và tập video trong cả pool
    v_top = [{c.video_id for c in cs[:TOP_SIGLIP]} for cs in cands_of]
    v_pool = [{c.video_id for c in cs} for cs in cands_of]
    ngoai_top = [i for i, g in enumerate(gt) if g["video_id"] not in v_top[i]]
    ngoai_pool = [i for i, g in enumerate(gt) if g["video_id"] not in v_pool[i]]

    print(f"\n=== (a) CHẨN ĐOÁN trên {len(gt)} câu GT ===")
    print(f"  video đúng NGOÀI top-{TOP_SIGLIP} ứng viên SigLIP : "
          f"{len(ngoai_top)}/{len(gt)} câu {ngoai_top}")
    print(f"  video đúng NGOÀI cả pool 400 ứng viên            : "
          f"{len(ngoai_pool)}/{len(gt)} câu {ngoai_pool}")
    print(f"  (số video phân biệt trong top-{TOP_SIGLIP}: trung vị "
          f"{np.median([len(s) for s in v_top]):.0f})")

    ket = {}
    print(f"\n{'biến thể':<24}{'hạng-1':>8}{'top-3':>7}{'top-10':>8}"
          f"{'trung vị':>10}{'CỨU top-3':>11}{'CỨU thật':>10}{'CỨU pool':>10}")
    print("-" * 91)
    for bt in BIEN_THE:
        idx = chi_muc_tieu_de(info, bt)
        lac, bi, hoa = [], [], []
        cuu_top, cuu_that, cuu_pool = [], [], []
        for i, g in enumerate(gt):
            hl, hb, nh = hang_cua(idx, truy_van_of[i], g["video_id"])
            lac.append(hl)
            bi.append(hb)
            hoa.append(nh)
            if hl <= 3 and i in ngoai_top:
                cuu_top.append(i)
            if hb <= 3 and i in ngoai_top:          # hạng bi quan: hoà điểm không tính
                cuu_that.append(i)
            if hb <= 3 and i in ngoai_pool:
                cuu_pool.append(i)
        a_l, a_b = np.array(lac), np.array(bi)
        ket[bt] = {"lac": lac, "bi": bi, "hoa": hoa, "cuu_top": cuu_top,
                   "cuu_that": cuu_that, "cuu_pool": cuu_pool, "idx": idx}
        tv = np.median(a_b[a_b < KHONG_HANG]) if (a_b < KHONG_HANG).any() else float("nan")
        print(f"{bt:<24}{int((a_b == 1).sum()):>8}{int((a_b <= 3).sum()):>7}"
              f"{int((a_b <= 10).sum()):>8}{tv:>10.0f}"
              f"{len(cuu_top):>11}{len(cuu_that):>10}{len(cuu_pool):>10}")

    print("\n  Mọi hạng ở bảng trên là hạng BI QUAN (video hoà điểm tính là đứng trước).")
    print(f"  CỨU top-3 = video đúng ở top-3 tiêu đề (hạng lạc quan, hoà điểm phá bằng")
    print(f"              bảng chữ cái) mà KHÔNG có trong top-{TOP_SIGLIP} SigLIP")
    print("  CỨU thật  = như trên nhưng dùng hạng BI QUAN — đây là con số quyết định")
    print("  CỨU pool  = cứu thật, và video còn không có ở đâu trong pool 400 ứng viên")

    print("\n  Chi tiết từng câu được kênh tiêu đề chạm tới (mọi biến thể):")
    for bt in BIEN_THE:
        for i in sorted(set(ket[bt]["cuu_top"])):
            hl, hb, nh = hang_cua(ket[bt]["idx"], truy_van_of[i], gt[i]["video_id"])
            that = "CỨU THẬT" if i in ket[bt]["cuu_that"] else \
                   f"HOÀ {nh} video -> may nhờ bảng chữ cái, không phải tín hiệu"
            print(f"    [{bt}] câu {i:2d} {gt[i]['video_id']}: hạng lạc quan {hl}, "
                  f"bi quan {hb} — {that}")
            print(f"        tiêu đề: {info[gt[i]['video_id']]['title'][:78]}")
    return ket


# ---------------------------------------------------------------------------
# 3) (b) cắm vào đuôi + chấm qua allocate_rows thật
# ---------------------------------------------------------------------------


def _plan() -> AllocationPlan:
    return AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)


def dong_nen(cands_list):
    """100 dòng/câu bằng đúng đường mã make_submission chạy trong trận (coverage)."""
    return [allocate_rows(c, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
            for c in cands_list]


def ung_vien_tieu_de(idx, truy_van, da_co: set, kf, n_video: int, n_frame: int):
    """(video, frame) của các video mà CHỈ kênh tiêu đề tìm ra.

    Frame lấy rải đều trên toàn bộ keyframe của video, không neo vào đâu cả:
    tiêu đề nói video này là video nào, nó không nói khoảnh khắc nằm ở giây thứ
    mấy.  Neo bừa vào đầu video là tự bịa một tín hiệu định vị không tồn tại.
    """
    out = []
    for v in hang_video(idx, truy_van):
        if v in da_co:
            continue
        a = kf.get(v)
        if a is None or len(a) == 0:
            continue
        # rải đều: lấy n_frame mốc phân vị, tránh hai đầu video (logo/credit)
        vi_tri = np.linspace(0, len(a) - 1, n_frame + 2)[1:-1] if n_frame < len(a) \
            else np.arange(len(a))
        for j in sorted({int(round(x)) for x in vi_tri}):
            out.append((v, int(a[j])))
        if len({v for v, _ in out}) >= n_video:
            break
    return out


def dong_co_tieu_de(rows_nen, idx, truy_van_of, cands_of, kf, n_video, n_frame):
    ra = []
    for rows, tv, cs in zip(rows_nen, truy_van_of, cands_of):
        extras = ung_vien_tieu_de(idx, tv, {c.video_id for c in cs}, kf, n_video, n_frame)
        ra.append(reserve_tail_rows(rows, extras, budget=MAX_ROWS))
    return ra


def co_video(rows_of, gt_sub) -> int:
    return sum(1 for rows, g in zip(rows_of, gt_sub)
               if any(v == g["video_id"] for v, _f in rows))


def tran_ly_thuyet(kf, windows) -> dict:
    """Trần của kênh tiêu đề, tính bằng số học chứ không bằng phép đo.

    Kênh tiêu đề trả lời câu hỏi "video NÀO", nhưng luật chấm hỏi "FRAME nào".
    Khi đã biết video mà không biết khoảnh khắc, dòng cắm vào chỉ là một lần
    bốc mù trên trục thời gian của video đó.  Với video dài trung vị ~8.000
    frame và cửa sổ đáp án nửa-bề-rộng 6..20, một dòng mù trúng khoảng 0,15% —
    và dòng đó nằm ở hạng 96..100 nên kể cả khi trúng nó chỉ đáng 0,2 (chỉ
    k=100 tính).  Nhân hai số đó ra kỳ vọng mỗi câu, rồi so với 1 sd của
    harness, là biết phép đo có khả năng nhìn thấy hiệu ứng này hay không.
    """
    do_dai = np.array([int(a[-1] - a[0]) for a in kf.values() if len(a) > 1])
    print("\n=== TRẦN LÝ THUYẾT của một dòng CẮM MÙ (số học, không phải phép đo) ===")
    print(f"  {len(do_dai)} video: độ dài trung vị {np.median(do_dai):.0f} frame, "
          f"keyframe/video trung vị {np.median([len(a) for a in kf.values()]):.0f}")
    ra = {}
    for w in windows:
        for f in (1, 3, 6):
            p = float(np.mean(np.minimum(1.0, f * (2 * w + 1) / do_dai)))
            ra[f"w{w}_f{f}"] = p
            print(f"  cửa sổ ±{w:<2d} · {f} dòng rải đều -> xác suất trúng {p*100:5.2f}%"
                  f"   kỳ vọng điểm/câu {p*0.2:.5f}")
    print("  (một dòng ở hạng 96..100 chỉ đáng 0,2 điểm câu — chỉ số hạng k=100 tính nó)")
    return ra


def do_luong(args, gt, cands_of, kf, truy_van_of, ket_a, cache_dir) -> int:
    windows = [int(w) for w in args.windows.split(",")]
    i_tune = [i for i in range(len(gt)) if i % 2 == 0]
    i_test = [i for i in range(len(gt)) if i % 2 == 1]
    print(f"\n=== (b) TUNE {len(i_tune)} câu (chỉ số chẵn) / TEST {len(i_test)} câu (lẻ) ===")

    rows_nen = nap_dong(cache_dir, f"nen_cov_{len(gt)}", args.refresh,
                        lambda: dong_nen(cands_of))

    lat = {}
    for phia, idxs, goc, so_ho, so_boc in (
        ("tune", i_tune, GOC_TUNE, args.tune_seeds, args.tune_draws),
        ("test", i_test, GOC_TEST, args.test_seeds, args.test_draws),
    ):
        gt_h = [gt[i] for i in idxs]
        lat[phia] = {
            "idx": idxs,
            "gt": gt_h,
            "nen": [rows_nen[i] for i in idxs],
            "ho": cac_lan_boc(goc, so_ho, so_boc, gt_h, kf),
            "harness": {"build": "tieude-v1", "windows": windows, "seeds": so_ho,
                        "draws": so_boc, "goc": goc, "phia": phia, "n_cau": len(gt_h)},
        }

    _doi_chieu_bo_cham(lat["tune"]["nen"][:5], lat["tune"]["gt"][:5],
                       [t[:3] for t in lat["tune"]["ho"][0][:5]], windows)
    print("  bộ chấm vector hoá khớp tuyệt đối bản sản xuất (5 câu × 3 bốc).")

    def diem_cac_ho(rows_of, gt_sub, ho):
        mats = ma_tran_dong(rows_of, gt_sub)
        return [cham_nhanh(mats, draws, windows) for draws in ho]

    cache = {p: CacheDiem(cache_dir / f"diem_{p}.json", lat[p]["harness"], args.refresh)
             for p in ("tune", "test")}

    nen_tune = cache["tune"].lay("nen", lambda: diem_cac_ho(
        lat["tune"]["nen"], lat["tune"]["gt"], lat["tune"]["ho"]))
    nen_tune_m, nen_tune_sd = float(np.mean(nen_tune)), float(np.std(nen_tune))
    print(f"  NỀN trên TUNE (coverage bản ship): {nen_tune_m:.4f} ±{nen_tune_sd:.4f}")

    # ---- quét trên TUNE -----------------------------------------------------
    to_hop = [(bt, nv, nf)
              for bt in BIEN_THE
              for nv in (1, 2, 3, 5)
              for nf in (1, 2, 3)]
    print(f"\n  quét {len(to_hop)} tổ hợp trên TUNE (biến thể × số video × số frame/video)")
    print(f"{'biến thể':<24}{'#video':>7}{'#frame':>7}{'dòng thêm':>11}"
          f"{'điểm':>9}{'±':>8}{'so nền':>9}{'video/100':>11}{'mất video':>11}")
    print("-" * 97)

    dong_cache: dict = {}

    def dong_cua(th, idxs):
        bt, nv, nf = th
        key = (bt, nv, nf)
        if key not in dong_cache:
            dong_cache[key] = nap_dong(
                cache_dir, f"td_{bt}_v{nv}_f{nf}_{len(gt)}", args.refresh,
                lambda: dong_co_tieu_de(rows_nen, ket_a[bt]["idx"], truy_van_of,
                                        cands_of, kf, nv, nf),
            )
        return [dong_cache[key][i] for i in idxs]

    kq_tune = {}
    for th in to_hop:
        rows_h = dong_cua(th, i_tune)
        ho = cache["tune"].lay(f"{th[0]}_v{th[1]}_f{th[2]}",
                               lambda rh=rows_h: diem_cac_ho(rh, lat["tune"]["gt"],
                                                             lat["tune"]["ho"]))
        m, sd = float(np.mean(ho)), float(np.std(ho))
        kq_tune[th] = (m, sd)
        them = np.mean([sum(1 for r in rw if tuple(r) not in {tuple(x) for x in nn})
                        for rw, nn in zip(rows_h, lat["tune"]["nen"])])
        # đuôi dành chỗ = CẮT các dòng cuối của bản nền: đếm câu mà video đúng
        # biến mất khỏi 100 dòng vì chính việc dành chỗ đó
        mat = sum(1 for rw, nn, g in zip(rows_h, lat["tune"]["nen"], lat["tune"]["gt"])
                  if any(v == g["video_id"] for v, _f in nn)
                  and not any(v == g["video_id"] for v, _f in rw))
        print(f"{th[0]:<24}{th[1]:>7}{th[2]:>7}{them:>11.1f}{m:>9.4f}{sd:>8.4f}"
              f"{100*(m/nen_tune_m-1):>+8.1f}%"
              f"{co_video(rows_h, lat['tune']['gt']):>8}/{len(i_tune)}{mat:>11}")

    chot = max(kq_tune, key=lambda k: kq_tune[k][0])
    print(f"\n  CHỐT trên TUNE: {chot[0]}, {chot[1]} video × {chot[2]} frame "
          f"-> {kq_tune[chot][0]:.4f} ({100*(kq_tune[chot][0]/nen_tune_m-1):+.1f}%)")

    # ---- TEST: đọc ĐÚNG MỘT LẦN, chỉ nền + tổ hợp đã chốt --------------------
    print(f"\n=== (b) TEST — đọc một lần, {len(i_test)} câu chỉ số lẻ ===")
    nen_test = cache["test"].lay("nen", lambda: diem_cac_ho(
        lat["test"]["nen"], lat["test"]["gt"], lat["test"]["ho"]))
    rows_test = dong_cua(chot, i_test)
    chot_test = cache["test"].lay(f"{chot[0]}_v{chot[1]}_f{chot[2]}",
                                  lambda: diem_cac_ho(rows_test, lat["test"]["gt"],
                                                      lat["test"]["ho"]))
    nen_m, nen_sd = float(np.mean(nen_test)), float(np.std(nen_test))
    cho_m, cho_sd = float(np.mean(chot_test)), float(np.std(chot_test))
    print(f"  nền        : {nen_m:.4f} ±{nen_sd:.4f}")
    print(f"  + tiêu đề  : {cho_m:.4f} ±{cho_sd:.4f}   ({100*(cho_m/nen_m-1):+.2f}% so nền)")

    # ---- (c) chỉ số về tính -------------------------------------------------
    print("\n=== (c) Số câu có VIDEO ĐÚNG trong 100 dòng ===")
    ket_c = {}
    for ten, idxs, rn, rt in (("TUNE", i_tune, lat["tune"]["nen"], dong_cua(chot, i_tune)),
                              ("TEST", i_test, lat["test"]["nen"], rows_test)):
        gt_h = [gt[i] for i in idxs]
        ket_c[ten] = (co_video(rn, gt_h), co_video(rt, gt_h), len(idxs))
        print(f"  {ten}: nền {ket_c[ten][0]}/{len(idxs)}  ->  "
              f"+tiêu đề {ket_c[ten][1]}/{len(idxs)}")

    # đâu là chỗ kênh tiêu đề CÓ THỂ ăn: video đúng nằm trong pool 400 mà bộ
    # phân bổ không cấp cho nó dòng nào.  Đây là toàn bộ cơ hội còn lại trên
    # bộ 60 câu này — nếu tiêu đề không xếp các câu đó vào top-3 thì hết cửa.
    thieu = [i for i in range(len(gt))
             if not any(v == gt[i]["video_id"] for v, _f in rows_nen[i])]
    print(f"\n  Cơ hội duy nhất còn lại: {len(thieu)}/{len(gt)} câu có video đúng TRONG "
          f"pool 400 mà bộ phân bổ không cấp dòng nào: {thieu}")
    for i in thieu:
        hs = {bt: hang_cua(ket_a[bt]["idx"], truy_van_of[i], gt[i]["video_id"])[1]
              for bt in BIEN_THE}
        print(f"    câu {i:2d} {gt[i]['video_id']}: hạng tiêu đề (bi quan) "
              + ", ".join(f"{bt}={h}" for bt, h in hs.items()))

    # ---- phán quyết: tiêu chí là KHÔNG ÂM ------------------------------------
    bien = max(nen_sd, 0.0005)
    delta = cho_m - nen_m
    print("\n=== PHÁN QUYẾT (b) — tiêu chí nghiệm thu là KHÔNG ÂM, không phải 'thắng' ===")
    if delta < -bien:
        phan = "AM"
        print(f"  ÂM: tụt {delta:.4f} (> 1 sd = {bien:.4f}) — kênh tiêu đề tiêu dòng "
              f"mà không trả lại gì. KHÔNG ship.")
    elif delta > 2 * bien:
        phan = "DUONG"
        print(f"  DƯƠNG vượt 2 sigma: {delta:+.4f} > {2*bien:.4f}.")
    else:
        phan = "HOA"
        print(f"  HOÀ: chênh {delta:+.4f}, trong khoảng ±2 sd ({2*bien:.4f}). "
              f"Không hại, cũng không giúp TRÊN BỘ NÀY.")

    tom_tat = {
        "chot_tune": {"bien_the": chot[0], "n_video": chot[1], "n_frame": chot[2]},
        "tune": {"nen": nen_tune_m, "nen_sd": nen_tune_sd},
        "test": {"nen": nen_m, "nen_sd": nen_sd, "tieu_de": cho_m, "tieu_de_sd": cho_sd,
                 "delta": delta, "phan_quyet": phan},
        "video_trong_100_dong": ket_c,
        "cau_thieu_dong": thieu,
        "windows": windows,
    }
    (cache_dir / "tom_tat.json").write_text(
        json.dumps(tom_tat, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  tóm tắt: {cache_dir / 'tom_tat.json'}")
    return 1 if phan == "AM" else 0


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_kenh_tieu_de"))
    ap.add_argument("--cache-ung-vien", default=str(ROOT / "data" / "cache_phu_quet_luoi"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--tune-seeds", type=int, default=3)
    ap.add_argument("--tune-draws", type=int, default=32)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--test-draws", type=int, default=48)
    ap.add_argument("--truy-van", choices=("vi", "vi_en"), default="vi",
                    help="văn bản đưa vào BM25 tiêu đề (metadata là tiếng Việt)")
    ap.add_argument("--chi-chan-doan", action="store_true", help="chỉ chạy (a) rồi dừng")
    ap.add_argument("--du-b", action="store_true", help="chạy (b) kể cả khi (a) = 0")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=== 0) Ứng viên đường sản xuất (cache của experiment_phu_quet_luoi) ===",
          flush=True)
    gt, cands_of, kf = nap_ung_vien(args.data, Path(args.cache_ung_vien), False)

    # ``nap_ung_vien`` chỉ giữ lại các trường cần cho chấm điểm, nên truy vấn
    # phải đọc lại từ ground_truth gốc, khớp theo (video_id, frame_idx).
    gt_full = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    tra = {(g["video_id"], int(g["frame_idx"])): g for g in gt_full}
    truy_van_of = []
    for g in gt:
        full = tra[(g["video_id"], int(g["frame_idx"]))]
        t = full["kis_query_vi"]
        if args.truy_van == "vi_en":
            t = t + " " + (full.get("kis_query_en") or "")
        truy_van_of.append(t)

    info = nap_media_info(Path(args.data))
    print(f"  media-info: {len(info)} video")

    ket_a = chan_doan(gt, cands_of, info, truy_van_of)
    tot_nhat = max(ket_a, key=lambda b: len(ket_a[b]["cuu_that"]))
    n_cuu = len(ket_a[tot_nhat]["cuu_that"])
    n_tho = max(len(v["cuu_top"]) for v in ket_a.values())
    n_cuu_pool = max(len(v["cuu_pool"]) for v in ket_a.values())
    print(f"\n  CON SỐ QUYẾT ĐỊNH (cứu thật, hạng bi quan): {n_cuu}/{len(gt)} câu "
          f"— biến thể tốt nhất: {tot_nhat}")
    print(f"  (đếm thô nếu phá hoà bằng bảng chữ cái: {n_tho}; "
          f"cứu ngoài pool 400: {n_cuu_pool})")

    tran = tran_ly_thuyet(kf, [int(w) for w in args.windows.split(",")])
    (cache_dir / "chan_doan.json").write_text(
        json.dumps(
            {
                "top_siglip": TOP_SIGLIP,
                "quyet_dinh_cuu_that": n_cuu,
                "dem_tho_pha_hoa_bang_chu_cai": n_tho,
                "cuu_ngoai_pool": n_cuu_pool,
                "theo_bien_the": {
                    bt: {k: v[k] for k in ("lac", "bi", "hoa", "cuu_top",
                                           "cuu_that", "cuu_pool")}
                    for bt, v in ket_a.items()
                },
                "tran_ly_thuyet_dong_mu": tran,
            },
            ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"  chẩn đoán: {cache_dir / 'chan_doan.json'}")

    if args.chi_chan_doan:
        print(f"\nXong sau {time.time()-t0:.0f}s.")
        return 0
    if n_cuu == 0 and not args.du_b:
        print("\n=== KẾT LUẬN ÂM — DỪNG LẠI ===")
        print("  Không câu nào được kênh tiêu đề cứu khỏi top-24 SigLIP trên bộ 60 câu GT.")
        print("  Cắm ứng viên tiêu đề vào đuôi chỉ có thể tiêu dòng, không thể trả lại gì")
        print("  TRÊN BỘ ĐO NÀY. Ghi vào docs/KENH_TIEU_DE.md, không ship.")
        print("  (chạy lại với --du-b nếu muốn xem phép đo hại-hay-không dù sao.)")
        print(f"\nXong sau {time.time()-t0:.0f}s.")
        return 0

    ma = do_luong(args, gt, cands_of, kf, truy_van_of, ket_a, cache_dir)
    print(f"\nXong sau {time.time()-t0:.0f}s. Cache: {cache_dir}")
    return ma


if __name__ == "__main__":
    raise SystemExit(main())
