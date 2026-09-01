"""Đo lại "VLM xếp lại keyframe NỘI-VIDEO" trên bộ đo khớp phân bố đề thật.

Cửa này bị đóng bằng một câu kết luận rất mạnh (docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md):

    "trong các slot đã truy xuất, SigLIP thuần đã đặt keyframe gần đáp án ở hạng
     nội-video trung vị 1,0 (hạng-1: 60%) — không còn chỗ cho bộ xếp lại nào"

Điều đó là **đặc điểm của bộ đo 60 câu cũ**, nơi câu hỏi được viết bằng cách nhìn
đúng cái keyframe mà bộ truy xuất đánh chỉ mục. Trên bộ đo mới (sinh từ đoạn
video, 50% câu hai cảnh) hạng nội-video trung vị là 2 (một cảnh) và 8 (hai cảnh),
và trần oracle định vị nội-video là +126%. Điều kiện đã đổi, nên phải đo lại.

HAI THỨ PHẢI SỬA SO VỚI PHÉP ĐO CŨ
----------------------------------

1. **Phép hoán vị cũ là phép ĐỒNG NHẤT trên bộ phân bổ sản xuất.** Bản "giữ-slot"
   (``experiment_sharp_rerank.xep_lai_giu_slot``) hoán vị các *đối tượng ứng viên*
   giữa các vị trí trong danh sách, rồi đưa vào ``allocate_hybrid_rows``. Nhưng bộ
   phân bổ sản xuất bây giờ là ``coverage``, và tiên nghiệm của nó là

       mass(v, x) = Σ_i w_i · exp(−½((x − f_i)/σ)²)     với w = softmax(điểm)

   — một **tổng trên tập** ứng viên. Đổi thứ tự các ứng viên trong danh sách
   không đổi một bit nào của tổng đó. Script này kiểm điều đó bằng assert chứ
   không bằng lời: hoán vị đối tượng cho ra 100 dòng **giống hệt** nền.

   Hệ quả: muốn xếp lại nội-video có tác dụng trên đường sản xuất thì phải đổi
   **ĐIỂM**, không phải thứ tự.

2. **Phép tương đương "giữ-slot" cho coverage là HOÁN VỊ ĐIỂM TRONG CÙNG VIDEO.**
   Trong mỗi video, giữ nguyên *đa tập điểm* của video đó, chỉ gán lại điểm nào
   thuộc về khung nào. Khi ấy tổng khối lượng của mỗi video **không đổi một ly**
   (softmax toàn cục thấy đúng đa tập điểm cũ), nên bề rộng phủ video không thể
   bị phá — chính là thứ đã gây artifact −35% của phép gom-khối. Chỉ có *hình
   dạng khối lượng bên trong video* dịch về phía khung mà tín hiệu mới ưa.

   Ở w = 0 phép này là phép đồng nhất (assert). Đó là cái neo an toàn.

CÁCH HỎI VLM: PHẢI LÀ CÂU HỎI ĐỊNH VỊ, KHÔNG PHẢI CÂU HỎI PHÂN LOẠI
-------------------------------------------------------------------
Bài học đã ghi: "VLM trả lời *khung này có khớp mô tả không*, trong khi thứ quyết
định điểm là *khung này có gần khoảnh khắc đúng nhất không*". Nhưng bắt VLM tự
trả lời "đây có phải khung ĐẦU TIÊN của cảnh B không" từ MỘT ảnh là bắt nó làm
việc bất khả — và bắt nó đọc một dãy ảnh đánh số thì rơi đúng vào lỗi đã phá bộ
sinh ground truth (model tả đúng nội dung nhưng **đánh sai số thứ tự ảnh**).

Chia việc đúng chỗ: **VLM chấm phân loại từng ảnh** (thứ nó làm tốt: "ảnh này có
phải cảnh B không"), còn **phép định vị do ta suy ra** từ điểm phân loại cộng với
trục thời gian mà ta đã biết chắc:

    loc(f) = B(f) · (1 − α·B(khung có candidate liền trước trong cùng video))

α = 0 là câu hỏi phân loại thuần (đối chứng lịch sử); α = 1 là "khung đầu tiên
của cảnh B" — B cao mà ngay trước đó chưa phải B. Không có chỗ nào cho model
đánh số nhầm, vì thứ tự thời gian lấy từ frame_idx chứ không hỏi model.

CHẠY
----
    python -u scripts/do_vlm_noi_video_moi.py --giai-doan co-che   # 0 đồng, chạy trước
    python -u scripts/do_vlm_noi_video_moi.py --giai-doan vlm      # tốn quota
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

from scripts.do_cap_thoi_gian_moi import GOC_TEST_MOI, GOC_TUNE_MOI, canh_cua  # noqa: E402
from scripts.experiment_cap_thoi_gian import KhoSims, _plan  # noqa: E402
from scripts.experiment_phu_quet_luoi import cac_lan_boc, cham_nhanh, ma_tran_dong  # noqa: E402
from scripts.make_submission import DEFAULT_N_FLAT, allocate_rows  # noqa: E402
from src.core.submission import MAX_ROWS, Candidate  # noqa: E402

#: M của lever cảnh B đã ship (make_submission --canh-b 100)
CANH_B_M = 100


# ---------------------------------------------------------------------------
# Hai phép hoán vị
# ---------------------------------------------------------------------------


def khoa_theo_chi_so(cands, loc_of, w):
    """{chỉ số ứng viên -> khoá xếp} = điểm + w·loc, chỉ cho khung đã được chấm.

    Khoá phải theo CHỈ SỐ chứ không theo cặp (video, khung): pool sản xuất có
    **khung trùng** — cùng một keyframe xuất hiện hai lần với hai điểm khác nhau
    (22/132 câu, tới 10 cặp mỗi câu; hợp của hai lần truy xuất vi/en). Đánh khoá
    theo cặp thì hai bản sao chung một khoá, phép hoán vị hoán đổi điểm giữa
    chúng, và bất biến "w=0 là phép đồng nhất" **vỡ** — đúng cái assert đã bắt.
    """
    ra = {}
    for i, c in enumerate(cands):
        k = (c.video_id, int(c.frame_idx))
        if k in loc_of:
            ra[i] = float(c.score) + w * float(loc_of[k])
    return ra


def hoan_vi_doi_tuong(cands, key_of):
    """Bản CŨ (``xep_lai_giu_slot``): hoán vị các ĐỐI TƯỢNG giữa các slot.

    Giữ nguyên tuyệt đối video nào ngồi ở vị trí nào trong danh sách; chỉ đổi
    khung nào của video đó ngồi ở vị trí nào. Với bộ phân bổ ``hybrid`` (thứ
    phép đo cũ dùng) đây là một phép biến đổi thật. Với ``coverage`` — bộ phân
    bổ sản xuất hiện tại — nó là phép đồng nhất, và script assert điều đó.
    """
    vi_tri = defaultdict(list)
    for i, c in enumerate(cands):
        vi_tri[c.video_id].append(i)
    moi = list(cands)
    for _vid, pos in vi_tri.items():
        co_diem = [i for i in pos if i in key_of]
        if len(co_diem) < 2:
            continue
        thu_tu = sorted(co_diem, key=lambda i: (-key_of[i], i))
        for chO, chM in zip(co_diem, thu_tu):
            moi[chO] = cands[chM]
    return moi


def hoan_vi_diem(cands, key_of):
    """Bản MỚI: hoán vị ĐIỂM trong cùng một video, giữ nguyên đa tập điểm.

    Bất biến bằng xây dựng: với mỗi video, đa tập {điểm} không đổi ⇒ tổng khối
    lượng softmax của video đó không đổi ⇒ bề rộng phủ video không đổi. Chỉ hình
    dạng khối lượng *bên trong* video dịch chuyển. Đây là phép đo sạch của đúng
    một giả thuyết: "tín hiệu X chọn khung trong video tốt hơn SigLIP".

    Khung KHÔNG được chấm giữ nguyên cặp (khung, điểm) của nó — chỉ các khung đã
    chấm mới đổi điểm cho nhau, nên một sweep chấm thiếu không thể lặng lẽ dìm
    những khung chưa kịp chấm xuống đáy.
    """
    vi_tri = defaultdict(list)
    for i, c in enumerate(cands):
        vi_tri[c.video_id].append(i)
    diem_moi = [float(c.score) for c in cands]
    for _vid, pos in vi_tri.items():
        co_diem = [i for i in pos if i in key_of]
        if len(co_diem) < 2:
            continue
        cac_diem = sorted((float(cands[i].score) for i in co_diem), reverse=True)
        thu_tu = sorted(co_diem, key=lambda i: (-key_of[i], i))
        for i, d in zip(thu_tu, cac_diem):
            diem_moi[i] = d
    return [
        Candidate(c.video_id, c.frame_idx, diem_moi[i], c.video_last_frame)
        for i, c in enumerate(cands)
    ]


# ---------------------------------------------------------------------------
# Nạp dữ liệu chung
# ---------------------------------------------------------------------------


def nap(args):
    data = Path(args.data)
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    sach = [moi[i] for i in giu]
    cands0 = [[Candidate(v, f, s, lf) for v, f, s, lf in uv[i]] for i in giu]

    nhan = []
    for m in sach:
        c = canh_cua(m)
        nhan.append(
            {"co_2_canh": bool(c), "canh_B_vi": c[2] if c else "", "canh_B_en": c[3] if c else ""}
        )
    bat = [i for i, d in enumerate(nhan) if d["co_2_canh"]]

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    kf_list, last_of, vid_of, frm_of = {}, {}, [], []
    ten_khung, hang_of = {}, {}
    for r, m in enumerate(meta):
        v, f = m["video_id"], int(m["frame_idx"])
        kf_list.setdefault(v, []).append(f)
        last_of[v] = max(last_of.get(v, 0), f)
        vid_of.append(v)
        frm_of.append(f)
        ten_khung[(v, f)] = m["frame_filename"]
        hang_of[(v, f)] = r
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}
    vid_of = np.array(vid_of)
    frm_of = np.array(frm_of)
    del meta, kf_list

    # ---- ứng viên ĐƯỜNG SẢN XUẤT HIỆN TẠI = 400 gốc + top-M cảnh B ----------
    # Đo bộ xếp lại trên pool KHÔNG có cảnh B sẽ lặp lại đúng sai lầm cũ theo
    # chiều ngược: ở gần nửa số câu hai cảnh, keyframe đáp án chưa bao giờ được
    # truy xuất, nên mọi bộ xếp lại đều bị chấm là vô dụng vì lý do không liên
    # quan gì tới nó.
    kho = KhoSims(args.data, False)
    cands = []
    for i, c0 in enumerate(cands0):
        if i not in bat or args.canh_b <= 0:
            cands.append(c0)
            continue
        s = kho.lay(nhan[i]["canh_B_vi"], nhan[i]["canh_B_en"])
        M = args.canh_b
        top = np.argpartition(-s, M)[:M]
        top = top[np.argsort(-s[top])]
        co = {(c.video_id, int(c.frame_idx)) for c in c0}
        them = []
        for j in top:
            key = (str(vid_of[j]), int(frm_of[j]))
            if key in co:
                continue
            co.add(key)
            them.append(Candidate(key[0], key[1], float(s[j]), last_of.get(key[0], key[1] + 1000)))
        cands.append(list(c0) + them)
    return sach, nhan, bat, cands, kf, ten_khung, hang_of, kho


def diem_tung_cau(sach, rows, kf, windows, goc, so_ho=2, so_boc=24):
    """Điểm nền của TỪNG câu — đại lượng TIỀN-CAN-THIỆP, dùng để chia khối."""
    ho = cac_lan_boc(goc, so_ho, so_boc, sach, kf)
    mats = ma_tran_dong(rows, sach)
    ra = np.zeros(len(sach))
    for draws in ho:
        for q in range(len(sach)):
            ra[q] += cham_nhanh([mats[q]], [draws[q]], windows)
    return ra / len(ho)


def chia_phan_tang(sach, bat, diem_nen):
    """TUNE/TEST phân tầng theo (qua cổng, shard) rồi CHIA KHỐI theo độ khó nền.

    Hai điều, và điều thứ hai đắt hơn vẻ ngoài:

    1. Không chia chẵn/lẻ thô. Bước sinh của bộ đo đặt câu hai cảnh vào đúng chỉ
       số chẵn, nên chia chẵn/lẻ dồn cả nhóm bị tác động về một bên.

    2. Trong từng tầng, **sắp theo điểm nền rồi luân phiên**. Bản chia chỉ theo
       (cổng, shard) cho ra hai nửa lệch nhau hơn hai lần trên chính nhóm hai
       cảnh (TUNE 0,087 vs TEST 0,194) — điểm mỗi câu ở đây rất lệch (phần lớn
       bằng 0, vài câu ~0,6), nên bốc thăm kiểu gì cũng dễ ra hai nửa khác hẳn
       nhau về độ khó. Nửa TUNE gần chạm sàn thì không còn chỗ để phân xử cấu
       hình, mà con số vẫn in ra bình thường.

    Điểm nền là đại lượng **tiền-can-thiệp** (đo trên chính đường nền, hạt giống
    riêng 77000, trước khi có bất kỳ tín hiệu VLM nào), nên chia khối theo nó
    không hề đụng vào ước lượng hiệu ứng — nó chỉ làm hai nửa so được với nhau.
    """
    tune, test = [], []
    nhom = defaultdict(list)
    for i, m in enumerate(sach):
        nhom[(i in bat, m.get("shard", "?"))].append(i)
    for k in sorted(nhom, key=lambda t: (t[0], str(t[1]))):
        thu_tu = sorted(nhom[k], key=lambda i: (-float(diem_nen[i]), i))
        for j, i in enumerate(thu_tu):
            # RẮN BÒ, không phải luân phiên thẳng. Luân phiên thẳng trên danh
            # sách đã sắp giảm dần thì câu CAO HƠN của mỗi cặp luôn rơi về TUNE
            # — đúng cái đã cho TUNE 0,178 vs TEST 0,100 ở lần chạy trước, tức
            # phép "chia khối cho cân" tự tay tạo ra độ lệch nó định xoá. Đảo
            # bên ở các cặp lẻ thì độ lệch triệt tiêu theo từng bộ bốn.
            cao = j % 2 == 0
            dao = (j // 2) % 2 == 1
            (tune if cao != dao else test).append(i)
    return sorted(tune), sorted(test)


# ---------------------------------------------------------------------------
# Giai đoạn 1 — cơ chế, 0 đồng
# ---------------------------------------------------------------------------


def giai_doan_co_che(args) -> int:
    sach, nhan, bat, cands, kf, _ten, _hang, _kho = nap(args)
    windows = [int(w) for w in args.windows.split(",")]
    print(f"bộ sạch {len(sach)} mục | hai cảnh {len(bat)} | cảnh B M={args.canh_b}")

    rows_nen = [
        allocate_rows(c, args.allocator, DEFAULT_N_FLAT, _plan())[:MAX_ROWS] for c in cands
    ]

    # ---- (a) phép hoán vị CŨ có phải phép đồng nhất trên coverage không? ----
    print("\n=== (a) Phép hoán vị giữ-slot CŨ trên bộ phân bổ sản xuất ===")
    rng = np.random.default_rng(90210)
    khac_cov = khac_hyb = 0
    for i in bat:
        # tín hiệu giả, chỉ để xem phép biến đổi có LÀM GÌ không
        key = {j: float(rng.random()) for j in range(len(cands[i]))}
        hv = hoan_vi_doi_tuong(cands[i], key)
        if allocate_rows(hv, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS] != allocate_rows(
            cands[i], "coverage", DEFAULT_N_FLAT, _plan()
        )[:MAX_ROWS]:
            khac_cov += 1
        if allocate_rows(hv, "hybrid", DEFAULT_N_FLAT, _plan())[:MAX_ROWS] != allocate_rows(
            cands[i], "hybrid", DEFAULT_N_FLAT, _plan()
        )[:MAX_ROWS]:
            khac_hyb += 1
    print(f"  hoán vị ĐỐI TƯỢNG bằng tín hiệu NGẪU NHIÊN, {len(bat)} câu hai cảnh:")
    print(f"    allocator coverage (sản xuất) : đổi dòng ở {khac_cov}/{len(bat)} câu")
    print(f"    allocator hybrid   (phép đo cũ): đổi dòng ở {khac_hyb}/{len(bat)} câu")
    if khac_cov == 0:
        print("  ⇒ trên đường sản xuất, phép hoán vị giữ-slot cũ là PHÉP ĐỒNG NHẤT.")
        print("    Tiên nghiệm coverage là tổng trên TẬP ứng viên: đổi thứ tự không đổi tổng.")
        print("    Muốn xếp lại nội-video có tác dụng thì phải đổi ĐIỂM, không phải thứ tự.")

    # ---- (b) trần của cơ chế "hoán vị ĐIỂM trong video" ---------------------
    print("\n=== (b) Trần oracle của hoán vị ĐIỂM trong cùng video ===")
    ho = cac_lan_boc(77000, args.test_seeds, args.test_draws, sach, kf)

    def cham(idx, rows_all):
        gt_s = [sach[i] for i in idx]
        mats = ma_tran_dong([rows_all[i] for i in idx], gt_s)
        return float(np.mean([cham_nhanh(mats, d, windows) for d in ho_sub(idx, gt_s)]))

    def ho_sub(idx, _gt_s):
        return [[h[i] for i in idx] for h in ho]

    # oracle: trong VIDEO ĐÚNG, dồn điểm lớn nhất về khung gần khoảnh khắc thật
    rows_orc = list(rows_nen)
    for i in range(len(sach)):
        vid, anchor = sach[i]["video_id"], int(sach[i]["frame_idx"])
        key = {
            j: -abs(int(c.frame_idx) - anchor)
            for j, c in enumerate(cands[i])
            if c.video_id == vid
        }
        if len(key) < 2:
            continue
        rows_orc[i] = allocate_rows(
            hoan_vi_diem(cands[i], key), args.allocator, DEFAULT_N_FLAT, _plan()
        )[:MAX_ROWS]

    # bất biến: w=0 (key = chính điểm) phải ra dòng GIỐNG HỆT nền
    for i in range(len(sach)):
        key0 = {j: float(c.score) for j, c in enumerate(cands[i])}
        r0 = allocate_rows(
            hoan_vi_diem(cands[i], key0), args.allocator, DEFAULT_N_FLAT, _plan()
        )[:MAX_ROWS]
        assert r0 == rows_nen[i], f"mục {i}: hoán vị điểm ở w=0 KHÔNG phải phép đồng nhất"
    print(f"  bất biến: hoán vị điểm với key = chính điểm ⇒ 100 dòng giống hệt nền "
          f"({len(sach)}/{len(sach)} mục, assert).")

    mot = [i for i in range(len(sach)) if i not in bat]
    print(f"\n  {'nhóm':>12}{'n':>5}{'nền':>10}{'oracle':>10}{'trần':>10}")
    print("  " + "-" * 47)
    for ten, idx in (("tất cả", list(range(len(sach)))), ("MỘT cảnh", mot), ("HAI cảnh", bat)):
        a, b = cham(idx, rows_nen), cham(idx, rows_orc)
        print(f"  {ten:>12}{len(idx):>5}{a:>10.4f}{b:>10.4f}{100*(b/a-1):>+9.1f}%")
    print("\n  Đây là trần của ĐÚNG cơ chế mà bộ xếp lại được phép dùng, không phải")
    print("  trần oracle tổng (scripts/tran_dinh_vi_noi_video.py) vốn đặt lại thẳng")
    print("  frame id của từng dòng và vì thế không bị bộ phân bổ chặn.")

    # ---- (c) trần KHẢ THI: oracle chỉ trong đúng tập khung sẽ gửi cho VLM ---
    # Trần ở (b) cho phép hoán vị mọi ứng viên của video đúng. VLM chỉ chấm được
    # top-V video × top-F khung. Nếu video đúng không nằm trong top-V, hoặc khung
    # gần đáp án không nằm trong top-F của video ấy, thì bộ xếp lại KHÔNG VỚI TỚI
    # dù nó hoàn hảo. Đây mới là con số quyết định có nên tiêu quota hay không.
    print("\n=== (c) Trần KHẢ THI theo ngân sách chấm khung ===")
    print(f"  {'V×F':>8}{'ảnh':>7}{'lô':>6}{'HAI cảnh':>11}{'trần':>9}{'với tới':>9}")
    print("  " + "-" * 50)
    nen_hai = cham(bat, rows_nen)
    orc_hai = cham(bat, rows_orc)
    for V, F in [(2, 10), (3, 12), (5, 12), (5, 20), (8, 20)]:
        rows_c = list(rows_nen)
        n_anh = 0
        for i in bat:
            chon = chon_khung_de_cham(cands[i], V, F)
            cho_phep = {(c.video_id, int(c.frame_idx)) for c in chon}
            n_anh += len(chon)
            vid, anchor = sach[i]["video_id"], int(sach[i]["frame_idx"])
            key = {
                j: -abs(int(c.frame_idx) - anchor)
                for j, c in enumerate(cands[i])
                if c.video_id == vid and (c.video_id, int(c.frame_idx)) in cho_phep
            }
            if len(key) < 2:
                continue
            rows_c[i] = allocate_rows(
                hoan_vi_diem(cands[i], key), args.allocator, DEFAULT_N_FLAT, _plan()
            )[:MAX_ROWS]
        d = cham(bat, rows_c)
        voi = 100 * (d - nen_hai) / (orc_hai - nen_hai) if orc_hai > nen_hai else 0.0
        print(f"  {f'{V}×{F}':>8}{n_anh:>7}{-(-n_anh//8):>6}{d:>11.4f}"
              f"{100*(d/nen_hai-1):>+8.1f}%{voi:>8.0f}%")
    print("\n  'với tới' = phần trăm khoảng cách nền→trần(b) mà ngân sách ấy chạm được.")
    return 0


# ---------------------------------------------------------------------------
# Giai đoạn 2 — VLM thật
# ---------------------------------------------------------------------------


def chon_khung_de_cham(cands_q, so_video, so_khung):
    """Các khung sẽ đưa cho VLM: top-``so_video`` video, mỗi video ``so_khung`` khung.

    Chọn theo ĐIỂM trong video (khối lượng softmax lớn nhất nằm ở đó) — hoán vị
    chỉ dịch được khối lượng mà nó chạm tới, nên chấm các khung điểm bét là mua
    quota để dịch những hạt bụi.
    """
    thu_tu, theo_video = [], defaultdict(list)
    for c in cands_q:
        if c.video_id not in theo_video:
            thu_tu.append(c.video_id)
        theo_video[c.video_id].append(c)
    ra = []
    for vid in thu_tu[:so_video]:
        nhom = sorted(theo_video[vid], key=lambda c: -float(c.score))[:so_khung]
        ra.extend(sorted(nhom, key=lambda c: int(c.frame_idx)))
    return ra


def suy_ra_loc(diem_B, khung_theo_video, alpha):
    """Điểm ĐỊNH VỊ suy ra từ điểm phân loại + trục thời gian.

    loc(f) = B(f) · (1 − α·B(khung được chấm liền trước trong cùng video))

    α = 0  → câu hỏi phân loại thuần ("khung này có phải cảnh B không")
    α = 1  → "khung ĐẦU TIÊN của cảnh B": B cao mà khung ngay trước đó chưa phải B

    Trục thời gian lấy từ frame_idx, không hỏi model — nên không có chỗ nào để
    model đánh nhầm số thứ tự ảnh, đúng cái lỗi đã phá bộ sinh ground truth.
    """
    loc = {}
    for _vid, ds in khung_theo_video.items():
        ds = sorted(ds)
        truoc = 0.0
        for f in ds:
            b = diem_B.get(f, 0.0)
            loc[f] = b * (1.0 - alpha * truoc)
            truoc = b
    return loc


def giai_doan_vlm(args) -> int:
    from src.core.vlm import VLMJudge, load_env

    sach, nhan, bat, cands, kf, ten_khung, hang_of, kho = nap(args)
    windows = [int(w) for w in args.windows.split(",")]
    load_env(Path(args.data).parent / ".env")
    load_env(".env")
    judge = VLMJudge(args.data, model=args.model)
    if not judge.ready and args.cach_hoi != "chi-siglip":
        print("Không có GEMINI_API_KEY (chế độ --cach-hoi chi-siglip không cần nó).")
        return 2

    rows_nen = [
        allocate_rows(c, args.allocator, DEFAULT_N_FLAT, _plan())[:MAX_ROWS] for c in cands
    ]
    d_nen = diem_tung_cau(sach, rows_nen, kf, windows, 77000)
    i_tune, i_test = chia_phan_tang(sach, bat, d_nen)
    bat_t = [i for i in i_tune if i in bat]
    bat_s = [i for i in i_test if i in bat]
    print(f"bộ sạch {len(sach)} | hai cảnh {len(bat)} "
          f"(TUNE {len(bat_t)} / TEST {len(bat_s)}) | cảnh B M={args.canh_b}")
    print(f"  cân bằng nhóm hai cảnh (điểm nền, hạt riêng): "
          f"TUNE {d_nen[bat_t].mean():.4f} | TEST {d_nen[bat_s].mean():.4f}")

    khung_of = {i: chon_khung_de_cham(cands[i], args.videos, args.frames) for i in bat}

    # ---- đối chứng 0 ĐỒNG: chính SigLIP, qua ĐÚNG cơ chế ấy ----------------
    # Bằng chứng cấu trúc nói neo TRÙNG KHÍT khung cảnh-B đầu tiên ở 64% số câu,
    # và ranh giới đó được xác định bằng chỗ simB vượt simA. Nếu SigLIP đã biết
    # cảnh B nằm ở đâu thì bộ xếp lại VLM phải THẮNG ĐƯỢC NÓ mới đáng tiền —
    # nếu không thì thứ mua được là lever miễn phí, còn VLM chỉ là bao bì đắt.
    # Cosine thô không có thang tuyệt đối nên chuẩn hoá min-max TRONG TỪNG VIDEO,
    # để cùng thang 0..1 với điểm VLM trước khi vào cùng một công thức loc.
    def tin_hieu_siglip(lay_van_ban, dau=1.0):
        ra_all = {}
        for i in bat:
            vi, en = lay_van_ban(i)
            s = kho.lay(vi, en)
            theo_video = defaultdict(list)
            for c in khung_of[i]:
                theo_video[c.video_id].append(int(c.frame_idx))
            ra = {}
            for v, fs in theo_video.items():
                gia = dau * np.array([float(s[hang_of[(v, f)]]) for f in fs])
                lo, hi = float(gia.min()), float(gia.max())
                chuan = (gia - lo) / (hi - lo) if hi > lo else np.zeros_like(gia)
                for f, x in zip(fs, chuan):
                    ra[(v, f)] = (float(x), "siglip")
            ra_all[i] = ra
        return ra_all

    # ---- đối chứng ARTIFACT: tín hiệu NGẪU NHIÊN qua đúng cơ chế ấy -------
    # Bài học −34,7%: một phép trộn có thể tự nó dịch điểm mà chẳng liên quan gì
    # tới tín hiệu. Nếu khoá NGẪU NHIÊN cũng làm điểm tăng thì thứ đo được là
    # đặc tính của bộ phân bổ, không phải của tín hiệu. Đây là phép kiểm bắt
    # buộc, không phải phần thêm cho đẹp.
    rng_nc = np.random.default_rng(31337)
    diem_ngau = {
        i: {(c.video_id, int(c.frame_idx)): (float(rng_nc.random()), "ngau nhien")
            for c in khung_of[i]}
        for i in bat
    }

    # ---- chấm khung: MỘT sweep cho mỗi cách hỏi ----------------------------
    nguon_vlm = {"canhB": lambda i: nhan[i]["canh_B_vi"], "de": lambda i: sach[i]["kis_query_vi"]}
    if args.cach_hoi == "ca-hai":
        cach_hoi = dict(nguon_vlm)
    elif args.cach_hoi == "chi-siglip":
        cach_hoi = {}
    else:
        cach_hoi = {args.cach_hoi: nguon_vlm[args.cach_hoi]}

    tong_anh = sum(len(v) for v in khung_of.values())
    print(f"sẽ chấm {tong_anh} ảnh × {len(cach_hoi)} cách hỏi VLM "
          f"(top-{args.videos} video, {args.frames} khung/video)"
          f" + đối chứng siglipB (0 đồng)")

    diem_of = {ten: {} for ten in cach_hoi}
    for ten, lay_query in cach_hoi.items():
        print(f"\n--- chấm khung, cách hỏi = {ten} ---", flush=True)
        for n, i in enumerate(bat):
            if not judge.usable:
                print("  !! hết quota ở MỌI model — dừng, phần đã chấm vẫn dùng được")
                break
            ds = [
                (c.video_id, int(c.frame_idx), ten_khung[(c.video_id, int(c.frame_idx))])
                for c in khung_of[i]
                if (c.video_id, int(c.frame_idx)) in ten_khung
            ]
            diem_of[ten][i] = judge.score(lay_query(i), ds)
            if (n + 1) % 5 == 0:
                print(f"  {n+1}/{len(bat)}  {judge.cost_note().splitlines()[0]}", flush=True)
    print(f"\n{judge.cost_note()}")

    # Chỉ giữ những câu MỌI nguồn tín hiệu đều có điểm — so nguồn này với nguồn
    # kia trên hai tập câu khác nhau thì con số chênh lệch không đọc được.
    if diem_of:
        xong = [i for i in bat if all(d.get(i) for d in diem_of.values())]
    else:
        xong = list(bat)
    if not xong:
        print("\nkhông câu nào có điểm — dừng")
        return 3
    # nguồn 0 đồng, thêm sau khi đã biết ``xong`` để mọi nguồn dùng CÙNG tập câu
    for ten, nguon in (
        ("siglipB", tin_hieu_siglip(lambda i: (nhan[i]["canh_B_vi"], nhan[i]["canh_B_en"]))),
        ("siglipA", tin_hieu_siglip(lambda i: canh_cua(sach[i])[:2])),
        ("nghichB", tin_hieu_siglip(
            lambda i: (nhan[i]["canh_B_vi"], nhan[i]["canh_B_en"]), dau=-1.0)),
        ("ngau_nhien", diem_ngau),
    ):
        diem_of[ten] = {i: nguon[i] for i in xong}

    # ---- VLM có thêm được gì TRÊN NỀN SigLIP không? ------------------------
    # Câu hỏi thật của lane này không phải "bộ xếp lại có ăn không" mà "bộ xếp
    # lại BẰNG VLM có ăn không". Nếu trộn hai tín hiệu không hơn SigLIP một mình
    # thì phần VLM đóng góp đúng bằng không, và mọi con số dương của nó chỉ là
    # con số của SigLIP đi vòng qua một hoá đơn.
    for ten in [t for t in cach_hoi if t in diem_of]:
        chung = {}
        for i in xong:
            a, b = diem_of[ten][i], diem_of["siglipB"][i]
            chung[i] = {
                k: (0.5 * float(a[k][0]) + 0.5 * float(b[k][0]), "tron")
                for k in a
                if k in b
            }
        diem_of[f"tron_{ten}"] = chung

    print(f"\nchấm xong {len(xong)}/{len(bat)} câu hai cảnh "
          f"({len(diem_of)} nguồn tín hiệu: {', '.join(diem_of)})")

    # ---- dựng dòng cho từng cấu hình --------------------------------------
    def dung_rows(ten, alpha, w):
        rows = list(rows_nen)
        for i in xong:
            sc = diem_of[ten][i]
            theo_video = defaultdict(list)
            for (v, f) in sc:
                theo_video[v].append(f)
            diem_B = {}
            for (v, f), (s, _why) in sc.items():
                diem_B[(v, f)] = float(s)
            loc_f = {}
            for v, fs in theo_video.items():
                sub = suy_ra_loc({f: diem_B[(v, f)] for f in fs}, {v: fs}, alpha)
                for f, x in sub.items():
                    loc_f[(v, f)] = x
            key = khoa_theo_chi_so(cands[i], loc_f, w)
            if len(key) < 2:
                continue
            rows[i] = allocate_rows(
                hoan_vi_diem(cands[i], key), args.allocator, DEFAULT_N_FLAT, _plan()
            )[:MAX_ROWS]
        return rows

    ngoai = [i for i in range(len(sach)) if i not in xong]
    ho_tune = cac_lan_boc(GOC_TUNE_MOI, args.tune_seeds, args.tune_draws,
                          [sach[i] for i in i_tune], kf)
    ho_test = cac_lan_boc(GOC_TEST_MOI, args.test_seeds, args.test_draws,
                          [sach[i] for i in i_test], kf)

    def cham(idx, rows_all, ho, chi=None):
        lay = [i for i in idx if chi is None or i in chi]
        vt = [k for k, i in enumerate(idx) if chi is None or i in chi]
        gt_s = [sach[i] for i in lay]
        mats = ma_tran_dong([rows_all[i] for i in lay], gt_s)
        return [cham_nhanh(mats, [d[k] for k in vt], windows) for d in ho]

    print("\n=== TUNE (chỉ nhóm HAI CẢNH đã chấm) ===")
    nen_t = cham(i_tune, rows_nen, ho_tune, set(xong))
    m_nen = float(np.mean(nen_t))
    print(f"  nền: {m_nen:.4f} ±{np.std(nen_t):.4f}   (n={len([i for i in i_tune if i in xong])})")
    print(f"\n{'cách hỏi':>10}{'α':>6}{'w':>8}{'điểm':>10}{'±':>8}{'so nền':>9}")
    print("-" * 51)
    ket, cache = {}, {}
    for ten in diem_of:
        for alpha in [float(x) for x in args.alphas.split(",")]:
            for w in [float(x) for x in args.weights.split(",")]:
                rows = dung_rows(ten, alpha, w)
                for i in ngoai:
                    assert rows[i] == rows_nen[i], f"mục {i} ngoài diện tác động đã đổi dòng"
                if w == 0.0:
                    # neo an toàn: ở w=0 khoá xếp CHÍNH LÀ điểm, nên phép hoán vị
                    # phải là phép đồng nhất trên MỌI mục, kể cả mục đã chấm
                    assert rows == rows_nen, "w=0 không cho ra đúng dòng nền"
                d = cham(i_tune, rows, ho_tune, set(xong))
                m = float(np.mean(d))
                ket[(ten, alpha, w)] = m
                cache[(ten, alpha, w)] = rows
                print(f"{ten:>10}{alpha:>6.1f}{w:>8.3f}{m:>10.4f}{np.std(d):>8.4f}"
                      f"{100*(m/m_nen-1):>+8.1f}%", flush=True)
    chot = max(ket, key=lambda k: ket[k])
    print(f"\nCHỐT trên TUNE: {chot} ({100*(ket[chot]/m_nen-1):+.1f}%)")
    print(f"bất biến: {len(ngoai)} mục ngoài diện ra dòng giống hệt nền (assert, mọi cấu hình).")

    if args.khong_doc_test:
        print("\n=== TEST: KHÔNG ĐỌC (--khong-doc-test) ===")
        print("  Lượt này chỉ so các NGUỒN tín hiệu và các ĐỐI CHỨNG với nhau trên TUNE.")
        print("  Mỗi lần đọc TEST tiêu một phần bảo đảm 'đọc đúng một lần', nên việc")
        print("  khảo sát nguồn/đối chứng phải sống trọn trong TUNE.")
        return 0

    print("\n=== TEST (đọc MỘT LẦN) ===")
    chi = set(xong)
    a_l = cham(i_test, rows_nen, ho_test, chi)
    b_l = cham(i_test, cache[chot], ho_test, chi)
    a, b = float(np.mean(a_l)), float(np.mean(b_l))
    print(f"  nền : {a:.4f} ±{np.std(a_l):.4f}")
    print(f"  chốt: {b:.4f} ±{np.std(b_l):.4f}   ({100*(b/a-1):+.1f}%)")

    lay = [i for i in i_test if i in chi]
    vt = [k for k, i in enumerate(i_test) if i in chi]
    gt_t = [sach[i] for i in lay]
    mn = ma_tran_dong([rows_nen[i] for i in lay], gt_t)
    mc = ma_tran_dong([cache[chot][i] for i in lay], gt_t)

    def tung_cau(mats):
        ra = np.zeros(len(gt_t))
        for draws in ho_test:
            for q in range(len(gt_t)):
                ra[q] += cham_nhanh([mats[q]], [draws[vt[q]]], windows)
        return ra / len(ho_test)

    dn, dc = tung_cau(mn), tung_cau(mc)
    rng = np.random.default_rng(4242)
    boc = rng.integers(0, len(gt_t), size=(4000, len(gt_t)))
    delta = dc[boc].mean(axis=1) - dn[boc].mean(axis=1)
    lo, hi = np.percentile(delta, [2.5, 97.5])
    # Đếm tất định, bổ cho khoảng tin cậy: một mức +80% do 2-3 câu nhảy vọt và
    # một mức +80% do hai phần ba số câu nhích lên là hai thứ khác hẳn nhau về
    # độ tin, mà cùng một khoảng tin cậy bootstrap thì không phân biệt được.
    hieu = dc - dn
    print(f"\n=== phân rã theo CÂU (đếm, không phải ước lượng) ===")
    print(f"  {int((hieu > 1e-12).sum())} câu TỐT lên | "
          f"{int((hieu < -1e-12).sum())} câu XẤU đi | "
          f"{int((np.abs(hieu) <= 1e-12).sum())} câu không đổi (trên {len(hieu)})")
    thu = np.sort(hieu)[::-1]
    if thu.size >= 3:
        print(f"  ba câu đóng góp nhiều nhất: {thu[0]:+.3f} {thu[1]:+.3f} {thu[2]:+.3f}"
              f"  = {100*thu[:3].sum()/max(hieu.sum(), 1e-12):.0f}% tổng mức tăng")
    print("\n=== bootstrap theo CÂU (nhóm bị tác động) ===")
    print(f"  n = {len(gt_t)} câu hai cảnh trên TEST")
    print(f"  chênh {dc.mean()-dn.mean():+.4f}; KTC 95% [{lo:+.4f}, {hi:+.4f}]; "
          f"P(≤0) = {(delta<=0).mean():.1%}")
    print("\n=== KẾT LUẬN ===")
    up = 100 * (b / a - 1) if a else 0.0
    if lo > 0:
        print(f"  DƯƠNG: TEST {up:+.1f}%, khoảng tin cậy không chứa 0.")
    elif hi < 0:
        print(f"  ÂM: TEST {up:+.1f}%, khoảng tin cậy nằm hoàn toàn dưới 0.")
    else:
        print(f"  HOÀ: {up:+.1f}%, khoảng tin cậy chứa 0 (P(≤0) = {(delta<=0).mean():.1%}).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--giai-doan", default="co-che", choices=("co-che", "vlm"))
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--allocator", default="coverage")
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--canh-b", type=int, default=CANH_B_M)
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--cach-hoi", default="ca-hai",
                    choices=("ca-hai", "canhB", "de", "chi-siglip"))
    ap.add_argument("--alphas", default="0,0.5,1.0")
    ap.add_argument("--weights", default="0,0.005,0.02,0.05,0.20,1.0")
    ap.add_argument("--khong-doc-test", action="store_true",
                    help="chỉ so trên TUNE, không tiêu thêm lần đọc TEST nào")
    ap.add_argument("--tune-seeds", type=int, default=3)
    ap.add_argument("--tune-draws", type=int, default=32)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--test-draws", type=int, default=48)
    args = ap.parse_args()
    if args.giai_doan == "co-che":
        return giai_doan_co_che(args)
    return giai_doan_vlm(args)


if __name__ == "__main__":
    raise SystemExit(main())
