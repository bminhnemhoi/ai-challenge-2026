"""Đo lại các CỬA ĐÃ ĐÓNG dưới mô hình bốc ĐÚNG hơn — có đóng nhầm cửa không?

Phát hiện của phản biện 1 (`docs/KE_HOACH_DINH_VI.md` §4.2a): **trục sigma của
`CoveragePlan` ĐỔI CHIỀU khi đổi giả định bốc khoảnh khắc thật** — bốc đều trong
ô keyframe thì sigma 60 thắng, bốc Gauss quanh neo thì sigma 15 thắng.

Điều đó biến một chi tiết của thiết bị đo thành thứ **quyết định dấu** của kết
luận. Và bộ chấm hiện tại bốc **ĐỀU** trên ô keyframe, trong khi với câu hai cảnh,
khung neo *được định nghĩa* là khung đầu tiên của cảnh B — một cú cắt. Khoảnh
khắc thật **không thể** nằm trước cú cắt đó, nhưng mô hình ĐỀU đặt gần **một nửa**
khối lượng xác suất vào cảnh A.

Nên câu hỏi phải trả lời trước khi tin bất kỳ kết luận âm nào: **có cửa nào bị
đóng chỉ vì thiết bị đo sai giả định không?**

Script này chấm lại, dưới CẢ HAI mô hình bốc (ĐỀU và SAU_NEO), đúng những thứ đã
bị đóng hoặc đã được ship:

  1. tham số CoveragePlan sigma (cửa đã đóng: TEST +1,5%, HOÀ)
  2. lever cảnh B (đã ship)
  3. lever hoán vị điểm nội-video (đã ship)

0 API, chỉ đọc cache ứng viên + metadata.

    python -u scripts/do_lai_cua_da_dong.py
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

from scripts.cong_do_ben_mo_hinh_boc import cac_lan_boc_kieu  # noqa: E402
from scripts.do_cap_thoi_gian_moi import canh_cua  # noqa: E402
from scripts.experiment_cap_thoi_gian import KhoSims, _plan  # noqa: E402
from scripts.experiment_phu_quet_luoi import cham_nhanh, ma_tran_dong  # noqa: E402
from scripts.make_submission import (  # noqa: E402
    DEFAULT_N_FLAT,
    allocate_rows,
    hoan_vi_theo_canh_b,
)
from src.core.submission import MAX_ROWS, Candidate, CoveragePlan  # noqa: E402

KIEU = ("DEU", "SAU_NEO")
GOC = 771000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--m", type=int, default=100)
    args = ap.parse_args()

    data = Path(args.data)
    windows = [int(w) for w in args.windows.split(",")]
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    sach = [moi[i] for i in giu]
    c0 = [[Candidate(v, f, s, lf) for v, f, s, lf in uv[i]] for i in giu]

    nhan = [canh_cua(m) for m in sach]
    bat = [i for i, c in enumerate(nhan) if c]
    print(f"bo sach {len(sach)} muc | cong hai canh BAT {len(bat)}")

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    kf_list, last_of, hang_of, vid_of, frm_of = {}, {}, {}, [], []
    for i, m in enumerate(meta):
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        last_of[m["video_id"]] = max(last_of.get(m["video_id"], 0), int(m["frame_idx"]))
        hang_of[(m["video_id"], int(m["frame_idx"]))] = i
        vid_of.append(m["video_id"])
        frm_of.append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}
    vid_of, frm_of = np.array(vid_of), np.array(frm_of)
    del meta, kf_list

    print("nap chi muc (mot lan, qua KhoSims) ...", flush=True)
    kho = KhoSims(args.data, False)
    simsB = {i: kho.lay(nhan[i][2], nhan[i][3]) for i in bat}

    def them_b(cands_list):
        ra = []
        for i, cc in enumerate(cands_list):
            if i not in simsB:
                ra.append(cc)
                continue
            s = simsB[i]
            k = min(args.m, len(s) - 1)
            top = np.argpartition(-s, k)[:k]
            top = top[np.argsort(-s[top])]
            co = {(c.video_id, int(c.frame_idx)) for c in cc}
            them = []
            for j in top:
                key = (str(vid_of[j]), int(frm_of[j]))
                if key in co:
                    continue
                co.add(key)
                them.append(Candidate(key[0], key[1], float(s[j]),
                                      last_of.get(key[0], key[1] + 1000)))
            ra.append(list(cc) + them)
        return ra

    def rows_cua(cands_list, plan_cov=None, hoan_vi=False):
        ra = []
        for i, cc in enumerate(cands_list):
            if hoan_vi and i in simsB:
                cc = hoan_vi_theo_canh_b(cc, simsB[i], hang_of)
            if plan_cov is not None:
                from src.core.submission import allocate_coverage_rows

                ra.append(allocate_coverage_rows(cc, plan=plan_cov,
                                                 tail_n_flat=DEFAULT_N_FLAT,
                                                 tail_plan=_plan())[:MAX_ROWS])
            else:
                ra.append(allocate_rows(cc, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS])
        return ra

    # --- các cấu hình cần chấm lại -----------------------------------------
    c_b = them_b(c0)
    cau_hinh = {
        "nen (san xuat hom nay)": rows_cua(c0),
        "+ ung vien canh B (da ship)": rows_cua(c_b),
        "+ hoan vi noi-video (da ship)": rows_cua(c_b, hoan_vi=True),
        "sigma 45 (CUA DA DONG)": rows_cua(c0, plan_cov=CoveragePlan(nhiet=0.01, sigma=45.0,
                                                                     nua_cua_so=6, luoi=5)),
        "sigma 15 (huong nguoc lai)": rows_cua(c0, plan_cov=CoveragePlan(nhiet=0.02, sigma=15.0,
                                                                        nua_cua_so=6, luoi=5)),
    }

    gt_b = [sach[i] for i in bat]
    print(f"\nNHOM HAI CANH (n={len(bat)}) — cham lai duoi CA HAI mo hinh boc\n")
    print(f"{'cau hinh':<32}" + "".join(f"{k:>22}" for k in KIEU))
    print("-" * (32 + 22 * len(KIEU)))
    diem = {}
    for kieu in KIEU:
        ho = cac_lan_boc_kieu(GOC, args.seeds, args.draws, gt_b, kf, kieu)
        for ten, rows in cau_hinh.items():
            mats = ma_tran_dong([rows[i] for i in bat], gt_b)
            diem[(ten, kieu)] = float(np.mean([cham_nhanh(mats, d, windows) for d in ho]))
    nen = {k: diem[("nen (san xuat hom nay)", k)] for k in KIEU}
    for ten in cau_hinh:
        o = ""
        for k in KIEU:
            d = diem[(ten, k)]
            o += f"{d:>12.4f} {100*(d/nen[k]-1):>+8.1f}%"
        print(f"{ten:<32}{o}")

    print("\n=== DOC KET QUA ===")
    for ten in cau_hinh:
        if ten.startswith("nen"):
            continue
        a = 100 * (diem[(ten, "DEU")] / nen["DEU"] - 1)
        b = 100 * (diem[(ten, "SAU_NEO")] / nen["SAU_NEO"] - 1)
        if a * b < 0:
            print(f"  !! {ten}: DOI DAU giua hai mo hinh ({a:+.1f}% vs {b:+.1f}%) "
                  "— ket luan cu phu thuoc gia dinh cua thiet bi do")
        else:
            print(f"  {ten}: cung dau ({a:+.1f}% / {b:+.1f}%)")
    print("\nSAU_NEO la mo hinh DUNG HON cho nhom hai canh: khung neo la khung dau tien")
    print("cua canh B (mot cu cat), nen khoanh khac that khong the nam TRUOC no.")
    print("Mo hinh DEU dat gan mot nua khoi luong xac suat vao canh A.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
