"""Trần THẬT của khâu đặt dòng — oracle bị chặn bởi danh sách ứng viên.

``tran_dinh_vi_noi_video.py`` cho +126%, nhưng nó **đặt thang quanh khoảnh khắc
THẬT** — một thứ không bộ phân bổ nào biết. Bộ phân bổ chỉ được đặt dòng quanh
các keyframe mà khâu truy xuất đã trả về; nếu trong 400 ứng viên không có
keyframe nào của video đúng nằm gần đáp án thì không tham số nào cứu được câu đó.

Nên +126% KHÔNG phải headroom của việc chỉnh tham số phân bổ. Script này đo cái
trần đúng: giữ nguyên VIDEO và THỨ HẠNG từng dòng y hệt oracle kia, nhưng thang
được đặt quanh **ứng viên gần đáp án nhất trong pool**, không phải quanh đáp án.

Ba mức, đọc cùng nhau mới ra kết luận:

    nền            — hệ thống sản xuất
    TRAN-POOL      — đặt dòng hoàn hảo, nhưng chỉ quanh ứng viên đã có
    TRAN-ORACLE    — đặt dòng quanh đáp án thật (cận trên tuyệt đối)

Khoảng nền → TRAN-POOL là tất cả những gì lane phân bổ có thể lấy.
Khoảng TRAN-POOL → TRAN-ORACLE thuộc về khâu SINH ứng viên, lane khác.

    python -u scripts/tran_dat_dong_theo_pool.py
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

from scripts.do_bo_do_moi import GOC_HAT  # noqa: E402
from scripts.do_phan_bo_sau import NEN_CP, sinh_dong  # noqa: E402
from scripts.experiment_phu_quet_luoi import cac_lan_boc, cham_nhanh, ma_tran_dong  # noqa: E402
from scripts.tran_dinh_vi_noi_video import oracle_rows  # noqa: E402
from src.core.submission import Candidate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--draws", type=int, default=48)
    args = ap.parse_args()

    data = Path(args.data)
    windows = [int(w) for w in args.windows.split(",")]
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    sach = [moi[i] for i in giu]
    cands = [[Candidate(v, f, s, lf) for v, f, s, lf in uv[i]] for i in giu]

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    kf_list: dict = {}
    last_of: dict = {}
    for m in meta:
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
        last_of[m["video_id"]] = max(last_of.get(m["video_id"], 0), int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}
    del meta, kf_list

    nen = [sinh_dong(c, "coverage", NEN_CP) for c in cands]

    orc, pool = [], []
    for r, g, cs in zip(nen, sach, cands):
        v, fr = g["video_id"], int(g["frame_idx"])
        last = last_of.get(v, fr + 1000)
        orc.append(oracle_rows(r, v, fr, last))
        trong = [c for c in cs if c.video_id == v]
        tam = min((abs(c.frame_idx - fr), int(c.frame_idx)) for c in trong)[1] if trong else fr
        # tâm = keyframe ứng viên GẦN đáp án nhất — thứ tốt nhất pool có để bám
        pool.append(oracle_rows(r, v, tam, last))

    nhom = (("bo SACH", list(range(len(sach)))),
            ("  |- MOT canh", [i for i, g in enumerate(sach) if not g.get("co_2_canh")]),
            ("  |- HAI canh", [i for i, g in enumerate(sach) if g.get("co_2_canh")]))

    print(f"{'nhom':<16}{'n':>4}{'nen':>9}{'TRAN-POOL':>12}{'':>8}"
          f"{'TRAN-ORACLE':>13}{'':>8}{'phan pool/oracle':>18}")
    print("-" * 90)
    for ten, idx in nhom:
        gt = [sach[i] for i in idx]
        ho = cac_lan_boc(GOC_HAT, args.seeds, args.draws, gt, kf)

        def d(rows_all):
            mats = ma_tran_dong([rows_all[i] for i in idx], gt)
            return float(np.mean([cham_nhanh(mats, x, windows) for x in ho]))

        a, p, o = d(nen), d(pool), d(orc)
        chia = (p - a) / (o - a) if o > a else float("nan")
        print(f"{ten:<16}{len(idx):>4}{a:>9.4f}{p:>12.4f}{100*(p/a-1):>+7.1f}%"
              f"{o:>13.4f}{100*(o/a-1):>+7.1f}%{chia:>17.0%}")

    print("\nTRAN-POOL  = giu nguyen video va thu hang tung dong; thang dat quanh UNG VIEN")
    print("             gan dap an nhat trong 400 ung vien. Day la CAN TREN cua moi cach")
    print("             chinh tham so phan bo — no khong the dat dong o cho khong co ung vien.")
    print("TRAN-ORACLE= thang dat quanh DAP AN THAT (tran_dinh_vi_noi_video.py).")
    print("Cot cuoi   = phan tram cua khoang oracle ma khau PHAN BO co quyen dong toi.")
    print("Phan con lai thuoc khau SINH ung vien, khong phai lane nay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
