"""100 dòng được đặt ở đâu, và xa nhất chúng CÓ THỂ tới đâu — hai phép đếm tất định.

``do_phan_bo_sau.py`` đếm "bao nhiêu dòng rơi gần đáp án". Script này hỏi tiếp
hai câu mà con số đó không trả lời được, và cả hai đều là phép ĐẾM, không có
khoảng tin cậy nào để bàn:

1. **Dòng gần nhất cách đáp án bao xa?** Nếu trung vị là vài frame thì bộ phân
   bổ đã đặt đúng chỗ và mọi việc chỉnh tham số chỉ là bào mép. Nếu trung vị là
   hàng chục frame thì nó rải trượt, và đó là chỗ có điểm để lấy.

2. **Bộ phân bổ có gì để mà đặt?** Mỗi dòng chỉ có thể xuất hiện quanh một ứng
   viên (phủ xác suất trải mass trong ±4σ quanh mỗi keyframe ứng viên). Nên nếu
   trong 400 ứng viên KHÔNG có keyframe nào của video đúng nằm gần đáp án, thì
   không tham số phân bổ nào cứu được câu đó — đó là nghẽn ở khâu SINH ứng viên,
   thuộc lane khác. Bảng thứ hai tách hai loại thất bại ấy ra.

    python -u scripts/chan_doan_dat_dong.py
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

from scripts.do_phan_bo_sau import NEN_CP, sinh_dong  # noqa: E402
from src.core.submission import Candidate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_bo_do_moi"))
    args = ap.parse_args()

    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    sach = [moi[i] for i in giu]
    cands = [[Candidate(v, f, s, lf) for v, f, s, lf in uv[i]] for i in giu]
    rows = [sinh_dong(c, "coverage", NEN_CP) for c in cands]
    nhom = (("MOT canh", [i for i, g in enumerate(sach) if not g.get("co_2_canh")]),
            ("HAI canh", [i for i, g in enumerate(sach) if g.get("co_2_canh")]))

    print("=== 1. DONG GAN NHAT cach khoanh khac that bao xa (chi cac cau co dong "
          "trong video dung) ===")
    print(f"{'nhom':<10}{'n':>4}{'co dong':>9}{'trung vi':>10}"
          f"{'<=20':>7}{'<=50':>7}{'<=100':>7}{'<=500':>7}")
    for ten, sel in nhom:
        d = []
        for i in sel:
            g = sach[i]
            fr, v = int(g["frame_idx"]), g["video_id"]
            k = [abs(f - fr) for vv, f in rows[i] if vv == v]
            if k:
                d.append(min(k))
        print(f"{ten:<10}{len(sel):>4}{len(d):>9}{int(np.median(d)):>10}"
              + "".join(f"{sum(1 for x in d if x <= w):>7}" for w in (20, 50, 100, 500)))

    print("\n=== 2. TRAN cua khau phan bo: 400 ung vien co gi gan dap an khong ===")
    print("(hang = vi tri tot nhat trong danh sach 400 ung vien cua mot keyframe")
    print(" thuoc video DUNG va cach dap an khong qua W frame)")
    for ten, sel in nhom:
        print(f"\n  {ten} (n={len(sel)})")
        print(f"{'':4}{'W':>6}{'co ung vien':>13}{'hang<=20':>10}{'hang<=50':>10}"
              f"{'hang<=100':>11}{'trung vi hang':>15}")
        for W in (20, 60, 120, 240):
            r = []
            for i in sel:
                g = sach[i]
                fr, v = int(g["frame_idx"]), g["video_id"]
                h = [k for k, c in enumerate(cands[i])
                     if c.video_id == v and abs(c.frame_idx - fr) <= W]
                if h:
                    r.append(min(h) + 1)
            print(f"{'':4}{W:>6}{f'{len(r)}/{len(sel)}':>13}"
                  f"{sum(1 for x in r if x <= 20):>10}{sum(1 for x in r if x <= 50):>10}"
                  f"{sum(1 for x in r if x <= 100):>11}"
                  f"{int(np.median(r)) if r else -1:>15}")
    print("\nDoc: cot 'co ung vien' la CAN TREN cung cua moi bo phan bo o ban kinh W.")
    print("Chenh giua no va bang 1 la phan bo phan bo con no; phan con lai no khong no.")

    # ---- 3. tách hai loại thất bại: phân bổ nợ được vs không nợ được -------
    W = 20
    print(f"\n=== 3. TACH HAI LOAI THAT BAI (ban kinh +-{W} frame) ===")
    print(f"{'nhom':<10}{'n':>4}{'da trung':>10}{'MAT do DAT DONG':>18}"
          f"{'khong co UV gan':>18}{'sai video':>11}")
    for ten, sel in nhom:
        trung = mat_dat = khong_uv = sai_vid = 0
        for i in sel:
            g = sach[i]
            fr, v = int(g["frame_idx"]), g["video_id"]
            co_dong = any(vv == v for vv, _ in rows[i])
            co_uv = any(c.video_id == v and abs(c.frame_idx - fr) <= W for c in cands[i])
            gan = any(vv == v and abs(f - fr) <= W for vv, f in rows[i])
            if gan:
                trung += 1
            elif not co_uv:
                khong_uv += 1
            elif not co_dong:
                sai_vid += 1
            else:
                mat_dat += 1
        print(f"{ten:<10}{len(sel):>4}{trung:>10}{mat_dat:>18}{khong_uv:>18}{sai_vid:>11}")
    print("\n  MAT do DAT DONG   = video dung CO dong, pool CO ung vien trong +-20,")
    print("                      ma khong dong nao roi vao +-20. Day la phan MA THAM SO")
    print("                      PHAN BO CO THE DOI — va chi phan nay.")
    print("  khong co UV gan   = trong 400 ung vien khong co keyframe nao cua video dung")
    print("                      gan dap an: nghen o khau SINH ung vien, lane khac.")
    print("  Luu y quan trong: oracle dinh vi noi-video KHONG bi chan boi cot nay vi no")
    print("  duoc dat frame id tuy y quanh dap an that. Nen tran +126% cua oracle KHONG")
    print("  phai la headroom cua viec chinh tham so phan bo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
