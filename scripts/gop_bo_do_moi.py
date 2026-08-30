"""Gộp 4 shard đã xác minh neo thành MỘT bộ đo, kèm cờ đánh dấu chỗ chưa sạch.

Bộ đo mới chỉ có giá trị nếu KHUNG NEO đúng — neo sai không làm nó yếu đi mà
làm nó **sai dấu**. Nên script này chỉ nhận mục đã qua ``kiem_neo_don_anh.py``
(một ảnh một request, không có chỉ số ảnh để nhầm) và ghi rõ mục nào đã bị dời
neo, để mọi kết luận sau này truy ngược được.

Ba nhóm đầu ra:

  * ``data/ground_truth_moi.json`` — mục dùng được để chấm
  * ``data/gt_moi_can_soat.json``  — mục còn cờ nghi ngờ, KHÔNG dùng để chấm
  * in ra bảng phân bố + cảnh báo nhiễu đã biết

Cảnh báo nhiễu đã biết, ghi thành trường chứ không giấu đi:

  * **shard c bị lẫn trục**: bước sinh dùng ``dai[i % len(dai)]`` cùng
    ``chi_tieu = (i % 2 == 0)``; với 2 dải thì hai trục trùng khít, nên trong
    shard c "câu hai cảnh" ĐỒNG NHẤT với "câu lấy từ L26" và "câu một cảnh"
    đồng nhất với L27. Mọi phép so *hai cảnh vs một cảnh* phải LOẠI shard c,
    nếu không nó đo đặc thù kênh chứ không đo cấu trúc thời gian.
  * **shard b thiếu trường cảnh_A/cảnh_B** — nhãn hai cảnh của nó là tự khai,
    chưa tách mệnh đề nên chưa đối chiếu được với ảnh.

    python scripts/gop_bo_do_moi.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--shard", default="a,b,c,d")
    args = ap.parse_args()

    data = Path(args.data)
    dung, soat = [], []
    for sh in [s.strip() for s in args.shard.split(",") if s.strip()]:
        f = data / f"gt_moi_shard_{sh}_neo_sua.json"
        if not f.is_file():
            print(f"shard {sh}: chua co {f.name} — chay kiem_neo_don_anh.py --ap-dung truoc")
            continue
        for muc in json.loads(f.read_text(encoding="utf-8")):
            m = dict(muc)
            m["shard"] = sh
            # shard c: trục "hai cảnh" trùng khít trục "dải video" — giữ mục lại
            # nhưng đánh dấu để mọi phép so hai-cảnh/một-cảnh loại nó ra
            m["lan_truc"] = (sh == "c")
            m["nhan_hai_canh_tu_khai"] = (sh == "b")
            tt = m.get("kiem_neo")
            (soat if tt == "nghi_ngo" else dung).append(m)

    (data / "ground_truth_moi.json").write_text(
        json.dumps(dung, ensure_ascii=False, indent=1), encoding="utf-8")
    (data / "gt_moi_can_soat.json").write_text(
        json.dumps(soat, ensure_ascii=False, indent=1), encoding="utf-8")

    doi = [m for m in dung if m.get("neo_sua_boi")]
    hai = [m for m in dung if m.get("co_2_canh")]
    sach = [m for m in dung if not m.get("lan_truc")]
    hai_sach = [m for m in sach if m.get("co_2_canh")]

    print(f"ground_truth_moi.json : {len(dung)} muc dung duoc")
    print(f"  trong do da DOI NEO : {len(doi)} muc "
          f"({', '.join(f'{m[chr(39)+chr(39)] if False else m['video_id']}' for m in doi[:8])})")
    print(f"  cau HAI CANH        : {len(hai)}/{len(dung)} = {len(hai)/max(1,len(dung)):.0%} "
          f"(de that BTC: 51%)")
    print(f"gt_moi_can_soat.json  : {len(soat)} muc con co nghi ngo, KHONG cham diem")
    print(f"\nBo SACH (loai shard c bi lan truc): {len(sach)} muc, "
          f"{len(hai_sach)} hai canh = {len(hai_sach)/max(1,len(sach)):.0%}")
    print("  -> dung bo SACH cho moi phep so hai-canh vs mot-canh")
    print(f"\nphan bo theo dai video: "
          f"{dict(sorted(Counter(m['video_id'].split('_')[0] for m in dung).items()))}")
    print(f"so video khac nhau: {len({m['video_id'] for m in dung})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
