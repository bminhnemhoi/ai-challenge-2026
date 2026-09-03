"""Phép-đếm-trước VLM xếp-lại nội-video — thước trận tay đôi, mở lại cửa cũ ĐÚNG LUẬT.

Cửa "VLM xếp lại nội-video" bị đóng trên bộ CŨ với lý do: tín hiệu 0 đồng ăn
nhiều hơn trên đúng cùng cơ chế (+80,5% vs +61,3% TEST) nên đóng góp biên ÂM.
Điều kiện ấy ĐÃ THAY ĐỔI (03/09): trên nhóm MỘT cảnh, cả bốn tín hiệu 0 đồng
(NNN/PRF/GQE/cut-score) đều chết trên thước trận-tay-đôi — không còn tín hiệu
0 đồng nào để "thắng biên" nữa. Mở lại phải qua đúng cổng rẻ nhất: phép đếm.

Thước (công bố trước, `QUYET_DINH_ENCODER_TRAKE.md` §3): trên các câu MỘT cảnh
sạch có keyframe-đáp-án đứng hạng 2–3 nội-video, VLM chấm ĐỘC LẬP từng khung
(một-ảnh-một-request, giao thức đã kiểm của `kiem_neo_don_anh.py` — không có
danh sách nên không có chỉ số để nhầm) cho khung hạng-1 cũ và keyframe-đáp-án:

    thắng ⟺ diem(đáp án) > diem(hạng-1)   —   cần ≥62% số trận
    hoà điểm KHÔNG tính là thắng (bảo thủ); in riêng số trận hoà.

Cache theo (video, frame, hash câu) nên chạy lại 0 đồng. Trận nào bị xoay
model giữa cặp (cạn quota giữa chừng) được cờ riêng để minh bạch.

    python -u scripts/dem_vlm_tran_tay_doi.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.do_cap_thoi_gian_moi import canh_cua  # noqa: E402
from scripts.kiem_neo_don_anh import hoi_mot_anh  # noqa: E402
from src.core.vlm import DEFAULT_MODEL, VLMJudge  # noqa: E402

NGUONG = 62.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache-uv", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_vlm_tran"))
    ap.add_argument("--hang-toi-da", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    data = Path(args.data)
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache_uv) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    assert len(uv) == len(moi), "cache uv lech bo do — chay lai do_bo_do_moi.py"

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    ten_of = {(m["video_id"], int(m["frame_idx"])): m["frame_filename"] for m in meta}
    kf = {}
    for m in meta:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf.items()}
    del meta

    # --- gom trận y hệt dem_prf_rocchio / dem_gqe_tran, trên bộ SẠCH mở rộng
    tran = []
    for i in giu:
        g = moi[i]
        if canh_cua(g):
            continue
        vid, dap = g["video_id"], int(g["frame_idx"])
        a = kf.get(vid)
        if a is None or not len(a):
            continue
        fd = int(a[int(np.argmin(np.abs(a - dap)))])
        trong = sorted(((float(s), int(f)) for v, f, s, _lf in uv[i] if v == vid),
                       key=lambda t: -t[0])
        hang = next((r for r, (_s, f) in enumerate(trong, 1) if f == fd), None)
        if hang is None or not (2 <= hang <= args.hang_toi_da):
            continue
        f1 = trong[0][1]
        t1, td = ten_of.get((vid, f1)), ten_of.get((vid, fd))
        if not t1 or not td:
            continue
        tran.append((g["kis_query_vi"].strip(), vid, f1, t1, fd, td))

    print(f"số trận (bộ sạch mở rộng, đáp án hạng 2..{args.hang_toi_da}): {len(tran)}")

    cdir = Path(args.cache)
    cdir.mkdir(parents=True, exist_ok=True)
    f_cache = cdir / "diem.json"
    cache = json.loads(f_cache.read_text(encoding="utf-8")) if f_cache.exists() else {}

    judge = VLMJudge(args.data, model=args.model)

    def cham(cau, vid, ten):
        k = hashlib.sha1(f"{vid}|{ten}|{cau}".encode()).hexdigest()[:20]
        if k in cache:
            return cache[k]
        blob = judge._fetch(vid, ten)
        r = hoi_mot_anh(judge, blob, cau)
        d = int(r.get("diem", -1))
        cache[k] = d
        f_cache.write_text(json.dumps(cache, indent=0), encoding="utf-8")
        time.sleep(1.0)
        return d

    thang, hoa, thua, xoay, loi = 0, 0, 0, 0, 0
    for j, (cau, vid, f1, t1, fd, td) in enumerate(tran, 1):
        n_het = len(judge.exhausted)
        try:
            d1 = cham(cau, vid, t1)
            dd = cham(cau, vid, td)
        except Exception as exc:  # noqa: BLE001
            print(f"  trận {j}: LỖI {type(exc).__name__}: {str(exc)[:60]}")
            loi += 1
            continue
        if len(judge.exhausted) != n_het:
            xoay += 1  # cặp này có xoay model giữa chừng — cờ minh bạch
        if d1 < 0 or dd < 0:
            loi += 1
            continue
        if dd > d1:
            thang += 1
        elif dd == d1:
            hoa += 1
        else:
            thua += 1
        if j % 5 == 0:
            print(f"  {j}/{len(tran)} | thắng {thang} hoà {hoa} thua {thua}",
                  flush=True)

    n = thang + hoa + thua
    print(f"\n=== KẾT QUẢ (n={n} trận chấm được; lỗi {loi}; cặp bị xoay model {xoay}) ===")
    if n == 0:
        print("  không chấm được trận nào — hết quota, chạy lại sau (cache giữ).")
        return 1
    print(f"  đáp án THẮNG: {thang}/{n} = {100*thang/n:.0f}%")
    print(f"  hoà điểm    : {hoa}/{n} = {100*hoa/n:.0f}% (không tính thắng)")
    print(f"  thua        : {thua}/{n} = {100*thua/n:.0f}%")
    ty = 100 * thang / n
    print(f"\n=== KẾT LUẬN (ngưỡng ≥{NGUONG:.0f}%, công bố trước) ===")
    if ty >= NGUONG:
        print("  ĐI TIẾP: VLM phân xử được trận tay đôi — đo đầy đủ 5 cổng")
        print("  (biến thể hẹp: chỉ hoán vị điểm nội-video theo diem VLM,")
        print("   tập dòng bất biến; TUNE/TEST chia tầng trên mục CHƯA đọc).")
    else:
        print("  ÂM: VLM cũng không phân xử được trận tay đôi ở nhóm một cảnh.")
        print("  Ghi bảng cửa đóng — nhóm một cảnh hết đường xếp-lại, đầu tư")
        print("  chuyển hẳn sang tầng SINH ứng viên và PHÂN BỔ dòng.")
    print(f"\nchi phí: {judge.calls} lượt gọi, {judge.tokens_in} vào / "
          f"{judge.tokens_out} ra (Gemini free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
