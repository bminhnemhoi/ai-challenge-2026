"""Diff CẤU TRÚC của truy xuất-thêm-cảnh-B trên ĐỀ THẬT (không có ground truth).

Phép đo trên bộ đo mới nói cảnh B đáng +23,3% ở nhóm câu qua cổng. Nhưng bộ đo
mới gồm câu do máy sinh; đề thật do BTC viết. Script này không nói bên nào đúng —
nó nói **cảnh B sẽ đổi những gì** khi chạy trên phân bố đề thật:

  * bao nhiêu câu qua cổng hai cảnh;
  * mỗi câu thêm được bao nhiêu ứng viên MỚI (không trùng 400 ứng viên gốc);
  * video ở dòng 1 có đổi không — đây là con số rủi ro: đổi video dòng 1 là
    đánh đổi R@1, thứ đắt nhất trong công thức chấm;
  * bao nhiêu dòng trong 100 dòng cuối cùng đến từ ứng viên cảnh B.

Nhãn hai cảnh đọc từ cache có sẵn (``data/cache_cap_thoi_gian/nhan_de``) nên
script này **không gọi LLM lần nào**.

    python -u scripts/diff_canh_b_de_that.py --de round1/queries round2/queries
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

from scripts.experiment_cap_thoi_gian import KhoSims, _plan  # noqa: E402
from scripts.make_submission import (  # noqa: E402
    DEFAULT_N_FLAT,
    allocate_rows,
    detect_task,
    ranked_hits,
    read_en_override,
    read_query_text,
    split_qa,
)
from src.core.submission import MAX_ROWS, Candidate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--de", nargs="+", default=["round1/queries", "round2/queries"])
    ap.add_argument("--nhan", default=str(ROOT / "data" / "cache_cap_thoi_gian" / "nhan_de"))
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--allocator", default="coverage")
    args = ap.parse_args()

    nhan_dir = Path(args.nhan)
    nhan = {}
    for p in nhan_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        nhan[d.get("stem") or p.stem] = d

    files = []
    for d in args.de:
        dd = Path(d)
        if not dd.is_dir():
            continue
        for p in sorted(dd.glob("*.txt")):
            if not p.name.lower().endswith((".en.txt", ".vi.txt")):
                files.append(p)
    if not files:
        print("khong tim thay cau nao")
        return 2

    bat = [p for p in files if (nhan.get(p.stem) or {}).get("co_2_canh")
           and (nhan.get(p.stem) or {}).get("canh_B_vi")]
    print(f"{len(files)} cau de that | cong hai canh BAT: {len(bat)} "
          f"({100*len(bat)/len(files):.0f}%) | nhan doc tu cache, 0 lan goi LLM")
    if not bat:
        return 0

    print("nap chi muc (mot lan) ...", flush=True)
    from src.core.kis_engine import KISEngine

    eng = KISEngine(args.data).load()
    kho = KhoSims(args.data, False)

    print(f"\n{'cau':<26}{'ung vien MOI':>13}{'dong tu canh B':>16}{'video dong 1':>28}")
    print("-" * 84)
    tong_moi, tong_dong, doi_video = [], [], 0
    for p in bat:
        text = read_query_text(p) or ""
        task = detect_task(p.name)
        probe = split_qa(text)[0] or text if task == "qa" else text
        # ranked_hits = dung duong san xuat (4 prompt + peak preference + object
        # boost), khong phai eng.search tho — hai cai xep hang khac nhau
        hits = ranked_hits(eng, probe, read_en_override(p))
        c0 = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame) for h in hits]
        co = {(c.video_id, int(c.frame_idx)) for c in c0}

        d = nhan[p.stem]
        s = kho.lay(d["canh_B_vi"], d.get("canh_B_en") or "")
        k = min(args.m, len(s) - 1)
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        them = []
        for j in top:
            md = eng.metadata[int(j)]
            key = (md["video_id"], int(md["frame_idx"]))
            if key in co:
                continue
            co.add(key)
            them.append(Candidate(key[0], key[1], float(s[int(j)]),
                                  eng.last_frame.get(key[0])))

        r0 = allocate_rows(c0, args.allocator, DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
        r1 = allocate_rows(c0 + them, args.allocator, DEFAULT_N_FLAT, _plan())[:MAX_ROWS]
        khac = sum(1 for x in r1 if x not in set(r0))
        v0 = r0[0][0] if r0 else "?"
        v1 = r1[0][0] if r1 else "?"
        doi = v0 != v1
        doi_video += doi
        tong_moi.append(len(them))
        tong_dong.append(khac)
        print(f"{p.stem:<26}{len(them):>13}{khac:>16}"
              f"{('DOI: ' + v0 + ' -> ' + v1) if doi else 'giu nguyen':>28}")

    a = np.array(tong_moi)
    b = np.array(tong_dong)
    print(f"\nung vien MOI moi cau : trung vi {np.median(a):.0f}, "
          f"min {a.min()}, max {a.max()}")
    print(f"dong doi trong 100   : trung vi {np.median(b):.0f}, "
          f"min {b.min()}, max {b.max()}")
    print(f"video dong 1 DOI     : {doi_video}/{len(bat)} cau")
    print("\nDoc so nay the nao: 'video dong 1 doi' la con so RUI RO — R@1 dat nhat")
    print("trong cong thuc cham. Doi it nghia la canh B chu yeu them lua chon o duoi,")
    print("doi nhieu nghia la no dang viet lai ca cau tra loi hang dau.")
    print("Diff CAU TRUC: no khong noi ben nao dung, chi noi canh B doi nhung gi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
