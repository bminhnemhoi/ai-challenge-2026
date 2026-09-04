"""Chấm TỔNG DUYỆT — hệ MỚI vs bản ĐÃ NỘP vòng 2 vs mốc người-xác-minh.

Mốc: các dòng ``query = Lxx_Vyyy:frame|frame2`` trong round2/picks_final*.txt
(người xem video chọn — chân lý tốt nhất đang có; KHÔNG phải đáp án BTC).

Với KIS/QA: hạng của dòng đầu tiên đúng video VÀ |frame − mốc| ≤ W (W=10/20/50)
+ hạng chỉ-cần-đúng-video. Với TRAKE: đúng video ở dòng 1 + số sự kiện có mốc
nằm trong ±10 của cột tương ứng. In bảng từng câu + tổng better/same/worse.

    python -u scripts/cham_tong_duyet.py --moi round2/tong_duyet_0409/csv --cu round2/base/csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()


def nap_moc(files):
    moc = {}
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(query-[\w-]+)\s*=\s*(L\d\d_V\d\d\d):([\d|]+)", line)
            if m:
                moc[m.group(1)] = (m.group(2), [int(x) for x in m.group(3).split("|")])
    return moc


def nap_csv(d, ten):
    p = Path(d) / f"{ten}.csv"
    if not p.exists():
        return None
    rows = []
    with open(p, encoding="utf-8-sig", newline="") as fh:
        for r in csv.reader(fh):
            if r:
                rows.append(r)
    return rows


def hang_kis(rows, vid, frame, w):
    for i, r in enumerate(rows, 1):
        try:
            if r[0] == vid and abs(int(r[1]) - frame) <= w:
                return i
        except (ValueError, IndexError):
            continue
    return None


def hang_video(rows, vid):
    for i, r in enumerate(rows, 1):
        if r and r[0] == vid:
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--moi", required=True)
    ap.add_argument("--cu", required=True)
    ap.add_argument("--picks", nargs="+", default=[
        str(ROOT / "round2" / "picks_final.txt"),
        str(ROOT / "round2" / "picks_final2.txt"),
        str(ROOT / "round2" / "picks_kis_final.txt"),
    ])
    args = ap.parse_args()

    moc = nap_moc(args.picks)
    print(f"mốc người-xác-minh: {len(moc)} câu\n")
    print(f"{'câu':<24}{'':<7}{'video':>7}{'±10':>6}{'±20':>6}{'±50':>6}")
    print("-" * 58)
    tot = {"better": 0, "same": 0, "worse": 0}
    for ten, (vid, frames) in sorted(moc.items()):
        rm = nap_csv(args.moi, ten)
        rc = nap_csv(args.cu, ten)
        if rm is None:
            print(f"{ten:<24} THIẾU CSV MỚI")
            continue
        la_trake = "trake" in ten
        if la_trake:
            v_m = rm[0][0] == vid if rm else False
            v_c = rc[0][0] == vid if rc else False
            hit_m = hit_c = 0
            if v_m and len(rm[0]) - 1 >= len(frames):
                hit_m = sum(1 for j, f in enumerate(frames)
                            if any(abs(int(r[1 + j]) - f) <= 10 for r in rm
                                   if r[0] == vid and len(r) > 1 + j))
            if rc and v_c and len(rc[0]) - 1 >= len(frames):
                hit_c = sum(1 for j, f in enumerate(frames)
                            if any(abs(int(r[1 + j]) - f) <= 10 for r in rc
                                   if r[0] == vid and len(r) > 1 + j))
            print(f"{ten:<24}{'MỚI':<7}{'OK' if v_m else 'SAI':>7}"
                  f"{hit_m}/{len(frames):>4}")
            print(f"{'':<24}{'CŨ':<7}{'OK' if v_c else 'SAI':>7}"
                  f"{hit_c}/{len(frames):>4}")
            key = (v_m, hit_m)
            key_c = (v_c, hit_c)
            tot["better" if key > key_c else ("same" if key == key_c else "worse")] += 1
            continue
        f0 = frames[0]
        for nhan, rows in (("MỚI", rm), ("CŨ", rc)):
            if rows is None:
                print(f"{ten:<24}{nhan:<7} THIẾU CSV")
                continue
            hv = hang_video(rows, vid)
            h10 = hang_kis(rows, vid, f0, 10)
            h20 = hang_kis(rows, vid, f0, 20)
            h50 = hang_kis(rows, vid, f0, 50)
            print(f"{ten if nhan == 'MỚI' else '':<24}{nhan:<7}"
                  f"{hv if hv else '—':>7}{h10 if h10 else '—':>6}"
                  f"{h20 if h20 else '—':>6}{h50 if h50 else '—':>6}")
        def diem(rows):
            h = hang_kis(rows, vid, f0, 20) if rows else None
            hv = hang_video(rows, vid) if rows else None
            return (-(h or 999), -(hv or 999))
        dm, dc = diem(rm), diem(rc)
        tot["better" if dm > dc else ("same" if dm == dc else "worse")] += 1

    print(f"\nTỔNG (mới so cũ trên mốc người): better {tot['better']} | "
          f"same {tot['same']} | worse {tot['worse']}")
    print("\nLưu ý đọc số: mốc = lựa chọn người vòng 2 (đã ăn 10,0 điểm), KHÔNG"
          " phải đáp án BTC; 'CŨ' là CSV máy-đã-nộp SAU khi người sửa — nên"
          " same trên các câu người đã sửa là kỳ vọng; giá trị nằm ở: MỚI tự"
          " động đạt được bao nhiêu phần của bản người-sửa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
