"""Đọc lưới TUNE của lane phân-bổ-sâu theo BIÊN, không chỉ theo đỉnh.

Chọn argmax trên 200 tổ hợp với n = 66 câu là một phép chọn rất dễ ăn nhiễu:
đỉnh cao nhất gần như luôn cao hơn mức thật của nó. Hai thứ chữa được điều đó
mà không tốn thêm dữ liệu:

* **Biên từng tham số** — điểm trung bình của mọi tổ hợp mang một giá trị, gộp
  trên các tham số còn lại. Một tham số thật sự quan trọng sẽ có biên dốc đều;
  một tham số chỉ ăn may sẽ có biên phẳng và đỉnh nằm lẻ loi.
* **Vùng lân cận của đỉnh** — nếu tổ hợp thắng đứng trên một cao nguyên (các
  hàng xóm cũng cao) thì đó là tín hiệu; nếu nó là gai nhọn giữa vùng thấp thì
  đó là nhiễu chọn.

Không có dữ liệu mới nào bị đọc ở đây — chỉ đọc lại cache TUNE đã ghi.

    python -u scripts/doc_luoi_phan_bo_sau.py
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

from scripts.do_phan_bo_sau import (  # noqa: E402
    LUOI_BUOC,
    LUOI_DEPTH,
    LUOI_NFLAT,
    LUOI_NHIET,
    LUOI_NUA,
    LUOI_SIGMA,
    NEN_CP,
)

TEN_CP = ("nhiet", "sigma", "nua_cua_so", "luoi")
LUOI_CP = (LUOI_NHIET, LUOI_SIGMA, LUOI_NUA, LUOI_BUOC)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ket", default=str(ROOT / "data" / "cache_phan_bo_sau"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    args = ap.parse_args()

    raw = json.loads((Path(args.ket) / "tune.json").read_text(encoding="utf-8"))
    cp = [r for r in raw if r["ho"] == "coverage"]
    hy = [r for r in raw if r["ho"] == "hybrid"]
    print(f"luoi TUNE: {len(cp)} to hop coverage + {len(hy)} to hop hybrid")

    # nhóm hai cảnh trong TUNE (cùng cách chia phân tầng như do_phan_bo_sau)
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    sach = [g for g in moi if not g.get("lan_truc")]
    hai = [i for i, g in enumerate(sach) if g.get("co_2_canh")]
    mot = [i for i, g in enumerate(sach) if not g.get("co_2_canh")]
    i_tune = sorted(hai[0::2] + mot[0::2])
    vt_hai = [k for k, i in enumerate(i_tune) if i in set(hai)]

    def d_hai(r):
        return float(np.array(r["per_q"])[vt_hai].mean())

    nen = next(r for r in cp if tuple(r["tham"]) == tuple(NEN_CP))
    print(f"nen san xuat (coverage {NEN_CP}): TUNE {nen['diem']:.4f} | "
          f"HAI canh {d_hai(nen):.4f}\n")

    print("=== BIEN tung tham so (trung binh moi to hop mang gia tri do) ===")
    for k, (ten, gia) in enumerate(zip(TEN_CP, LUOI_CP)):
        print(f"\n  {ten}")
        print(f"{'':6}{'gia tri':>10}{'TUNE tb':>10}{'TUNE max':>10}{'HAI canh tb':>13}")
        for g in gia:
            sub = [r for r in cp if r["tham"][k] == g]
            if not sub:
                continue
            m = np.mean([r["diem"] for r in sub])
            mx = max(r["diem"] for r in sub)
            mh = np.mean([d_hai(r) for r in sub])
            danh = "  <- nen" if g == NEN_CP[k] else ""
            print(f"{'':6}{g:>10g}{m:>10.4f}{mx:>10.4f}{mh:>13.4f}{danh}")

    top = sorted(cp, key=lambda r: -r["diem"])
    best = top[0]
    print(f"\n=== DINH: {dict(zip(TEN_CP, best['tham']))} -> {best['diem']:.4f} "
          f"({100*(best['diem']/nen['diem']-1):+.1f}% so nen) ===")
    print("\nHang xom truc tiep (doi DUNG MOT tham so mot buoc):")
    bt = best["tham"]
    for k, (ten, gia) in enumerate(zip(TEN_CP, LUOI_CP)):
        j = list(gia).index(bt[k])
        for jj in (j - 1, j + 1):
            if 0 <= jj < len(gia):
                th = list(bt)
                th[k] = gia[jj]
                r = next((x for x in cp if tuple(x["tham"]) == tuple(th)), None)
                if r:
                    print(f"  {ten:>10} {bt[k]:g} -> {gia[jj]:g} : {r['diem']:.4f} "
                          f"({100*(r['diem']/best['diem']-1):+.1f}% so dinh)")

    print("\n=== hybrid: bang day du ===")
    print(f"{'n_flat':>8}" + "".join(f"{f'dc={d:g}':>10}" for d in LUOI_DEPTH))
    for nf in LUOI_NFLAT:
        hang = []
        for d in LUOI_DEPTH:
            r = next((x for x in hy if tuple(x["tham"]) == (nf, d)), None)
            hang.append(f"{r['diem']:>10.4f}" if r else f"{'-':>10}")
        print(f"{nf:>8}" + "".join(hang))
    print(f"\nMoi o hybrid deu so voi nen coverage {nen['diem']:.4f} tren cung TUNE.")

    print("\n=== phan phoi diem toan luoi coverage ===")
    d = np.array([r["diem"] for r in cp])
    print(f"  min {d.min():.4f} | p25 {np.percentile(d,25):.4f} | trung vi "
          f"{np.median(d):.4f} | p75 {np.percentile(d,75):.4f} | max {d.max():.4f}")
    print(f"  so to hop >= nen ({nen['diem']:.4f}): {(d >= nen['diem']).sum()}/{len(d)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
