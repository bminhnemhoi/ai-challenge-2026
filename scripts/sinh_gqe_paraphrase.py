"""GQE bước 1 — sinh k=2 paraphrase tiếng Việt cho 132 câu sạch, cache theo câu.

Đặc tả: `docs/QUYET_DINH_ENCODER_TRAKE.md` §3.1 (khoản chi ~$0,3 đã duyệt).
GQE (arXiv:2408.07249): mở rộng truy vấn bằng LLM rồi ensemble — nguồn TỪ VỰNG
MỚI, đúng chỗ SigLIP nhạy bất thường với cách chọn từ.

Luật sinh (chống drift kiểu VIREO): giữ nguyên MỌI chi tiết định danh (tên,
số, màu, chữ trên băng rôn, hành động); chỉ đổi từ vựng + cú pháp; cấm thêm
thông tin; độ dài cỡ câu gốc. Ra JSON {"p1": ..., "p2": ...}.

Chỉ gọi API, KHÔNG nạp model nào — chạy song song an toàn với encode PE.
Sau bước này PHẢI soi tay ~20 paraphrase trước khi đếm (in sẵn 20 mẫu đầu).

    python -u scripts/sinh_gqe_paraphrase.py            # sinh + in 20 mẫu
    python -u scripts/sinh_gqe_paraphrase.py --chi-in   # chỉ in mẫu từ cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request as _rq
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from src.core.vlm import load_env  # noqa: E402

MODEL = "gpt-5.2"
GIA_VAO, GIA_RA = 1.25 / 1e6, 10.0 / 1e6  # USD/token
NGAN_SACH = 1.0  # USD, trần cứng

PROMPT = """Viết lại câu truy vấn video sau thành ĐÚNG 2 dị bản tiếng Việt.

LUẬT (bắt buộc):
- GIỮ NGUYÊN mọi chi tiết định danh: tên riêng, con số, màu sắc, chữ xuất hiện
  trên màn hình/băng rôn, hành động, vật thể. Không dịch, không làm tròn số.
- CHỈ đổi từ vựng (từ đồng nghĩa) và cấu trúc câu. CẤM thêm chi tiết mới,
  CẤM bỏ chi tiết, CẤM suy diễn.
- Độ dài xấp xỉ câu gốc. Văn phong tự nhiên như người mô tả cảnh video.

Câu gốc: {cau}

Trả về DUY NHẤT một JSON: {{"p1": "...", "p2": "..."}}"""


def goi(cau: str, key: str, tien: dict) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(cau=cau.strip())}],
        "max_completion_tokens": 2000,
    }).encode()
    r = json.load(_rq.urlopen(_rq.Request(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=body), timeout=180))
    u = r.get("usage") or {}
    tien["vao"] += int(u.get("prompt_tokens") or 0)
    tien["ra"] += int(u.get("completion_tokens") or 0)
    txt = (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError(f"khong parse duoc: {txt[:120]}")
    d = json.loads(m.group(0))
    if not (d.get("p1") and d.get("p2")):
        raise ValueError("thieu p1/p2")
    return {"p1": str(d["p1"]).strip(), "p2": str(d["p2"]).strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_gqe"))
    ap.add_argument("--chi-in", action="store_true")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    import os
    key = os.environ.get("OPENAI_API_KEY")

    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    giu = [g for g in moi if not g.get("lan_truc")]
    cau_hoi = [g["kis_query_vi"] for g in giu]

    cdir = Path(args.cache)
    cdir.mkdir(parents=True, exist_ok=True)
    f_cache = cdir / "paraphrase.json"
    cache = json.loads(f_cache.read_text(encoding="utf-8")) if f_cache.exists() else {}

    tien = {"vao": 0, "ra": 0}
    if not args.chi_in:
        if not key:
            raise RuntimeError("khong co OPENAI_API_KEY trong .env")
        moi_sinh = 0
        for i, cau in enumerate(cau_hoi, 1):
            h = hashlib.sha1(cau.strip().encode()).hexdigest()[:16]
            if h in cache:
                continue
            usd = tien["vao"] * GIA_VAO + tien["ra"] * GIA_RA
            if usd > NGAN_SACH:
                print(f"CHAM TRAN ngan sach ${NGAN_SACH} — dung o cau {i}")
                break
            for lan in range(3):
                try:
                    cache[h] = goi(cau, key, tien)
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"  cau {i} lan {lan + 1}: {type(exc).__name__} "
                          f"{str(exc)[:80]}")
                    time.sleep(10 * (lan + 1))
            moi_sinh += 1
            if moi_sinh % 20 == 0:
                f_cache.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
                print(f"  {i}/{len(cau_hoi)} | ${tien['vao']*GIA_VAO + tien['ra']*GIA_RA:.3f}",
                      flush=True)
        f_cache.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                           encoding="utf-8")
        du = sum(1 for c in cau_hoi
                 if hashlib.sha1(c.strip().encode()).hexdigest()[:16] in cache)
        print(f"\nxong: {du}/{len(cau_hoi)} câu có paraphrase | mới sinh {moi_sinh} | "
              f"chi phí lượt này ${tien['vao']*GIA_VAO + tien['ra']*GIA_RA:.3f}")

    # --- in 20 mẫu đầu để soi tay (bắt buộc trước khi đếm)
    print("\n=== 20 MẪU SOI TAY (câu gốc / p1 / p2) ===")
    dem = 0
    for cau in cau_hoi:
        h = hashlib.sha1(cau.strip().encode()).hexdigest()[:16]
        if h not in cache:
            continue
        dem += 1
        if dem > 20:
            break
        print(f"\n[{dem}] GỐC: {cau.strip()[:150]}")
        print(f"    P1 : {cache[h]['p1'][:150]}")
        print(f"    P2 : {cache[h]['p2'][:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
