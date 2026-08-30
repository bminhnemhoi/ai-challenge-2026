"""Gắn nhãn "câu này có cấu trúc HAI CẢNH nối tiếp không?" cho 60 câu ground truth.

Đây là **cổng** của lane truy vấn cặp thời gian (docs/NGHIEN_CUU_SOTA.md §③).
Công thức ``s_temp(i) = HM(s_A(i), max_{j cùng video, i<j<=i+W} s_B(j))`` chỉ được
bật cho câu thật sự mô tả hai cảnh NỐI TIẾP; câu một cảnh phải giữ nguyên 100%
đường cũ.  Nếu số câu qua cổng quá ít thì lane này không có lực thống kê — và
điều đó phải biết TRƯỚC khi viết bộ chấm, không phải sau.

Mỗi câu là MỘT request text (không ảnh), nhiệt độ 0 để tái lập, cache xuống
``data/cache_cap_thoi_gian/nhan/<idx>.json`` nên chạy lại không tốn quota.

    python scripts/gan_nhan_hai_canh.py            # gắn nhãn (dùng cache nếu có)
    python scripts/gan_nhan_hai_canh.py --refresh  # gọi lại toàn bộ
    python scripts/gan_nhan_hai_canh.py --bang     # chỉ in bảng từ cache

Chế độ thứ hai — gắn nhãn ĐỀ THẬT của BTC (không có ground truth, chỉ để đếm tỉ
lệ câu có cấu trúc hai cảnh trên phân bố ra đề thật):

    python scripts/gan_nhan_hai_canh.py --de round1/queries round2/queries
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

CACHE = ROOT / "data" / "cache_cap_thoi_gian" / "nhan"

#: Nhiệt 0 + prompt cố định => nhãn tái lập được.  Phiên bản prompt nằm trong
#: cache, đổi prompt là cache tự hỏng (không lẫn nhãn của hai prompt khác nhau).
PROMPT_VERSION = 2

_PROMPT = """Bạn đang phân loại câu mô tả dùng để TÌM MỘT KHOẢNH KHẮC trong video.

CÂU MÔ TẢ (tiếng Việt):
{vi}

BẢN TIẾNG ANH (do người viết):
{en}

CÂU HỎI: câu trên mô tả MỘT cảnh duy nhất (mọi thứ nêu ra đều nhìn thấy được
trong CÙNG MỘT khung hình), hay mô tả HAI cảnh NỐI TIẾP nhau theo thời gian
(cảnh A xảy ra TRƯỚC, rồi cảnh B xảy ra SAU, và KHÔNG thể thấy cả hai trong
cùng một khung hình)?

CHỈ trả co_2_canh = true khi CẢ HAI điều kiện đúng:
  1. Câu có dấu hiệu TRÌNH TỰ THỜI GIAN rõ ràng: "rồi", "sau đó", "tiếp theo",
     "trước khi", "sau khi", "chuyển cảnh sang", "camera lia sang", "cắt sang",
     "bắt đầu bằng ... rồi ...", "vừa ... thì ...".
  2. Hai phần đó KHÁC NHAU về nội dung hình ảnh tới mức KHÔNG thể nằm chung
     một khung hình.

TUYỆT ĐỐI KHÔNG coi là hai cảnh:
  - nhiều người/vật cùng có mặt trong một khung ("A đứng cạnh B", "A phía sau B")
  - một hành động liên tục trong một cảnh ("đang trộn thịt với hành lá")
  - bối cảnh + chi tiết của cùng một khung ("trên nền sọc đỏ trắng có logo",
    "với dòng chữ ở góc dưới bên phải")
  - quan hệ KHÔNG GIAN ("phía sau", "bên cạnh", "ở góc", "trên nền")
  - liệt kê thuộc tính ("áo trắng, đeo tạp dề, tóc buộc")

Nếu co_2_canh = true: viết canh_A và canh_B thành hai câu mô tả ĐỘC LẬP, mỗi câu
tự đứng được như một truy vấn tìm khung hình riêng (nêu lại chủ thể, đừng dùng
đại từ), và kèm bản tiếng Anh tự nhiên cho mỗi cảnh.

Trả về DUY NHẤT một object JSON, không giải thích thêm:
{{"co_2_canh": true hoặc false, "do_tin": 0-100,
  "canh_A_vi": "", "canh_A_en": "", "canh_B_vi": "", "canh_B_en": "",
  "ly_do": "một câu ngắn giải thích quyết định"}}
Nếu co_2_canh = false thì bốn trường cảnh để chuỗi rỗng."""


def _goi(client, types_mod, models, prompt: str):
    """Một request text, xoay vòng model chống 429 (cùng luật với src/core/vlm.py)."""
    from src.core.vlm import _is_daily_quota

    last_err = None
    het_quota = set()
    for model in models:
        if model in het_quota:
            continue
        for attempt in range(3):
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types_mod.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1200,
                        response_mime_type="application/json",
                    ),
                )
                m = re.search(r"\{.*\}", r.text or "", re.S)
                if not m:
                    last_err = f"{model}: không parse được JSON"
                    break
                return json.loads(m.group(0)), model
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                last_err = f"{model}: {type(exc).__name__}: {msg[:90]}"
                if _is_daily_quota(msg):
                    het_quota.add(model)
                    break
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep((8, 22, 40)[min(attempt, 2)])
                    continue
                if "503" in msg or "UNAVAILABLE" in msg:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                break
    raise RuntimeError(last_err or "không rõ")


def _chuan_hoa(raw: dict) -> dict:
    """Ép nhãn về đúng hình dạng, và HẠ CỜ khi hai cảnh không dùng được.

    Một nhãn ``co_2_canh=true`` mà thiếu văn bản cảnh B thì không tính được
    ``max_j s_B(j)`` — để nó bật là để một câu chạy công thức trên chuỗi rỗng.
    """
    co = bool(raw.get("co_2_canh"))
    a_vi = str(raw.get("canh_A_vi") or "").strip()
    b_vi = str(raw.get("canh_B_vi") or "").strip()
    a_en = str(raw.get("canh_A_en") or "").strip()
    b_en = str(raw.get("canh_B_en") or "").strip()
    if co and not (a_vi and b_vi):
        co = False
    return {
        "co_2_canh": co,
        "do_tin": float(raw.get("do_tin") or 0),
        "canh_A_vi": a_vi if co else "",
        "canh_A_en": a_en if co else "",
        "canh_B_vi": b_vi if co else "",
        "canh_B_en": b_en if co else "",
        "ly_do": str(raw.get("ly_do") or "")[:300],
    }


def nap_nhan(data_dir: Path = ROOT / "data"):
    """Đọc toàn bộ nhãn đã cache; thiếu câu nào trả None ở chỗ đó."""
    gt = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))
    out = []
    for i in range(len(gt)):
        p = CACHE / f"{i:02d}.json"
        if not p.exists():
            out.append(None)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append(d if d.get("prompt_version") == PROMPT_VERSION else None)
    return gt, out


def _khach(args):
    """Gemini client + danh sách model, hoặc None nếu thiếu key."""
    from src.core.vlm import DEFAULT_MODEL, FALLBACK_MODELS, load_env

    load_env(ROOT / ".env")
    import os

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    from google import genai
    from google.genai import types

    return (genai.Client(api_key=key), types,
            [DEFAULT_MODEL] + [m for m in FALLBACK_MODELS if m != DEFAULT_MODEL])


def cmd_de(args) -> int:
    """Cùng prompt, cùng nhiệt 0, nhưng chạy trên ĐỀ THẬT của BTC.

    60 câu ground truth do đội tự viết; đề thật do BTC viết.  Nếu tỉ lệ câu có
    cấu trúc hai cảnh khác nhau giữa hai tập thì bộ đo 60 câu KHÔNG đại diện cho
    thứ sẽ gặp trong trận — và đó là điều phải biết trước khi kết luận gì về
    lane này.  Không có ground truth ở đây nên chỉ ĐẾM, không chấm.
    """
    from scripts.make_submission import decode_text

    kh = _khach(args)
    out_dir = ROOT / "data" / "cache_cap_thoi_gian" / "nhan_de"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for d in args.de:
        for p in sorted(Path(d).glob("*.txt")):
            if not p.name.lower().endswith((".en.txt", ".vi.txt")):
                files.append(p)
    print(f"{len(files)} câu đề thật trong {', '.join(args.de)}")

    ket = []
    for p in files:
        cache = out_dir / f"{p.stem}.json"
        rec = None
        if cache.exists() and not args.refresh:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if d.get("prompt_version") == PROMPT_VERSION:
                rec = d
        if rec is None:
            if kh is None:
                print("thiếu GEMINI_API_KEY trong .env")
                return 2
            text = (decode_text(p.read_bytes()) or "").strip()
            prompt = _PROMPT.format(vi=text, en="(không có)")
            try:
                raw, model = _goi(kh[0], kh[1], kh[2], prompt)
            except Exception as exc:  # noqa: BLE001
                print(f"  {p.stem}: LỖI {exc}")
                continue
            rec = _chuan_hoa(raw)
            rec.update({"stem": p.stem, "model": model, "prompt_version": PROMPT_VERSION,
                        "query": text[:1200]})
            cache.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        ket.append((p.stem, rec))

    bat = [(s, r) for s, r in ket if r["co_2_canh"]]
    print(f"\n=== ĐỀ THẬT: {len(bat)}/{len(ket)} câu có cổng BẬT "
          f"({100*len(bat)/max(len(ket),1):.0f}%) ===")
    for s, r in bat:
        print(f"  {s:<22} A: {r['canh_A_vi'][:52]}")
        print(f"  {'':<22} B: {r['canh_B_vi'][:52]}")
    theo_vong = {}
    for s, r in ket:
        v = "vòng 1" if "-p1-" in s else "vòng 2"
        a, b = theo_vong.get(v, (0, 0))
        theo_vong[v] = (a + r["co_2_canh"], b + 1)
    for v, (a, b) in sorted(theo_vong.items()):
        print(f"  {v}: {a}/{b}")
    print(f"\ncache: {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--bang", action="store_true", help="chỉ in bảng từ cache, không gọi API")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--de", nargs="+", default=None,
                    help="thư mục đề thật của BTC -> chỉ đếm tỉ lệ, không chấm")
    args = ap.parse_args()

    if args.de:
        return cmd_de(args)

    data_dir = Path(args.data)
    gt, nhan = nap_nhan(data_dir)
    n = args.limit or len(gt)

    thieu = [i for i in range(n) if nhan[i] is None or args.refresh]
    if thieu and not args.bang:
        kh = _khach(args)
        if kh is None:
            print("thiếu GEMINI_API_KEY trong .env")
            return 2
        client, types, models = kh
        CACHE.mkdir(parents=True, exist_ok=True)

        print(f"gọi Gemini cho {len(thieu)} câu (nhiệt 0, cache lại) ...", flush=True)
        for k, i in enumerate(thieu):
            g = gt[i]
            prompt = _PROMPT.format(vi=g["kis_query_vi"], en=g.get("kis_query_en") or "(không có)")
            try:
                raw, model = _goi(client, types, models, prompt)
            except Exception as exc:  # noqa: BLE001
                print(f"  câu {i}: LỖI {exc}")
                continue
            rec = _chuan_hoa(raw)
            rec.update({"n": i, "model": model, "prompt_version": PROMPT_VERSION,
                        "kis_query_vi": g["kis_query_vi"]})
            (CACHE / f"{i:02d}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            nhan[i] = rec
            if (k + 1) % 10 == 0:
                print(f"  ... {k + 1}/{len(thieu)}", flush=True)

    # ---------------------------------------------------------------- bảng
    bat = [i for i in range(n) if nhan[i] and nhan[i]["co_2_canh"]]
    thieu_sau = [i for i in range(n) if nhan[i] is None]
    print(f"\n=== NHÃN HAI CẢNH: {len(bat)}/{n} câu có cổng BẬT ===")
    if thieu_sau:
        print(f"  ! {len(thieu_sau)} câu chưa có nhãn: {thieu_sau}")
    print(f"{'câu':>4} {'nửa':>5} {'tin':>4}  cảnh A  ||  cảnh B")
    for i in bat:
        d = nhan[i]
        print(f"{i:>4} {'chẵn' if i % 2 == 0 else 'lẻ':>5} {d['do_tin']:>4.0f}  "
              f"{d['canh_A_vi'][:60]}  ||  {d['canh_B_vi'][:60]}")
    chan = sum(1 for i in bat if i % 2 == 0)
    print(f"\nphân bố: {chan} câu chỉ số chẵn / {len(bat) - chan} câu chỉ số lẻ")
    if len(bat) < 8:
        print("\n  !! DƯỚI 8 CÂU — không đủ lực thống kê cho luật 2 sigma.")
        print("     Báo cáo phải là per-query, không được ép thành kết luận tổng.")
    print(f"\ncache: {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
