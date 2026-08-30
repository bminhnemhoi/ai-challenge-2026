"""Xác minh KHUNG NEO của bộ đo mới bằng cách hỏi MỘT ẢNH MỘT REQUEST.

Vấn đề phải sửa, và nó là lỗi của phép đo chứ không phải của hệ thống: bước sinh
đưa cả đoạn 9–12 keyframe vào **một** request. Gemini tả nội dung đúng nhưng
**đánh sai số thứ tự ảnh**, nên khung neo trỏ lệch một cú cắt. Lane sinh mở ảnh
ra xem và bắt được 3/8 câu hai cảnh của shard a bị lệch đúng kiểu đó; tệ hơn,
bộ tự kiểm đầu tiên cũng lệch ±1 vì **mắc đúng nguyên nhân đó**.

Khung neo là ĐÁP ÁN của bộ đo. Neo sai không làm bộ đo yếu đi — nó làm bộ đo
**sai dấu**: hệ thống trả về đúng chỗ lại bị chấm là trượt, và mọi cải tiến đo
trên đó đều đọc ngược.

Cách sửa tận gốc, không cần mắt người: **bỏ hẳn việc đánh số ảnh.** Mỗi lần hỏi
chỉ gửi ĐÚNG MỘT ảnh cùng câu mô tả, và hỏi "ảnh này có khớp mô tả không, 0–100".
Không còn danh sách thì không còn chỗ để nhầm chỉ số. Quét cửa sổ ±``--quanh``
keyframe quanh neo, mỗi khung một request, rồi so điểm:

  * khung điểm cao nhất trùng neo hiện tại      -> XÁC NHẬN
  * một khung lân cận cao hơn rõ rệt (>= --chenh) -> ĐỀ NGHỊ DỜI (ghi ra, không tự ghi đè)
  * mọi khung đều thấp (< --nguong)               -> NGHI NGỜ: mô tả không khớp đâu cả

Không mục nào bị xoá và không mục nào bị ghi đè tự động — script chỉ **xếp loại**
và ghi bằng chứng, vì một bộ kiểm tự động ghi đè hàng loạt chính là cách lỗi
lệch-một-khung lan ra cả bộ đo.

    python -u scripts/kiem_neo_don_anh.py                 # cả 4 shard
    python -u scripts/kiem_neo_don_anh.py --shard a,c     # chỉ vài shard
    python -u scripts/kiem_neo_don_anh.py --ap-dung       # ghi neo đề nghị vào file mới
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from src.core.vlm import DEFAULT_MODEL, VLMJudge  # noqa: E402

PROMPT = """Đây là MỘT khung hình duy nhất lấy từ một video tin tức tiếng Việt.

Mô tả cần kiểm: "{mo_ta}"

Khung hình này có đúng là cảnh được mô tả không?

Chấm điểm:
- 90-100: khớp rõ ràng, mọi chi tiết chính đều thấy được trong ảnh
- 60-89 : đúng loại cảnh nhưng thiếu hoặc lệch một vài chi tiết
- 30-59 : cùng bối cảnh chung nhưng KHÔNG phải cảnh này
- 0-29  : khác hẳn

Chỉ chấm những gì THẤY trong ảnh. Đừng suy đoán từ ngữ cảnh video.
Liệt kê ở "khop" các chi tiết của mô tả mà bạn THẤY, ở "thieu" các chi tiết
mô tả có mà ảnh KHÔNG có.

Trả về DUY NHẤT JSON:
{{"diem": 0-100, "khop": "...", "thieu": "..."}}"""


def hoi_mot_anh(judge, blob, mo_ta):
    """Một ảnh, một request — không có danh sách nên không có chỉ số để nhầm."""
    from google.genai import types

    from src.core.vlm import RETRY_WAIT, _is_daily_quota

    client = judge._get_client()
    prompt = PROMPT.format(mo_ta=mo_ta.strip()[:600])
    last = None
    for model in judge._model_order():
        if model in judge.exhausted:
            continue
        for lan in range(3):
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=[types.Part.from_bytes(data=blob, mime_type="image/jpeg"), prompt],
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=600),
                )
                u = getattr(r, "usage_metadata", None)
                if u:
                    judge.calls += 1
                    judge.tokens_in += u.prompt_token_count or 0
                    judge.tokens_out += u.candidates_token_count or 0
                m = re.search(r"\{.*\}", r.text or "", re.S)
                if not m:
                    return {"diem": -1, "loi": "khong parse duoc"}
                return json.loads(m.group(0))
            except Exception as exc:  # noqa: BLE001
                last = exc
                msg = str(exc)
                if _is_daily_quota(msg):
                    judge.exhausted.add(model)
                    break
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(RETRY_WAIT[min(lan, len(RETRY_WAIT) - 1)])
                    continue
                if "503" in msg or "UNAVAILABLE" in msg:
                    time.sleep(2.0 * (lan + 1))
                    continue
                break
    raise last or RuntimeError("het model kha dung")


def hoi_openai(model: str, blob, mo_ta: str, tien: dict) -> dict:
    """Cùng câu hỏi, qua gpt-5.x. Một ảnh, một request — vẫn không có chỉ số nào để nhầm.

    Đáng tiền ở đúng việc này: xác minh neo là hạ tầng dùng lâu dài, làm sai một
    lần thì mọi phép đo sau đều đọc ngược. Ảnh gửi đi là bản 512px nên mỗi lượt
    rẻ hơn nhiều so với đường trả lời Q&A (ở đó gpt-5.2 THUA Gemini free và
    không đáng dùng — xem docs/NGHIEN_CUU_SOTA.md §1①).
    """
    import base64
    import os
    import urllib.request as _rq

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("khong co OPENAI_API_KEY trong .env")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(blob).decode()}},
            {"type": "text", "text": PROMPT.format(mo_ta=mo_ta.strip()[:600])},
        ]}],
        "max_completion_tokens": 2000,
    }).encode()
    r = json.load(_rq.urlopen(_rq.Request(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        data=body), timeout=180))
    u = r.get("usage") or {}
    tien["goi"] += 1
    tien["vao"] += int(u.get("prompt_tokens") or 0)
    tien["ra"] += int(u.get("completion_tokens") or 0)
    txt = (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {"diem": -1, "loi": "khong parse duoc"}
    return json.loads(m.group(0))


def mo_ta_neo(muc: dict) -> str:
    """Mô tả DÙNG ĐỂ KIỂM neo.

    Với câu hai cảnh, neo phải trỏ vào lúc **cảnh B bắt đầu**, nên phải kiểm
    bằng chính cảnh B — kiểm bằng cả câu (gồm cả cảnh A) thì khung nào trong
    đoạn cũng "khớp một nửa" và phép kiểm mất hết sức phân giải.
    """
    if muc.get("co_2_canh"):
        b = muc.get("canh_B") or muc.get("canh_B_vi") or ""
        if isinstance(b, dict):
            b = b.get("vi") or b.get("mo_ta") or ""
        if str(b).strip():
            return str(b)
    return muc.get("kis_query_vi", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--shard", default="a,b,c,d")
    ap.add_argument("--quanh", type=int, default=2, help="số keyframe mỗi bên quanh neo")
    ap.add_argument("--chenh", type=int, default=15, help="chênh điểm tối thiểu để đề nghị dời neo")
    ap.add_argument("--nguong", type=int, default=55, help="dưới mức này coi như không khung nào khớp")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--provider", choices=("gemini", "openai"), default="gemini")
    ap.add_argument("--openai-model", default="gpt-5.2")
    ap.add_argument("--ap-dung", action="store_true",
                    help="ghi các neo ĐỀ NGHỊ vào <shard>_neo_sua.json (vẫn không đụng file gốc)")
    args = ap.parse_args()

    data = Path(args.data)
    judge = VLMJudge(args.data, model=args.model)
    if not judge.ready:
        print("Khong co GEMINI_API_KEY trong .env")
        return 2

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    by_n = {(m["video_id"], int(m["n"])): m for m in meta}
    tien = {"goi": 0, "vao": 0, "ra": 0}
    cache_dir = data / "cache_kiem_neo"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tong = {"xac_nhan": 0, "de_nghi_doi": 0, "nghi_ngo": 0, "thieu_anh": 0}
    for sh in [s.strip() for s in args.shard.split(",") if s.strip()]:
        f = data / f"gt_moi_shard_{sh}.json"
        if not f.is_file():
            print(f"shard {sh}: khong co {f.name}")
            continue
        muc_list = json.loads(f.read_text(encoding="utf-8"))
        print(f"\n=== shard {sh}: {len(muc_list)} muc ===", flush=True)
        ket = []
        for i, muc in enumerate(muc_list):
            vid = muc["video_id"]
            n0 = int(muc["n"])
            mo_ta = mo_ta_neo(muc)
            if not mo_ta.strip():
                ket.append({"i": i, "trang_thai": "khong_co_mo_ta"})
                continue

            diem = {}
            for d in range(-args.quanh, args.quanh + 1):
                m = by_n.get((vid, n0 + d))
                if not m:
                    continue
                khoa = hashlib.sha1(
                    f"{vid}|{n0 + d}|{mo_ta}".encode("utf-8")).hexdigest()[:20]
                cf = cache_dir / f"{khoa}.json"
                if cf.is_file():
                    diem[d] = json.loads(cf.read_text(encoding="utf-8"))
                    continue
                blob = judge._fetch(vid, m["frame_filename"])
                if not blob:
                    continue
                try:
                    j = (hoi_openai(args.openai_model, blob, mo_ta, tien)
                         if args.provider == "openai" else hoi_mot_anh(judge, blob, mo_ta))
                except Exception as exc:  # noqa: BLE001
                    print(f"  ! muc {i}: {type(exc).__name__}: {str(exc)[:70]}")
                    continue
                cf.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
                diem[d] = j

            if not diem:
                tong["thieu_anh"] += 1
                ket.append({"i": i, "trang_thai": "khong_tai_duoc_anh"})
                continue

            d_tot = max(diem, key=lambda k: diem[k].get("diem", -1))
            v_tot = diem[d_tot].get("diem", -1)
            v_neo = diem.get(0, {}).get("diem", -1)
            if v_tot < args.nguong:
                tt = "nghi_ngo"
            elif d_tot == 0 or v_tot - v_neo < args.chenh:
                tt = "xac_nhan"
            else:
                tt = "de_nghi_doi"
            tong[tt] += 1
            ket.append({
                "i": i, "video_id": vid, "n_neo": n0, "trang_thai": tt,
                "diem_neo": v_neo, "n_de_nghi": n0 + d_tot if tt == "de_nghi_doi" else n0,
                "diem_de_nghi": v_tot,
                "diem_quanh": {str(k): diem[k].get("diem", -1) for k in sorted(diem)},
                "thieu": diem.get(0, {}).get("thieu", "")[:150],
                "mo_ta_kiem": mo_ta[:150],
            })
            dau = {"xac_nhan": "OK ", "de_nghi_doi": "DOI", "nghi_ngo": "??? "}[tt]
            print(f"  {dau} muc {i:2d} {vid} n={n0} neo={v_neo:3} "
                  f"tot={v_tot:3} tai d={d_tot:+d}  {mo_ta[:44]}", flush=True)

        (data / f"gt_moi_shard_{sh}_kiem_neo.json").write_text(
            json.dumps(ket, ensure_ascii=False, indent=1), encoding="utf-8")

        if args.ap_dung:
            sua = []
            for muc, k in zip(muc_list, ket):
                m2 = dict(muc)
                if k.get("trang_thai") == "de_nghi_doi":
                    mm = by_n.get((muc["video_id"], k["n_de_nghi"]))
                    if mm:
                        m2.update(n=int(mm["n"]), frame_idx=int(mm["frame_idx"]),
                                  frame_filename=mm["frame_filename"],
                                  pts_time=float(mm["pts_time"]))
                        m2["neo_sua_boi"] = "kiem_neo_don_anh"
                        m2["neo_cu"] = int(muc["n"])
                m2["kiem_neo"] = k.get("trang_thai")
                sua.append(m2)
            (data / f"gt_moi_shard_{sh}_neo_sua.json").write_text(
                json.dumps(sua, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n=== TONG: {tong['xac_nhan']} xac nhan | {tong['de_nghi_doi']} de nghi doi neo | "
          f"{tong['nghi_ngo']} nghi ngo (khong khung nao khop) | {tong['thieu_anh']} thieu anh")
    if args.provider == "openai":
        gia = tien["vao"] / 1e6 * 1.25 + tien["ra"] / 1e6 * 10.0
        print(f"{tien['goi']} lan goi {args.openai_model}, {tien['vao']:,} token vao, "
              f"{tien['ra']:,} token ra  ~ ${gia:.2f}")
    else:
        print(judge.cost_note())
    if tong["de_nghi_doi"] or tong["nghi_ngo"]:
        print("\nMuc 'de nghi doi' va 'nghi ngo' KHONG duoc dung de cham diem cho toi khi")
        print("giai quyet xong — neo sai lam bo do SAI DAU chu khong phai yeu di.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
