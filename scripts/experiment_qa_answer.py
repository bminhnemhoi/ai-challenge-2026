"""Đo ĐỘ CHÍNH XÁC ĐÁP ÁN của bước trả lời Q&A, có chia TUNE/TEST chống overfit.

Vì sao đây là phép đo đáng tiền nhất trong hệ thống: một câu Q&A có khung hình
ĐÚNG mà đáp án SAI vẫn được **0 tuyệt đối** (luật 2.1.2). Vòng sơ tuyển 2 sai
4/9 câu Q&A ở chế độ tự động — tức khoảng 4 điểm trên tổng 10 điểm đạt được,
nhiều hơn mọi thứ mà một tuần tối ưu truy xuất có thể mang lại.

Phép đo tách bạch **bước ĐỌC** khỏi **bước TÌM**: mọi biến thể đều được xem đúng
khung hình ground truth. Nó trả lời đúng một câu hỏi — *khi đã nhìn thấy đúng
chỗ, model đọc đúng đáp án bao nhiêu phần trăm* — nên kết quả không bị nhiễu bởi
chất lượng truy xuất.

Các biến thể (mỗi cái thêm ĐÚNG MỘT thứ so với cái trước, để biết cái gì ăn):

    goc         512px (đúng như answer_qa.py đang chạy), prompt hiện tại
    net         ảnh gốc ~1900px          <- thao tác tay đã cứu 4 câu vòng 2
    net_loi     + lời thoại ±30 giây     <- kênh NII-UIT (VBS 2026)
    net_loi_cu  + prompt ép cụ thể       <- chống đáp án hạng-mục-chung
    net_loi_cu_lan  + 2 khung lân cận    <- chống keyframe lệch nửa nhịp

Chia TUNE/TEST 30/30 theo chỉ số chẵn/lẻ, đúng luật đã ship cho allocator: chọn
biến thể trên TUNE, đọc TEST **đúng một lần**. Mọi lời gọi API đều cache xuống
đĩa (data/cache_qa_answer/) nên chạy lại không tốn tiền và TEST không bị đọc lén.

    python scripts/experiment_qa_answer.py --bien-the goc,net --limit 6   # thử
    python scripts/experiment_qa_answer.py                                 # đủ
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from src.core.submission import _default_answer_match, sanitise_field  # noqa: E402
from src.core.vlm import CDN, UA, DEFAULT_MODEL, VLMJudge  # noqa: E402

# --------------------------------------------------------------------------- prompt

PROMPT_GOC = """Bạn đang trả lời một câu hỏi về một đoạn video tiếng Việt.

{cau_hoi}

Dưới đây là {n} khung hình lấy từ video, đánh số 1..{n}.
Hãy ĐỌC KỸ mọi chữ hiện trên các khung hình (băng rôn, phụ đề, bảng, tờ giấy,
tiêu đề chương trình) và trả lời câu hỏi.

Quy tắc:
- Chỉ trả lời từ những gì THẤY ĐƯỢC trong ảnh. Không suy đoán, không dùng kiến
  thức bên ngoài.
- Đáp án phải NGẮN GỌN, đúng như chữ hiện trên hình (giữ nguyên dấu tiếng Việt).
- Nếu không khung hình nào cho thấy câu trả lời, trả về chuỗi rỗng cho "answer".

Trả về DUY NHẤT JSON:
{{"answer": "...", "frame": <số khung hình chứa câu trả lời, hoặc 0>, "confidence": 0-100, "seen": "chữ bạn đọc được"}}"""

#: Ép ĐỘ CỤ THỂ. Hồ sơ lỗi vòng 2 không phải "model không biết" mà là "model trả
#: lời đúng nhưng ở mức hạng mục" — hỏi loài quả thì trả "trái cây", hỏi số thì
#: trả "vài". Bộ chấm của BTC so khớp ngữ nghĩa với một đáp án CỤ THỂ, nên một
#: câu trả lời chung chung là số 0 y hệt một câu trả lời sai.
PROMPT_CU_THE = """Bạn đang trả lời một câu hỏi về một đoạn video tiếng Việt.

{cau_hoi}
{loi_thoai}
Dưới đây là {n} khung hình lấy từ video, đánh số 1..{n}.
Hãy ĐỌC KỸ mọi chữ hiện trên khung hình (băng rôn, phụ đề, bảng, tờ giấy, tiêu
đề chương trình), quan sát kỹ vật thể, rồi trả lời.

Quy tắc:
- Chỉ trả lời từ những gì THẤY trong ảnh hoặc NGHE thấy trong lời thoại kèm theo.
  Không dùng kiến thức bên ngoài.
- Trả DANH TỪ CỤ THỂ NHẤT đọc/thấy được: tên riêng, tên loài, nhãn hiệu, con số
  chính xác, màu cụ thể. TRÁNH từ hạng mục chung ("trái cây", "một loại xe",
  "nhiều người") — đáp án chung chung bị chấm 0 y như đáp án sai.
- Ngắn gọn: một cụm từ, giữ nguyên dấu tiếng Việt. Không giải thích.
- Nếu thật sự không xác định được, trả chuỗi rỗng cho "answer".

Trả về DUY NHẤT JSON:
{{"answer": "...", "nguon": "nhìn thấy|nghe thấy|đọc thấy", "frame": <số khung hình, hoặc 0>, "confidence": 0-100, "seen": "chữ/chi tiết bạn căn cứ"}}"""

#: Không bao giờ bỏ trống. Đây không phải mẹo vặt mà là hệ quả trực tiếp của
#: luật chấm: đáp án rỗng CHẮC CHẮN 0 điểm, đáp án đoán sai cũng 0 điểm — nên
#: đoán là trội tuyệt đối. Phép đo lần đầu cho thấy 9/20 câu sai là máy tự bỏ
#: trống, tức prompt cũ đang DẠY model vứt điểm. Cùng họ với luật "hedge theo
#: dòng" đã dùng: khi sai không bị phạt, im lặng là lựa chọn tệ nhất.
PROMPT_DOAN = """Bạn đang trả lời một câu hỏi về một đoạn video tiếng Việt.

{cau_hoi}
{loi_thoai}
Dưới đây là {n} khung hình lấy từ video, đánh số 1..{n}.

Quy tắc:
- TRẢ LỜI ĐÚNG CÂU HỎI ĐƯỢC HỎI. Chữ trên màn hình chỉ dùng khi nó trả lời câu
  hỏi đó; đừng chép tiêu đề/băng rôn nếu câu hỏi hỏi về màu sắc, vật thể hay
  hành động.
- Trả DANH TỪ CỤ THỂ NHẤT thấy được: tên riêng, tên loài, nhãn hiệu, con số
  chính xác (đọc kỹ dấu thập phân), màu cụ thể. TRÁNH từ hạng mục chung.
- Ngắn gọn: một cụm từ, giữ nguyên dấu tiếng Việt. Không giải thích.
- **LUÔN LUÔN đưa ra đáp án.** Nếu không chắc, vẫn đoán phương án hợp lý nhất
  và hạ "confidence" xuống. Bỏ trống chắc chắn bị 0 điểm, còn đoán thì vẫn có
  cơ hội đúng — không bao giờ trả chuỗi rỗng.

Trả về DUY NHẤT JSON:
{{"answer": "...", "nguon": "nhìn thấy|nghe thấy|đọc thấy", "frame": <số khung hình, hoặc 0>, "confidence": 0-100, "seen": "chữ/chi tiết bạn căn cứ"}}"""

BIEN_THE = {
    #  tên            net    lời thoại  prompt        số khung lân cận
    "goc": dict(net=False, loi=False, prompt="goc", lan_can=0),
    "net": dict(net=True, loi=False, prompt="goc", lan_can=0),
    "net_loi": dict(net=True, loi=True, prompt="goc", lan_can=0),
    "net_loi_cu": dict(net=True, loi=True, prompt="cu_the", lan_can=0),
    "net_loi_cu_lan": dict(net=True, loi=True, prompt="cu_the", lan_can=2),
    # biến thể sinh ra từ chính phép đo trên: cấm bỏ trống + cấm chép băng rôn
    "net_loi_doan": dict(net=True, loi=True, prompt="doan", lan_can=0),
    "net_loi_doan_lan": dict(net=True, loi=True, prompt="doan", lan_can=2),
    "goc_loi_doan": dict(net=False, loi=True, prompt="doan", lan_can=0),
}

# --------------------------------------------------------------------------- ảnh


def tai_anh(video_id: str, filename: str, max_side: int) -> bytes | None:
    """Khung hình như CDN công bố (~1280px), chỉ thu nhỏ khi vượt ``max_side``."""
    from PIL import Image

    try:
        raw = urllib.request.urlopen(
            urllib.request.Request(f"{CDN}/{video_id}/{filename}", headers=UA), timeout=60
        ).read()
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=95)
        return buf.getvalue()
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- lời thoại


def nap_loi_thoai(data_dir: Path) -> dict:
    """{video_id: [(giây, câu), ...]} từ data/captions — 873/873 video."""
    d = data_dir / "captions"
    out = {}
    if not d.is_dir():
        return out
    for p in d.glob("*.json"):
        try:
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
    return out


def loi_thoai_quanh(caps, video_id: str, giay: float, cua_so: float = 30.0) -> str:
    seg = caps.get(video_id) or []
    lay = [str(t[1]) for t in seg if isinstance(t, (list, tuple)) and len(t) >= 2
           and abs(float(t[0]) - giay) <= cua_so]
    txt = " ".join(lay).strip()
    return txt[:1500]


# --------------------------------------------------------------------------- gọi model


def _khoa(bien: str, g: dict, model: str) -> str:
    raw = f"{model}|{bien}|{g['video_id']}|{g['frame_idx']}|{g['vqa_question']}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def hoi(judge, model, blobs, prompt):
    """Một lượt hỏi, dùng ĐÚNG cơ chế chống 429 của VLMJudge.

    Free tier đếm request theo PHÚT và theo MODEL. Lần chạy đầu của thí nghiệm
    này gọi thẳng một model duy nhất và mất 35/60 câu vào 429 — con số 25/60
    đúng khi ấy là con số của hạn mức, không phải của model. Xoay vòng qua các
    model lite ngang cơ (``_model_order``) nhân tốc độ dùng được lên, còn 429
    theo-ngày thì phải ghi sổ và bỏ qua thay vì ngủ chờ (``_is_daily_quota``).
    """
    from google.genai import types

    from src.core.vlm import RETRY_WAIT, _is_daily_quota

    client = judge._get_client()
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in blobs]
    last = None
    for m_name in judge._model_order():
        if m_name in judge.exhausted:
            continue
        for lan in range(3):
            try:
                r = client.models.generate_content(
                    model=m_name,
                    contents=[*parts, prompt],
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=1200),
                )
                u = getattr(r, "usage_metadata", None)
                if u:
                    judge.calls += 1
                    judge.tokens_in += u.prompt_token_count or 0
                    judge.tokens_out += u.candidates_token_count or 0
                mm = re.search(r"\{.*\}", r.text or "", re.S)
                if not mm:
                    return {"answer": "", "raw": (r.text or "")[:200], "model": m_name}
                try:
                    j = json.loads(mm.group(0))
                    j["model"] = m_name
                    return j
                except Exception:  # noqa: BLE001
                    return {"answer": "", "raw": mm.group(0)[:200], "model": m_name}
            except Exception as exc:  # noqa: BLE001
                last = exc
                msg = str(exc)
                if _is_daily_quota(msg):
                    judge.exhausted.add(m_name)
                    break
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(RETRY_WAIT[min(lan, len(RETRY_WAIT) - 1)])
                    continue
                if "503" in msg or "UNAVAILABLE" in msg:
                    time.sleep(2.0 * (lan + 1))
                    continue
                break
    raise last or RuntimeError("het model kha dung")


def chay_bien_the(bien, gt, judge, caps, meta_by_key, cache_dir, model, args):
    """Trả [{dap_an, dung, ...}] cho từng câu; cache mỗi câu một file."""
    cfg = BIEN_THE[bien]
    ket = []
    for g in gt:
        f = cache_dir / f"{bien}_{_khoa(bien, g, model)}.json"
        if f.is_file() and not args.refresh:
            # chấm lại tại chỗ đọc: hai cờ khớp là hàm thuần của (đáp án, chuẩn)
            # nên đổi bộ so khớp không phải gọi lại API lần nào
            rec = json.loads(f.read_text(encoding="utf-8"))
            rec["dung"] = bool(_default_answer_match(rec.get("dap_an", ""), rec.get("chuan", "")))
            rec["dung_rong"] = bool(khop_rong(rec.get("dap_an", ""), rec.get("chuan", "")))
            ket.append(rec)
            continue

        # --- khung hình: khung GT, cộng lân cận nếu biến thể yêu cầu
        keys = [(g["video_id"], int(g["frame_idx"]))]
        if cfg["lan_can"]:
            n = int(g["n"])
            for d in range(1, cfg["lan_can"] + 1):
                for nn in (n - d, n + d):
                    k = (g["video_id"], nn)
                    if k in meta_by_key:
                        keys.append(k)
        blobs = []
        for vid, fi in keys:
            m = meta_by_key.get((vid, fi))
            if m is None:
                continue
            b = (tai_anh(vid, m["frame_filename"], args.max_side) if cfg["net"]
                 else judge._fetch(vid, m["frame_filename"]))
            if b:
                blobs.append(b)
        if not blobs:
            ket.append({"dap_an": "", "loi": "khong tai duoc anh"})
            continue

        cau_hoi = f"Bối cảnh: {g['vqa_context']}\nCâu hỏi: {g['vqa_question']}"
        kieu = cfg["prompt"]
        if kieu == "goc":
            ch = cau_hoi
            if cfg["loi"]:
                lt = loi_thoai_quanh(caps, g["video_id"], float(g["pts_time"]))
                if lt:
                    ch += f'\nLời thoại quanh khoảnh khắc này: "{lt}"'
            prompt = PROMPT_GOC.format(cau_hoi=ch, n=len(blobs))
        else:
            lt = loi_thoai_quanh(caps, g["video_id"], float(g["pts_time"])) if cfg["loi"] else ""
            mau = PROMPT_CU_THE if kieu == "cu_the" else PROMPT_DOAN
            prompt = mau.format(
                cau_hoi=cau_hoi, n=len(blobs),
                loi_thoai=(f"\nLời thoại quanh khoảnh khắc này:\n\"{lt}\"\n" if lt else ""),
            )

        try:
            j = hoi(judge, model, blobs, prompt)
        except Exception as exc:  # noqa: BLE001
            print(f"    ! {g['video_id']}:{g['frame_idx']} {type(exc).__name__}: {str(exc)[:80]}")
            time.sleep(2)
            ket.append({"dap_an": "", "loi": type(exc).__name__})
            continue

        rec = {
            "dap_an": sanitise_field(j.get("answer", "")),
            "chuan": g["vqa_answer"],
            "tin_cay": j.get("confidence", 0),
            "nguon": j.get("nguon", ""),
            "thay": str(j.get("seen", ""))[:120],
            "n_anh": len(blobs),
        }
        rec["dung"] = bool(_default_answer_match(rec["dap_an"], rec["chuan"]))
        rec["dung_rong"] = bool(khop_rong(rec["dap_an"], rec["chuan"]))
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        ket.append(rec)
    return ket


#: từ chức năng tiếng Việt trong câu trả lời ngắn — bỏ đi thì "Màu đỏ và trắng"
#: và "trắng và đỏ" mới so được với nhau
_BO = {"màu", "và", "có", "là", "của", "một", "cái", "chiếc", "loại", "kiểu",
       "bằng", "với", "ở", "the", "a", "of"}


def _tu(s: str):
    import unicodedata

    s = unicodedata.normalize("NFC", (s or "").lower())
    s = re.sub(r"[.,;:!?\"'()\-/]", " ", s)
    return {t for t in s.split() if t and t not in _BO}


def khop_rong(pred: str, chuan: str) -> bool:
    """Proxy SÁT hơn với bộ chấm ngữ nghĩa của BTC so với so-chuỗi-con.

    ``_default_answer_match`` của sản xuất chỉ xét chuỗi con, nên nó chấm SAI
    một đáp án chỉ đảo trật tự từ: "trắng và đỏ" vs đáp án chuẩn "Màu đỏ và
    trắng" (câu 35 trong phép đo đầu). Tài liệu cũ tuyên bố bộ so khớp ấy "chỉ
    có thể đếm thừa, không đếm thiếu" — điều đó KHÔNG đúng với các cụm đảo thứ
    tự, và nó làm mọi phép đo cải tiến bị nhiễu về phía bi quan.

    Ở đây so theo TẬP TỪ nội dung: khớp khi một bên là tập con của bên kia.
    Vẫn là proxy, nhưng sai lệch của nó đối xứng hơn và được báo cáo song song
    với số chặt để đọc được cả hai phía.
    """
    if _default_answer_match(pred, chuan):
        return True
    a, b = _tu(pred), _tu(chuan)
    if not a or not b:
        return False
    return a <= b or b <= a


def do_chinh_xac(ket, idx, rong=False):
    n = len(idx)
    if not n:
        return 0.0
    khoa = "dung_rong" if rong else "dung"
    return sum(1 for i in idx if ket[i].get(khoa)) / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_qa_answer"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-side", type=int, default=1900)
    ap.add_argument("--bien-the", default=",".join(BIEN_THE))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    gt = json.loads((data / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("vqa_question") and g.get("vqa_answer")]
    if args.limit:
        gt = gt[: args.limit]

    judge = VLMJudge(args.data, model=args.model)
    if not judge.ready:
        print("Khong co GEMINI_API_KEY trong .env")
        return 2

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    meta_by_key = {(m["video_id"], int(m["n"])): m for m in meta}
    # ground_truth dùng frame_idx; bảng tra theo (video, frame_idx) và theo (video, n)
    by_frame = {(m["video_id"], int(m["frame_idx"])): m for m in meta}
    meta_by_key = {**meta_by_key, **by_frame}

    caps = nap_loi_thoai(data)
    print(f"{len(gt)} câu Q&A ground truth | lời thoại: {len(caps)} video | model: {args.model}")

    i_tune = [i for i in range(len(gt)) if i % 2 == 0]
    i_test = [i for i in range(len(gt)) if i % 2 == 1]
    print(f"TUNE {len(i_tune)} câu (chỉ số chẵn) — chọn biến thể; "
          f"TEST {len(i_test)} câu (lẻ) — đọc MỘT lần\n")

    cache_dir = Path(args.cache)
    tens = [b.strip() for b in args.bien_the.split(",") if b.strip() in BIEN_THE]
    ket_qua = {}
    for bien in tens:
        t0 = time.time()
        print(f"chạy biến thể {bien} ...", flush=True)
        ket = chay_bien_the(bien, gt, judge, caps, meta_by_key, cache_dir, args.model, args)
        ket_qua[bien] = ket
        print(f"  xong sau {time.time()-t0:.0f}s  ({sum(1 for k in ket if k.get('dung'))}/{len(gt)} đúng tổng)")

    print(f"\n{'biến thể':<17}{'TUNE':>8}{'TEST':>8}{'cả 60':>8} | "
          f"{'TUNE~':>7}{'TEST~':>7}{'cả 60~':>8}{'rỗng':>6}{'lỗi':>5}")
    print(f"{'':17}{'so khớp chặt':^24} | {'so khớp tập từ (~)':^22}")
    print("-" * 78)
    hong = {}
    for bien in tens:
        k = ket_qua[bien]
        hong[bien] = sum(1 for r in k if r.get("loi"))
        rong = sum(1 for r in k if not r.get("loi") and not (r.get("dap_an") or "").strip())
        print(f"{bien:<17}{do_chinh_xac(k, i_tune):8.1%}{do_chinh_xac(k, i_test):8.1%}"
              f"{do_chinh_xac(k, range(len(gt))):8.1%} | "
              f"{do_chinh_xac(k, i_tune, True):7.1%}{do_chinh_xac(k, i_test, True):7.1%}"
              f"{do_chinh_xac(k, range(len(gt)), True):8.1%}{rong:6d}{hong[bien]:5d}")
    if any(hong.values()):
        print("\n! CÓ LỖI GỌI API — các câu hỏng bị tính là SAI, nên các con số trên là")
        print("  CẬN DƯỚI, chưa kết luận được. Chạy lại (câu đã xong nằm trong cache).")

    # chọn trên TUNE bằng số so-khớp-tập-từ (proxy sát bộ chấm BTC hơn)
    chot = max(tens, key=lambda b: do_chinh_xac(ket_qua[b], i_tune, True))
    goc = tens[0]
    print(f"\nCHỐT trên TUNE (theo số ~): {chot}")
    d_tune = do_chinh_xac(ket_qua[chot], i_tune, True) - do_chinh_xac(ket_qua[goc], i_tune, True)
    d_test = do_chinh_xac(ket_qua[chot], i_test, True) - do_chinh_xac(ket_qua[goc], i_test, True)
    print(f"  so với {goc}:  TUNE {d_tune:+.1%}   TEST {d_test:+.1%}")
    n_test = len(i_test)
    # sai số nhị thức thô cho tỷ lệ trên n_test câu
    p = do_chinh_xac(ket_qua[goc], i_test, True)
    se = (p * (1 - p) / max(1, n_test)) ** 0.5
    print(f"  1 sd nhị thức trên TEST ({n_test} câu) ≈ {se:.1%}; "
          f"{'GIỮ ĐƯỢC' if d_test > 2 * se else 'CHƯA VƯỢT 2 sd — coi như hoà'}")

    print("\n--- các câu biến thể chốt vẫn SAI (nơi còn điểm để lấy) ---")
    for i, g in enumerate(gt):
        r = ket_qua[chot][i]
        if not r.get("dung_rong"):
            phia = "TUNE" if i % 2 == 0 else "TEST"
            print(f"  [{i:2d} {phia}] hỏi: {g['vqa_question'][:60]}")
            print(f"        chuẩn: {r.get('chuan','')!r}   máy: {r.get('dap_an','')!r}"
                  f"   ({r.get('nguon','')} {r.get('tin_cay','')}%)")
    print(f"\n{judge.cost_note()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
