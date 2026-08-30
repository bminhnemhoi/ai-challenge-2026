"""Sinh ground truth MỚI bằng cách XEM MỘT ĐOẠN VIDEO, không phải nhìn một keyframe.

Vì sao có script này (docs/NGHIEN_CUU_SOTA.md §3b): bộ đo 60 câu hiện tại có
**0/60** câu mô tả hai cảnh nối tiếp, còn đề THẬT của BTC có **28/55 = 51%**.
Nguyên nhân gốc không phải cỡ mẫu mà là **cách viết**: 60 câu kia được viết bằng
cách nhìn MỘT keyframe, BTC ra đề bằng cách xem MỘT ĐOẠN. Hệ quả: mọi kỹ thuật
khai thác cấu trúc thời gian đều VĨNH VIỄN không đo được trên bộ đo cũ.

Nên ở đây mỗi câu sinh ra từ **8-12 keyframe liên tiếp** đưa cho VLM cùng một
lượt, theo đúng thứ tự thời gian, kèm lời thoại của chính đoạn đó nếu có.

KỶ LUẬT CHỐNG TỰ LỪA — nếu vi phạm thì bộ đo thành cái gương, đo lại chính mình:

* Video và đoạn được chọn **ngẫu nhiên có phân tầng** (seed cố định theo shard nên
  tái lập được). Không có điểm SigLIP, không có ranked_hits, không có bảng xếp
  hạng nào chạm vào quy trình này — prompt gửi đi chỉ có ẢNH và LỜI THOẠI.
* Không lọc bỏ câu vì hệ thống tìm không ra. Câu khó là câu đáng giá nhất; lọc
  chúng đi là cách nhanh nhất biến bộ đo thành cái gương.
* Có **một** phép loại duy nhất, và nó xảy ra TRƯỚC khi câu tồn tại: đoạn nào VLM
  báo ``dung_duoc=false`` (10 khung hình gần như y hệt nhau, không có khoảnh khắc
  nào để định vị) thì bốc đoạn khác trong CÙNG video bằng chính RNG đã seed. Đây
  không phải lọc theo độ khó — một đoạn tĩnh hoàn toàn thì không tồn tại câu nào
  "định vị một khoảnh khắc cụ thể" cả. Mọi lần bốc lại đều ghi vào ``bo_qua``
  của bản ghi để kiểm toán được.

Tỉ lệ hai cảnh được **giao chỉ tiêu 50/50** trước khi nhìn thấy video (chẵn/lẻ
theo chỉ số), cho khớp 51% của đề thật. Chỉ tiêu là thứ ĐẶT RA, không phải thứ
ĐO ĐƯỢC: tỉ lệ thật phải kiểm bằng bộ gắn nhãn độc lập, chạy riêng:

    python scripts/sinh_gt_doan_video.py --dai L21,L22 --shard a --so 16
    python scripts/sinh_gt_doan_video.py --shard a --xuat-de data/cache_gt_moi/de_a
    python scripts/gan_nhan_hai_canh.py --de data/cache_gt_moi/de_a

Cache mỗi video xuống ``data/cache_gt_moi/`` nên chạy lại không tốn quota, và
model xoay vòng theo đúng luật ``src/core/vlm.py`` (lần trước mất 35/60 câu vào
rate limit vì gọi thẳng một model).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

#: Đổi prompt là cache tự hỏng — không được trộn câu của hai prompt khác nhau
#: vào cùng một bộ đo, vì khi đó "tỉ lệ hai cảnh" không còn nghĩa gì.
PROMPT_VERSION = 1

CDN = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main"

#: dài hơn cho câu hai cảnh: 12 keyframe ≈ 35 giây, gần như chắc chắn bắc qua một
#: cú cắt cảnh; 9 keyframe là đủ cho câu một cảnh mà vẫn thấy được ngữ cảnh động
DAI_DOAN = {True: 12, False: 9}

#: bao nhiêu lần bốc lại đoạn trong cùng một video trước khi bỏ video đó
SO_LAN_THU = 3

#: Đáp án Q&A là "danh từ cụ thể" — quy tắc đã đo được trên vòng trước. Đây là
#: lưới chặn thô cho trường hợp mô hình trả về đúng cái từ hạng mục chung mà
#: quy tắc cấm; nó KHÔNG loại câu, chỉ bắt viết lại một lần rồi gắn cảnh báo.
_TU_HANG_MUC_CHUNG = {
    "xe", "cai xe", "chiec xe", "phuong tien", "nguoi", "mot nguoi", "con nguoi",
    "thuc an", "do an", "mon an", "thuc pham", "bang", "cai bang", "tam bang",
    "con vat", "dong vat", "cay", "cai cay", "nha", "can nha", "toa nha",
    "dung cu", "trang phuc", "quan ao", "do vat", "vat", "may moc", "thiet bi",
    "cong trinh", "dia diem", "khong ro", "khong xac dinh",
    # đo trên shard a: mô hình lách quy tắc bằng cách hỏi "đeo phụ kiện gì trên
    # mặt?" rồi trả "kính" — đúng là danh từ, nhưng nó là TỪ HẠNG MỤC và câu hỏi
    # tự lộ đáp án, không phân biệt được khung hình này với khung hình khác
    "kinh", "mu", "non", "khau trang", "ao", "quan", "giay", "dep", "tui",
    "phu kien", "dong ho", "xe may", "o to", "xe dap", "tau", "thuyen",
    "chu", "bien", "bang hieu", "logo", "man hinh", "may tinh", "dien thoai",
}


def _bo_dau(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]+", " ", s.replace("đ", "d")).strip()


def _dap_an_chung_chung(ans: str) -> bool:
    """True khi đáp án chỉ là một từ hạng mục, không định danh được gì."""
    t = re.sub(r"\s+", " ", _bo_dau(ans))
    t = re.sub(r"^(mau|so|chu|ten|la|cai|chiec|con|mot) ", "", t).strip()
    return (not t) or t in _TU_HANG_MUC_CHUNG


# --------------------------------------------------------------------- dữ liệu
def nap_khung(data_dir: Path):
    """{video_id: [row, ...]} sắp theo n tăng dần, đã bỏ các khung trắng.

    ``blank_frame_indices.json`` là chỉ số DÒNG trong metadata.json, không phải
    số keyframe — một đoạn có khung trắng ở giữa thì VLM nhìn thấy một ô đen và
    câu sinh ra sẽ tả sai chỗ chuyển cảnh.
    """
    md = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    trang = set()
    p = data_dir / "blank_frame_indices.json"
    if p.exists():
        for i in json.loads(p.read_text(encoding="utf-8")):
            if 0 <= int(i) < len(md):
                trang.add((md[int(i)]["video_id"], int(md[int(i)]["n"])))
    byv: dict[str, list] = defaultdict(list)
    for r in md:
        byv[r["video_id"]].append(r)
    for v in byv:
        byv[v].sort(key=lambda r: int(r["n"]))
    return byv, trang


def nap_loi_thoai(data_dir: Path, video_id: str):
    """[(giây, câu), ...] hoặc [] — chỉ 217/873 video thật sự có lời thoại."""
    p = data_dir / "captions" / f"{video_id}.json"
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for row in d if isinstance(d, list) else []:
        try:
            out.append((float(row[0]), str(row[1])))
        except Exception:  # noqa: BLE001
            continue
    return out


def _loi_thoai_doan(cap, t0: float, t1: float) -> str:
    noi = [s for t, s in cap if t0 - 2.0 <= t <= t1 + 2.0]
    if not noi:
        return ""
    return " ".join(noi)[:900]


# ------------------------------------------------------------- ví dụ văn phong
def vi_du_van_phong(data_dir: Path, k2: int = 5, k1: int = 4):
    """Câu đề THẬT của BTC, tách theo nhãn hai cảnh — đây là giọng cần học.

    Lấy từ chính cache đã dùng để đo con số 51%, nên ví dụ và thước đo nói cùng
    một thứ tiếng. Chỉ lấy stem ``-kis`` (đề TRAKE/Q&A có cấu trúc nhiều dòng
    riêng, đưa vào đây sẽ dạy sai định dạng).
    """
    d = data_dir / "cache_cap_thoi_gian" / "nhan_de"
    hai, mot = [], []
    for p in sorted(d.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        q = (r.get("query") or "").strip()
        if not q or not str(r.get("stem", "")).endswith("-kis"):
            continue
        (hai if r.get("co_2_canh") else mot).append(" ".join(q.split())[:420])
    return hai[:k2], mot[:k1]


# -------------------------------------------------------------------- prompt
_YEU_CAU_2 = """YÊU CẦU LẦN NÀY — câu HAI CẢNH NỐI TIẾP:
Câu phải mô tả cảnh A xảy ra TRƯỚC rồi cảnh B xảy ra SAU, hai cảnh khác nhau tới
mức KHÔNG thể nhìn thấy cả hai trong cùng một khung hình, và phải dùng từ chỉ
trình tự ("sau đó", "tiếp theo", "rồi", "chuyển sang", "bắt đầu bằng ... rồi").
Trong {n} khung hình trên hãy tìm chỗ CHUYỂN CẢNH thật sự rồi viết bám vào nó.
Quan hệ KHÔNG GIAN trong cùng một khung ("A đứng cạnh B", "phía sau là C") và
một hành động liên tục trong một cảnh đều KHÔNG tính là hai cảnh."""

_YEU_CAU_1 = """YÊU CẦU LẦN NÀY — câu MỘT CẢNH:
Câu chỉ được mô tả những gì nhìn thấy trong MỘT khung hình duy nhất. TUYỆT ĐỐI
KHÔNG dùng từ chỉ trình tự thời gian ("sau đó", "tiếp theo", "rồi", "chuyển
sang"). Hãy chọn trong {n} khung hình trên một khoảnh khắc giàu chi tiết và tả
thật chính xác chi tiết của riêng khung hình đó."""

_NEO_2 = ("Câu hai cảnh: neo vào khung hình ĐẦU TIÊN của cảnh B — đúng khoảnh khắc\n"
          "cảnh B bắt đầu. Đó chính là thứ ban tổ chức hỏi.")
_NEO_1 = "Câu một cảnh: neo vào đúng khung hình mà câu mô tả."

_PROMPT = """Bạn là người RA ĐỀ cho cuộc thi tìm khoảnh khắc trong video tiếng Việt.

Dưới đây là {n} khung hình LIÊN TIẾP cắt từ MỘT đoạn video, đánh số 1..{n} theo
đúng thứ tự thời gian (1 sớm nhất, {n} muộn nhất), cách nhau khoảng {dt:.1f} giây,
đoạn dài khoảng {tong:.0f} giây.
{loi_thoai}
NHIỆM VỤ: viết MỘT câu truy vấn tiếng Việt theo đúng văn phong ban tổ chức.

{yeu_cau}

VĂN PHONG BAN TỔ CHỨC — đây là đề THẬT của họ. Học CÁCH VIẾT, đừng chép nội dung:
{vi_du}

RÀNG BUỘC:
1. Câu phải ĐỊNH VỊ MỘT KHOẢNH KHẮC CỤ THỂ trong đoạn này — người đọc phải chỉ
   ra được đúng một chỗ. Tuyệt đối không mô tả chung chung cả video.
2. CHỈ nói thứ NHÌN THẤY ĐƯỢC trong các khung hình trên (hoặc nghe rõ trong lời
   thoại kèm theo). Không suy đoán, không bịa tên riêng, không bịa số liệu.
3. Nêu chi tiết phân biệt được: màu sắc, số lượng, chữ hiện trên hình, tư thế,
   hướng di chuyển, vị trí trong khung.
4. KHÔNG được nhắc "khung hình số 3", "ảnh thứ 5" trong câu truy vấn.
5. Dài 1-3 câu, như đề thật.

KHUNG NEO: chọn số thứ tự khung hình mà câu truy vấn trỏ tới.
{neo}

CÂU HỎI KÈM (Q&A): thêm một câu hỏi trả lời được CHỈ BẰNG khung neo đó.
Đáp án BẮT BUỘC là một DANH TỪ CỤ THỂ: tên riêng, màu cụ thể, con số, hoặc chữ
đọc được trên hình. TUYỆT ĐỐI KHÔNG dùng từ hạng mục chung ("cái xe", "một
người", "thức ăn", "cái bảng"). vqa_context là một câu nêu bối cảnh chung.

Nếu đoạn này KHÔNG viết được câu đạt yêu cầu (các khung hình gần như y hệt nhau,
không có khoảnh khắc nào để định vị{them}), hãy trả dung_duoc=false kèm lý do —
ĐỪNG cố viết một câu mơ hồ.

Trả về DUY NHẤT một object JSON, không giải thích:
{{"dung_duoc": true hoặc false, "ly_do_bo": "",
  "kis_query_vi": "", "kis_query_en": "",
  "khung_neo": <số nguyên 1..{n}>,
  "canh_A": "", "canh_B": "",
  "vqa_context": "", "vqa_question": "", "vqa_answer": ""}}
kis_query_en là bản dịch tiếng Anh tự nhiên của kis_query_vi.
canh_A/canh_B chỉ điền khi câu có hai cảnh nối tiếp, ngược lại để chuỗi rỗng."""


def dung_prompt(n: int, dt: float, tong: float, loi: str, hai_canh: bool,
                vd2, vd1) -> str:
    khoi = []
    if hai_canh:
        for i, s in enumerate(vd2, 1):
            khoi.append(f"  (hai cảnh {i}) {s}")
    else:
        for i, s in enumerate(vd1, 1):
            khoi.append(f"  (một cảnh {i}) {s}")
    # Cho xem cả hai kiểu: nếu chỉ thấy ví dụ hai cảnh thì câu một cảnh cũng bị
    # kéo theo giọng trình tự, và ngược lại — mà đó đúng là thứ đang đo.
    khac = vd1[:2] if hai_canh else vd2[:2]
    for i, s in enumerate(khac, 1):
        khoi.append(f"  (KHÔNG phải kiểu lần này, chỉ để đối chiếu {i}) {s}")
    return _PROMPT.format(
        n=n, dt=dt, tong=tong,
        loi_thoai=(f"\nLỜI THOẠI TRONG ĐOẠN (tự động, có thể sai chính tả):\n"
                   f"«{loi}»\n" if loi else "\n(Đoạn này không có lời thoại.)\n"),
        yeu_cau=(_YEU_CAU_2 if hai_canh else _YEU_CAU_1).format(n=n),
        vi_du="\n".join(khoi),
        neo=_NEO_2 if hai_canh else _NEO_1,
        them=(", hoặc cả đoạn chỉ có đúng một cảnh liên tục không hề chuyển cảnh"
              if hai_canh else ""),
    )


# ----------------------------------------------------------------- gọi model
def _goi(judge, anh_bytes, prompt: str, max_out: int = 2200):
    """Một request nhiều ảnh, xoay vòng model theo đúng luật của VLMJudge.

    ``judge._model_order()`` xoay vòng theo lượt gọi nên hai request liên tiếp
    không đập vào cùng một cửa sổ phút của cùng một model; ``judge.exhausted``
    nhớ model đã hết quota NGÀY để không ngủ 40 giây trên một cánh cửa đã khoá.
    """
    from google.genai import types

    from src.core.vlm import RETRY_WAIT, _is_daily_quota

    client = judge._get_client()
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in anh_bytes]
    last = None
    for model in judge._model_order():
        if model in judge.exhausted:
            continue
        for attempt in range(3):
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=[*parts, prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.35,          # đề cần đa dạng, không cần tái lập từng chữ
                        max_output_tokens=max_out,
                        response_mime_type="application/json",
                    ),
                )
                judge.calls += 1
                u = getattr(r, "usage_metadata", None)
                if u:
                    judge.tokens_in += u.prompt_token_count or 0
                    judge.tokens_out += u.candidates_token_count or 0
                m = re.search(r"\{.*\}", r.text or "", re.S)
                if not m:
                    last = f"{model}: không parse được JSON"
                    break
                return json.loads(m.group(0)), model
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                last = f"{model}: {type(exc).__name__}: {msg[:90]}"
                if _is_daily_quota(msg):
                    judge.exhausted.add(model)
                    break
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(RETRY_WAIT[min(attempt, len(RETRY_WAIT) - 1)])
                    continue
                if "503" in msg or "UNAVAILABLE" in msg:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                break
    raise RuntimeError(last or "không rõ")


# ------------------------------------------------------------------- một đoạn
def thu_mot_doan(judge, data_dir: Path, rows, cap, n_tu: int, n_den: int,
                 hai_canh: bool, vd2, vd1):
    """Gửi cả đoạn cho VLM, trả (bản ghi thô, model) hoặc (None, lý do bỏ)."""
    doan = [r for r in rows if n_tu <= int(r["n"]) <= n_den]
    blobs, dung = [], []
    for r in doan:
        b = judge._fetch(r["video_id"], r["frame_filename"])
        if b:
            blobs.append(b)
            dung.append(r)
    if len(blobs) < 6:
        return None, f"chỉ tải được {len(blobs)} khung hình"

    t0, t1 = float(dung[0]["pts_time"]), float(dung[-1]["pts_time"])
    dt = (t1 - t0) / max(len(dung) - 1, 1)
    prompt = dung_prompt(len(dung), dt, t1 - t0,
                         _loi_thoai_doan(cap, t0, t1), hai_canh, vd2, vd1)
    raw, model = _goi(judge, blobs, prompt)

    if not raw.get("dung_duoc"):
        return None, f"VLM bỏ đoạn: {str(raw.get('ly_do_bo') or '')[:120]}"

    # Đáp án Q&A phải là danh từ cụ thể — cho viết lại ĐÚNG MỘT lần, rồi thôi.
    # Bỏ hẳn câu vì đáp án chung chung là bắt đầu lọc bộ đo theo ý mình.
    canh_bao = ""
    if _dap_an_chung_chung(str(raw.get("vqa_answer") or "")):
        try:
            raw2, model2 = _goi(
                judge, blobs,
                prompt + "\n\nLƯU Ý SỬA: đáp án vqa_answer bạn vừa định đưa ra là"
                         " một TỪ HẠNG MỤC CHUNG, không đạt. Hãy đổi vqa_question"
                         " sang thứ trả lời được bằng TÊN RIÊNG, MÀU CỤ THỂ, CON SỐ"
                         " hoặc CHỮ đọc được trên khung neo. Giữ nguyên"
                         " kis_query_vi/kis_query_en/khung_neo.")
            if raw2.get("dung_duoc") and not _dap_an_chung_chung(
                    str(raw2.get("vqa_answer") or "")):
                raw, model = raw2, model2
            else:
                canh_bao = "vqa_answer vẫn chung chung sau 1 lần sửa"
        except Exception as exc:  # noqa: BLE001
            canh_bao = f"không sửa được vqa_answer: {str(exc)[:60]}"

    try:
        k = int(raw.get("khung_neo"))
    except (TypeError, ValueError):
        return None, "khung_neo không phải số"
    if not 1 <= k <= len(dung):
        return None, f"khung_neo {k} ngoài khoảng 1..{len(dung)}"

    neo = dung[k - 1]
    raw["_neo"] = neo
    raw["_n_tu"] = int(dung[0]["n"])
    raw["_n_den"] = int(dung[-1]["n"])
    raw["_canh_bao"] = canh_bao
    return (raw, model), ""


def _muc(raw: dict, model: str, hai_canh: bool, shard: str, bo_qua: list) -> dict:
    neo = raw["_neo"]
    return {
        "kis_query_vi": " ".join(str(raw.get("kis_query_vi") or "").split()),
        "kis_query_en": " ".join(str(raw.get("kis_query_en") or "").split()),
        "vqa_context": " ".join(str(raw.get("vqa_context") or "").split()),
        "vqa_question": " ".join(str(raw.get("vqa_question") or "").split()),
        "vqa_answer": " ".join(str(raw.get("vqa_answer") or "").split()),
        "video_id": neo["video_id"],
        "frame_idx": int(neo["frame_idx"]),
        "n": int(neo["n"]),
        "frame_filename": neo["frame_filename"],
        "pts_time": float(neo["pts_time"]),
        "cdn_url": f"{CDN}/{neo['video_id']}/{neo['frame_filename']}",
        "sinh_tu": {"video_id": neo["video_id"], "n_tu": raw["_n_tu"],
                    "n_den": raw["_n_den"]},
        "co_2_canh": bool(hai_canh),          # CHỈ TIÊU đặt ra, không phải nhãn đo
        "canh_A": str(raw.get("canh_A") or "").strip(),
        "canh_B": str(raw.get("canh_B") or "").strip(),
        "model": model,
        "shard": shard,
        "prompt_version": PROMPT_VERSION,
        "bo_qua": bo_qua,                      # dấu vết mọi đoạn đã bốc rồi bỏ
        "canh_bao": raw.get("_canh_bao") or "",
    }


# --------------------------------------------------------------------- xuất đề
def cmd_xuat_de(args) -> int:
    """Đổ kis_query_vi ra .txt để bộ gắn nhãn độc lập chấm bằng chế độ ``--de``.

    Cùng prompt, cùng nhiệt 0 mà con số 28/55 của đề thật được đo — nên tỉ lệ
    hai cảnh của bộ mới so được thẳng với nó. Đây là lý do phải đi vòng qua
    file .txt thay vì tự tin vào cờ ``co_2_canh`` do bước sinh tự khai.
    """
    ra = Path(args.ra or ROOT / "data" / f"gt_moi_shard_{args.shard}.json")
    muc = json.loads(ra.read_text(encoding="utf-8"))
    d = Path(args.xuat_de)
    d.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(muc):
        (d / f"gtmoi-{args.shard}-{i:02d}-kis.txt").write_text(
            m["kis_query_vi"], encoding="utf-8")
    print(f"{len(muc)} câu -> {d}")
    print(f"kiểm: python scripts/gan_nhan_hai_canh.py --de {d}")
    return 0


# ----------------------------------------------------------------- kiểm khung neo
_PROMPT_NEO = """Dưới đây là {n} khung hình LIÊN TIẾP của một đoạn video, đánh số
1..{n} theo đúng thứ tự thời gian.

CÂU TRUY VẤN:
{q}

CÂU HỎI: trong {n} khung hình trên, khung nào là ĐÚNG khoảnh khắc mà câu truy vấn
trỏ tới?
{luat}

Chỉ được chọn theo thứ NHÌN THẤY trong ảnh. Nếu không khung nào khớp, trả khung=0.

Trả về DUY NHẤT: {{"khung": <0..{n}>, "ly_do": "một câu ngắn"}}"""

_LUAT_NEO_2 = ("Câu này mô tả HAI cảnh nối tiếp. Hãy chọn khung hình ĐẦU TIÊN của\n"
               "cảnh SAU (cảnh B) — khung đầu tiên mà cảnh B đã hiện ra trên màn hình.")
_LUAT_NEO_1 = "Câu này mô tả MỘT cảnh. Hãy chọn khung hình thể hiện đúng cảnh đó."


def cmd_kiem_neo(args) -> int:
    """Hỏi lại VLM: khung neo đã lưu có ĐÚNG là khoảnh khắc câu trỏ tới không?

    Khung neo là ĐÁP ÁN của bộ đo. Đo tay trên shard a: 3/8 câu hai cảnh có neo
    lệch đúng MỘT cú cắt — VLM tả cảnh B chính xác rồi trỏ vào cảnh bên cạnh
    (cảnh A cuối, một shot xen giữa, thậm chí trỏ NGƯỢC lại cảnh A). Câu vẫn đọc
    trôi chảy nên lỗi này vô hình nếu chỉ đọc văn bản — phải mở ảnh ra mới thấy.

    Neo sai không làm bộ đo yếu đi, nó làm bộ đo SAI DẤU: hệ thống trả về đúng
    chỗ thì bị chấm trượt. Nên bước này chạy TRƯỚC khi ai đó chấm điểm trên bộ mới.

    Đây là bộ LỌC, không phải bộ sửa, và lý do là một phép đo chứ không phải sự
    thận trọng suông. Chạy trên shard a SAU khi 3 neo đã được sửa tay: bộ kiểm
    đòi kéo mục 04 về 213 và mục 10 về 105 — đúng hai khung mà mắt người đã xác
    nhận VẪN LÀ CẢNH A (213 còn là trang web Koryo Tours, 105 còn là bàn bo mạch).
    Tức là **chính bộ kiểm cũng lệch ±1**, cùng một căn nguyên với lỗi lúc sinh:
    khi nhét 9-12 ảnh vào một request, Gemini tả đúng nội dung cảnh nhưng đánh
    sai số thứ tự ảnh chứa cảnh đó.

    Hệ quả phải nói thẳng: KHÔNG có đường tự động nào chốt được khung neo. Bất
    đồng ở đây chỉ có nghĩa "mở ảnh ra xem", không bao giờ có nghĩa "bộ kiểm
    đúng". Ai tin bộ kiểm và ghi đè hàng loạt sẽ hỏng đúng những mục đang đúng.
    """
    ra = Path(args.ra or ROOT / "data" / f"gt_moi_shard_{args.shard}.json")
    muc = json.loads(ra.read_text(encoding="utf-8"))
    data_dir = Path(args.data)
    byv, _ = nap_khung(data_dir)

    from src.core.vlm import VLMJudge

    judge = VLMJudge(data_dir=data_dir)
    if not judge.ready:
        print("thiếu GEMINI_API_KEY trong .env")
        return 2

    lech, hop, loi = [], 0, 0
    for i, m in enumerate(muc):
        st = m["sinh_tu"]
        rows = [r for r in byv[m["video_id"]]
                if st["n_tu"] <= int(r["n"]) <= st["n_den"]]
        blobs, dung = [], []
        for r in rows:
            b = judge._fetch(r["video_id"], r["frame_filename"])
            if b:
                blobs.append(b)
                dung.append(r)
        if not blobs:
            print(f"[{i:02d}] không tải được khung hình")
            loi += 1
            continue
        p = _PROMPT_NEO.format(
            n=len(dung), q=m["kis_query_vi"],
            luat=_LUAT_NEO_2 if m["co_2_canh"] else _LUAT_NEO_1)
        try:
            raw, _model = _goi(judge, blobs, p, max_out=400)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i:02d}] LỖI {str(exc)[:80]}")
            loi += 1
            continue
        try:
            k = int(raw.get("khung"))
        except (TypeError, ValueError):
            k = 0
        n_vlm = int(dung[k - 1]["n"]) if 1 <= k <= len(dung) else 0
        if n_vlm == int(m["n"]):
            hop += 1
        else:
            lech.append((i, int(m["n"]), n_vlm, str(raw.get("ly_do") or "")[:70]))
            print(f"[{i:02d}] {m['video_id']} LỆCH: đã lưu n={m['n']}, VLM chọn n={n_vlm}"
                  f"  ({raw.get('ly_do','')[:60]})")
            print(f"     {m['kis_query_vi'][:100]}")

    n = len(muc)
    print(f"\n=== KIỂM KHUNG NEO shard {args.shard}: {hop}/{n} khớp, "
          f"{len(lech)} lệch, {loi} lỗi ===")
    if lech:
        print("MỞ ẢNH RA XEM từng mục lệch rồi sửa tay — đừng tin bên nào trong hai bên:")
        for i, cu, moi, _ly in lech:
            v = muc[i]["video_id"]
            print(f"  mục {i:02d} {v}: data/frames/{v}/ khung {cu} (đã lưu) vs {moi} (VLM)")
    print(judge.cost_note())
    return 0


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dai", default="L21,L22", help='dải video, ví dụ "L21,L22"')
    ap.add_argument("--shard", default="a", help="tên lane; quyết định seed và tên file")
    ap.add_argument("--so", type=int, default=16, help="số mục cần sinh")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--ra", default=None)
    ap.add_argument("--xuat-de", default=None,
                    help="đổ kis_query_vi ra .txt cho gan_nhan_hai_canh.py --de")
    ap.add_argument("--refresh", action="store_true", help="bỏ cache, gọi lại")
    ap.add_argument("--kiem-neo", action="store_true",
                    help="hỏi lại VLM xem khung neo đã lưu có đúng không (in chỗ lệch)")
    args = ap.parse_args()

    if args.xuat_de:
        return cmd_xuat_de(args)
    if args.kiem_neo:
        return cmd_kiem_neo(args)

    data_dir = Path(args.data)
    ra = Path(args.ra or ROOT / "data" / f"gt_moi_shard_{args.shard}.json")
    cache_dir = data_dir / "cache_gt_moi"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dai = [s.strip() for s in args.dai.split(",") if s.strip()]
    byv, trang = nap_khung(data_dir)
    vd2, vd1 = vi_du_van_phong(data_dir)
    print(f"ví dụ văn phong từ đề thật: {len(vd2)} câu hai cảnh, {len(vd1)} câu một cảnh")

    # Phân tầng: chia đều chỉ tiêu cho từng dải, rồi xáo ngẫu nhiên trong dải.
    # Seed buộc vào tên shard nên hai lane không bao giờ bốc trùng video và
    # chạy lại vẫn ra đúng bộ cũ.
    rng = random.Random(f"gt-moi::{args.shard}::{args.dai}")
    pool = {}
    for d in dai:
        vs = sorted(v for v in byv if v.startswith(d))
        rng.shuffle(vs)
        pool[d] = vs
        print(f"  {d}: {len(vs)} video")

    # Chỉ tiêu hai cảnh 50/50, gán TRƯỚC khi nhìn thấy bất kỳ khung hình nào.
    #
    # Dải phải xoay theo TỪNG CẶP chứ không theo từng mục. Nếu xoay theo từng mục
    # thì i chẵn (hai cảnh) luôn rơi vào dải đầu và i lẻ (một cảnh) luôn rơi vào
    # dải sau — hai trục trùng khít nhau, và "câu hai cảnh" hoá ra chỉ là "câu
    # lấy từ L21". Mọi khác biệt đo được sau này sẽ không tách nổi cấu trúc thời
    # gian khỏi đặc thù kênh. Chia theo cặp cho mỗi dải đủ cả hai loại.
    chi_tieu = [(i % 2 == 0) for i in range(args.so)]
    phan = [dai[(i // 2) % len(dai)] for i in range(args.so)]

    from src.core.vlm import VLMJudge

    judge = VLMJudge(data_dir=data_dir)
    if not judge.ready:
        print("thiếu GEMINI_API_KEY trong .env")
        return 2

    ket: list[dict] = []
    con = {d: list(pool[d]) for d in dai}
    for i in range(args.so):
        muon_2 = chi_tieu[i]
        d = phan[i]
        got = None
        while con[d] and got is None:
            vid = con[d].pop(0)
            cp = cache_dir / f"{args.shard}_{vid}.json"
            if cp.exists() and not args.refresh:
                c = json.loads(cp.read_text(encoding="utf-8"))
                if (c.get("prompt_version") == PROMPT_VERSION
                        and c.get("co_2_canh") == muon_2 and c.get("ok")):
                    got = c["muc"]
                    print(f"[{i:02d}] {vid} (cache) hai_canh={muon_2}")
                    break
                if c.get("prompt_version") == PROMPT_VERSION and not c.get("ok") \
                        and c.get("co_2_canh") == muon_2:
                    print(f"[{i:02d}] {vid} bỏ (cache): {c.get('ly_do','')[:70]}")
                    continue

            rows = [r for r in byv[vid] if (vid, int(r["n"])) not in trang]
            L = DAI_DOAN[muon_2]
            if len(rows) < L + 4:
                continue
            # RNG của đoạn buộc vào (shard, video, lần thử) chứ KHÔNG lấy từ dòng
            # rng chung: nếu lấy từ dòng chung thì một video trúng cache sẽ không
            # rút số, và mọi video SAU nó bốc trúng đoạn khác — chạy lại từ đầu ra
            # một bộ đo khác. Bộ đo phải tái lập được từng chữ, kể cả sau khi xoá
            # cache một nửa.
            bo_qua, out = [], None
            for lan in range(SO_LAN_THU):
                r_doan = random.Random(f"gt-moi::{args.shard}::{vid}::{lan}")
                s = r_doan.randrange(0, len(rows) - L)
                n_tu, n_den = int(rows[s]["n"]), int(rows[s + L - 1]["n"])
                try:
                    out, ly = thu_mot_doan(judge, data_dir, rows, nap_loi_thoai(data_dir, vid),
                                           n_tu, n_den, muon_2, vd2, vd1)
                except Exception as exc:  # noqa: BLE001
                    out, ly = None, f"LỖI {type(exc).__name__}: {str(exc)[:90]}"
                if out is not None:
                    break
                bo_qua.append({"n_tu": n_tu, "n_den": n_den, "ly_do": ly})
                print(f"     {vid} n={n_tu}-{n_den} bỏ: {ly[:80]}")
                if judge.exhausted and not judge.usable:
                    print("  !! HẾT QUOTA mọi model — dừng, đừng ghi bộ đo dở dang")
                    return 3
            if out is None:
                cp.write_text(json.dumps(
                    {"prompt_version": PROMPT_VERSION, "co_2_canh": muon_2, "ok": False,
                     "ly_do": bo_qua[-1]["ly_do"] if bo_qua else "không rõ",
                     "bo_qua": bo_qua}, ensure_ascii=False, indent=1), encoding="utf-8")
                continue
            raw, model = out
            m = _muc(raw, model, muon_2, args.shard, bo_qua)
            cp.write_text(json.dumps({"prompt_version": PROMPT_VERSION, "ok": True,
                                      "co_2_canh": muon_2, "muc": m},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
            got = m
            print(f"[{i:02d}] {vid} n={m['sinh_tu']['n_tu']}-{m['sinh_tu']['n_den']}"
                  f" neo={m['n']} hai_canh={muon_2} <{model}>")
            print(f"     {m['kis_query_vi'][:110]}")
        if got is None:
            print(f"[{i:02d}] KHÔNG sinh được mục nào trong dải {d}")
            continue
        ket.append(got)

    ra.write_text(json.dumps(ket, ensure_ascii=False, indent=1), encoding="utf-8")
    hai = sum(1 for m in ket if m["co_2_canh"])
    print(f"\n=== {len(ket)}/{args.so} mục -> {ra} ===")
    print(f"chỉ tiêu hai cảnh (TỰ KHAI, chưa kiểm): {hai}/{len(ket)}")
    cb = [m for m in ket if m.get("canh_bao")]
    if cb:
        print(f"  ! {len(cb)} mục có cảnh báo: "
              + "; ".join(f"{m['video_id']}: {m['canh_bao']}" for m in cb[:5]))
    print(judge.cost_note())
    print("\nBƯỚC BẮT BUỘC TIẾP THEO — kiểm bằng bộ gắn nhãn độc lập:")
    print(f"  python scripts/sinh_gt_doan_video.py --shard {args.shard}"
          f" --xuat-de data/cache_gt_moi/de_{args.shard}")
    print(f"  python scripts/gan_nhan_hai_canh.py --de data/cache_gt_moi/de_{args.shard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
