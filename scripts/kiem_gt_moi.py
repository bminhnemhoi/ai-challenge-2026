"""Kiểm ĐỘC LẬP bộ ground truth mới (data/gt_moi_shard_*.json) bằng cách MỞ ẢNH RA XEM.

Người sinh ra bộ này đã tự chấm cho mình ``do_tin_nhan = 100`` trên mọi mục.
Một điểm tự chấm không phải là bằng chứng, nên script này không đọc trường đó và
không tin trường ``co_2_canh`` — nó dựng lại từng kết luận từ chính các khung hình
trong ``data/frames``.

Năm phép đo, mỗi phép là một câu hỏi trả lời được bằng ảnh:

1. **Neo**  — cho VLM xem ĐÚNG khung neo và chấm 0-100 mức khớp với mô tả (với câu
   hai cảnh thì chấm theo ``canh_B``, vì khung neo phải là cảnh SAU).
2. **Cảnh A** — cho xem 6-8 khung TRƯỚC khung neo trong cùng đoạn và chấm ``canh_A``.
   Cảnh A không có thật là lỗi NẶNG HƠN câu một cảnh: nó dạy kỹ thuật thời gian đi
   sai hướng. Đồng thời chấm ``canh_A`` NGAY TRÊN khung neo — nếu cả A và B cùng
   nằm trong một khung thì đó là "hai cảnh giả", không phải trình tự thời gian.
3. **Q&A**  — hỏi VLM đúng ``vqa_question`` khi CHỈ nhìn khung neo, rồi so câu trả
   lời độc lập đó với ``vqa_answer`` bằng một lượt so nghĩa riêng (không đưa đáp án
   vào lượt hỏi ảnh, để khỏi mớm).
4. **Định vị** — câu có ghim một khoảnh khắc hay chỉ tả chung chủ đề cả video.
5. **Độ dễ** — chấm chính mô tả đó trên ~18 khung KHÁC rải đều cùng video. Nhiều
   khung khác cũng khớp thì câu không định vị được. Ghi nhận, KHÔNG loại (chỉ tiêu
   này đo độ mơ hồ, và loại theo nó là bắt đầu tự chọn câu dễ).

Trước cả năm phép đó là **phép số 0**: gắn lại nhãn hai cảnh bằng chính bộ nhãn
độc lập ``scripts/gan_nhan_hai_canh.py`` (cùng prompt, cùng PROMPT_VERSION đã cho
ra con số 28/55 của đề thật), chạy trên ``kis_query_vi`` chứ không đọc cờ
``co_2_canh`` của người sinh. Có hai lý do bắt buộc:

* shard b khai ``co_2_canh=true`` cho 8 mục mà KHÔNG có trường ``canh_A``/``canh_B``
  nào — không tách được cảnh thì không có gì để đối chiếu với ảnh;
* tỉ lệ hai cảnh chỉ so được với 51% của đề thật nếu ĐO BẰNG CÙNG MỘT THƯỚC. Lấy
  cờ tự khai so với nhãn của bộ gán nhãn là so hai thước khác nhau.

Nhãn cache vào ``data/cache_kiem_gt/nhan/`` — thư mục RIÊNG, không đụng vào
``data/cache_cap_thoi_gian/nhan_de/`` là bằng chứng của con số đề thật.

Cache mỗi mục xuống ``data/cache_kiem_gt/`` nên chạy lại không tốn quota.

    python scripts/kiem_gt_moi.py                 # kiểm toàn bộ 4 shard
    python scripts/kiem_gt_moi.py --limit 4       # thử 4 mục đầu
    python scripts/kiem_gt_moi.py --bang          # chỉ in bảng từ cache
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

CACHE = ROOT / "data" / "cache_kiem_gt"
CACHE_NHAN = CACHE / "nhan"
PHIEN_BAN = 2

#: Khung neo dưới ngưỡng này thì mô tả không nói về khung được neo -> loại.
NEO_DAT = 0.50
#: Cảnh A phải xuất hiện ở ít nhất một khung TRƯỚC neo với mức này.
A_DAT = 0.50
#: Cảnh A mà khớp CHÍNH khung neo tới mức này thì A và B cùng một khung -> hai cảnh giả.
A_TRUNG_NEO = 0.70
#: Ngưỡng đếm "khung khác cũng khớp" cho phép đo độ dễ.
DE_NGUONG = 0.70
#: Bao nhiêu khung khác trong cùng video được lấy mẫu cho phép đo độ dễ.
#: 15 + khung neo = 16 = đúng 2 lô, vì ba model lite đã hết quota ngày và mỗi
#: lượt gọi thừa là một mục không kiểm được.
SO_KHUNG_KHAC = 15
#: Bao nhiêu khung trước neo được xem để tìm cảnh A (7 + neo = 1 lô).
SO_KHUNG_TRUOC = 7
#: "Mô tả chung cả video" chỉ được coi là LỖI khi có bằng chứng ẢNH đi kèm:
#: chừng này khung KHÁC trong cùng video cũng khớp mô tả đó.
MO_HO_LOAI = 5
#: Khung cách neo dưới ngần này coi như cùng một cú máy, không tính là "khung khác".
CUNG_CU_MAY = 3


# --------------------------------------------------------------------- dữ liệu
def nap_shard(data_dir: Path, shards="abcd") -> list:
    """Hợp nhất các shard, gắn số thứ tự toàn cục để tra ngược được."""
    out = []
    for s in shards:
        p = data_dir / f"gt_moi_shard_{s}.json"
        if not p.exists():
            print(f"  ! thiếu {p.name}")
            continue
        for k, it in enumerate(json.loads(p.read_text(encoding="utf-8"))):
            it = dict(it)
            it["_shard"] = s
            it["_idx_shard"] = k
            it["_ma"] = f"{s}{k:02d}"
            out.append(it)
    return out


def ban_do_khung(data_dir: Path):
    """{video: {n: (frame_idx, filename)}} và {video: [n đã sắp xếp]}."""
    md = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    theo_video: dict = defaultdict(dict)
    for r in md:
        theo_video[r["video_id"]][int(r["n"])] = (int(r["frame_idx"]), r["frame_filename"])
    ds = {v: sorted(d) for v, d in theo_video.items()}
    return theo_video, ds


def _ung_vien(khung: dict, video: str, ns) -> list:
    """(video, frame_idx, filename) cho judge.score."""
    return [(video, khung[video][n][0], khung[video][n][1]) for n in ns if n in khung[video]]


def mau_khung_khac(ds_n, anchor: int, k: int) -> list:
    """~k khung rải đều khắp video, bỏ những khung cùng cú máy với neo."""
    xa = [n for n in ds_n if abs(n - anchor) > CUNG_CU_MAY]
    if len(xa) <= k:
        return xa
    buoc = len(xa) / k
    return [xa[int(i * buoc)] for i in range(k)]


def mau_khung_truoc(ds_n, anchor: int, n_tu: int, k: int) -> list:
    """Đến k khung TRƯỚC neo. Ưu tiên trong đoạn đã khai báo; nếu đoạn quá ngắn
    thì nới ngược về trước — thà xem rộng hơn rồi ghi lại là cảnh A nằm NGOÀI đoạn
    còn hơn kết luận "không có cảnh A" chỉ vì cửa sổ hẹp."""
    truoc = [n for n in ds_n if n < anchor]
    if not truoc:
        return []
    trong = [n for n in truoc if n >= n_tu]
    if len(trong) >= 3:
        chon = trong
    else:
        chon = truoc[-max(k, 6):]
    if len(chon) > k:
        buoc = len(chon) / k
        chon = [chon[int(i * buoc)] for i in range(k)]
    return chon


# ------------------------------------------------------------------ gọi model
def _goi(judge, contents, max_tokens=800):
    """Một request (ảnh hoặc text) theo đúng luật xoay model của src/core/vlm.py."""
    from google.genai import types
    from src.core.vlm import RETRY_WAIT, _is_daily_quota

    client = judge._get_client()
    last = None
    for model in judge._model_order():
        if model in judge.exhausted:
            continue
        for attempt in range(3):
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.0, max_output_tokens=max_tokens,
                        response_mime_type="application/json"),
                )
                judge.calls += 1
                u = getattr(r, "usage_metadata", None)
                if u:
                    judge.tokens_in += u.prompt_token_count or 0
                    judge.tokens_out += u.candidates_token_count or 0
                return r.text or "", model
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


def _json(text: str, mo="{", dong="}"):
    m = re.search(re.escape(mo) + r".*" + re.escape(dong), text or "", re.S)
    if not m:
        raise ValueError("không parse được JSON")
    return json.loads(m.group(0))


_P_VQA = """Bạn CHỈ được nhìn đúng một khung hình dưới đây, không có gì khác.

BỐI CẢNH (do người ra đề ghi): {ctx}
CÂU HỎI: {q}

Trả lời NGẮN GỌN đúng thứ nhìn thấy trong khung hình này. Nếu khung hình không đủ
thông tin để trả lời (chữ quá mờ, vật thể không có trong khung, phải xem khung khác
mới biết) thì tra_loi_duoc = false và giải thích vì sao.

Trả về DUY NHẤT một object JSON:
{{"tra_loi_duoc": true/false, "dap_an": "câu trả lời ngắn", "ly_do": "một câu"}}"""

_P_TUONG_DUONG = """So nghĩa từng cặp câu trả lời cho cùng một câu hỏi về một bức ảnh.

Hai câu trả lời TƯƠNG ĐƯƠNG nếu chỉ cùng một sự vật/giá trị, dù khác cách viết
("Màu be" = "be", "2" = "hai", "PETROLIMEX" = "Petrolimex", "kẻ caro" = "sọc ca rô").
KHÔNG tương đương nếu khác màu, khác số lượng, khác chữ, hoặc một bên né trả lời.

{cap}

Trả về DUY NHẤT một mảng JSON, mỗi phần tử {{"i":<số>,"tuong_duong":true/false,"ly_do":"ngắn"}}"""

_P_SOI = """Bạn đang SOI LỖI một mô tả được viết cho đúng bức ảnh này.

MÔ TẢ: {mo_ta}
{phan}
Tách mô tả thành từng chi tiết kiểm được (màu sắc, số lượng, chữ hiện trên hình,
vật thể, tư thế, góc máy) rồi đối chiếu TỪNG chi tiết với bức ảnh. Chỉ dựa vào
thứ NHÌN THẤY trong ảnh, đừng suy đoán từ chủ đề chung.

Một chi tiết là MÂU THUẪN khi ảnh cho thấy điều NGƯỢC LẠI (mô tả nói áo xanh mà
ảnh là áo vàng, nói hai người mà ảnh có một người). Chi tiết chỉ KHÔNG THẤY được
(bị che, ngoài khung, chữ quá nhỏ) thì KHÔNG phải mâu thuẫn.

Trả về DUY NHẤT một object JSON:
{{"mau_thuan": true/false,
  "chi_tiet_sai": ["mô tả nói X nhưng ảnh là Y", ...],
  "chi_tiet_khong_thay": ["..."],
  "ly_do": "một câu"}}"""

_P_DINH_VI = """Bạn đang lọc các câu mô tả dùng để TÌM MỘT KHOẢNH KHẮC trong một video dài.

Với TỪNG câu bên dưới, quyết định: câu này có ghim ĐƯỢC một khoảnh khắc cụ thể
(có chi tiết nhìn thấy được chỉ đúng ở một cú máy: màu áo, chữ trên hình, số lượng,
tư thế, vật thể cụ thể), hay chỉ TẢ CHUNG chủ đề cả video / một loại cảnh lặp đi
lặp lại suốt video (ví dụ "phát thanh viên đang dẫn bản tin trong trường quay" mà
không có chi tiết riêng nào)?

{cau}

Trả về DUY NHẤT một mảng JSON, mỗi phần tử
{{"i":<số>,"dinh_vi":true/false,"ly_do":"một câu ngắn"}}"""


def gan_nhan_lai(judge, muc: list, refresh=False) -> None:
    """Phép số 0: gắn lại nhãn hai cảnh bằng bộ nhãn độc lập, không đọc cờ tự khai.

    Dùng thẳng ``_PROMPT`` và ``_chuan_hoa`` của scripts/gan_nhan_hai_canh.py để
    nhãn ở đây và nhãn 28/55 của đề thật ra từ cùng một thước đo. Ghi vào mục dưới
    khoá ``_nhan``; câu nào thiếu ``canh_A``/``canh_B`` thì lấy luôn bản tách của
    bộ nhãn để phép đo ảnh số 2 có cái mà chấm.
    """
    from scripts.gan_nhan_hai_canh import _PROMPT, PROMPT_VERSION, _chuan_hoa

    CACHE_NHAN.mkdir(parents=True, exist_ok=True)
    can = 0
    for it in muc:
        p = CACHE_NHAN / f"{it['_ma']}.json"
        rec = None
        if p.exists() and not refresh:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("prompt_version") == PROMPT_VERSION:
                rec = d
        if rec is None:
            try:
                txt, model = _goi(judge, _PROMPT.format(
                    vi=it["kis_query_vi"], en=it.get("kis_query_en") or "(không có)"),
                    max_tokens=1200)
                rec = _chuan_hoa(_json(txt))
            except Exception as exc:  # noqa: BLE001
                print(f"  ! nhãn {it['_ma']}: {str(exc)[:90]}")
                continue
            rec.update({"ma": it["_ma"], "model": model, "prompt_version": PROMPT_VERSION})
            p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            can += 1
        it["_nhan"] = rec
    if can:
        print(f"  gắn lại nhãn hai cảnh cho {can} câu (bộ nhãn độc lập)")


def _canh_ab(it: dict):
    """(canh_A, canh_B, nguồn) — ưu tiên trường của người sinh, thiếu thì lấy nhãn."""
    a = str(it.get("canh_A") or "").strip()
    b = str(it.get("canh_B") or "").strip()
    if a and b:
        return a, b, "người sinh"
    n = it.get("_nhan") or {}
    if n.get("canh_A_vi") and n.get("canh_B_vi"):
        return n["canh_A_vi"].strip(), n["canh_B_vi"].strip(), "bộ nhãn độc lập"
    return a, b, "thiếu"


# ------------------------------------------------------------------- kiểm 1 mục
def kiem_mot(it: dict, judge, khung, ds) -> dict:
    video = it["video_id"]
    anchor = int(it["n"])
    khai_bao = bool(it.get("co_2_canh"))
    nhan = it.get("_nhan") or {}
    doc_lap = bool(nhan.get("co_2_canh"))
    q_a, q_b, nguon_ab = _canh_ab(it)
    # Kiểm cảnh A bất cứ khi nào MỘT trong hai thước nói có hai cảnh: cờ tự khai
    # sai theo chiều nào cũng phải lộ ra, và chỉ có ảnh mới phân xử được.
    hai_canh = (khai_bao or doc_lap) and bool(q_a and q_b)
    ds_n = ds.get(video, [])
    kq: dict = {"ma": it["_ma"], "video_id": video, "n": anchor,
                "co_2_canh_khai_bao": khai_bao, "co_2_canh_doc_lap": doc_lap,
                "nguon_canh_AB": nguon_ab, "nhan_ly_do": str(nhan.get("ly_do", ""))[:200],
                "canh_bao": []}
    if khai_bao and not (it.get("canh_A") and it.get("canh_B")):
        kq["canh_bao"].append(
            "shard khai co_2_canh=true nhưng KHÔNG có trường canh_A/canh_B — "
            f"phải tách lại bằng {nguon_ab}")
    if khai_bao != doc_lap:
        kq["canh_bao"].append(
            f"cờ hai cảnh LỆCH: người sinh ghi {khai_bao}, bộ nhãn độc lập ghi {doc_lap}")

    if anchor not in khung.get(video, {}):
        kq["loi"] = f"khung neo n={anchor} không có trong metadata của {video}"
        return kq

    neo_idx, neo_fn = khung[video][anchor]

    # --- 1 + 5: chấm mô tả trên khung neo VÀ trên ~15 khung khác cùng video ----
    # Câu hai cảnh phải chấm theo cảnh SAU: chấm bằng cả câu là phạt khung neo vì
    # nửa câu đang tả một cú máy khác.
    q_neo = q_b if (hai_canh and q_b) else it["kis_query_vi"].strip()
    khac = mau_khung_khac(ds_n, anchor, SO_KHUNG_KHAC)
    diem = judge.score(q_neo, _ung_vien(khung, video, [anchor] + khac))
    kq["mo_ta_cham_theo"] = ("canh_B (" + nguon_ab + ")") if (hai_canh and q_b) else "kis_query_vi"
    d_neo = diem.get((video, neo_idx))
    kq["diem_neo"] = round(100 * d_neo[0], 1) if d_neo else None
    kq["ly_do_neo"] = d_neo[1] if d_neo else "KHÔNG chấm được (ảnh hỏng hoặc hết quota)"

    diem_khac = []
    for n in khac:
        if n in khung[video]:
            d = diem.get((video, khung[video][n][0]))
            if d:
                diem_khac.append((n, round(100 * d[0], 1)))
    kq["so_khung_khac_da_xem"] = len(diem_khac)
    kq["so_khung_khac_khop"] = sum(1 for _n, s in diem_khac if s >= 100 * DE_NGUONG)
    kq["khung_khac_cao_nhat"] = sorted(diem_khac, key=lambda t: -t[1])[:3]

    # --- 2: cảnh A có thật, và có TRƯỚC neo không -----------------------------
    if khai_bao and not hai_canh:
        kq["canh_bao"].append("khai co_2_canh=true nhưng KHÔNG tách được cảnh A/B "
                              "— không có gì để đối chiếu với ảnh")
    if hai_canh:
        n_tu = int((it.get("sinh_tu") or {}).get("n_tu", anchor))
        truoc = mau_khung_truoc(ds_n, anchor, n_tu, SO_KHUNG_TRUOC)
        kq["khung_truoc_da_xem"] = truoc
        kq["canh_A_da_dung"] = q_a[:200]
        if not truoc:
            kq["canh_bao"].append("không có khung nào TRƯỚC neo trong video")
            kq["diem_A_max"] = None
        else:
            d_a = judge.score(q_a, _ung_vien(khung, video, truoc + [anchor]))
            ds_a = []
            for n in truoc:
                if n in khung[video]:
                    d = d_a.get((video, khung[video][n][0]))
                    if d:
                        ds_a.append((n, round(100 * d[0], 1)))
            kq["diem_A_tung_khung"] = ds_a
            if ds_a:
                # Hoà điểm thì lấy khung GẦN NEO NHẤT: cảnh A là cú máy ngay
                # trước cú cắt, không phải khung đầu tiên tình cờ cùng điểm.
                n_best, s_best = max(ds_a, key=lambda t: (t[1], t[0]))
                kq["diem_A_max"] = s_best
                kq["khung_A_tot_nhat"] = n_best
                kq["A_trong_doan"] = bool(n_best >= n_tu)
            else:
                kq["diem_A_max"] = None
            da = d_a.get((video, neo_idx))
            kq["diem_A_tren_neo"] = round(100 * da[0], 1) if da else None

    # --- 3: Q&A trả lời được từ khung neo, và đáp án có đúng không -------------
    from google.genai import types

    blob = judge._fetch(video, neo_fn)
    if blob is None:
        kq["canh_bao"].append("KHÔNG tải được khung neo — phần Q&A không kiểm được")
        kq["vqa"] = None
    else:
        try:
            txt, model = _goi(judge, [types.Part.from_bytes(data=blob, mime_type="image/jpeg"),
                                      _P_VQA.format(ctx=(it.get("vqa_context") or "(không có)")[:400],
                                                    q=it["vqa_question"])])
            raw = _json(txt)
            kq["vqa"] = {"tra_loi_duoc": bool(raw.get("tra_loi_duoc")),
                         "dap_an_doc_lap": str(raw.get("dap_an") or "")[:120],
                         "ly_do": str(raw.get("ly_do") or "")[:200],
                         "dap_an_khai_bao": it.get("vqa_answer", ""),
                         "model": model}
        except Exception as exc:  # noqa: BLE001
            kq["vqa"] = None
            kq["canh_bao"].append(f"gọi Q&A lỗi: {str(exc)[:80]}")
    return kq


def soi_chi_tiet(judge, ket: list, muc: list, khung) -> None:
    """Lượt SOI: chấm 0-100 quá dễ dãi với chi tiết, phải hỏi thẳng "sai chỗ nào".

    Bằng chứng cho lượt này: câu b00 tả cảnh A là "đoàn đua mặc áo XANH quay từ
    trên cao"; khung 13 đúng là cảnh quay từ trên cao thành hàng dọc nhưng áo
    VÀNG — lượt chấm 0-100 vẫn cho 95. Một mô tả sai màu vẫn kéo được điểm cao vì
    nó đúng loại cảnh, nên phải có một lượt hỏi riêng về TỪNG chi tiết.

    Soi khung neo cho mọi mục, và soi thêm khung cảnh A của câu hai cảnh.
    """
    from google.genai import types

    for k, it in zip(ket, muc):
        if k is None or k.get("soi") is not None:
            continue
        viec = []
        q_a, q_b, _ng = _canh_ab(it)
        hai = str(k.get("mo_ta_cham_theo", "")).startswith("canh_B")
        # Soi bằng CẢ CÂU, không phải bằng canh_B. Lý do là một lỗi thật: câu a06
        # tả người phỏng vấn "đeo kính cận" trong khi khung neo không có kính —
        # nhưng chi tiết ấy bị rơi mất khi người sinh viết canh_B, nên soi canh_B
        # thì lỗi biến mất. Phần tả cảnh TRƯỚC được dặn xếp vào "không thấy".
        viec.append(("neo", k["n"], it["kis_query_vi"],
                     ("\nBức ảnh này là cảnh SAU trong mô tả (cảnh B). Phần mô tả nói về "
                      "cảnh TRƯỚC đó xếp vào chi_tiet_khong_thay, KHÔNG tính là mâu thuẫn."
                      f"\nCẢNH SAU là: {q_b}") if hai else ""))
        if k.get("khung_A_tot_nhat") is not None and q_a:
            viec.append(("canh_A", k["khung_A_tot_nhat"], q_a, ""))
        soi = {}
        for ten, n, mo_ta, phan in viec:
            if n not in khung.get(k["video_id"], {}):
                continue
            blob = judge._fetch(k["video_id"], khung[k["video_id"]][n][1])
            if blob is None:
                continue
            try:
                txt, _m = _goi(judge, [types.Part.from_bytes(data=blob, mime_type="image/jpeg"),
                                       _P_SOI.format(mo_ta=str(mo_ta)[:900], phan=phan)],
                               max_tokens=900)
                raw = _json(txt)
                soi[ten] = {"n": n, "mau_thuan": bool(raw.get("mau_thuan")),
                            "chi_tiet_sai": [str(x)[:160] for x in
                                             (raw.get("chi_tiet_sai") or [])][:5],
                            "ly_do": str(raw.get("ly_do") or "")[:200]}
            except Exception as exc:  # noqa: BLE001
                print(f"  ! soi {k['ma']}/{ten}: {str(exc)[:80]}")
        if soi:
            k["soi"] = soi
            print(f"  soi {k['ma']}: " + ", ".join(
                f"{t}={'MÂU THUẪN' if s['mau_thuan'] else 'ok'}" for t, s in soi.items()),
                flush=True)


def _chuan(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"^(màu|mau|chữ|chu)\s+", "", s)
    return re.sub(r"[^0-9a-zà-ỹ ]+", "", s).strip()


def _bo_dau(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", _chuan(s))
                   if unicodedata.category(c) != "Mn").replace("đ", "d")


def so_dap_an(judge, ket: list, muc: list) -> None:
    """So nghĩa đáp án độc lập với vqa_answer — lô 8 cặp một lượt text.

    Một trường hợp phải xử riêng: hai bản đọc CHỈ khác nhau ở dấu tiếng Việt
    ("TRỌNG HIỂN" / "TRỌNG HIẾN"). Ở 512px chính máy kiểm cũng không đọc chắc hơn
    máy sinh, nên tuyên "sai" ở đây là loại một câu tốt bằng bằng chứng không có.
    Cho qua, nhưng gắn cờ để người soát mở ảnh gốc quyết.
    """
    can = []
    for k, it in zip(ket, muc):
        v = k.get("vqa")
        if not v or not v["tra_loi_duoc"]:
            continue
        a, b = _chuan(v["dap_an_doc_lap"]), _chuan(v["dap_an_khai_bao"])
        if a and b and (a == b or a in b or b in a):
            v["tuong_duong"] = True
            v["so_boi"] = "khớp chuỗi"
            continue
        if a and b and _bo_dau(a) == _bo_dau(b):
            v["tuong_duong"] = True
            v["so_boi"] = "CHỈ KHÁC DẤU — máy kiểm không đọc chắc hơn máy sinh ở 512px"
            k.setdefault("canh_bao", []).append(
                f"đáp án khác dấu: đề '{v['dap_an_khai_bao']}' / máy đọc "
                f"'{v['dap_an_doc_lap']}' — cần người mở ảnh gốc xác nhận")
            continue
        can.append((k, it))
    for i in range(0, len(can), 8):
        lo = can[i:i + 8]
        cap = "\n\n".join(
            f"[{j}] CÂU HỎI: {it['vqa_question']}\n    ĐÁP ÁN A (người ra đề): {k['vqa']['dap_an_khai_bao']}"
            f"\n    ĐÁP ÁN B (máy nhìn ảnh): {k['vqa']['dap_an_doc_lap']}"
            for j, (k, it) in enumerate(lo))
        try:
            txt, _m = _goi(judge, _P_TUONG_DUONG.format(cap=cap), max_tokens=1200)
            rows = _json(txt, "[", "]")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! so đáp án lô {i}: {exc}")
            continue
        for r in rows:
            try:
                j = int(r["i"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= j < len(lo):
                lo[j][0]["vqa"]["tuong_duong"] = bool(r.get("tuong_duong"))
                lo[j][0]["vqa"]["so_boi"] = str(r.get("ly_do") or "")[:150]


def cham_dinh_vi(judge, ket: list, muc: list) -> None:
    """Câu có ghim một khoảnh khắc hay tả chung cả video — lô 6 câu một lượt text."""
    can = [(k, it) for k, it in zip(ket, muc) if "dinh_vi" not in k]
    for i in range(0, len(can), 6):
        lo = can[i:i + 6]
        cau = "\n\n".join(f"[{j}] {it['kis_query_vi']}" for j, (_k, it) in enumerate(lo))
        try:
            txt, _m = _goi(judge, _P_DINH_VI.format(cau=cau), max_tokens=1200)
            rows = _json(txt, "[", "]")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! định vị lô {i}: {exc}")
            continue
        for r in rows:
            try:
                j = int(r["i"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= j < len(lo):
                lo[j][0]["dinh_vi"] = bool(r.get("dinh_vi"))
                lo[j][0]["dinh_vi_ly_do"] = str(r.get("ly_do") or "")[:200]


# ------------------------------------------------------------------ phán quyết
def phan_quyet(k: dict) -> dict:
    """Gộp năm phép đo thành ĐẠT / LOẠI, kèm lý do đọc được bằng mắt người.

    "Không đo được" KHÔNG phải là "loại". Một mục mà quota chết giữa chừng chưa
    bị chứng minh là sai; xếp nó vào loại là để một sự cố hạ tầng giả dạng thành
    kết luận về chất lượng bộ đo. Những mục ấy ra trạng thái riêng CHUA_KIEM.
    """
    ly_do, thieu = [], []
    if k.get("loi"):
        return {"dat": False, "ly_do_loai": [k["loi"]], "nhom_loi": "du_lieu",
                "chua_kiem": False}

    dn = k.get("diem_neo")
    if dn is None:
        thieu.append("không chấm được khung neo (ảnh hỏng / hết quota)")
    elif dn < 100 * NEO_DAT:
        ly_do.append(f"khung neo KHÔNG khớp mô tả ({dn:.0f}/100): {k.get('ly_do_neo','')}")

    # Cờ hai cảnh CÓ BẰNG CHỨNG ẢNH: chỉ bật khi cảnh A thật sự tìm thấy ở một
    # khung TRƯỚC neo, và không đồng thời nằm luôn trong khung neo.
    co_2_canh_kiem = False
    da = k.get("diem_A_max")
    atn = k.get("diem_A_tren_neo")
    if "diem_A_max" in k:
        if da is None:
            thieu.append("không kiểm được cảnh A")
        elif da < 100 * A_DAT:
            if k.get("co_2_canh_khai_bao"):
                ly_do.append(f"cảnh A KHÔNG có thật trước khung neo (cao nhất {da:.0f}/100) "
                             "— câu hai cảnh giả còn tệ hơn câu một cảnh")
            else:
                k.setdefault("canh_bao", []).append(
                    f"bộ nhãn cho là hai cảnh nhưng ảnh không đỡ (A cao nhất {da:.0f}/100)")
        elif (atn is not None and atn >= 100 * A_TRUNG_NEO
              and (dn or 0) >= 100 * A_TRUNG_NEO):
            ly_do.append(f"hai cảnh GIẢ: cảnh A khớp luôn chính khung neo ({atn:.0f}/100) "
                         "— A và B nằm cùng một khung, không phải trình tự thời gian")
        else:
            co_2_canh_kiem = True
    elif k.get("co_2_canh_khai_bao"):
        # Khai hai cảnh mà không tách nổi thành hai mô tả độc lập: không có gì
        # để đối chiếu với ảnh, nên cờ này không được tính vào tỉ lệ hai cảnh.
        k.setdefault("canh_bao", []).append(
            "cờ co_2_canh=true KHÔNG có bằng chứng nào đỡ — hạ xuống false")
    k["co_2_canh_kiem"] = co_2_canh_kiem

    # Lượt soi chi tiết là CỜ, KHÔNG phải cổng — và đây là kết luận từ số liệu,
    # không phải từ thiện chí. Đem soi ra làm cổng thì ba mục đầu của shard a bị
    # loại sạch, trong đó a00 là mục tôi đã tự mở ảnh ra xem và thấy đúng (soi
    # đòi phải nhìn thấy NGỌN LỬA trong đúng khung đó mới cho là "vụ hỏa hoạn").
    # Nhưng nó cũng bắt được lỗi thật mà điểm 95 bỏ qua: a02 tả "xe bồn màu xanh
    # lá quân đội" trong khi thân bồn TRẮNG, chỉ ca-bin mới xanh quân đội.
    # Một dụng cụ vừa bắt đúng vừa bắt oan thì chỗ của nó là bàn người soát.
    soi = k.get("soi") or {}
    if not soi:
        k.setdefault("canh_bao", []).append("CHƯA soi chi tiết — chỉ có điểm 0-100")
    can_soat = []
    for ten, nhan_ten in (("neo", "khung neo"), ("canh_A", "khung cảnh A")):
        s = soi.get(ten) or {}
        if s.get("mau_thuan"):
            can_soat.append(f"{nhan_ten} (khung {s.get('n')}): "
                            + ("; ".join(s.get("chi_tiet_sai") or []) or s.get("ly_do", "")))
    if can_soat:
        k["can_nguoi_soat"] = can_soat
        k.setdefault("canh_bao", []).extend("SOI: " + x for x in can_soat)

    v = k.get("vqa")
    if v is None:
        thieu.append("không kiểm được Q&A")
    elif not v.get("tra_loi_duoc"):
        ly_do.append(f"Q&A KHÔNG trả lời được từ khung neo: {v.get('ly_do','')}")
    elif "tuong_duong" not in v:
        thieu.append("không so được đáp án")
    elif not v.get("tuong_duong"):
        ly_do.append(f"vqa_answer SAI: đề ghi '{v.get('dap_an_khai_bao')}', "
                     f"nhìn ảnh ra '{v.get('dap_an_doc_lap')}' ({v.get('so_boi','')})")

    # Tiêu chí 4 chỉ được LOẠI khi có bằng chứng ẢNH đỡ lưng. Một lượt chấm
    # thuần văn bản gọi "mô tả chung" trong khi 14/15 khung khác của chính video
    # đó KHÔNG khớp là phán đoán tiên nghiệm cãi lại số liệu — và loại theo nó là
    # cách vứt câu tốt đi để có con số đẹp.
    mo_ho = k.get("so_khung_khac_khop", 0) or 0
    if k.get("dinh_vi") is False:
        if mo_ho >= MO_HO_LOAI:
            ly_do.append(f"mô tả CHUNG cả video: {mo_ho}/{k.get('so_khung_khac_da_xem',0)} "
                         f"khung KHÁC cùng video cũng khớp — {k.get('dinh_vi_ly_do','')}")
        else:
            k.setdefault("canh_bao", []).append(
                f"chấm văn bản cho là mô tả chung ({k.get('dinh_vi_ly_do','')}) nhưng ảnh "
                f"không đỡ: chỉ {mo_ho}/{k.get('so_khung_khac_da_xem',0)} khung khác khớp "
                "— giữ lại")

    nhom = "" if not ly_do else (
        "qa" if all(x.startswith(("Q&A", "vqa_answer")) for x in ly_do) else "kis")
    return {"dat": not ly_do and not thieu, "ly_do_loai": ly_do, "nhom_loi": nhom,
            "chua_kiem": bool(thieu) and not ly_do, "thieu": thieu,
            "can_nguoi_soat": k.get("can_nguoi_soat", [])}


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--shard", default="abcd")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bang", action="store_true", help="chỉ in bảng từ cache, không gọi API")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--khong-soi", action="store_true",
                    help="bỏ lượt soi chi tiết (chấm 0-100 dễ bỏ lọt lỗi màu/số lượng)")
    args = ap.parse_args()

    data_dir = Path(args.data)
    muc = nap_shard(data_dir, args.shard)
    if args.limit:
        muc = muc[:args.limit]
    print(f"{len(muc)} mục từ shard {', '.join(args.shard)}")
    khung, ds = ban_do_khung(data_dir)
    CACHE.mkdir(parents=True, exist_ok=True)

    from src.core.vlm import VLMJudge

    judge = VLMJudge(data_dir=data_dir)
    if not judge.ready and not args.bang:
        print("thiếu GEMINI_API_KEY trong .env")
        return 2

    # Phép số 0 — gắn lại nhãn hai cảnh trước, vì phép đo ảnh số 2 cần văn bản
    # cảnh A/B mà 8 mục của shard b không có.
    if not args.bang:
        gan_nhan_lai(judge, muc, refresh=args.refresh)
    else:
        for it in muc:
            p = CACHE_NHAN / f"{it['_ma']}.json"
            if p.exists():
                it["_nhan"] = json.loads(p.read_text(encoding="utf-8"))

    ket = []
    can_goi = []
    for it in muc:
        p = CACHE / f"{it['_ma']}.json"
        d = None
        if p.exists() and not args.refresh:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("phien_ban") != PHIEN_BAN:
                d = None
        if d and d.get("diem_A_tung_khung"):
            # Áp lại luật hoà điểm mới trên điểm đã cache (không tốn lượt gọi);
            # nếu khung cảnh A đổi thì kết quả soi cũ soi nhầm khung, phải bỏ.
            n_moi = max(d["diem_A_tung_khung"], key=lambda t: (t[1], t[0]))[0]
            if n_moi != d.get("khung_A_tot_nhat"):
                d["khung_A_tot_nhat"] = n_moi
                d.pop("soi", None)
        ket.append(d)
        if d is None:
            can_goi.append(it["_ma"])

    if can_goi and not args.bang:
        print(f"cần kiểm {len(can_goi)} mục (mở ảnh thật, chấm bằng VLM) ...", flush=True)
        for i, it in enumerate(muc):
            if ket[i] is not None:
                continue
            try:
                k = kiem_mot(it, judge, khung, ds)
            except Exception as exc:  # noqa: BLE001
                print(f"  {it['_ma']}: LỖI {type(exc).__name__}: {str(exc)[:100]}")
                continue
            k["phien_ban"] = PHIEN_BAN
            ket[i] = k
            print(f"  {it['_ma']} {it['video_id']} n={it['n']}  neo={k.get('diem_neo')}"
                  f"  A={k.get('diem_A_max','-')}  khác khớp={k.get('so_khung_khac_khop')}",
                  flush=True)
    # Ba lượt còn lại đều bỏ qua mục đã có kết quả, nên chạy lại script là rẻ —
    # và chúng phải nằm NGOÀI nhánh trên, để lượt soi chi tiết thêm sau vẫn chạy
    # được trên cache cũ mà không phải chấm lại toàn bộ ảnh.
    if not args.bang:
        con = [(k, it) for k, it in zip(ket, muc) if k is not None]
        if con:
            so_dap_an(judge, [k for k, _ in con], [it for _, it in con])
            cham_dinh_vi(judge, [k for k, _ in con], [it for _, it in con])
            if not args.khong_soi:
                soi_chi_tiet(judge, [k for k, _ in con], [it for _, it in con], khung)
        for k, it in zip(ket, muc):
            if k is not None:
                (CACHE / f"{it['_ma']}.json").write_text(
                    json.dumps(k, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n" + judge.cost_note())

    thieu = [it["_ma"] for k, it in zip(ket, muc) if k is None]
    ket_co = [(k, it) for k, it in zip(ket, muc) if k is not None]

    # ------------------------------------------------------------------- xuất
    dat, loai, chua = [], [], []
    for k, it in ket_co:
        pq = phan_quyet(k)
        k["phan_quyet"] = pq
        ban = {key: v for key, v in it.items() if not key.startswith("_")}
        # Cờ trong bộ đo xuất ra là cờ CÓ BẰNG CHỨNG, không phải cờ tự khai. Giữ
        # nguyên cờ cũ dưới tên khác để đối chiếu được về sau.
        ban["co_2_canh_khai_bao"] = bool(it.get("co_2_canh"))
        ban["co_2_canh"] = k.get("co_2_canh_kiem", False)
        a, b, nguon = _canh_ab(it)
        if ban["co_2_canh"]:
            ban["canh_A"], ban["canh_B"], ban["nguon_canh_AB"] = a, b, nguon
        ban["kiem_doc_lap"] = {
            "co_2_canh_doc_lap": k.get("co_2_canh_doc_lap"),
            "nhan_ly_do": k.get("nhan_ly_do"),
            "diem_neo": k.get("diem_neo"), "cham_theo": k.get("mo_ta_cham_theo"),
            "diem_A_max": k.get("diem_A_max"), "khung_A_tot_nhat": k.get("khung_A_tot_nhat"),
            "diem_A_tren_neo": k.get("diem_A_tren_neo"), "A_trong_doan": k.get("A_trong_doan"),
            "vqa": k.get("vqa"), "dinh_vi": k.get("dinh_vi"),
            "so_khung_khac_da_xem": k.get("so_khung_khac_da_xem"),
            "so_khung_khac_khop": k.get("so_khung_khac_khop"),
            "khung_khac_cao_nhat": k.get("khung_khac_cao_nhat"),
            "canh_bao_kiem": k.get("canh_bao", []),
            "can_nguoi_soat": pq.get("can_nguoi_soat", []),
        }
        if pq["dat"]:
            dat.append(ban)
        elif pq["chua_kiem"]:
            ban["chua_kiem_duoc"] = pq["thieu"]
            chua.append(ban)
        else:
            ban["ly_do_loai"] = pq["ly_do_loai"]
            ban["nhom_loi"] = pq["nhom_loi"]
            loai.append(ban)

    (data_dir / "ground_truth_moi.json").write_text(
        json.dumps(dat, ensure_ascii=False, indent=1), encoding="utf-8")
    (data_dir / "gt_moi_bi_loai.json").write_text(
        json.dumps(loai, ensure_ascii=False, indent=1), encoding="utf-8")
    if chua:
        (data_dir / "gt_moi_chua_kiem.json").write_text(
            json.dumps(chua, ensure_ascii=False, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ bảng
    n = len(ket_co)
    print(f"\n{'='*78}\nKẾT QUẢ KIỂM ĐỘC LẬP — {n} mục đã kiểm"
          + (f", {len(thieu)} mục CHƯA kiểm được: {thieu}" if thieu else ""))
    print("cột 2c: cờ hai cảnh  khai/nhãn độc lập/đã kiểm bằng ảnh")
    print("cột soi: chi tiết mâu thuẫn với ảnh — N=khung neo, A=khung cảnh A")
    print(f"{'mã':>4} {'video':<10} {'n':>4} {'2c':>6} {'neo':>4} {'A':>4} {'A@neo':>5} "
          f"{'soi':>4} {'Q&A':>4} {'ghim':>4} {'khác':>5}  phán quyết")
    for k, _it in ket_co:
        v = k.get("vqa") or {}
        qa = "-" if not v else ("OK" if v.get("tra_loi_duoc") and v.get("tuong_duong") else "SAI")
        c2 = "".join("x" if k.get(f) else "." for f in
                     ("co_2_canh_khai_bao", "co_2_canh_doc_lap", "co_2_canh_kiem"))
        s = k.get("soi") or {}
        mt = ("?" if not s else
              ("".join(t[0].upper() for t in ("neo", "canh_A")
                       if (s.get(t) or {}).get("mau_thuan")) or "ok"))
        print(f"{k['ma']:>4} {k['video_id']:<10} {k['n']:>4} {c2:>6} "
              f"{(k.get('diem_neo') if k.get('diem_neo') is not None else -1):>4.0f} "
              f"{(k.get('diem_A_max') if k.get('diem_A_max') is not None else -1):>4.0f} "
              f"{(k.get('diem_A_tren_neo') if k.get('diem_A_tren_neo') is not None else -1):>5.0f} "
              f"{mt:>4} {qa:>4} {('x' if k.get('dinh_vi') else '.'):>4} "
              f"{k.get('so_khung_khac_khop', -1):>2}/{k.get('so_khung_khac_da_xem', 0):<2} "
              f" {'ĐẠT' if k['phan_quyet']['dat'] else ('CHƯA KIỂM: ' + '; '.join(k['phan_quyet']['thieu'])[:60] if k['phan_quyet']['chua_kiem'] else 'LOẠI: ' + '; '.join(k['phan_quyet']['ly_do_loai'])[:70])}")

    hai_dat = sum(1 for b in dat if b.get("co_2_canh"))
    hai_goc = sum(1 for _k, it in ket_co if it.get("co_2_canh"))
    hai_nhan = sum(1 for k, _ in ket_co if k.get("co_2_canh_doc_lap"))
    hai_kiem = sum(1 for k, _ in ket_co if k.get("co_2_canh_kiem"))
    lech = [k["ma"] for k, _ in ket_co
            if bool(k.get("co_2_canh_khai_bao")) != bool(k.get("co_2_canh_kiem"))]
    da_quyet = len(dat) + len(loai)
    print(f"\n--- THỐNG KÊ ---")
    print(f"đã kiểm xong      : {da_quyet}/{n}"
          + (f"  ({len(chua)} mục CHƯA đo đủ, không tính vào tỉ lệ)" if chua else ""))
    print(f"đạt               : {len(dat)}/{da_quyet} ({100*len(dat)/max(da_quyet,1):.0f}%)")
    print(f"loại              : {len(loai)}/{da_quyet} ({100*len(loai)/max(da_quyet,1):.0f}%)")
    nhom = defaultdict(int)
    for b in loai:
        nhom[b["nhom_loi"]] += 1
    for g, c in sorted(nhom.items()):
        print(f"  vì {g:<11}: {c}")
    # Tách riêng lỗi cấu trúc hai cảnh — đây là thứ cả bộ đo này sinh ra để đo.
    hai_hong = sum(1 for b in loai if b.get("co_2_canh")
                   and any("cảnh A" in x or "hai cảnh" in x for x in b["ly_do_loai"]))
    print(f"  trong đó cảnh A không có thật / hai cảnh giả: {hai_hong}")
    print(f"\nTỈ LỆ HAI CẢNH (đề thật của BTC: 28/55 = 51%, đo bằng cùng bộ nhãn)")
    print(f"  người sinh tự khai        : {hai_goc}/{n} ({100*hai_goc/max(n,1):.0f}%)")
    print(f"  bộ nhãn độc lập trên câu  : {hai_nhan}/{n} ({100*hai_nhan/max(n,1):.0f}%)")
    print(f"  CÓ BẰNG CHỨNG ẢNH         : {hai_kiem}/{n} ({100*hai_kiem/max(n,1):.0f}%)")
    print(f"  sau khi lọc, trong bộ đo  : {hai_dat}/{max(len(dat),1)} "
          f"({100*hai_dat/max(len(dat),1):.0f}%)")
    if lech:
        print(f"  cờ tự khai LỆCH với bằng chứng ở {len(lech)} mục: {', '.join(lech)}")
    chung_van_ban = sum(1 for k, _ in ket_co if k.get("dinh_vi") is False)
    da_soi = sum(1 for k, _ in ket_co if k.get("soi"))
    mt_neo = sum(1 for k, _ in ket_co if ((k.get("soi") or {}).get("neo") or {}).get("mau_thuan"))
    mt_a = sum(1 for k, _ in ket_co if ((k.get("soi") or {}).get("canh_A") or {}).get("mau_thuan"))
    print(f"\nsoi chi tiết (CỜ, không phải cổng): {da_soi}/{n} mục đã soi — "
          f"{mt_neo} mục bị chỉ ra mâu thuẫn ở khung neo, {mt_a} mục ở khung cảnh A")
    can_soat = [(k["ma"], k["phan_quyet"]["can_nguoi_soat"]) for k, _ in ket_co
                if k["phan_quyet"].get("can_nguoi_soat") and k["phan_quyet"]["dat"]]
    if can_soat:
        print(f"  {len(can_soat)}/{len(dat)} mục ĐẠT vẫn có chi tiết bị soi cờ — "
              "người phải mở ảnh ra phân xử, KHÔNG tự động loại:")
        for ma, ds_soat in can_soat:
            print(f"    {ma}: {ds_soat[0][:96]}")
    print(f"chấm văn bản gọi là 'mô tả chung': {chung_van_ban} mục — "
          f"chỉ {sum(1 for b in loai if any('mô tả CHUNG' in x for x in b['ly_do_loai']))} "
          f"mục trong số đó có ảnh đỡ lưng (>= {MO_HO_LOAI} khung khác cũng khớp) nên bị loại")
    theo_dai = defaultdict(lambda: [0, 0])
    for b in dat:
        theo_dai[b["video_id"][:3]][0] += 1
    for _k, it in ket_co:
        theo_dai[it["video_id"][:3]][1] += 1
    print("phân bố theo dải  : " + "  ".join(
        f"{d}:{a}/{b}" for d, (a, b) in sorted(theo_dai.items())))
    # Bốn shard ra từ bốn quy trình khác nhau (shard a có neo_kiem_mat trên cả 16
    # mục, shard b không có cả trường canh_A/canh_B), nên tỉ lệ đạt phải tách ra
    # theo shard — gộp lại là giấu mất chính chỗ quy trình hỏng.
    theo_shard = defaultdict(lambda: [0, 0, 0])
    for k, it in ket_co:
        r = theo_shard[it["_shard"]]
        r[1] += 1
        if k["phan_quyet"]["dat"]:
            r[0] += 1
        if k.get("co_2_canh_kiem"):
            r[2] += 1
    print("theo shard        : " + "  ".join(
        f"{s}:{a}/{b} đạt, {c} hai cảnh" for s, (a, b, c) in sorted(theo_shard.items())))
    mo_ho = [k["ma"] for k, _ in ket_co
             if k.get("so_khung_khac_khop", 0) >= 5 and k["phan_quyet"]["dat"]]
    if mo_ho:
        print(f"CẢNH BÁO mơ hồ (>=5/{SO_KHUNG_KHAC} khung khác cùng video cũng khớp, "
              f"vẫn giữ): {', '.join(mo_ho)}")
    print(f"\nghi: {data_dir/'ground_truth_moi.json'}  ({len(dat)} mục)")
    print(f"ghi: {data_dir/'gt_moi_bi_loai.json'}  ({len(loai)} mục)")
    if chua:
        print(f"ghi: {data_dir/'gt_moi_chua_kiem.json'}  ({len(chua)} mục CHƯA đo đủ — "
              "chạy lại khi quota hồi để chốt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
