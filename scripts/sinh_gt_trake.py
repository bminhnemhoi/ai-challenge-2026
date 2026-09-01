"""Sinh ground truth TRAKE — nhánh duy nhất còn bị chặn hoàn toàn.

TRAKE chiếm **8,5% đề thật** (2/25 câu vòng 2) nhưng có **0 câu trong MỌI bộ đo**.
Không có bộ đo thì không đo được gì, nên mọi ý tưởng TRAKE đều nằm im — kể cả
những ý tưởng đã phác sẵn trong `docs/CHAN_DOAN_TRAKE.md`.

Luật chấm TRAKE (mục 2.1.3): **0 nếu sai video**, ngược lại
``(1/N)·Σ_j I(id_j ∈ [s_j, e_j])`` — chấm **từng sự kiện một**, và **KHÔNG có
ràng buộc thứ tự**. Nên một mục ground truth TRAKE là: một video, N mô tả sự
kiện, và N frame_idx tương ứng.

## Thiết kế tránh đúng cái lỗi đã trả giá

Bộ sinh KIS từng hỏng vì nhét 9–12 ảnh vào **một** request: model tả nội dung
đúng nhưng **đánh sai số thứ tự ảnh**, nên khung neo lệch một cú cắt — và neo là
đáp án, nên bộ đo **sai dấu** chứ không phải yếu đi. Ở TRAKE lỗi ấy nhân lên N
lần, nên ở đây **không bao giờ** hỏi model "sự kiện này ở ảnh số mấy":

    bước 1  cho xem cả đoạn, hỏi N mô tả sự kiện — **thuần văn bản**, không số
    bước 2  với mỗi sự kiện, chấm TỪNG khung MỘT ẢNH MỘT REQUEST, lấy argmax
    bước 3  loại mục có sự kiện không khung nào đạt ngưỡng

Bước 2 chính là cơ chế của ``kiem_neo_don_anh.py`` đã chứng minh đúng (56/64 xác
nhận, 8 dời neo, và hai ca biên độ lớn nhất đã mở ảnh kiểm bằng mắt — cả hai đều
đúng). Không có danh sách thì không có chỗ để nhầm chỉ số.

    python -u scripts/sinh_gt_trake.py --so 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from src.core.vlm import DEFAULT_MODEL, VLMJudge  # noqa: E402

PROMPT_SINH = """Đây là {n} khung hình LIÊN TIẾP theo thời gian, cắt từ một video
tin tức tiếng Việt.

Hãy viết một câu hỏi kiểu TRAKE cho đoạn này: chọn **{k} khoảnh khắc** khác nhau
xảy ra nối tiếp nhau trong đoạn, và mô tả từng khoảnh khắc.

Quy tắc:
- Mỗi mô tả phải chỉ tới **một khoảnh khắc xác định**, không phải cả đoạn. Viết
  như đề thi: cụ thể, nhìn là nhận ra ngay ("khoảnh khắc người đàn ông áo trắng
  đưa tay bắt tay", chứ không phải "cảnh phỏng vấn").
- {k} khoảnh khắc phải PHÂN BIỆT được với nhau — đừng viết hai mô tả mà cùng một
  khung hình đều khớp.
- **TUYỆT ĐỐI KHÔNG nhắc tới số thứ tự ảnh.** Chỉ mô tả nội dung nhìn thấy.
- Thêm một câu "bối cảnh" chung một dòng, đúng văn phong đề thi.

Trả về DUY NHẤT JSON:
{{"boi_canh": "...", "su_kien": ["mô tả 1", "mô tả 2", ...]}}"""

PROMPT_CHAM = """Đây là MỘT khung hình duy nhất từ một video tin tức tiếng Việt.

Khoảnh khắc cần tìm: "{mo_ta}"

Khung hình này có ĐÚNG là khoảnh khắc đó không?

- 90-100: đúng khoảnh khắc, mọi chi tiết chính đều thấy
- 60-89 : đúng loại cảnh nhưng lệch vài chi tiết
- 30-59 : cùng bối cảnh nhưng KHÔNG phải khoảnh khắc này
- 0-29  : khác hẳn

Trả về DUY NHẤT JSON: {{"diem": 0-100, "ly_do": "một câu ngắn"}}"""


def hoi(judge, blobs, prompt, max_tokens=900):
    """Gọi có xoay vòng model chống 429 — bài học đã trả giá hai lần."""
    from google.genai import types

    from src.core.vlm import RETRY_WAIT, _is_daily_quota

    client = judge._get_client()
    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in blobs]
    cuoi = None
    for mo in judge._model_order():
        if mo in judge.exhausted:
            continue
        for lan in range(3):
            try:
                r = client.models.generate_content(
                    model=mo, contents=[*parts, prompt],
                    config=types.GenerateContentConfig(temperature=0.0,
                                                       max_output_tokens=max_tokens),
                )
                u = getattr(r, "usage_metadata", None)
                if u:
                    judge.calls += 1
                    judge.tokens_in += u.prompt_token_count or 0
                    judge.tokens_out += u.candidates_token_count or 0
                m = re.search(r"\{.*\}", r.text or "", re.S)
                if not m:
                    return None
                return json.loads(m.group(0))
            except Exception as exc:  # noqa: BLE001
                cuoi = exc
                msg = str(exc)
                if _is_daily_quota(msg):
                    judge.exhausted.add(mo)
                    break
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(RETRY_WAIT[min(lan, len(RETRY_WAIT) - 1)])
                    continue
                if "503" in msg or "UNAVAILABLE" in msg:
                    time.sleep(2.0 * (lan + 1))
                    continue
                break
    raise cuoi or RuntimeError("het model kha dung")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--so", type=int, default=20, help="so muc TRAKE can sinh")
    ap.add_argument("--k", type=int, default=3, help="so su kien moi muc")
    ap.add_argument("--doan", type=int, default=14, help="so keyframe moi doan")
    ap.add_argument("--nguong", type=int, default=70, help="diem toi thieu de nhan mot su kien")
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--ra", default=str(ROOT / "data" / "gt_trake.json"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_gt_trake"))
    args = ap.parse_args()

    data = Path(args.data)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    judge = VLMJudge(args.data, model=DEFAULT_MODEL)
    if not judge.ready:
        print("khong co GEMINI_API_KEY")
        return 2

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    theo_video: dict = {}
    for m in meta:
        theo_video.setdefault(m["video_id"], []).append(m)
    for v in theo_video:
        theo_video[v].sort(key=lambda m: int(m["n"]))
    del meta

    # Chon video NGAU NHIEN phan tang theo dai — khong chon theo diem SigLIP,
    # neu khong bo do chi phan anh chinh cai no dung de cham.
    rng = np.random.default_rng(args.seed)
    dai = sorted({v.split("_")[0] for v in theo_video})
    ung = []
    for i in range(args.so * 3):
        d = dai[i % len(dai)]
        cs = [v for v in theo_video if v.startswith(d) and len(theo_video[v]) >= args.doan + 4]
        if cs:
            ung.append(cs[int(rng.integers(0, len(cs)))])
    ung = list(dict.fromkeys(ung))[: args.so * 2]

    ra, bo_qua = [], []
    for vid in ung:
        if len(ra) >= args.so:
            break
        ks = theo_video[vid]
        b = int(rng.integers(0, max(1, len(ks) - args.doan)))
        doan = ks[b : b + args.doan]

        blobs, giu = [], []
        for m in doan:
            bl = judge._fetch(vid, m["frame_filename"])
            if bl:
                blobs.append(bl)
                giu.append(m)
        if len(blobs) < args.doan // 2:
            bo_qua.append((vid, "khong tai du anh"))
            continue

        h = hashlib.sha1(f"sinh|{vid}|{giu[0]['n']}|{args.k}".encode()).hexdigest()[:20]
        f = cache / f"{h}.json"
        if f.is_file():
            de = json.loads(f.read_text(encoding="utf-8"))
        else:
            try:
                de = hoi(judge, blobs, PROMPT_SINH.format(n=len(blobs), k=args.k))
            except Exception as exc:  # noqa: BLE001
                print(f"  {vid}: LOI sinh {type(exc).__name__}")
                break
            if not de or len(de.get("su_kien") or []) != args.k:
                bo_qua.append((vid, "sinh khong dung so su kien"))
                continue
            f.write_text(json.dumps(de, ensure_ascii=False), encoding="utf-8")

        # --- buoc 2: dinh vi TUNG su kien bang MOT ANH MOT REQUEST
        moc, diem_moc, hong = [], [], False
        for sk in de["su_kien"]:
            best, best_m = -1, None
            for bl, m in zip(blobs, giu):
                hc = hashlib.sha1(f"cham|{vid}|{m['n']}|{sk}".encode()).hexdigest()[:20]
                fc = cache / f"{hc}.json"
                if fc.is_file():
                    d = json.loads(fc.read_text(encoding="utf-8"))
                else:
                    try:
                        d = hoi(judge, [bl], PROMPT_CHAM.format(mo_ta=sk[:500]), 400) or {}
                    except Exception:  # noqa: BLE001
                        d = {}
                    fc.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
                v = int(d.get("diem", -1) or -1)
                if v > best:
                    best, best_m = v, m
            if best < args.nguong or best_m is None:
                hong = True
                bo_qua.append((vid, f"su kien khong co khung nao dat: diem cao nhat {best}"))
                break
            moc.append(best_m)
            diem_moc.append(best)

        if hong:
            continue
        # hai su kien tro vao CUNG mot khung => khong phan biet duoc, loai
        if len({int(m["frame_idx"]) for m in moc}) < args.k:
            bo_qua.append((vid, "hai su kien tro vao cung mot khung"))
            continue

        ra.append({
            "video_id": vid,
            "boi_canh": de.get("boi_canh", ""),
            "su_kien": de["su_kien"],
            "frames": [int(m["frame_idx"]) for m in moc],
            "n": [int(m["n"]) for m in moc],
            "diem_khop": diem_moc,
            "sinh_tu": {"n_tu": int(giu[0]["n"]), "n_den": int(giu[-1]["n"])},
            "dang": "trake",
        })
        print(f"[{len(ra):2d}] {vid} n={[int(m['n']) for m in moc]} "
              f"diem={diem_moc} | {de['su_kien'][0][:60]}", flush=True)

    Path(args.ra).write_text(json.dumps(ra, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(ra)} muc TRAKE -> {args.ra}")
    print(f"bo qua {len(bo_qua)} video:")
    for v, ly in bo_qua[:8]:
        print(f"  {v}: {ly}")
    print(f"\n{judge.cost_note()}")
    print("\nBUOC BAT BUOC TIEP THEO: mo anh kiem bang mat vai muc — buoc 2 chi noi")
    print("'khung nay khop mo ta nhat trong doan', KHONG noi 'mo ta nay dung'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
