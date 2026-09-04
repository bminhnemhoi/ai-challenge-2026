"""Nhân tử 0/1 của Q&A trên cả bộ đo — hai lever đã ship có mở khoá kênh đáp án không?

`docs/KE_HOACH_DINH_VI.md` §4.1 là ưu tiên số một, và §1 để ngỏ đúng một câu hỏi:

> "thứ cần cho đường Q&A đúng chính là lever §1, và nó có một kênh thứ hai chưa
> ai tính vào giá trị của nó."

Điểm Q&A = điểm định vị × **1[đáp án đúng]**. Bốn lane định vị chỉ đo thừa số
thứ nhất. Nếu lever cảnh B + lever hoán vị đưa được khung ĐÚNG lên đầu danh sách,
thì model đọc đáp án từ khung đúng — và cùng một lever ăn điểm **hai lần**.

Ba tập khung, mỗi tập đi qua ĐÚNG đường sinh đáp án của sản xuất
(`answer_qa.tra_loi_tu_ung_vien`: khung neo + 2 lân cận + 2 video dự phòng, ảnh
gốc, lời thoại, prompt cấm bỏ trống):

    (a) NEN       ứng viên gốc, chưa có lever nào
    (b) CANH_B    + ứng viên cảnh B (lever đã ship)
    (c) HOAN_VI   + hoán vị điểm nội-video (lever đã ship)

Đại lượng: P(đáp án đúng). Chia TUNE/TEST **phân tầng theo `co_2_canh`**,
bootstrap **theo câu**, báo cáo **riêng nhóm HAI cảnh** — nhóm duy nhất mà hai
lever chạm tới; nhóm MỘT cảnh là **đối chứng bất biến** (hai lever không đụng nó,
nên nó PHẢI ra kết quả y hệt ở cả ba tập — kiểm bằng assert trên số dòng).

    python -u scripts/do_nhan_tu_qa_bo_moi.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.answer_qa import nap_loi_thoai, tra_loi_tu_ung_vien  # noqa: E402
from scripts.do_cap_thoi_gian_moi import canh_cua  # noqa: E402
from scripts.experiment_cap_thoi_gian import KhoSims  # noqa: E402
from scripts.experiment_qa_answer import khop_rong  # noqa: E402
from scripts.experiment_cap_thoi_gian import _plan  # noqa: E402
from scripts.make_submission import (  # noqa: E402
    DEFAULT_N_FLAT,
    allocate_rows,
    hoan_vi_theo_canh_b,
)
from src.core.submission import Candidate, _default_answer_match  # noqa: E402
from src.core.vlm import DEFAULT_MODEL, VLMJudge  # noqa: E402

TAP = ("NEN", "CANH_B", "HOAN_VI")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache-uv", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_nhan_tu_qa"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--m", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vote3", action="store_true",
                    help="VOTE3: 3 mau temp=1.0 kem OCR, da so theo cum, so voi mot-phat-OCR")
    ap.add_argument("--nhieu-nen", action="store_true",
                    help="do san nhieu: chay lai NEN bo cache tren 40 muc phan tang")
    ap.add_argument("--ab-ocr", action="store_true",
                    help="R3: A/B chen chu OCR cua khung-doc vao prompt roi dung")
    ap.add_argument("--probe-sai", type=int, default=0,
                    help="R4: chay lai N lan temp=1.0 cac muc sai-dap-an/dung-video roi dung")
    ap.add_argument("--xuat-sai", default=None,
                    help="ghi JSON trang thai dung/sai tung muc cua tap NEN (0 API khi cache day)")
    args = ap.parse_args()

    data = Path(args.data)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache_uv) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    sach = [moi[i] for i in giu]
    c0 = [[Candidate(v, f, s, lf) for v, f, s, lf in uv[i]] for i in giu]
    if args.limit:
        sach, c0 = sach[: args.limit], c0[: args.limit]

    nhan = [canh_cua(m) for m in sach]
    bat = [i for i, c in enumerate(nhan) if c]
    print(f"bo sach {len(sach)} muc | cong hai canh BAT {len(bat)}")

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    meta_key = {(m["video_id"], int(m["frame_idx"])): m for m in meta}
    by_n = {(m["video_id"], int(m["n"])): m for m in meta}
    hang_of = {(m["video_id"], int(m["frame_idx"])): i for i, m in enumerate(meta)}
    last_of, vid_of, frm_of = {}, [], []
    for m in meta:
        last_of[m["video_id"]] = max(last_of.get(m["video_id"], 0), int(m["frame_idx"]))
        vid_of.append(m["video_id"])
        frm_of.append(int(m["frame_idx"]))
    vid_of, frm_of = np.array(vid_of), np.array(frm_of)
    del meta

    print("nap chi muc (mot lan, qua KhoSims) ...", flush=True)
    kho = KhoSims(args.data, False)
    simsB = {i: kho.lay(nhan[i][2], nhan[i][3]) for i in bat}
    caps = nap_loi_thoai(data)
    judge = VLMJudge(args.data, model=args.model)
    if not judge.ready:
        print("khong co GEMINI_API_KEY")
        return 2

    def dung_tap(ten):
        ra = []
        for i, cc in enumerate(c0):
            if ten == "NEN" or i not in simsB:
                ra.append(cc)
                continue
            s = simsB[i]
            k = min(args.m, len(s) - 1)
            top = np.argpartition(-s, k)[:k]
            top = top[np.argsort(-s[top])]
            co = {(c.video_id, int(c.frame_idx)) for c in cc}
            them = []
            for j in top:
                key = (str(vid_of[j]), int(frm_of[j]))
                if key in co:
                    continue
                co.add(key)
                them.append(Candidate(key[0], key[1], float(s[j]),
                                      last_of.get(key[0], key[1] + 1000)))
            cc2 = list(cc) + them
            if ten == "HOAN_VI":
                cc2 = hoan_vi_theo_canh_b(cc2, s, hang_of)
            ra.append(cc2)
        return ra

    def khung_de_doc(cands_q, last_of_):
        """DUNG duong san xuat: chon khung doc dap an THEO DIEM.

        Hai lever da ship the hien qua DIEM (canh B noi ung vien vao CUOI danh
        sach; hoan vi doi diem chu khong doi vi tri), nen doc theo thu tu danh
        sach thi ca ba tap cho dap an giong het nhau — do chinh la ket qua lan
        do truoc (48,5% o ca ba, neo doi 0/66 cau).

        Da thu mot phuong an khac — doc tu dau ra allocator (frame_rows) — va no
        TE HON: bo phu xac suat sinh DIEM LUOI, chi 8-20/100 dong la keyframe
        that, nen "dong dau la keyframe" thuong la ung vien yeu hon han.
        """
        return sorted(cands_q, key=lambda c: -float(c.score))

    ket = {}
    for ten in TAP:
        cl = [khung_de_doc(c, last_of) for c in dung_tap(ten)]
        dung = []
        for i, g in enumerate(sach):
            cau = f"Bối cảnh: {g['vqa_context']}\nCâu hỏi: {g['vqa_question']}"
            # KHOA CACHE = thu THAT SU quyet dinh dap an: bo khung se gui di +
            # cau hoi. KHONG phai nhan cua tap.
            #
            # Ly do: tra_loi_tu_ung_vien XOAY VONG model chong 429, nen cung mot
            # cau hoi voi cung bo khung van co the roi vao hai model khac nhau o
            # hai tap, va bat bien "nhom MOT canh khong doi" hong ngay — smoke
            # test da bat duoc dung loi nay (0/3 giong). Khoa theo bo khung lam
            # cac tap co bo khung TRUNG NHAU dung chung mot ket qua theo dinh
            # nghia, vua dung vua re gap ba o nhom mot canh.
            khung_key = "|".join(f"{c.video_id}:{int(c.frame_idx)}" for c in cl[i][:24])
            h = hashlib.sha1(f"{args.model}|{cau}|{khung_key}".encode("utf-8")).hexdigest()[:24]
            f = cache / f"{h}.json"
            if f.is_file():
                rec = json.loads(f.read_text(encoding="utf-8"))
            else:
                da, ghi = tra_loi_tu_ung_vien(judge, args.model, cl[i], meta_key,
                                              by_n, caps, cau)
                rec = {"dap_an": da, "ghi": ghi, "dong1": cl[i][0].video_id}
                f.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            chuan = g.get("vqa_answer") or ""
            rec["dung"] = bool(_default_answer_match(rec["dap_an"], chuan)
                               or khop_rong(rec["dap_an"], chuan))
            rec["video_dung"] = rec["dong1"] == g["video_id"]
            dung.append(rec)
            if (i + 1) % 20 == 0:
                print(f"  {ten}: {i+1}/{len(sach)}", flush=True)
        ket[ten] = dung
        if ten == "NEN" and args.probe_sai:
            # R4 — probe bất ổn định: chỉ mục SAI-đáp-án nhưng ĐÚNG-video (nhóm
            # duy nhất voting cứu được). Ép temp=1.0 bằng vá cục bộ tiến trình
            # này — KHÔNG đụng answer_qa sản xuất. Không cache (mỗi lần là một
            # mẫu mới). Ngưỡng tiền-đăng-ký: ≥1/3 số mục có ≥1 lần đúng.
            from google.genai import types as _t
            goc_cfg = _t.GenerateContentConfig
            _t.GenerateContentConfig = (
                lambda **kw: goc_cfg(**{**kw, "temperature": 1.0}))
            muc_tieu = [i for i, r in enumerate(dung)
                        if not r["dung"] and r["video_dung"]]
            print(f"\n=== PROBE R4: {len(muc_tieu)} mục × {args.probe_sai} lần "
                  f"temp=1.0 ===", flush=True)
            lat = 0
            for i in muc_tieu:
                g = sach[i]
                cau = f"Bối cảnh: {g['vqa_context']}\nCâu hỏi: {g['vqa_question']}"
                chuan = g.get("vqa_answer") or ""
                ky = []
                for _lan in range(args.probe_sai):
                    try:
                        da, _ghi = tra_loi_tu_ung_vien(judge, args.model, cl[i],
                                                       meta_key, by_n, caps, cau)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  muc#{i}: LOI {type(exc).__name__}", flush=True)
                        ky.append("?")
                        continue
                    ok = bool(_default_answer_match(da, chuan) or khop_rong(da, chuan))
                    ky.append("D" if ok else "s")
                lat += "D" in ky
                print(f"  muc#{i}: {''.join(ky)}", flush=True)
            _t.GenerateContentConfig = goc_cfg
            can = (len(muc_tieu) + 2) // 3
            print(f"\nPROBE R4: {lat}/{len(muc_tieu)} mục có ≥1 lần đúng "
                  f"(ngưỡng ≥{can}) -> "
                  f"{'VOTING CÓ CỬA' if lat >= can else 'ÂM — voting không có gì để cứu'}")
            return 0
        if ten == "NEN" and args.vote3:
            # VOTE3 — 3 mẫu temp=1.0 VỚI OCR-prompt (đúng cấu hình sản xuất
            # 59eb3fe), đa số theo CỤM khớp (khop_rong đôi một; hoà → mẫu 1).
            # Đối chứng = nhánh A/B OCR một-phát (đã cache). Ngưỡng đăng ký:
            # net ≥ +2 VÀ đúng→sai ≤ sàn nhiễu (10% số câu đang đúng).
            from google.genai import types as _t
            from src.core.ocr import OCRIndex
            goc_cfg = _t.GenerateContentConfig
            _t.GenerateContentConfig = (
                lambda **kw: goc_cfg(**{**kw, "temperature": 1.0}))
            oidx = OCRIndex(args.data, langs=["vi", "en"])

            def chu_ocr_v(cands_doc):
                mau = []
                for c in cands_doc[:2]:
                    cv = oidx._video(c.video_id)
                    a_kf = sorted(int(k) for k in cv.keys()) if cv else []
                    gan = [f for f in a_kf if abs(f - int(c.frame_idx)) <= 150]
                    for f in ([int(c.frame_idx)] + gan)[:3]:
                        for x in cv.get(str(f)) or []:
                            t = (x[1] if isinstance(x, (list, tuple)) and len(x) > 1
                                 else str(x))
                            if t and t not in mau:
                                mau.append(str(t))
                return " | ".join(mau)[:500]

            print(f"\n=== VOTE3 (temp=1.0, kèm OCR) trên {len(sach)} mục ===",
                  flush=True)
            len_v, xuong_v, n_so = 0, 0, 0
            for i, g in enumerate(sach):
                cau = f"Bối cảnh: {g['vqa_context']}\nCâu hỏi: {g['vqa_question']}"
                ocr_t = chu_ocr_v(cl[i])
                cau2 = cau + ("\nChữ đọc được trên các khung hình (OCR tự động, "
                              "có thể sai chính tả, chỉ dùng làm gợi ý): "
                              + ocr_t if ocr_t else "")
                khung_key = "|".join(f"{c.video_id}:{int(c.frame_idx)}"
                                     for c in cl[i][:24])
                # đối chứng một-phát (cache A/B)
                h2 = hashlib.sha1(f"{args.model}|{cau2}|{khung_key}"
                                  .encode("utf-8")).hexdigest()[:24]
                f2 = cache / f"{h2}.json"
                if not f2.is_file():
                    continue  # chỉ so trên mục đã có nhánh đối chứng
                rec2 = json.loads(f2.read_text(encoding="utf-8"))
                chuan = g.get("vqa_answer") or ""
                ok_1 = bool(_default_answer_match(rec2["dap_an"], chuan)
                            or khop_rong(rec2["dap_an"], chuan))
                mau3 = []
                for k3 in range(3):
                    h3 = hashlib.sha1(f"VOTE{k3}|{args.model}|{cau2}|{khung_key}"
                                      .encode("utf-8")).hexdigest()[:24]
                    f3 = cache / f"{h3}.json"
                    if f3.is_file():
                        mau3.append(json.loads(f3.read_text(encoding="utf-8"))["dap_an"])
                        continue
                    try:
                        da3, _g3 = tra_loi_tu_ung_vien(judge, args.model, cl[i],
                                                       meta_key, by_n, caps, cau2)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  muc#{i}: LOI {type(exc).__name__}", flush=True)
                        da3 = ""
                    f3.write_text(json.dumps({"dap_an": da3}, ensure_ascii=False),
                                  encoding="utf-8")
                    mau3.append(da3)
                # đa số theo cụm: mẫu nào khớp với ≥1 mẫu khác thì cụm đó thắng
                chon = mau3[0]
                for x in mau3:
                    if sum(1 for y in mau3 if y is not x
                           and (khop_rong(x, y) or _default_answer_match(x, y))) >= 1:
                        chon = x
                        break
                ok_v = bool(_default_answer_match(chon, chuan)
                            or khop_rong(chon, chuan))
                n_so += 1
                if ok_v and not ok_1:
                    len_v += 1
                if ok_1 and not ok_v:
                    xuong_v += 1
                if n_so % 20 == 0:
                    print(f"  {n_so} mục | lên {len_v} xuống {xuong_v}", flush=True)
            _t.GenerateContentConfig = goc_cfg
            print(f"\nVOTE3 vs một-phát-OCR (n={n_so}): lên {len_v} | xuống "
                  f"{xuong_v} | NET {len_v - xuong_v:+d}")
            print("NGƯỠNG: net ≥ +2 VÀ xuống ≤ 10% số câu đang đúng -> "
                  + ("QUA" if (len_v - xuong_v >= 2) else "KHÔNG QUA"))
            return 0
        if ten == "NEN" and args.nhieu_nen:
            # SÀN NHIỄU của chính nền — đăng ký TRƯỚC khi đọc thêm số A/B nào:
            # chạy lại y hệt NEN (không OCR, không temp patch) bỏ cache trên mẫu
            # phân tầng; đếm tự-lật hai chiều. Nếu đúng→sai của A/B ≤ sàn nhiễu
            # thì vế "0 lật" đang đo nhiễu nền, không đo cơ chế.
            rng = np.random.default_rng(20260904)
            m_dung = [i for i, r in enumerate(dung) if r["dung"]]
            m_sai = [i for i, r in enumerate(dung) if not r["dung"]]
            lay = (list(rng.choice(m_dung, size=min(20, len(m_dung)), replace=False))
                   + list(rng.choice(m_sai, size=min(20, len(m_sai)), replace=False)))
            print(f"\n=== SÀN NHIỄU NỀN: chạy lại {len(lay)} mục (20 đúng + 20 sai) ===",
                  flush=True)
            xuong, len_ = 0, 0
            for i in sorted(lay):
                g = sach[i]
                cau = f"Bối cảnh: {g['vqa_context']}\nCâu hỏi: {g['vqa_question']}"
                try:
                    da3, _g3 = tra_loi_tu_ung_vien(judge, args.model, cl[i],
                                                   meta_key, by_n, caps, cau)
                except Exception as exc:  # noqa: BLE001
                    print(f"  muc#{i}: LOI {type(exc).__name__}", flush=True)
                    continue
                chuan = g.get("vqa_answer") or ""
                ok3 = bool(_default_answer_match(da3, chuan) or khop_rong(da3, chuan))
                if dung[i]["dung"] and not ok3:
                    xuong += 1
                if not dung[i]["dung"] and ok3:
                    len_ += 1
            print(f"\nSÀN NHIỄU: tự-lật đúng→sai {xuong}/20 | sai→đúng {len_}/20")
            print("Đọc: down-rate A/B (4/77 = 5,2%) so với sàn đúng→sai này;")
            print("nếu A/B ≤ sàn + 1 mục thì 4 cú lật là nhiễu nền, không phải do OCR.")
            return 0
        if ten == "NEN" and args.ab_ocr:
            # R3 bậc A/B — chèn chữ OCR của CHÍNH các khung model sẽ đọc vào
            # prompt (KOCRBench: +12 điểm % Gemini). Nhánh không-OCR = cache NEN.
            # Thước: net ≥ +1 câu VÀ 0 câu đúng lật sai; báo riêng nhóm video-đúng.
            from src.core.ocr import OCRIndex
            oidx = OCRIndex(args.data, langs=["vi", "en"])

            def chu_ocr(cands_doc):
                mau = []
                for c in cands_doc[:2]:
                    cv = oidx._video(c.video_id)
                    a_kf = sorted(int(k) for k in cv.keys()) if cv else []
                    gan = [f for f in a_kf if abs(f - int(c.frame_idx)) <= 150]
                    for f in ([int(c.frame_idx)] + gan)[:3]:
                        for x in cv.get(str(f)) or []:
                            t = x[1] if isinstance(x, (list, tuple)) and len(x) > 1 else str(x)
                            if t and t not in mau:
                                mau.append(str(t))
                return " | ".join(mau)[:500]

            print(f"\n=== A/B R3: chèn OCR vào prompt, {len(sach)} mục ===", flush=True)
            net_len, net_xuong = 0, 0
            theo_vd = {True: [0, 0], False: [0, 0]}  # video_dung -> [len, xuong]
            for i, g in enumerate(sach):
                cau = f"Bối cảnh: {g['vqa_context']}\nCâu hỏi: {g['vqa_question']}"
                ocr_t = chu_ocr(cl[i])
                cau2 = cau + ("\nChữ đọc được trên các khung hình (OCR tự động, "
                              "có thể sai chính tả, chỉ dùng làm gợi ý): "
                              + ocr_t if ocr_t else "")
                khung_key = "|".join(f"{c.video_id}:{int(c.frame_idx)}"
                                     for c in cl[i][:24])
                h2 = hashlib.sha1(f"{args.model}|{cau2}|{khung_key}"
                                  .encode("utf-8")).hexdigest()[:24]
                f2 = cache / f"{h2}.json"
                if f2.is_file():
                    rec2 = json.loads(f2.read_text(encoding="utf-8"))
                else:
                    da2, ghi2 = tra_loi_tu_ung_vien(judge, args.model, cl[i],
                                                    meta_key, by_n, caps, cau2)
                    rec2 = {"dap_an": da2, "ghi": ghi2, "dong1": cl[i][0].video_id}
                    f2.write_text(json.dumps(rec2, ensure_ascii=False),
                                  encoding="utf-8")
                chuan = g.get("vqa_answer") or ""
                ok2 = bool(_default_answer_match(rec2["dap_an"], chuan)
                           or khop_rong(rec2["dap_an"], chuan))
                ok1 = dung[i]["dung"]
                vd = dung[i]["video_dung"]
                if ok2 and not ok1:
                    net_len += 1
                    theo_vd[vd][0] += 1
                if ok1 and not ok2:
                    net_xuong += 1
                    theo_vd[vd][1] += 1
                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/{len(sach)} | lên {net_len} xuống {net_xuong}",
                          flush=True)
            print(f"\nA/B R3: sai→đúng {net_len} | đúng→sai {net_xuong} | "
                  f"NET {net_len - net_xuong:+d}")
            print(f"  nhóm video-ĐÚNG: lên {theo_vd[True][0]} xuống {theo_vd[True][1]}"
                  f" | video-SAI: lên {theo_vd[False][0]} xuống {theo_vd[False][1]}")
            print("NGƯỠNG: net ≥ +1 VÀ đúng→sai = 0 -> "
                  + ("QUA" if (net_len - net_xuong >= 1 and net_xuong == 0)
                     else "KHÔNG QUA (đọc kỹ hai chiều trước khi kết luận)"))
            return 0
        if ten == "NEN" and args.xuat_sai:
            Path(args.xuat_sai).write_text(json.dumps(
                [{"chi_so_sach": i, "dung": r["dung"], "video_dung": r["video_dung"],
                  "dap_an_may": r["dap_an"]} for i, r in enumerate(dung)],
                ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  -> da xuat trang thai dung/sai NEN: {args.xuat_sai}")
        n_ok = sum(1 for r in dung if r["dung"])
        print(f"{ten}: {n_ok}/{len(dung)} dap an dung", flush=True)

    # bất biến: nhóm MỘT cảnh không được đụng tới ở cả ba tập
    mot = [i for i in range(len(sach)) if i not in bat]
    for ten in ("CANH_B", "HOAN_VI"):
        khac = [i for i in mot if ket[ten][i]["dap_an"] != ket["NEN"][i]["dap_an"]]
        print(f"bat bien nhom MOT canh ({ten} vs NEN): {len(mot)-len(khac)}/{len(mot)} giong"
              + (f"  !! {len(khac)} khac — dieu tra" if khac else "  -> DAT"))

    i_tune = sorted(bat[0::2] + mot[0::2])
    i_test = sorted(bat[1::2] + mot[1::2])

    def ty_le(ten, idx, khoa="dung"):
        return sum(1 for i in idx if ket[ten][i][khoa]) / max(1, len(idx))

    print(f"\n{'tap':<10}{'TUNE':>9}{'TEST':>9}{'ca bo':>9} | "
          f"{'HAI canh':>10}{'MOT canh':>10}{'video dung':>12}")
    print("-" * 72)
    b_test = [i for i in i_test if i in bat]
    m_test = [i for i in i_test if i in mot]
    for ten in TAP:
        print(f"{ten:<10}{ty_le(ten,i_tune):>9.1%}{ty_le(ten,i_test):>9.1%}"
              f"{ty_le(ten,range(len(sach))):>9.1%} | "
              f"{ty_le(ten,b_test):>10.1%}{ty_le(ten,m_test):>10.1%}"
              f"{ty_le(ten,b_test,'video_dung'):>12.1%}")

    print("\n=== bootstrap theo CAU, rieng nhom HAI canh cua TEST ===")
    rng = np.random.default_rng(4242)
    a = np.array([1.0 if ket["NEN"][i]["dung"] else 0.0 for i in b_test])
    for ten in ("CANH_B", "HOAN_VI"):
        b = np.array([1.0 if ket[ten][i]["dung"] else 0.0 for i in b_test])
        lay = rng.integers(0, len(b_test), size=(4000, len(b_test)))
        d = b[lay].mean(axis=1) - a[lay].mean(axis=1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {ten:8s} vs NEN: {a.mean():.1%} -> {b.mean():.1%} "
              f"= {b.mean()-a.mean():+.1%} | KTC [{lo:+.3f}, {hi:+.3f}] | P(<=0) = {(d<=0).mean():.1%}")
    print(f"\n{judge.cost_note()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
