"""Đo lại "VLM xếp lại frame trong video" — lần này hỏi bằng CÂU HỎI PHÂN BIỆT.

Bảng tín hiệu trong `docs/KIEN_TRUC_VA_HUONG_CAI_THIEN.md` ghi ý tưởng này là
**−4,6%**, và vì thế nó bị gạch. Nhưng phép đo đó (`experiment_frame_promote.py:192`)
đưa cho VLM **nguyên văn đề bài**:

    sc = judge.score(g["kis_query_vi"], cands)

Vòng sơ tuyển 1 cho thấy đó chính là cách hỏi sai. Đề bài mô tả *cả bối cảnh lẫn
khoảnh khắc*, nên khi chấm bằng nguyên văn đề, VLM chấm theo bối cảnh và cho cả
video điểm cao. Đo trên `L24_V035`, cùng một video, cùng một model:

    hỏi bằng nguyên văn đề      72/193 khung ≥ 0,60   (37%)
    hỏi bằng câu hỏi phân biệt  24/193 khung ≥ 0,60   (12%)

Một cao nguyên 37% thì xếp hạng bằng nó là vô nghĩa. Đó có thể mới là thứ đo ra
−4,6%, chứ không phải bản thân ý tưởng.

Điều này đáng đo lại vì nó nhắm đúng nhóm thất bại lớn nhất: **14 trong 22 câu
trượt có keyframe đúng nằm sẵn trong danh sách ứng viên, chỉ là xếp hạng quá thấp
bên trong video của nó**. Trần lý thuyết của việc sửa đúng chỗ này là 0,345 → 0,740.

Khác biệt duy nhất so với thí nghiệm cũ: một lệnh gọi LLM cho mỗi câu, biến đề bài
thành một câu hỏi có/không nhắm vào chi tiết **thoáng qua**. Phần còn lại giữ nguyên
để so sánh cho công bằng.

    python scripts/experiment_sharp_rerank.py --limit 20
    python scripts/experiment_sharp_rerank.py            # cả 60 câu ground truth
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.make_submission import DEFAULT_DEPTH_COST, DEFAULT_N_FLAT, ranked_hits  # noqa: E402
from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    allocate_hybrid_rows,
    final_score,
    r_score_kis,
)

SINH_CAU_HOI = """Đây là một câu hỏi tìm khoảnh khắc trong video:

{query}

Hãy viết MỘT câu hỏi để chấm từng khung hình, giúp phân biệt đúng khoảnh khắc đó
với những khung hình khác **trong cùng video**.

Quy tắc:
- Chỉ hỏi về chi tiết THOÁNG QUA — thứ chỉ đúng ở đúng giây đó (một bàn tay đang
  chạm vào vật, miệng đang ngậm, dùi đang gõ). ĐỪNG hỏi về bối cảnh, vì bối cảnh
  đúng ở cả video.
- Dạng CÓ/KHÔNG, ép mô hình phải quyết.
- Nói rõ khi nào chấm 100 và khi nào chấm 0. Câu "chỉ thấy bối cảnh mà không thấy
  <chi tiết> thì chấm 0" là bắt buộc.

Trả về DUY NHẤT câu hỏi, không giải thích."""


_CAU_HOI_CACHE = ROOT / "data" / "vlm" / "sharp_questions_gt.json"


def sinh_cau_hoi_phan_biet(client, model, query: str) -> str:
    """Đề bài -> câu hỏi phân biệt. Hỏng thì trả lại chính đề bài (tức là về hành vi cũ).

    Cache xuống đĩa theo nguyên văn đề. Không cache thì mỗi lần chạy lại tốn 60
    gọi LLM VÀ — tệ hơn — chỉ cần câu sinh ra lệch một dấu cách là toàn bộ điểm
    chấm khung hình đã cache (bucket theo sha1 của câu hỏi) thành mồ côi.
    """
    import json as _json

    try:
        cache = _json.loads(_CAU_HOI_CACHE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        cache = {}
    if query in cache:
        return cache[query]

    from google.genai import types

    try:
        r = client.models.generate_content(
            model=model,
            contents=[SINH_CAU_HOI.format(query=query.strip()[:1200])],
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=400),
        )
        out = " ".join((r.text or "").split())
        if len(out) > 40:
            cache[query] = out
            _CAU_HOI_CACHE.parent.mkdir(parents=True, exist_ok=True)
            _CAU_HOI_CACHE.write_text(
                _json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            return out
        return query
    except Exception:  # noqa: BLE001
        return query


def do_sac_net(scores: dict) -> float:
    """Tỷ lệ khung được chấm >= 0,60. Càng thấp càng phân biệt tốt.

    Đây là thứ phân biệt một phép chấm hữu ích với một cao nguyên vô dụng, và là
    lý do phép đo cũ thất bại — nên nó được in ra cùng điểm, không giấu đi.
    """
    if not scores:
        return float("nan")
    return sum(1 for s, _w in scores.values() if s >= 0.6) / len(scores)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--videos", type=int, default=3, help="số video dẫn đầu được chấm lại")
    ap.add_argument("--frames", type=int, default=10, help="số khung mỗi video")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--weights", default="0.02,0.05,0.20,1.00")
    args = ap.parse_args()

    from src.core.kis_engine import KISEngine
    from src.core.vlm import VLMJudge, load_env

    load_env(Path(args.data).parent / ".env")
    load_env(".env")

    judge = VLMJudge(args.data, model=args.model)
    if not judge.ready:
        print("Không có GEMINI_API_KEY — thí nghiệm này cần nó.")
        return 2
    client = judge._get_client()

    print("nạp chỉ mục ...", flush=True)
    eng = KISEngine(args.data).load()
    meta = {(m["video_id"], m["frame_idx"]): m for m in eng.metadata}
    gt = json.loads((Path(args.data) / "ground_truth.json").read_text(encoding="utf-8"))
    gt = [g for g in gt if g.get("video_id") in eng.last_frame]
    if args.limit:
        gt = gt[: args.limit]

    print(f"{len(gt)} câu, truy xuất ...", flush=True)
    hits_of = [ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en")) for g in gt]

    kf: dict = {}
    for m in eng.metadata:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    for v in kf:
        kf[v] = np.array(sorted(kf[v]))

    def draw(seed):
        # khoảnh khắc thật KHÔNG bám keyframe — 93% ground truth trùng keyframe,
        # đo thẳng trên đó sẽ ra kết luận ngược (xem experiment_allocation.py --snapped)
        rng = np.random.default_rng(seed)
        out = []
        for g in gt:
            a = kf[g["video_id"]]
            i = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
            lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
            hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
            out.append(int(rng.integers(lo, max(lo + 1, hi))))
        return out

    draws = [draw(21000 + s) for s in range(args.draws)]
    windows = [int(w) for w in args.windows.split(",")]
    plan = AllocationPlan(breadth_cost=1.0, depth_cost=DEFAULT_DEPTH_COST, step=10)

    def cham(order_of):
        per_w = []
        for half in windows:
            tot = 0.0
            for qi, g in enumerate(gt):
                cs = [Candidate(h.video_id, h.frame_idx, h.score, h.video_last_frame)
                      for h in order_of[qi]]
                rows = allocate_hybrid_rows(cs, n_flat=DEFAULT_N_FLAT, plan=plan)[:MAX_ROWS]
                for truth in draws:
                    span = (truth[qi] - half, truth[qi] + half)
                    tot += final_score([r_score_kis(v, f, g["video_id"], span) for v, f in rows])
            per_w.append(tot / (len(gt) * len(draws)))
        return per_w

    base_w = cham(hits_of)
    base = sum(base_w) / len(base_w)
    print(f"\nnền: {base:.3f}   " + "  ".join(f"W={w}:{v:.3f}" for w, v in zip(windows, base_w)))

    # ---- chấm lại các khung dẫn đầu, bằng HAI cách hỏi, trên CÙNG tập khung -----
    print("\nchấm khung ...", flush=True)
    sharp_of, raw_of, lead_of = [], [], []
    for qi, g in enumerate(gt):
        lead, seen_v = [], []
        for h in hits_of[qi]:
            if h.video_id not in seen_v:
                if len(seen_v) >= args.videos:
                    continue
                seen_v.append(h.video_id)
            if sum(1 for x in lead if x.video_id == h.video_id) < args.frames:
                lead.append(h)
        cands = [(h.video_id, h.frame_idx, meta[(h.video_id, h.frame_idx)]["frame_filename"])
                 for h in lead if (h.video_id, h.frame_idx) in meta]
        lead_of.append(lead)
        raw_of.append(judge.score(g["kis_query_vi"], cands))
        cauhoi = sinh_cau_hoi_phan_biet(client, args.model, g["kis_query_vi"])
        sharp_of.append(judge.score(cauhoi, cands))
        if (qi + 1) % 10 == 0:
            print(f"  {qi+1}/{len(gt)}  {judge.cost_note().splitlines()[0]}", flush=True)

    sac_raw = float(np.nanmean([do_sac_net(s) for s in raw_of]))
    sac_sharp = float(np.nanmean([do_sac_net(s) for s in sharp_of]))
    print(f"\nĐỘ SẮC (tỷ lệ khung ≥0,60 — càng thấp càng phân biệt tốt)")
    print(f"  hỏi bằng nguyên văn đề     : {sac_raw:5.1%}")
    print(f"  hỏi bằng câu hỏi phân biệt : {sac_sharp:5.1%}")

    def xep_lai_toan_cuc(scores_of, w):
        """Cộng điểm VLM vào rồi xếp lại CẢ danh sách — cách thí nghiệm cũ làm."""
        out = []
        for qi, hits in enumerate(hits_of):
            sc = scores_of[qi]
            scored = [(h.score + w * sc.get((h.video_id, h.frame_idx), (0.0, ""))[0], i, h)
                      for i, h in enumerate(hits)]
            scored.sort(key=lambda t: (-t[0], t[1]))
            out.append([t[2] for t in scored])
        return out

    def xep_lai_trong_video(scores_of, w):
        """Giữ NGUYÊN thứ tự video, chỉ đảo thứ tự keyframe bên trong mỗi video.

        Đây mới đúng là giả thuyết cần đo. Xếp lại toàn cục đồng thời phá thứ tự
        video — mà xếp hạng video là thứ đã đo được là đáng giá (+41% nếu hoàn hảo).
        Trộn hai tác động vào một phép đo thì không đọc được kết quả: điểm giảm
        cũng không biết vì đảo frame sai hay vì làm hỏng thứ tự video.
        """
        out = []
        for qi, hits in enumerate(hits_of):
            sc = scores_of[qi]
            thu_tu_video, nhom = [], defaultdict(list)
            for h in hits:
                if h.video_id not in nhom:
                    thu_tu_video.append(h.video_id)
                nhom[h.video_id].append(h)
            moi = []
            for vid in thu_tu_video:
                trong = nhom[vid]
                trong_sorted = sorted(
                    ((h.score + w * sc.get((h.video_id, h.frame_idx), (0.0, ""))[0], i, h)
                     for i, h in enumerate(trong)),
                    key=lambda t: (-t[0], t[1]),
                )
                moi.extend(t[2] for t in trong_sorted)
            out.append(moi)
        return out

    def xep_lai_giu_slot(scores_of, w):
        """Hoán vị GIỮ SLOT — bản sửa sau khi phản biện phát hiện artifact.

        Bản gom-khối (xep_lai_trong_video) kéo TOÀN BỘ frame của video 1 lên đầu
        danh sách, làm 30 dòng phẳng đầu (n_flat) sụp từ ~10 video xuống 1-2
        video — tự nó gây −38% dù không đổi thứ tự frame nào bên trong video.
        Con số −34,7% từng được đọc là "VLM phá điểm" nhiều khả năng đo nhầm
        chính cơ chế này.

        Ở đây cấu trúc slot giữ nguyên tuyệt đối: vị trí i vốn thuộc video v thì
        vẫn thuộc video v; chỉ có VIỆC FRAME NÀO của v ngồi ở slot nào là xếp
        lại theo tín hiệu mới. Độ rộng phủ video của bộ phân bổ không đổi một
        ly — đây mới là phép đo sạch của giả thuyết "tín hiệu X chọn frame
        trong video tốt hơn SigLIP".
        """
        out = []
        for qi, hits in enumerate(hits_of):
            sc = scores_of[qi]
            slots = defaultdict(list)
            for i, h in enumerate(hits):
                slots[h.video_id].append(i)
            moi = list(hits)
            for vid, pos in slots.items():
                mems = sorted(
                    (hits[i] for i in pos),
                    key=lambda h: -(h.score + w * sc.get((h.video_id, h.frame_idx), (0.0, ""))[0]),
                )
                for i, h in zip(pos, mems):
                    moi[i] = h
            out.append(moi)
        return out

    # ---- hạng nội-video của keyframe gần đáp án — độ đo KHÔNG dính bộ phân bổ --
    def hang_noi_video(scores_of, w):
        ranks = []
        for qi, g in enumerate(gt):
            gv = g["video_id"]
            arr = kf[gv]
            dung = int(arr[np.argmin(np.abs(arr - int(g["frame_idx"])))])
            trong = [h for h in hits_of[qi] if h.video_id == gv]
            if not any(h.frame_idx == dung for h in trong):
                continue
            thu = sorted(
                trong,
                key=lambda h: -(h.score + w * scores_of[qi].get((h.video_id, h.frame_idx), (0.0, ""))[0]),
            )
            ranks.append(1 + next(i for i, h in enumerate(thu) if h.frame_idx == dung))
        if not ranks:
            return float("nan"), 0.0
        return float(np.median(ranks)), sum(1 for r in ranks if r == 1) / len(ranks)

    md0, top0 = hang_noi_video([{} for _ in gt], 0.0)
    print(f"\nHẠNG NỘI-VIDEO của keyframe gần đáp án (trong các slot đã truy xuất):")
    print(f"  SigLIP thuần                     : trung vị {md0:.1f}, hạng-1 {top0:.0%}")
    for nhan_r, scores_r in (("nguyên văn đề", raw_of), ("câu hỏi phân biệt", sharp_of)):
        for w_r in (0.2, 1.0):
            md, top = hang_noi_video(scores_r, w_r)
            print(f"  + {nhan_r:18s} w={w_r:<4}: trung vị {md:.1f}, hạng-1 {top:.0%}")

    print(f"\n{'cách xếp lại':24s}{'cách hỏi':22s}{'w':>6}{'điểm':>9}{'so nền':>10}")
    print("-" * 71)
    ket = [(base, "nền (không chấm lại)", "", 0.0)]
    for kieu, fn in (
        ("giữ slot (sạch)", xep_lai_giu_slot),
        ("gom khối (artifact)", xep_lai_trong_video),
        ("toàn cục", xep_lai_toan_cuc),
    ):
        for w in [float(x) for x in args.weights.split(",")]:
            for nhan, scores_of in (("nguyên văn đề", raw_of), ("câu hỏi phân biệt", sharp_of)):
                m = sum(cham(fn(scores_of, w))) / len(windows)
                ket.append((m, nhan, kieu, w))
                print(f"{kieu:24s}{nhan:22s}{w:6.2f}{m:9.3f}{100*(m/base-1):+9.1f}%", flush=True)

    ket.sort(reverse=True)
    tot, nhan, kieu, w = ket[0]
    print(f"\nTốt nhất: {nhan} / {kieu or '—'} (w={w}) -> {tot:.3f}  ({100*(tot/base-1):+.1f}%)")
    if nhan == "nền (không chấm lại)":
        print("Kết luận: chấm lại bằng VLM vẫn không ăn, kể cả khi hỏi đúng cách"
              " và chỉ đảo bên trong video.")
    print(f"\n{judge.cost_note()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
