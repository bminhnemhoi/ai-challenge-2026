"""doc2query cho video: sinh câu hỏi KIS tổng hợp từ transcript, đo xem hồ sơ
văn bản của video có giúp xếp VIDEO ĐÚNG lên cao hơn điểm SigLIP trực tiếp không.

Ý tưởng (lane "sinh câu hỏi tương tự + hồ sơ đặc trưng"): với mỗi video, nhờ
Gemini đọc transcript + OCR rồi viết 5 câu truy vấn kiểu đề thi KIS; nhúng các
câu đó bằng CHÍNH text encoder SigLIP-2 đang dùng; hồ sơ video = trung bình các
vector. Khi có câu hỏi thật, cosine(câu hỏi, hồ sơ video) là một kênh điểm cấp
VIDEO nằm ngoài ảnh — nếu nó tách được video đúng khỏi video sai thì trộn với
trọng số nhỏ.

LƯU Ý TRUNG THỰC ghi ngay từ đầu: recall video đã 60/60 trong top-400
(docs/HUONG_DI_TIEP.md), nên kỳ vọng hợp lý của lane này là ~0. Phép đo ở đây
để ĐÓNG câu hỏi bằng số liệu. Ngân sách Gemini: 20 gói (1 gói/video, có cache).

Các pha (cache giữa các pha, chạy lại pha nào cũng rẻ trừ retrieve):

    python scripts/experiment_doc2query.py retrieve   # ranked_hits + qvec cho 60 câu GT (chậm, 1 lần)
    python scripts/experiment_doc2query.py select     # chọn 20 video: 10 nhóm thất bại + 10 nhóm đang đúng
    python scripts/experiment_doc2query.py generate   # 20 gói Gemini, cache data/doc2query/<video>.json
    python scripts/experiment_doc2query.py measure    # nhúng hồ sơ + toàn bộ phép đo

Phép đo gồm ba tầng, tách bạch phần rò rỉ:

  1. AUC/tách biệt: cosine(câu thật, hồ sơ video ĐÚNG) so với hồ sơ 19 video sai
     — kênh hồ sơ có mang tín hiệu không, chưa nói gì tới trộn.
  2. Nhận diện 20-chọn-1 (không rò rỉ): trong đúng 20 video có hồ sơ, xếp hạng
     bằng (a) SigLIP trực tiếp, (b) hồ sơ, (c) trộn — R@1/R@5 cấp video.
  3. Điểm cuối bằng harness chuẩn (ranked_hits + allocate_hybrid_rows +
     final_score, khoảnh khắc thật KHÔNG bám keyframe, nhiều họ hạt giống).
     Chỉ 20/873 video có hồ sơ nên cộng thưởng cho riêng chúng là RÒ RỈ về phía
     video GT — số này chỉ được đọc như CẬN TRÊN; bản "đã canh giữa" (trừ trung
     bình cosine trên 20 hồ sơ) bớt rò rỉ hơn nhưng vẫn là cận trên.
"""

from __future__ import annotations

import argparse
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

from src.core.submission import (  # noqa: E402
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    allocate_hybrid_rows,
    final_score,
    r_score_kis,
)

OUT_DIR = ROOT / "data" / "doc2query"
HITS_CACHE = OUT_DIR / "_hits_cache.json"
QVECS_CACHE = OUT_DIR / "_qvecs.npy"
SELECTION = OUT_DIR / "_selection.json"
PROFILES = OUT_DIR / "_profiles.npz"
TRANSCRIPTS = ROOT.parent / "transcripts_full"

#: mỗi video đúng 1 gói Gemini; đây là trần cứng của pha generate
GEMINI_BUDGET = 20


# ------------------------------------------------------------------ tiện ích


def load_gt(data_dir: Path):
    gt = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))
    return gt


# ------------------------------------------------------------------ retrieve


def cmd_retrieve(args) -> int:
    """Chạy ĐƯỜNG SẢN XUẤT (ranked_hits) cho 60 câu GT, cache hit + vector câu."""
    from scripts.make_submission import ranked_hits
    from src.core.kis_engine import KISEngine

    print("nạp chỉ mục SigLIP-2 (chậm, một lần) ...", flush=True)
    eng = KISEngine(args.data).load()
    gt = [g for g in load_gt(Path(args.data)) if g["video_id"] in eng.last_frame]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_hits, qvecs = [], []
    for i, g in enumerate(gt):
        t0 = time.time()
        hits = ranked_hits(eng, g["kis_query_vi"], g.get("kis_query_en"))
        qvecs.append(eng.query_vector(g["kis_query_vi"], g.get("kis_query_en")))
        all_hits.append(
            [[h.video_id, int(h.frame_idx), float(h.score), int(h.video_last_frame)] for h in hits]
        )
        print(f"  [{i+1:2d}/{len(gt)}] {g['video_id']}  {len(hits)} hit  {time.time()-t0:.1f}s", flush=True)

    HITS_CACHE.write_text(
        json.dumps({"queries": [g["video_id"] for g in gt], "hits": all_hits}), encoding="utf-8"
    )
    np.save(QVECS_CACHE, np.stack(qvecs))
    print(f"đã cache {len(gt)} câu -> {HITS_CACHE.name}, {QVECS_CACHE.name}")
    return 0


# -------------------------------------------------------------------- select


def video_rank(hits, gt_video: str):
    """Hạng của video đúng trong thứ tự video xuất hiện lần đầu (1-based)."""
    seen = []
    for v, _f, _s, _l in hits:
        if v not in seen:
            seen.append(v)
        if v == gt_video:
            return len(seen)
    return None


def cmd_select(args) -> int:
    cache = json.loads(HITS_CACHE.read_text(encoding="utf-8"))
    gt = load_gt(Path(args.data))
    assert [g["video_id"] for g in gt] == cache["queries"], "cache lệch ground truth"

    scored = []
    for qi, g in enumerate(gt):
        r = video_rank(cache["hits"][qi], g["video_id"])
        scored.append((qi, g["video_id"], r if r is not None else 10**9))

    # nhóm thất bại: hạng video tệ nhất; nhóm đang đúng: hạng 1
    fail = sorted(scored, key=lambda t: (-t[2], t[1]))[: args.n_fail]
    good_pool = [t for t in scored if t[2] == 1]
    good = sorted(good_pool, key=lambda t: t[1])[: args.n_good]
    chosen = {t[1]: ("fail" if t in fail else "good") for t in fail + good}

    sel = {
        "videos": [
            {"video_id": v, "group": grp, "video_rank": next(t[2] for t in scored if t[1] == v)}
            for v, grp in chosen.items()
        ]
    }
    SELECTION.write_text(json.dumps(sel, indent=1), encoding="utf-8")
    print(f"đã chọn {len(sel['videos'])} video -> {SELECTION.name}")
    for r in sel["videos"]:
        print(f"  {r['video_id']}  {r['group']:4s}  hạng video nền = {r['video_rank']}")
    return 0


# ------------------------------------------------------------------ generate

_PROMPT = (
    "Bạn là người ra đề cho cuộc thi truy xuất khoảnh khắc video (Textual KIS).\n"
    "Dưới đây là TIÊU ĐỀ, LỜI THOẠI (ASR) và CHỮ TRÊN MÀN HÌNH (OCR, có thể trống) "
    "của MỘT video bản tin tiếng Việt.\n\n"
    "TIÊU ĐỀ: {title}\n\nLỜI THOẠI (cắt {n_words} từ đầu):\n{transcript}\n\n"
    "OCR:\n{ocr}\n\n"
    "Hãy viết đúng 5 câu truy vấn kiểu đề thi KIS mà đáp án nằm trong video này. "
    "Yêu cầu với TỪNG câu:\n"
    "- mô tả MỘT cảnh cụ thể có thể nhìn thấy (người/vật/hành động/bối cảnh) "
    "suy ra hợp lý từ nội dung, ưu tiên chi tiết thị giác: màu sắc, số lượng, hành động;\n"
    "- 1-2 câu, giọng văn như đề thi; KHÔNG nhắc 'video này', KHÔNG nêu tiêu đề;\n"
    "- 5 câu phủ 5 đoạn/chủ đề KHÁC NHAU của video.\n\n"
    'Trả về DUY NHẤT một mảng JSON: [{{"vi": "câu tiếng Việt", "en": "natural English rendering"}}, ...]'
)


def _ocr_lines(video_id: str, data_dir: Path, k: int = 10):
    p = data_dir / "ocr" / f"{video_id}.json"
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    texts = []
    for _frame, items in raw.items():
        for t, conf in items:
            t = str(t).strip()
            if float(conf) >= 0.75 and len(t) >= 5 and not t.isdigit():
                texts.append(t)
    # ưu tiên chuỗi lặp lại nhiều (banner tin), rồi chuỗi dài
    from collections import Counter

    cnt = Counter(texts)
    uniq = sorted(cnt, key=lambda t: (-cnt[t], -len(t)))
    return uniq[:k]


def _gen_one(client, types_mod, models, video_id: str, data_dir: Path):
    tpath = TRANSCRIPTS / f"{video_id}.json"
    tr = json.loads(tpath.read_text(encoding="utf-8"))
    words = (tr.get("full_text") or "").split()[:3000]
    ocr = _ocr_lines(video_id, data_dir)
    prompt = _PROMPT.format(
        title=tr.get("title") or "(không rõ)",
        n_words=len(words),
        transcript=" ".join(words),
        ocr="\n".join(f"- {t}" for t in ocr) if ocr else "(không có)",
    )

    from src.core.vlm import _is_daily_quota

    last_err = None
    for model in models:
        for attempt in range(3):
            try:
                r = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types_mod.GenerateContentConfig(
                        temperature=0.4, max_output_tokens=3000,
                        response_mime_type="application/json",
                    ),
                )
                m = re.search(r"\[.*\]", r.text or "", re.S)
                qs = json.loads(m.group(0)) if m else []
                qs = [q for q in qs if isinstance(q, dict) and q.get("vi")]
                if len(qs) >= 3:
                    return {"video_id": video_id, "model": model, "n_ocr": len(ocr),
                            "questions": qs[:5]}
                last_err = f"{model}: chỉ parse được {len(qs)} câu"
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                last_err = f"{model}: {type(exc).__name__}: {msg[:90]}"
                if _is_daily_quota(msg):
                    break
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep((8, 22, 40)[min(attempt, 2)])
                    continue
                if "503" in msg or "UNAVAILABLE" in msg:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                break
    raise RuntimeError(last_err or "không rõ")


def cmd_generate(args) -> int:
    from src.core.vlm import DEFAULT_MODEL, FALLBACK_MODELS, load_env

    load_env(ROOT / ".env")
    import os

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("thiếu GEMINI_API_KEY"); return 2
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    models = [DEFAULT_MODEL] + [m for m in FALLBACK_MODELS if m != DEFAULT_MODEL]

    sel = json.loads(SELECTION.read_text(encoding="utf-8"))["videos"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    calls = 0
    for row in sel:
        vid = row["video_id"]
        out = OUT_DIR / f"{vid}.json"
        if out.exists():
            print(f"  {vid}: đã có cache, bỏ qua"); continue
        if calls >= GEMINI_BUDGET:
            print(f"  {vid}: DỪNG — chạm trần {GEMINI_BUDGET} gói"); break
        t0 = time.time()
        try:
            rec = _gen_one(client, types, models, vid, Path(args.data))
        except Exception as exc:  # noqa: BLE001
            print(f"  {vid}: LỖI {exc}"); calls += 1; continue
        calls += 1
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {vid}: {len(rec['questions'])} câu ({rec['model']}, {rec['n_ocr']} dòng OCR, "
              f"{time.time()-t0:.1f}s)", flush=True)
    print(f"tổng gói Gemini đã dùng lần chạy này: {calls}")
    return 0


# ------------------------------------------------------------------- measure


def _draw_moments(gt, kf, seed):
    """Khoảnh khắc thật KHÔNG bám keyframe: bốc ngẫu nhiên trong khe giữa hai keyframe."""
    rng = np.random.default_rng(seed)
    out = []
    for g in gt:
        a = kf[g["video_id"]]
        i = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
        lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
        hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
        out.append(int(rng.integers(lo, max(lo + 1, hi))))
    return out


def cmd_measure(args) -> int:
    from src.core.kis_engine import KISEngine

    data_dir = Path(args.data)
    gt = load_gt(data_dir)
    cache = json.loads(HITS_CACHE.read_text(encoding="utf-8"))
    assert [g["video_id"] for g in gt] == cache["queries"]
    qvecs = np.load(QVECS_CACHE)
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))["videos"]
    prof_vids = [r["video_id"] for r in sel if (OUT_DIR / f"{r['video_id']}.json").exists()]
    print(f"{len(prof_vids)}/{len(sel)} video có câu hỏi tổng hợp trong cache")

    print("nạp chỉ mục SigLIP-2 ...", flush=True)
    eng = KISEngine(args.data).load()

    # ---- hồ sơ video: trung bình vector 4-prompt của từng câu hỏi tổng hợp ----
    if PROFILES.exists() and not args.reembed:
        z = np.load(PROFILES, allow_pickle=True)
        pv, pmat = list(z["videos"]), z["profiles"]
        if pv != prof_vids:
            pv = None
    else:
        pv = None
    if pv is None:
        rows = []
        for vid in prof_vids:
            qs = json.loads((OUT_DIR / f"{vid}.json").read_text(encoding="utf-8"))["questions"]
            vecs = [eng.query_vector(q["vi"], q.get("en") or None) for q in qs]
            m = np.mean(vecs, axis=0)
            rows.append(m / max(float(np.linalg.norm(m)), 1e-6))
            print(f"  nhúng {vid}: {len(qs)} câu", flush=True)
        pv, pmat = prof_vids, np.stack(rows)
        np.savez(PROFILES, videos=np.array(pv, dtype=object), profiles=pmat)

    prof_of = {v: pmat[i] for i, v in enumerate(pv)}
    cos = qvecs @ pmat.T  # (60 câu, n hồ sơ)

    # ---- 1) tách biệt: hồ sơ video đúng so với hồ sơ video sai ---------------
    gt_idx = {g["video_id"]: qi for qi, g in enumerate(gt)}
    pos, neg = [], []
    for qi, g in enumerate(gt):
        for pj, v in enumerate(pv):
            (pos if v == g["video_id"] else neg).append(cos[qi, pj])
    pos, neg = np.array(pos), np.array(neg)
    auc = float(np.mean([np.mean(neg < p) + 0.5 * np.mean(neg == p) for p in pos]))
    print("\n== 1) Kênh hồ sơ có mang tín hiệu không ==")
    print(f"  cosine(câu thật, hồ sơ ĐÚNG) : {pos.mean():.4f} ±{pos.std():.4f}  (n={len(pos)})")
    print(f"  cosine(câu thật, hồ sơ SAI)  : {neg.mean():.4f} ±{neg.std():.4f}  (n={len(neg)})")
    print(f"  AUC tách đúng/sai            : {auc:.4f}   (0.5 = không có tín hiệu)")

    # ---- 2) nhận diện 20-chọn-1, không rò rỉ ---------------------------------
    # điểm SigLIP trực tiếp: max cosine(qvec, frame) trên TOÀN BỘ frame của video
    rows_of = {}
    vid_arr = eng.video_id
    for v in pv:
        rows_of[v] = np.where(vid_arr == v)[0]
    direct = np.zeros((len(gt), len(pv)), dtype=np.float32)
    for qi in range(len(gt)):
        for pj, v in enumerate(pv):
            direct[qi, pj] = float(
                np.max(np.asarray(eng.embeddings[rows_of[v]], dtype=np.float32) @ qvecs[qi])
            )

    sub = [qi for qi, g in enumerate(gt) if g["video_id"] in prof_of]
    print(f"\n== 2) Nhận diện video đúng trong {len(pv)} video có hồ sơ ({len(sub)} câu) ==")
    print(f"{'kênh':<28}{'R@1':>6}{'R@5':>6}")

    def rk(score_mat, name):
        r1 = r5 = 0
        for qi in sub:
            order = np.argsort(-score_mat[qi])
            truth = pv.index(gt[qi]["video_id"])
            rank = int(np.where(order == truth)[0][0]) + 1
            r1 += rank == 1
            r5 += rank <= 5
        print(f"{name:<28}{r1:>4d}/{len(sub)}{r5:>4d}/{len(sub)}")
        return r1, r5

    rk(direct, "SigLIP trực tiếp")
    rk(cos, "hồ sơ doc2query")
    for w in (0.05, 0.1, 0.2):
        rk(direct + w * cos, f"trộn w={w}")

    # ---- 3) điểm cuối bằng harness chuẩn -------------------------------------
    print("\n== 3) Điểm cuối harness (khoảnh khắc KHÔNG bám keyframe) ==")
    print("   LƯU Ý: chỉ 20/873 video có hồ sơ -> thưởng chỉ rơi vào chúng = RÒ RỈ")
    print("   về phía video GT; đọc các số dưới đây là CẬN TRÊN của lane.\n")
    kf = {}
    for m in eng.metadata:
        kf.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    for v in kf:
        kf[v] = np.array(sorted(kf[v]))

    plan = AllocationPlan(breadth_cost=1.0, depth_cost=0.5, step=10)
    mean_cos = cos.mean(axis=1)  # để canh giữa thưởng theo từng câu

    def rows_with_bonus(w: float, centered: bool):
        out = []
        for qi in range(len(gt)):
            hs = cache["hits"][qi]
            def bonus(v):
                if v not in prof_of:
                    return 0.0
                c = cos[qi, pv.index(v)]
                return w * float(c - mean_cos[qi] if centered else c)
            order = sorted(range(len(hs)), key=lambda i: (-(hs[i][2] + bonus(hs[i][0])), i))
            cands = [Candidate(hs[i][0], hs[i][1], hs[i][2], hs[i][3]) for i in order]
            out.append(allocate_hybrid_rows(cands, n_flat=30, plan=plan)[:MAX_ROWS])
        return out

    arms = {"nền sản xuất (thứ tự cache)": [
        allocate_hybrid_rows([Candidate(*h) for h in cache["hits"][qi]], n_flat=30, plan=plan)[:MAX_ROWS]
        for qi in range(len(gt))
    ]}
    arms["sắp lại theo điểm (w=0)"] = rows_with_bonus(0.0, False)
    for w in (0.05, 0.1, 0.2):
        arms[f"trộn thô w={w}"] = rows_with_bonus(w, False)
        arms[f"trộn canh giữa w={w}"] = rows_with_bonus(w, True)

    windows = [int(x) for x in args.windows.split(",")]

    def cham(rows_of, draws, qis):
        per_w = []
        for half in windows:
            tot = 0.0
            for qi in qis:
                g = gt[qi]
                for truth in draws:
                    span = (truth[qi] - half, truth[qi] + half)
                    tot += final_score(
                        [r_score_kis(v, f, g["video_id"], span) for v, f in rows_of[qi]]
                    )
            per_w.append(tot / (len(qis) * len(draws)))
        return sum(per_w) / len(per_w)

    tapcau = {"60 câu": list(range(len(gt))), f"{len(sub)} câu có hồ sơ": sub}
    fam = {tc: {name: [] for name in arms} for tc in tapcau}
    for s in range(args.seeds):
        draws = [_draw_moments(gt, kf, 41000 + s * 1000 + t) for t in range(args.draws)]
        for name, rows in arms.items():
            for tc, qis in tapcau.items():
                fam[tc][name].append(cham(rows, draws, qis))
        print(f"  họ hạt giống {s+1}/{args.seeds} xong", flush=True)

    for tc in tapcau:
        base = float(np.mean(fam[tc]["nền sản xuất (thứ tự cache)"]))
        sd0 = float(np.std(fam[tc]["nền sản xuất (thứ tự cache)"]))
        print(f"\n--- {tc} ---")
        print(f"{'nhánh':<30}{'điểm':>9}{'±':>8}{'so nền':>9}")
        print("-" * 58)
        for name, vals in fam[tc].items():
            m, sd = float(np.mean(vals)), float(np.std(vals))
            print(f"{name:<30}{m:>9.4f}{sd:>8.4f}{100*(m/base-1):>+8.1f}%")
        bien = max(sd0, 0.0005)
        print(f"quy tắc hoà: chênh < 2×{bien:.4f} so với nền = HOÀ, không kết luận.")
    return 0


# ---------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["retrieve", "select", "generate", "measure"])
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--n-fail", type=int, default=10)
    ap.add_argument("--n-good", type=int, default=10)
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--draws", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reembed", action="store_true", help="nhúng lại hồ sơ dù đã có cache")
    args = ap.parse_args()
    return {"retrieve": cmd_retrieve, "select": cmd_select,
            "generate": cmd_generate, "measure": cmd_measure}[args.phase](args)


if __name__ == "__main__":
    raise SystemExit(main())
