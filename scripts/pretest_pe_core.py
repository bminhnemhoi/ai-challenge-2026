"""Pre-test encoder thứ hai (PE-Core) — cổng GO/NO-GO trước khi đốt GPU Colab.

Giao thức và NGƯỠNG GO đã chốt TRƯỚC khi chạy trong docs/PRETEST_ENCODER.md §2–§3.
Tóm tắt:

* 40 mục của bộ SẠCH 132 (20 MỘT cảnh / 20 HAI cảnh), RandomState(92026).
* Phía SigLIP: đọc thẳng ``data/cache_tin_hieu_noi_video/sims_sach.npy`` —
  sims dựng bằng đúng ngữ nghĩa sản xuất (kỷ luật đo #6), KHÔNG encode lại.
* Phía PE-Core: encode toàn bộ keyframe hợp lệ của video ĐÚNG từng mục từ
  ``data/frames`` (512px — hạn chế đã ghi), preprocess mặc định của model
  (squash 336×336 — trùng triết lý squash 384 của SigLIP sản xuất).
* Hạng nội-video = 1 + #{khung có sim > sim(khung đáp án)}, cùng quy ước cho
  cả hai encoder, trên CÙNG tập khung (valid ∩ có file ảnh).
* GO nếu (trung vị hạng HAI cảnh ≤ 3) HOẶC (hạng-1 MỘT cảnh ≥ 55%) ở cấu hình
  văn bản tốt hơn trong hai cấu hình đã khai báo trước (A ensemble-vi /
  B ensemble-en).

Chạy:
    python -u scripts/pretest_pe_core.py --giai-doan chuan-bi   # 0 model, in mẫu + nền SigLIP
    python -u scripts/pretest_pe_core.py --giai-doan encode     # nặng CPU, resumable theo video
    python -u scripts/pretest_pe_core.py --giai-doan cham       # text + hạng + bootstrap + GO/NO-GO
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

DATA = ROOT / "data"
CACHE = DATA / "cache_pretest_pe"
HF_CACHE = str(DATA / "hf_cache_pe_core")

SEED = 92026
N_MOI_NHOM = 20
BOOT = 4000
MODEL_L = "hf-hub:timm/PE-Core-L-14-336"
MODEL_B = "hf-hub:timm/PE-Core-B-16-224"

#: y hệ số sản xuất (src/core/kis_engine.py PROMPT_WEIGHTS)
W_A = (0.45, 0.35, 0.10, 0.10)
#: sản xuất bỏ vi, chuẩn hoá lại: (0.45, 0.10, 0.10)/0.65
W_B = (0.45 / 0.65, 0.10 / 0.65, 0.10 / 0.65)

#: 5 video đầu (theo thứ tự mẫu ổn định) encode thêm bản center-crop
N_VIDEO_CROP = 5


# ---------------------------------------------------------------------------
# Mẫu + metadata
# ---------------------------------------------------------------------------


def nap_mau():
    """Bộ sạch 132 + chỉ số 40 mục mẫu (ổn định theo seed, không phụ thuộc máy)."""
    gt = json.loads((DATA / "ground_truth_moi.json").read_text(encoding="utf-8"))
    sach = [g for g in gt if not g.get("lan_truc")]
    assert len(sach) == 132, f"bo sach {len(sach)} != 132"
    mot = [i for i, g in enumerate(sach) if not g["co_2_canh"]]
    hai = [i for i, g in enumerate(sach) if g["co_2_canh"]]
    assert len(mot) == 66 and len(hai) == 66
    rng = np.random.RandomState(SEED)
    chon_mot = sorted(rng.choice(mot, N_MOI_NHOM, replace=False).tolist())
    chon_hai = sorted(rng.choice(hai, N_MOI_NHOM, replace=False).tolist())
    return sach, chon_mot, chon_hai


def nap_meta():
    meta = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    valid = np.load(DATA / "cache_tin_hieu_noi_video" / "valid.npy")
    assert len(meta) == len(valid)
    rows_of = defaultdict(list)
    for r, m in enumerate(meta):
        rows_of[m["video_id"]].append(r)
    return meta, valid, rows_of


def khung_hop_le(video_id, meta, valid, rows_of, can_anh: bool):
    """Các hàng metadata dùng để xếp hạng trong video: valid ∩ (có file ảnh)."""
    ra, thieu = [], []
    for r in rows_of[video_id]:
        if not valid[r]:
            continue
        if can_anh:
            f = DATA / "frames" / video_id / meta[r]["frame_filename"]
            if not f.exists():
                thieu.append(r)
                continue
        ra.append(r)
    return ra, thieu


def hang_cua(sims_row, rows, row_dap_an):
    """1 + #{khung có sim > sim(đáp án)} — cùng quy ước cho cả hai encoder."""
    s_ans = float(sims_row[row_dap_an])
    return 1 + int(sum(1 for r in rows if float(sims_row[r]) > s_ans))


# ---------------------------------------------------------------------------
# Giai đoạn 0: nền SigLIP (0 model)
# ---------------------------------------------------------------------------


def bang_nhom(ranks):
    r = np.array(ranks, dtype=float)
    return {
        "n": len(r),
        "trung_vi": float(np.median(r)),
        "hang1": float((r == 1).mean()),
        "top5": float((r <= 5).mean()),
    }


def cmd_chuan_bi(args):
    sach, chon_mot, chon_hai = nap_mau()
    meta, valid, rows_of = nap_meta()
    sims = np.load(DATA / "cache_tin_hieu_noi_video" / "sims_sach.npy", mmap_mode="r")
    assert sims.shape[0] == 132

    # --- kiểm nền: toàn bộ 132 mục, mọi khung valid (không cần file ảnh) ---
    full = {"mot": [], "hai": []}
    for i, g in enumerate(sach):
        rows, _ = khung_hop_le(g["video_id"], meta, valid, rows_of, can_anh=False)
        row_ans = tim_hang_dap_an(g, meta, rows_of)
        if row_ans not in rows:
            rows = rows + [row_ans]
        h = hang_cua(sims[i], rows, row_ans)
        full["hai" if g["co_2_canh"] else "mot"].append(h)
    print("KIEM NEN SigLIP (132 muc, moi khung valid) — doi chieu voi so da cong bo:")
    for k, ten in (("mot", "MOT canh"), ("hai", "HAI canh")):
        b = bang_nhom(full[k])
        print(f"  {ten}: n={b['n']}  trung vi {b['trung_vi']:.1f}  "
              f"hang-1 {b['hang1']:.1%}  top-5 {b['top5']:.1%}")
    print("  (cong bo: MOT trung vi 2,0 / hang-1 ~43%; HAI trung vi 6,0 / hang-1 11%)")

    # --- mẫu 40 mục: đếm khối lượng encode ---
    tong_khung = 0
    for i in chon_mot + chon_hai:
        g = sach[i]
        rows, thieu = khung_hop_le(g["video_id"], meta, valid, rows_of, can_anh=True)
        tong_khung += len(rows)
        if thieu:
            print(f"  ! {g['video_id']}: thieu {len(thieu)} file anh")
    print(f"\nMAU: {len(chon_mot)} MOT + {len(chon_hai)} HAI, "
          f"tong {tong_khung} khung can encode (~{tong_khung/0.9/60:.0f} phut @0,9 anh/s)")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "mau.json").write_text(json.dumps({
        "seed": SEED, "chon_mot": chon_mot, "chon_hai": chon_hai,
        "video": sorted({sach[i]["video_id"] for i in chon_mot + chon_hai}),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"da ghi {CACHE / 'mau.json'}")
    return 0


def tim_hang_dap_an(g, meta, rows_of):
    for r in rows_of[g["video_id"]]:
        if meta[r]["n"] == g["n"]:
            assert meta[r]["frame_idx"] == g["frame_idx"], \
                f"{g['video_id']} n={g['n']}: frame_idx lech"
            return r
    raise KeyError(f"khong tim thay khung dap an {g['video_id']} n={g['n']}")


# ---------------------------------------------------------------------------
# Giai đoạn 1: encode ảnh (nặng, resumable theo video)
# ---------------------------------------------------------------------------


FP16_CKPT = Path(HF_CACHE) / "pe_core_l_fp16.safetensors"


def _chuyen_fp16():
    """Đổi checkpoint fp32 (2,7GB) sang fp16 (1,35GB) — một lần, cho máy 16GB RAM.

    Windows: ``safe_open`` trên file 2,7GB tính TOÀN BỘ kích thước file vào
    commit charge (copy-on-write); cộng với model fp32 2,7GB đã dựng thì vượt
    commit còn trống (~5,8GB) → os error 1455 ổn định qua nhiều lần thử.
    Bản fp16 hạ đỉnh bộ nhớ lúc nạp từ ~5,5GB xuống ~4,1GB.

    Sai số fp16 trên TRỌNG SỐ (~5e-4 tương đối) — ghi nhận trong
    docs/PRETEST_ENCODER.md §5; áp dụng đồng đều cho cả tháp ảnh lẫn tháp chữ.
    """
    from safetensors.torch import load_file, save_file

    goc = sorted(Path(HF_CACHE).glob(
        "models--timm--PE-Core-L-14-336/snapshots/*/open_clip_model.safetensors"))
    assert goc, "khong thay checkpoint safetensors goc trong cache"
    print(f"chuyen fp16 (mot lan): {goc[-1].name} -> {FP16_CKPT.name}", flush=True)
    sd = load_file(str(goc[-1]))
    sd16 = {k: (v.half() if v.dtype.is_floating_point else v) for k, v in sd.items()}
    del sd
    tmp = FP16_CKPT.with_suffix(".tmp.safetensors")
    save_file(sd16, str(tmp))
    tmp.replace(FP16_CKPT)
    # bản .pt: torch.load stream từng tensor, không mmap cả file — cần cho máy 16GB
    import torch

    torch.save(sd16, str(FP16_CKPT.with_suffix(".pt")))
    print(f"da ghi {FP16_CKPT} (+.pt) ({FP16_CKPT.stat().st_size/1e9:.2f} GB)", flush=True)


def nap_model(ten):
    import open_clip
    import torch

    torch.set_num_threads(max(1, (torch.get_num_threads() * 3) // 2))
    if ten == MODEL_L and not FP16_CKPT.exists():
        # máy 16GB: nạp fp32 thẳng đã hỏng ổn định (os error 1455, có lần segfault)
        # — đổi checkpoint sang fp16 TRƯỚC, đừng thử lại đường fp32
        _chuyen_fp16()
        import gc

        gc.collect()
    if ten == MODEL_L:
        import open_clip.factory as _F
        from safetensors.torch import load_file as _lf

        _goc = _F.load_state_dict

        def _vá(path, device="cpu", weights_only=True):  # noqa: ANN001
            # giữ fp16 (1,35GB) — load_state_dict của Module tự cast vào param fp32.
            # Dùng bản .pt (torch.load stream từng tensor, KHÔNG mmap cả file):
            # mmap safetensors 1,34GB + model fp32 vẫn vượt commit → access violation.
            pt = FP16_CKPT.with_suffix(".pt")
            if pt.exists():
                return torch.load(str(pt), map_location="cpu", weights_only=True)
            return _lf(str(FP16_CKPT))

        _F.load_state_dict = _vá
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                ten, cache_dir=HF_CACHE)
        finally:
            _F.load_state_dict = _goc
        print("da nap PE-Core-L qua duong fp16 it bo nho", flush=True)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(ten, cache_dir=HF_CACHE)
    tokenizer = open_clip.get_tokenizer(ten, cache_dir=HF_CACHE)
    model.eval()
    return model, preprocess, tokenizer


def preprocess_crop():
    """Bản center-crop chuẩn CLIP để kiểm độ nhạy (giao thức §3.3)."""
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.CenterCrop(336),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def encode_video(video_id, rows, meta, model, prep, ra_file, batch=16):
    import torch
    from PIL import Image

    if ra_file.exists():
        E = np.load(ra_file)
        if E.shape[0] == len(rows):
            return E
        print(f"  ! cache {ra_file.name} lech ({E.shape[0]} vs {len(rows)}) — encode lai")
    vecs = []
    t0 = time.time()
    for a in range(0, len(rows), batch):
        ims = []
        for r in rows[a:a + batch]:
            f = DATA / "frames" / video_id / meta[r]["frame_filename"]
            ims.append(prep(Image.open(f).convert("RGB")))
        with torch.inference_mode():
            v = model.encode_image(torch.stack(ims))
        v = v / v.norm(dim=-1, keepdim=True)
        vecs.append(v.cpu().numpy().astype(np.float32))
    E = np.concatenate(vecs, axis=0)
    tmp = ra_file.with_suffix(".tmp.npy")
    np.save(tmp, E)
    tmp.replace(ra_file)
    print(f"  {video_id}: {len(rows)} khung, {time.time()-t0:.0f}s", flush=True)
    return E


def cmd_encode(args):
    sach, chon_mot, chon_hai = nap_mau()
    meta, valid, rows_of = nap_meta()
    mau = chon_mot + chon_hai
    (CACHE / "img").mkdir(parents=True, exist_ok=True)

    ten_model = MODEL_B if args.model_b else MODEL_L
    model, prep, _ = nap_model(ten_model)
    (CACHE / "model.txt").write_text(ten_model, encoding="utf-8")

    videos = []  # thứ tự mẫu ổn định, khử trùng lặp
    for i in mau:
        v = sach[i]["video_id"]
        if v not in videos:
            videos.append(v)
    print(f"encode {len(videos)} video ({ten_model})", flush=True)
    for k, v in enumerate(videos):
        rows, _ = khung_hop_le(v, meta, valid, rows_of, can_anh=True)
        encode_video(v, rows, meta, model, prep, CACHE / "img" / f"{v}.npy")
        if k < N_VIDEO_CROP:
            encode_video(v, rows, meta, model, preprocess_crop(),
                         CACHE / "img" / f"{v}.crop.npy")
    print("encode XONG", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Giai đoạn 2: text + hạng + bootstrap + GO/NO-GO
# ---------------------------------------------------------------------------


def vec_text(model, tokenizer, texts):
    import torch

    with torch.inference_mode():
        v = model.encode_text(tokenizer(texts))
    v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype(np.float32)


def dem_token_clip(text):
    """Số token BPE CLIP của văn bản (chưa kể SOT/EOT) — đo mức cắt cụt ở 32."""
    from open_clip.tokenizer import _tokenizer

    return len(_tokenizer.encode(text))


def cmd_cham(args):
    sach, chon_mot, chon_hai = nap_mau()
    meta, valid, rows_of = nap_meta()
    sims_sig = np.load(DATA / "cache_tin_hieu_noi_video" / "sims_sach.npy", mmap_mode="r")

    ten_model = (CACHE / "model.txt").read_text(encoding="utf-8").strip()
    model, _, tokenizer = nap_model(ten_model)

    mau = [(i, "mot") for i in chon_mot] + [(i, "hai") for i in chon_hai]
    cau_hinh = ["A_ensemble_vi", "B_ensemble_en", "en_thuan", "vi_thuan"]
    ranks = {c: {"mot": [], "hai": []} for c in cau_hinh}
    ranks_sig = {"mot": [], "hai": []}
    per_item = []
    n_cut = {"en": 0, "vi": 0}

    for i, nhom in mau:
        g = sach[i]
        v = g["video_id"]
        rows, _ = khung_hop_le(v, meta, valid, rows_of, can_anh=True)
        row_ans = tim_hang_dap_an(g, meta, rows_of)
        if row_ans not in rows:
            rows = rows + [row_ans]
            print(f"  ! {v}: khung dap an ngoai tap valid/anh — them vao rieng")
        E = np.load(CACHE / "img" / f"{v}.npy")
        pos = {r: k for k, r in enumerate(rows)}
        assert E.shape[0] == len(rows), f"{v}: cache anh lech"

        en, vi = g["kis_query_en"], g["kis_query_vi"]
        if dem_token_clip(en) > 30:
            n_cut["en"] += 1
        if dem_token_clip(vi) > 30:
            n_cut["vi"] += 1
        V = vec_text(model, tokenizer, [
            en, vi,
            f"a high quality video keyframe of {en}",
            f"a photo of {en}",
        ])
        q = {
            "A_ensemble_vi": W_A[0] * V[0] + W_A[1] * V[1] + W_A[2] * V[2] + W_A[3] * V[3],
            "B_ensemble_en": W_B[0] * V[0] + W_B[1] * V[2] + W_B[2] * V[3],
            "en_thuan": V[0],
            "vi_thuan": V[1],
        }
        muc = {"i": i, "nhom": nhom, "video": v, "n_khung": len(rows)}
        for c in cau_hinh:
            qq = q[c] / max(float(np.linalg.norm(q[c])), 1e-6)
            s = E @ qq
            s_ans = float(s[pos[row_ans]])
            h = 1 + int((s > s_ans).sum())
            ranks[c][nhom].append(h)
            muc[f"hang_{c}"] = h
        h_sig = hang_cua(sims_sig[i], rows, row_ans)
        ranks_sig[nhom].append(h_sig)
        muc["hang_siglip"] = h_sig
        per_item.append(muc)

    # ------------------------------------------------------------------ bảng
    print(f"\n=== KET QUA pre-test ({ten_model}) — 20 MOT / 20 HAI, seed {SEED} ===")
    print(f"cau en bi cat >30 token CLIP: {n_cut['en']}/40 ; vi: {n_cut['vi']}/40")
    ket = {"model": ten_model, "seed": SEED, "n_cut": n_cut, "per_item": per_item,
           "bang": {}}
    for nhom, ten in (("mot", "MOT canh"), ("hai", "HAI canh")):
        print(f"\n--- nhom {ten} (n={len(ranks_sig[nhom])}) ---")
        b = bang_nhom(ranks_sig[nhom])
        print(f"  {'SigLIP (san xuat)':22s} trung vi {b['trung_vi']:5.1f}  "
              f"hang-1 {b['hang1']:6.1%}  top-5 {b['top5']:6.1%}")
        ket["bang"][f"siglip_{nhom}"] = b
        for c in cau_hinh:
            b = bang_nhom(ranks[c][nhom])
            thang = sum(1 for a, s in zip(ranks[c][nhom], ranks_sig[nhom]) if a < s)
            thua = sum(1 for a, s in zip(ranks[c][nhom], ranks_sig[nhom]) if a > s)
            print(f"  PE {c:19s} trung vi {b['trung_vi']:5.1f}  "
                  f"hang-1 {b['hang1']:6.1%}  top-5 {b['top5']:6.1%}  "
                  f"(thang/thua/hoa {thang}/{thua}/{len(ranks_sig[nhom])-thang-thua})")
            b["thang"], b["thua"] = thang, thua
            ket["bang"][f"{c}_{nhom}"] = b

    # ------------------------------------------- bootstrap theo câu (ghép cặp)
    rng = np.random.RandomState(SEED)
    print("\nbootstrap theo cau, 4000 lan (hieu PE − SigLIP, cau hinh du thi):")
    for c in ("A_ensemble_vi", "B_ensemble_en"):
        for nhom in ("mot", "hai"):
            a = np.array(ranks[c][nhom], dtype=float)
            s = np.array(ranks_sig[nhom], dtype=float)
            n = len(a)
            d_h1, d_tv = [], []
            for _ in range(BOOT):
                idx = rng.randint(0, n, n)
                d_h1.append((a[idx] == 1).mean() - (s[idx] == 1).mean())
                d_tv.append(np.median(a[idx]) - np.median(s[idx]))
            lo1, hi1 = np.percentile(d_h1, [2.5, 97.5])
            lo2, hi2 = np.percentile(d_tv, [2.5, 97.5])
            print(f"  {c} {nhom}: Δhang-1 KTC95 [{lo1:+.1%}, {hi1:+.1%}] ; "
                  f"Δtrung-vi KTC95 [{lo2:+.1f}, {hi2:+.1f}]")
            ket["bang"][f"boot_{c}_{nhom}"] = {
                "d_hang1_ktc": [float(lo1), float(hi1)],
                "d_trungvi_ktc": [float(lo2), float(hi2)],
            }

    # ------------------------------------------------- kiểm độ nhạy center-crop
    crop_vids = [f.stem.replace(".crop", "") for f in (CACHE / "img").glob("*.crop.npy")]
    if crop_vids:
        print(f"\nkiem do nhay center-crop ({len(crop_vids)} video):")
        for c in ("A_ensemble_vi", "B_ensemble_en"):
            cap = []
            for muc in per_item:
                if muc["video"] not in crop_vids:
                    continue
                g = sach[muc["i"]]
                v = g["video_id"]
                rows, _ = khung_hop_le(v, meta, valid, rows_of, can_anh=True)
                row_ans = tim_hang_dap_an(g, meta, rows_of)
                if row_ans not in rows:
                    rows = rows + [row_ans]
                E = np.load(CACHE / "img" / f"{v}.crop.npy")
                pos = {r: k for k, r in enumerate(rows)}
                en, vi = g["kis_query_en"], g["kis_query_vi"]
                V = vec_text(model, tokenizer, [
                    en, vi,
                    f"a high quality video keyframe of {en}",
                    f"a photo of {en}",
                ])
                if c == "A_ensemble_vi":
                    qv = W_A[0] * V[0] + W_A[1] * V[1] + W_A[2] * V[2] + W_A[3] * V[3]
                else:
                    qv = W_B[0] * V[0] + W_B[1] * V[2] + W_B[2] * V[3]
                qv = qv / max(float(np.linalg.norm(qv)), 1e-6)
                s = E @ qv
                h = 1 + int((s > float(s[pos[row_ans]])).sum())
                cap.append((muc[f"hang_{c}"], h))
            if cap:
                print(f"  {c}: squash {[a for a, _ in cap]} vs crop {[b for _, b in cap]}")
                ket["bang"][f"crop_{c}"] = cap

    # ----------------------------------------------------------- GO / NO-GO
    # Quy ước hạng của pre-test (TOÀN video) khác quy ước của số công bố
    # (TRONG pool 400) — ngưỡng đã dịch giữ nguyên ý định, chốt TRƯỚC khi chấm
    # trong docs/PRETEST_ENCODER.md §2b:
    #   1. hai cảnh: PE ≤ ½ trung vị SigLIP trên CÙNG mẫu (ý "6→3" = giảm nửa)
    #   2. một cảnh: hạng-1 PE ≥ hạng-1 SigLIP cùng mẫu + 12pp (ý "43→55")
    # Ngưỡng tuyệt đối gốc (≤3; ≥55%) báo cáo song song — khắt khe hơn.
    tv_hai = min(ket["bang"]["A_ensemble_vi_hai"]["trung_vi"],
                 ket["bang"]["B_ensemble_en_hai"]["trung_vi"])
    h1_mot = max(ket["bang"]["A_ensemble_vi_mot"]["hang1"],
                 ket["bang"]["B_ensemble_en_mot"]["hang1"])
    tv_hai_sig = ket["bang"]["siglip_hai"]["trung_vi"]
    h1_mot_sig = ket["bang"]["siglip_mot"]["hang1"]
    go1 = tv_hai <= 0.5 * tv_hai_sig
    go2 = h1_mot >= h1_mot_sig + 0.12
    go = go1 or go2
    print(f"\nNGUONG DICH (§2b, chot truoc khi cham):")
    print(f"  1. trung vi HAI canh <= 1/2 SigLIP cung mau "
          f"({0.5 * tv_hai_sig:.1f})? PE {tv_hai:.1f} -> {'DAT' if go1 else 'KHONG'}")
    print(f"  2. hang-1 MOT canh >= SigLIP cung mau + 12pp "
          f"({h1_mot_sig + 0.12:.1%})? PE {h1_mot:.1%} -> {'DAT' if go2 else 'KHONG'}")
    print(f"NGUONG TUYET DOI GOC (khat khe hon duoi quy uoc toan-video):")
    print(f"  trung vi HAI canh <= 3 ? {tv_hai:.1f} "
          f"-> {'DAT' if tv_hai <= 3 else 'KHONG'}")
    print(f"  hang-1 MOT canh >= 55%? {h1_mot:.1%} "
          f"-> {'DAT' if h1_mot >= 0.55 else 'KHONG'}")
    print(f"\n==> KET LUAN: {'GO' if go else 'NO_GO'}")
    ket["nguong"] = {
        "quy_uoc": "hang tren TOAN video (khong phai trong-pool)",
        "go1_hai_dich": [tv_hai, 0.5 * tv_hai_sig, bool(go1)],
        "go2_mot_dich": [h1_mot, h1_mot_sig + 0.12, bool(go2)],
        "goc_tuyet_doi": [tv_hai <= 3.0, h1_mot >= 0.55],
    }
    ket["ket_luan"] = "GO" if go else "NO_GO"
    (CACHE / "ket_qua.json").write_text(
        json.dumps(ket, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"da ghi {CACHE / 'ket_qua.json'}")
    return 0


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--giai-doan", choices=["chuan-bi", "encode", "cham", "tat-ca"],
                    default="tat-ca")
    ap.add_argument("--model-b", action="store_true",
                    help="dung PE-Core-B-16-224 (du phong khi L tai hong)")
    args = ap.parse_args()
    if args.giai_doan in ("chuan-bi", "tat-ca"):
        rc = cmd_chuan_bi(args)
        if rc:
            return rc
    if args.giai_doan in ("encode", "tat-ca"):
        rc = cmd_encode(args)
        if rc:
            return rc
    if args.giai_doan in ("cham", "tat-ca"):
        return cmd_cham(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
