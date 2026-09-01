"""Điểm CẮT tương đối-trong-video làm đặc trưng xếp hạng cho nhóm MỘT cảnh.

`docs/KE_HOACH_DINH_VI.md` §4.5 để lại đúng một việc 0 đồng đáng thử. Lane paper
kết luận "cosine SigLIP liền kề không đo được cú cắt", nhưng phản biện chỉ ra kết
luận ấy dựa trên một ngưỡng **tuyệt đối** (0,5) nằm **dưới trung vị của cặp ngẫu
nhiên khác video** — tức nó đòi hai khung cùng bản tin phải khác nhau hơn hai
khung của hai video hoàn toàn khác nhau. Trên thang **tương đối trong từng
video**, cú cắt thật nằm ở phân vị 0,24 còn đối chứng nằm ở 0,44: **có tín hiệu,
nhưng yếu**.

Nhóm MỘT cảnh là nhóm duy nhất **chưa có tín hiệu nội-video nào** (lever cảnh B
và lever hoán vị đều chỉ chạm câu hai cảnh) và có thâm hụt thứ tự lớn nhất.
Giả thuyết: khoảnh khắc mà người ra đề mô tả thường là khung **mở đầu một cú
cắt** — cảnh vừa chuyển sang thứ đang được tả.

Cơ chế áp dụng dùng lại **đúng phép hoán vị đã được chứng minh an toàn** ở §1:
giữ nguyên đa tập điểm của mỗi video, chỉ đổi xem điểm nào thuộc khung nào. Nên
bề rộng phủ video bất biến theo xây dựng, và R@100 cũng bất biến (tập dòng không
đổi, chỉ thứ tự trong video đổi).

Báo cáo SONG SONG hai mô hình bốc theo luật của `docs/MO_HINH_BOC.md`.

    python -u scripts/do_diem_cat_mot_canh.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._console import safe_console  # noqa: E402

safe_console()

from scripts.cong_do_ben_mo_hinh_boc import cac_lan_boc_kieu  # noqa: E402
from scripts.do_cap_thoi_gian_moi import canh_cua  # noqa: E402
from scripts.experiment_cap_thoi_gian import _plan, nap_truc_video  # noqa: E402
from scripts.experiment_phu_quet_luoi import cham_nhanh, ma_tran_dong  # noqa: E402
from scripts.make_submission import DEFAULT_N_FLAT, allocate_rows  # noqa: E402
from src.core.submission import MAX_ROWS, Candidate  # noqa: E402

KIEU = ("DEU", "SAU_NEO")
GOC_TUNE = 881000
GOC_TEST = 882000


def diem_cat_theo_video(emb, truc, vids_can):
    """{(video, frame): phân vị điểm cắt trong video}.

    Điểm cắt thô = 1 − cos(khung này, khung liền TRƯỚC trong cùng video).
    Rồi đổi sang **phân vị trong chính video đó** — đây là điểm mấu chốt: thang
    tuyệt đối của cosine không so được giữa các video (bản tin studio có mọi
    khung giống nhau; phóng sự ngoài trời thì không), nên ngưỡng tuyệt đối là
    phép so sai. Phân vị làm mỗi video tự làm chuẩn cho mình.
    """
    ra = {}
    for v in vids_can:
        if v not in truc:
            continue
        frames, rows = truc[v]
        if len(rows) < 3:
            continue
        X = np.asarray(emb[np.asarray(rows)], dtype=np.float32)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(n, 1e-8)
        cos = np.sum(X[1:] * X[:-1], axis=1)
        tho = np.concatenate(([0.0], 1.0 - cos))  # khung đầu video: không có trước
        thu = np.argsort(np.argsort(tho))
        pv = thu / max(1, len(tho) - 1)
        for f, p in zip(frames, pv):
            ra[(v, int(f))] = float(p)
    return ra


def hoan_vi_theo_cat(cands, diem_cat, w, so_video=3, so_khung=12):
    """Hoán vị điểm trong video theo điểm cắt — cùng cơ chế an toàn của §1."""
    if w <= 0 or len(cands) < 2:
        return cands
    thu_tu, theo_video = [], {}
    for i, c in enumerate(cands):
        if c.video_id not in theo_video:
            thu_tu.append(c.video_id)
            theo_video[c.video_id] = []
        theo_video[c.video_id].append(i)

    key_of = {}
    for vid in thu_tu[:so_video]:
        pos = sorted(theo_video[vid], key=lambda i: -float(cands[i].score))[:so_khung]
        if len(pos) < 2:
            continue
        for i in pos:
            p = diem_cat.get((cands[i].video_id, int(cands[i].frame_idx)))
            if p is None:
                p = 0.0
            key_of[i] = float(cands[i].score) + w * p
    if len(key_of) < 2:
        return cands

    diem_moi = [float(c.score) for c in cands]
    for _v, pos in theo_video.items():
        co = [i for i in pos if i in key_of]
        if len(co) < 2:
            continue
        cac = sorted((float(cands[i].score) for i in co), reverse=True)
        for i, d in zip(sorted(co, key=lambda i: (-key_of[i], i)), cac):
            diem_moi[i] = d
    return [Candidate(c.video_id, c.frame_idx, diem_moi[i], c.video_last_frame)
            for i, c in enumerate(cands)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--moi", default=str(ROOT / "data" / "ground_truth_moi.json"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache_bo_do_moi"))
    ap.add_argument("--windows", default="6,10,20")
    ap.add_argument("--tune-seeds", type=int, default=3)
    ap.add_argument("--tune-draws", type=int, default=32)
    ap.add_argument("--test-seeds", type=int, default=4)
    ap.add_argument("--test-draws", type=int, default=48)
    args = ap.parse_args()

    data = Path(args.data)
    windows = [int(w) for w in args.windows.split(",")]
    moi = json.loads(Path(args.moi).read_text(encoding="utf-8"))
    uv = json.loads((Path(args.cache) / "uv_moi.json").read_text(encoding="utf-8"))
    giu = [i for i, g in enumerate(moi) if not g.get("lan_truc")]
    sach = [moi[i] for i in giu]
    cands = [[Candidate(v, f, s, lf) for v, f, s, lf in uv[i]] for i in giu]

    # nhóm MỘT cảnh: đúng nhóm chưa có tín hiệu nội-video nào
    mot = [i for i, m in enumerate(sach) if not canh_cua(m)]
    print(f"bo sach {len(sach)} muc | nhom MOT canh {len(mot)}")

    meta = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    kf_list = {}
    for m in meta:
        kf_list.setdefault(m["video_id"], []).append(int(m["frame_idx"]))
    kf = {v: np.array(sorted(a), dtype=np.int64) for v, a in kf_list.items()}
    del meta, kf_list

    print("nap embeddings (mmap) + truc video ...", flush=True)
    emb = np.load(data / "embeddings_siglip2_384.npy", mmap_mode="r")
    truc = nap_truc_video(data, False)
    vids_can = {c.video_id for i in mot for c in cands[i][:200]}
    print(f"tinh diem cat cho {len(vids_can)} video ...", flush=True)
    diem_cat = diem_cat_theo_video(emb, truc, vids_can)
    print(f"  co diem cat cho {len(diem_cat):,} keyframe")

    # --- chẩn đoán TRƯỚC khi nhìn điểm: keyframe đáp án có nằm ở chỗ cắt không?
    pv_dap_an, pv_ngau = [], []
    rng = np.random.default_rng(7)
    for i in mot:
        g = sach[i]
        a = kf.get(g["video_id"])
        if a is None or not len(a):
            continue
        k = int(a[int(np.argmin(np.abs(a - int(g["frame_idx"]))))])
        p = diem_cat.get((g["video_id"], k))
        if p is None:
            continue
        pv_dap_an.append(p)
        j = int(rng.integers(0, len(a)))
        q = diem_cat.get((g["video_id"], int(a[j])))
        if q is not None:
            pv_ngau.append(q)
    pa, pn = np.array(pv_dap_an), np.array(pv_ngau)
    print(f"\nCHAN DOAN (truoc khi nhin diem): phan vi diem cat cua keyframe DAP AN")
    print(f"  dap an  : trung vi {np.median(pa):.3f}  (n={len(pa)})")
    print(f"  ngau nhien cung video: trung vi {np.median(pn):.3f}  (n={len(pn)})")
    print(f"  -> {'CO tin hieu' if np.median(pa) > np.median(pn) + 0.05 else 'KHONG thay tin hieu ro'}")

    # --- TUNE/TEST phân tầng trong nhóm một cảnh
    i_tune = mot[0::2]
    i_test = mot[1::2]
    nen = [allocate_rows(c, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS] for c in cands]

    def rows_w(w):
        ra = []
        for i, c in enumerate(cands):
            cc = hoan_vi_theo_cat(c, diem_cat, w) if i in set(mot) else c
            ra.append(allocate_rows(cc, "coverage", DEFAULT_N_FLAT, _plan())[:MAX_ROWS])
        return ra

    # bất biến: w = 0 phải ra dòng giống hệt nền, mọi câu
    r0 = rows_w(0.0)
    assert all(r0[i] == nen[i] for i in range(len(sach))), "w=0 doi dong — co ch bug"
    print("bat bien: w=0 ra dong giong het nen o 132/132 muc (assert) -> DAT")

    LUOI_W = (0.01, 0.03, 0.1, 0.3)
    gt_tune = [sach[i] for i in i_tune]
    ho_tune = cac_lan_boc_kieu(GOC_TUNE, args.tune_seeds, args.tune_draws, gt_tune, kf, "DEU")

    def cham(idx, rows, ho, gt_sub):
        mats = ma_tran_dong([rows[i] for i in idx], gt_sub)
        return float(np.mean([cham_nhanh(mats, d, windows) for d in ho]))

    nen_t = cham(i_tune, nen, ho_tune, gt_tune)
    print(f"\nNEN tren TUNE (nhom mot canh, n={len(i_tune)}): {nen_t:.4f}")
    print(f"{'w':>7}{'diem':>10}{'so nen':>10}")
    print("-" * 27)
    ket, rc = {}, {}
    for w in LUOI_W:
        rc[w] = rows_w(w)
        ket[w] = cham(i_tune, rc[w], ho_tune, gt_tune)
        print(f"{w:>7.2f}{ket[w]:>10.4f}{100*(ket[w]/nen_t-1):>+9.1f}%", flush=True)

    chot = max(ket, key=lambda k: ket[k])
    print(f"\nCHOT tren TUNE: w={chot} ({100*(ket[chot]/nen_t-1):+.1f}%)")

    print("\n=== TEST (doc DUNG MOT LAN), song song hai mo hinh boc ===")
    gt_test = [sach[i] for i in i_test]
    for kieu in KIEU:
        ho = cac_lan_boc_kieu(GOC_TEST, args.test_seeds, args.test_draws, gt_test, kf, kieu)
        a = cham(i_test, nen, ho, gt_test)
        b = cham(i_test, rc[chot], ho, gt_test)
        # bootstrap theo câu
        mn = ma_tran_dong([nen[i] for i in i_test], gt_test)
        mc = ma_tran_dong([rc[chot][i] for i in i_test], gt_test)

        def tung_cau(mats):
            r = np.zeros(len(gt_test))
            for draws in ho:
                for q in range(len(gt_test)):
                    r[q] += cham_nhanh([mats[q]], [draws[q]], windows)
            return r / len(ho)

        dn, dc = tung_cau(mn), tung_cau(mc)
        rg = np.random.default_rng(4242)
        lay = rg.integers(0, len(gt_test), size=(4000, len(gt_test)))
        delta = dc[lay].mean(axis=1) - dn[lay].mean(axis=1)
        lo, hi = np.percentile(delta, [2.5, 97.5])
        print(f"  {kieu:8s}: nen {a:.4f} -> chot {b:.4f} = {100*(b/a-1):+.1f}%"
              f" | KTC [{lo:+.4f}, {hi:+.4f}] | P(<=0) = {(delta<=0).mean():.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
