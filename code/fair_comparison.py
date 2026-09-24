"""The fair comparison of allocator families: every family tuned by the same
nested cross-validation on the primary set (Sect. 5 of the paper).

    python code/fair_comparison.py --workers 4        # both pools, about an hour on 4 cores
    python code/fair_comparison.py --workers 4 --pools raw   # raw pool only, about half

A copy of scripts/e16_so_cong_bang.py (allocators, grids, scorer, statistics)
and of the per-set analysis of scripts/e16_so_cong_bang_chinh.py (``analyse_set``
for all items, ``analyse_test`` for the held-out half), fed from the release.
Left out: the post-hoc 120-cell extension of the ranking-based grid and the
statistic of the earlier nested-tuning report; everything else of those two
blocks is here, including the verdict rule.

Families and grids (5 families, 1,448 cells):
  COV   greedy coverage of the belief (the shipped allocator), tau x sigma x h x g = 384
  HYB   ranking-based allocator, n_flat x depth cost x step = 200
  UG    uniform-grid ladders on the top N videos, N x step = 320
  NMS   peak-capped temporal NMS + ladders, d x peaks x step = 400
  FLAT  the coverage greedy with a flat belief, sigma x h x g = 144
UG, NMS and FLAT use no belief ("belief-free").

Estimator on ALL items: nested 5-fold cross-validation, folds stratified by the
two-scene label, 20 fold partitions (seeds 20260930 + r), the same folds for
every family; each item's score is out of fold, averaged over the partitions.
On the held-out half: each family tuned once on TUNE and scored on TEST.
Scores under four moment models (DEU, SAU_NEO, GAUSS12, TAM_GIAC) and windows
+-6/10/20 (official) and +-50. Paired bootstrap 4,000 (seed 4242), sign-flip
10,000 (seed 606).

Output: <out>/fair_comparison.json and a comparison of every number with
expected/fair_comparison.json (the file the paper's numbers come from).
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REL = HERE.parent
sys.path.insert(0, str(HERE))

import allocators as AL  # noqa: E402
import alloc_tables as T  # noqa: E402
import protocol as P  # noqa: E402
import stats as S  # noqa: E402
from submission import (  # noqa: E402
    _MASS_QUANTUM,
    MAX_ROWS,
    AllocationPlan,
    Candidate,
    allocate_hybrid_rows,
    frame_ladder,
)

# ---------------------------------------------------------------------------------------------
# grids (fixed in the pre-registration of the original experiment)
# ---------------------------------------------------------------------------------------------
G_COV = [(t, s, h, g) for t in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05)
         for s in (15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0)
         for h in (6, 10, 15, 25) for g in (5, 10)]
G_HYB = [(n, d, st) for n in (5, 10, 15, 20, 30, 40, 50, 70)
         for d in (0.25, 0.5, 0.75, 1.0, 1.5) for st in (6, 8, 10, 14, 20)]
G_UG = [(n, st) for n in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 33, 50, 100)
        for st in (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 25, 30, 41, 55)]
G_NMS = [(d, p, st) for d in (6, 10, 20, 30, 45, 60, 90, 150)
         for p in (1, 2, 3, 4, 5, 7, 10, 15, 20, 33) for st in (10, 13, 17, 20, 25)]
G_FLAT = [(s, h, g) for s in (10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0,
                              360.0, 500.0)
          for h in (6, 8, 10, 15, 20, 25) for g in (5, 10)]
GRIDS = {"COV": G_COV, "HYB": G_HYB, "UG": G_UG, "NMS": G_NMS, "FLAT": G_FLAT}
FAMS = ("COV", "HYB", "UG", "NMS", "FLAT")
BELIEF_FREE = ("UG", "NMS", "FLAT")
FROZEN = {"COV": (0.02, 30.0, 6, 5), "HYB": (30, 0.5, 10)}
E6_FROZEN = {"raw": {"UG": (1, 20), "NMS": (6, 10, 20), "FLAT": (30.0, 6, 5)},
             "prod": {"UG": (1, 20), "NMS": (10, 3, 13), "FLAT": (30.0, 6, 5)}}
#: the 200-cell coverage grid of the earlier nested-tuning report (a subset of G_COV)
GRID_COV200 = [(n, s, w, g) for n in (0.01, 0.015, 0.02, 0.03, 0.05) for s in (15.0, 20.0, 30.0, 45.0, 60.0)
               for w in (6, 10, 15, 25) for g in (5, 10)]
MODELS = P.MOMENT_MODELS
WINS = (6, 10, 20, 50)          # the first three are the official triple
POOLS = ("raw", "prod")
FOLD_SEED = 20260930
N_REP = 20
K_FOLDS = 5
MATCH_SIZE = 144
MATCH_DRAWS = 20
MATCH_SEED = 16016
T975_DF4 = 2.7764451051977987
PLAN_TAIL = AllocationPlan(breadth_cost=1.0, depth_cost=0.5, step=10)
TAIL_N_FLAT = 30


# ---------------------------------------------------------------------------------------------
# allocators
# ---------------------------------------------------------------------------------------------


def tail_fill(rows, candidates, budget=MAX_ROWS):
    """The shipped allocator's own tail fill, shared by every belief-free family."""
    rows = list(rows)
    if len(rows) >= budget:
        return rows[:budget]
    seen = set(rows)
    fill = AllocationPlan(budget=budget, breadth_cost=PLAN_TAIL.breadth_cost,
                          depth_cost=PLAN_TAIL.depth_cost, step=PLAN_TAIL.step,
                          max_depth=PLAN_TAIL.max_depth)
    for key in allocate_hybrid_rows(candidates, n_flat=TAIL_N_FLAT, plan=fill):
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
        if len(rows) >= budget:
            break
    return rows[:budget]


def cov_rows_fast(cands, nhiet, sigma, nua_cua_so, luoi, flat=False, budget=MAX_ROWS):
    """allocate_coverage_rows, bit for bit, with the per-video window sums cached.
    ``flat`` replaces the softmax belief by w_i = 1/N and changes nothing else."""
    if not cands:
        return []
    diem = np.array([round(float(c.score), 4) for c in cands], dtype=np.float64)
    if flat:
        w = np.full(len(cands), 1.0 / len(cands), dtype=np.float64)
    else:
        w = np.exp((diem - diem.max()) / max(nhiet, 1e-9))
        w /= w.sum()
    theo: dict = {}
    for c, wi in zip(cands, w):
        last = int(c.video_last_frame) if c.video_last_frame is not None else 1 << 31
        theo.setdefault(c.video_id, []).append((int(c.frame_idx), float(wi), last))
    s4 = 4 * int(sigma)
    vids, trucs, cons = [], [], []
    for vid, items in theo.items():
        last = max(x[2] for x in items)
        lo = max(0, min(x[0] for x in items) - s4)
        hi = max(lo, min(last, max(x[0] for x in items) + s4))
        truc = np.arange(lo, hi + 1, luoi, dtype=np.int64)
        if truc.size == 0:
            continue
        mass = np.zeros(truc.size, dtype=np.float64)
        for f, wi, _ in items:
            mass += wi * np.exp(-0.5 * ((truc - f) / sigma) ** 2)
        mass = np.floor(mass / _MASS_QUANTUM + 0.5) * _MASS_QUANTUM
        vids.append(vid)
        trucs.append(truc)
        cons.append(mass.copy())
    nua = max(1, nua_cua_so // luoi)
    V = len(vids)
    bval = np.zeros(V, dtype=np.float64)
    bj = np.zeros(V, dtype=np.int64)

    def upd(k):
        con = cons[k]
        n = con.size
        tich = np.cumsum(np.concatenate(([0.0], con)))
        ar = np.arange(n)
        gia = tich[np.minimum(n, ar + nua + 1)] - tich[np.maximum(0, ar - nua)]
        j = int(np.argmax(gia))
        bval[k] = gia[j]
        bj[k] = j

    for k in range(V):
        upd(k)
    rows, used = [], set()
    while len(rows) < budget and V:
        k = int(np.argmax(bval))
        if not bval[k] > 0.0:
            break
        key = (vids[k], int(trucs[k][bj[k]]))
        if key in used:
            k, tv = None, 0.0
            for kk in range(V):
                kkey = (vids[kk], int(trucs[kk][bj[kk]]))
                if bval[kk] > tv and kkey not in used:
                    k, tv = kk, float(bval[kk])
            if k is None:
                break
            key = (vids[k], int(trucs[k][bj[k]]))
        rows.append(key)
        used.add(key)
        j = int(bj[k])
        cons[k][max(0, j - nua): j + nua + 1] = 0.0
        upd(k)
    if len(rows) < budget:
        seen = set(rows)
        fill = AllocationPlan(budget=budget, breadth_cost=PLAN_TAIL.breadth_cost,
                              depth_cost=PLAN_TAIL.depth_cost, step=PLAN_TAIL.step,
                              max_depth=PLAN_TAIL.max_depth)
        for key in allocate_hybrid_rows(cands, n_flat=AL.DEFAULT_N_FLAT, plan=fill):
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
            if len(rows) >= budget:
                break
    return rows[:budget]


def uniform_grid_rows(candidates, n_videos, step, budget=MAX_ROWS):
    """No belief. Top-N videos in candidate-rank order, fixed-step ladders, round-robin."""
    order, first = [], {}
    for c in candidates:
        if c.video_id not in first:
            first[c.video_id] = c
            order.append(c.video_id)
    vids = order[:n_videos]
    lads = [frame_ladder(int(first[v].frame_idx), budget, step, lo=0, hi=first[v].video_last_frame)
            for v in vids]
    rows, seen = [], set()
    depth = 0
    while len(rows) < budget:
        moved = False
        for v, lad in zip(vids, lads):
            if depth >= len(lad):
                continue
            moved = True
            key = (v, int(lad[depth]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
            if len(rows) >= budget:
                return rows
        if not moved:
            break
        depth += 1
    return tail_fill(rows, candidates, budget)


def nms_peaks(candidates, d, cap):
    """Temporal NMS over the similarity-ranked candidates; the first ``cap`` survivors."""
    order = sorted(range(len(candidates)), key=lambda i: (-round(float(candidates[i].score), 4), i))
    kept, per_video = [], {}
    for i in order:
        c = candidates[i]
        f = int(c.frame_idx)
        if all(abs(f - g) > d for g in per_video.get(c.video_id, [])):
            per_video.setdefault(c.video_id, []).append(f)
            kept.append(c)
            if len(kept) >= cap:
                break
    return kept


def nms2_rows(candidates, d, cap, step=13, budget=MAX_ROWS):
    """Peak-capped NMS: at most ``cap`` peaks, then round-robin ladders."""
    kept = nms_peaks(candidates, d, cap)
    lads = [frame_ladder(int(c.frame_idx), budget, step, lo=0, hi=c.video_last_frame) for c in kept]
    rows, seen = [], set()
    depth = 0
    while len(rows) < budget:
        moved = False
        for c, lad in zip(kept, lads):
            if depth >= len(lad):
                continue
            moved = True
            key = (c.video_id, int(lad[depth]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
            if len(rows) >= budget:
                return rows
        if not moved:
            break
        depth += 1
    return tail_fill(rows, candidates, budget)


def rows_for(fam, cell, cands):
    if fam == "COV":
        t, s, h, g = cell
        return cov_rows_fast(cands, t, s, h, g)
    if fam == "FLAT":
        s, h, g = cell
        return cov_rows_fast(cands, 0.02, s, h, g, flat=True)
    if fam == "HYB":
        n, d, st = cell
        return allocate_hybrid_rows(cands, n_flat=n,
                                    plan=AllocationPlan(breadth_cost=1.0, depth_cost=d, step=st))[:MAX_ROWS]
    if fam == "UG":
        n, st = cell
        return uniform_grid_rows(cands, n, st)
    if fam == "NMS":
        d, p, st = cell
        return nms2_rows(cands, d, p, st)
    raise KeyError(fam)


# ---------------------------------------------------------------------------------------------
# scoring: every model and every window in one pass
# ---------------------------------------------------------------------------------------------


def score_multi(rows_of, gt_vids, truths_list):
    """(M, Q, W, F): per model, query, window, seed family -- the official value averaged over draws."""
    M = len(truths_list)
    F, Q, _D = truths_list[0].shape
    out = np.zeros((M, Q, len(WINS), F))
    for q in range(Q):
        r = rows_of[q][:MAX_ROWS]
        idx = np.array([i for i, (v, _f) in enumerate(r) if v == gt_vids[q]], dtype=np.int64)
        if idx.size == 0:
            continue
        f = np.array([int(r[i][1]) for i in idx], dtype=np.int64)
        for mi, Tm in enumerate(truths_list):
            d = np.abs(f[None, None, :] - Tm[:, q, :, None])
            for wi, h in enumerate(WINS):
                hit = d <= h
                co = hit.any(axis=2)
                rank = np.where(co, idx[hit.argmax(axis=2)] + 1, 0)
                out[mi, q, wi, :] = P.BUCKET[rank].mean(axis=1)
    return out


def official(t):
    """(..., Q, W, F) -> (..., Q): windows 6/10/20 only."""
    x = np.ascontiguousarray(t[..., :3, :])
    return x.mean(axis=(-2, -1))


def per_window(t):
    return t.mean(axis=-1)


# ---------------------------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------------------------

_W: dict = {}


def _init(pools, gt_vids, truths):
    _W.update({"pools": pools, "gt_vids": gt_vids, "truths": truths})


def _job(job):
    pool, fam, ci = job
    rows = [rows_for(fam, GRIDS[fam][ci], c) for c in _W["pools"][pool]]
    t = score_multi(rows, _W["gt_vids"], _W["truths"])
    return pool, fam, ci, official(t), per_window(t)


# ---------------------------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------------------------


def make_folds(two, seed, k=K_FOLDS):
    """Stratified round-robin assignment of item positions 0..n-1 to folds 0..k-1."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(two), -1, dtype=np.int64)
    for mask in (~two, two):
        idx = np.flatnonzero(mask)
        idx = idx[rng.permutation(len(idx))]
        for j, i in enumerate(idx):
            fold[i] = j % k
    assert (fold >= 0).all()
    return fold


def nested_one(off, fold):
    chosen = np.zeros(K_FOLDS, dtype=np.int64)
    oof = np.zeros(off.shape[1])
    for k in range(K_FOLDS):
        te = np.flatnonzero(fold == k)
        tr = np.flatnonzero(fold != k)
        gi = int(np.argmax(off[:, tr].mean(axis=1)))
        chosen[k] = gi
        oof[te] = off[gi, te]
    return chosen, oof


def gather(tensor, chosen, folds):
    R, n = folds.shape
    out = np.zeros((R, n) + tensor.shape[2:])
    for r in range(R):
        out[r] = tensor[chosen[r][folds[r]], np.arange(n)]
    return out


def tci(x):
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    t = T975_DF4 if len(x) == 5 else 2.093024054408263
    return [m, m - t * se, m + t * se]


contrast = S.contrast_fair


def _rankdata(x):
    """Average ranks (1-based), as scipy.stats.rankdata(method='average')."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman's rho: Pearson correlation of the average ranks (scipy's definition);
    uses scipy when it is installed and the numpy version above otherwise."""
    try:
        from scipy.stats import spearmanr
        r = spearmanr(a, b)
        return float(r.statistic if hasattr(r, "statistic") else r.correlation)
    except ImportError:
        ra, rb = _rankdata(a), _rankdata(b)
        ra, rb = ra - ra.mean(), rb - rb.mean()
        return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def jround(x):
    return json.loads(json.dumps(x))


# ---------------------------------------------------------------------------------------------
# the per-set analysis (ALL items)
# ---------------------------------------------------------------------------------------------


def analyse_set(two, OFF, PW, idx, pools, gt_vids):
    """OFF[(p, f)]: (C, M, n); PW[(p, f)]: (C, M, n, W), over the set's n items."""
    cov200 = np.array([idx["COV"][c] for c in GRID_COV200])
    n = len(two)
    folds = np.stack([make_folds(two, FOLD_SEED + r) for r in range(N_REP)])
    bal = [[int((folds[0] == k).sum()) for k in range(K_FOLDS)],
           [int(two[folds[0] == k].sum()) for k in range(K_FOLDS)]]
    SR = {"n": n, "two_scene": int(two.sum()), "fold_sizes_primary": bal, "pools": {}}
    for p in [p for p in POOLS if (p, "COV") in OFF]:
        PR = {}
        off = {f: OFF[(p, f)] for f in FAMS}
        pw = {f: PW[(p, f)] for f in FAMS}
        off["COV200"] = off["COV"][cov200]
        pw["COV200"] = pw["COV"][cov200]
        fams_all = FAMS + ("COV200",)
        grid_of = dict(GRIDS)
        grid_of["COV200"] = [G_COV[i] for i in cov200]
        chosen, oof = {}, {}
        for mi, m in enumerate(MODELS):
            chosen[m], oof[m] = {}, {}
            for f in fams_all:
                ch = np.zeros((N_REP, K_FOLDS), dtype=np.int64)
                oo = np.zeros((N_REP, n))
                for r in range(N_REP):
                    ch[r], oo[r] = nested_one(off[f][:, mi, :], folds[r])
                chosen[m][f], oof[m][f] = ch, oo
        fz = {"COV": idx["COV"][FROZEN["COV"]], "HYB": idx["HYB"][FROZEN["HYB"]]}
        for fam in BELIEF_FREE:
            fz[f"{fam}_E6"] = idx[fam][E6_FROZEN[p][fam]]

        def fam_frozen(key, mi):
            return off[key.split("_")[0]][fz[key], mi, :]

        for mi, m in enumerate(MODELS):
            M_ = {}
            hyb0 = fam_frozen("HYB", mi)
            avg = {f: oof[m][f].mean(axis=0) for f in fams_all}
            table = {}
            for f in fams_all:
                per_part = oof[m][f].mean(axis=1)
                insample = off[f][:, mi, :].mean(axis=1)
                table[f] = {
                    "oof_mean": float(avg[f].mean()),
                    "partition_mean_sd_min_max": [float(per_part.mean()), float(per_part.std(ddof=1)),
                                                  float(per_part.min()), float(per_part.max())],
                    "primary_partition_mean": float(per_part[0]),
                    "rel_vs_frozen_HYB": float(avg[f].mean() / hyb0.mean() - 1.0),
                    "insample_best_cell": list(grid_of[f][int(np.argmax(insample))]),
                    "insample_best_mean": float(insample.max()),
                    "optimism_insample_minus_oof": float(insample.max() - avg[f].mean()),
                }
            for key in fz:
                x = fam_frozen(key, mi)
                table[f"frozen_{key}"] = {"mean": float(x.mean()),
                                          "rel_vs_frozen_HYB": float(x.mean() / hyb0.mean() - 1.0)}
            M_["families"] = table
            bf_best = max(BELIEF_FREE, key=lambda f: avg[f].mean())
            M_["best_belief_free"] = bf_best
            cmp_ = {}
            for f in ("HYB", "UG", "NMS", "FLAT"):
                c = contrast(avg["COV"], avg[f])
                c["primary_partition"] = contrast(oof[m]["COV"][0], oof[m][f][0])
                dpp = oof[m]["COV"].mean(axis=1) - oof[m][f].mean(axis=1)
                c["per_partition_diff_mean_sd_min_max"] = [float(dpp.mean()), float(dpp.std(ddof=1)),
                                                           float(dpp.min()), float(dpp.max())]
                c["partitions_positive"] = int((dpp > 0).sum())
                fold_d = []
                for k in range(K_FOLDS):
                    te = folds[0] == k
                    fold_d.append(float(oof[m]["COV"][0][te].mean() - oof[m][f][0][te].mean()))
                c["primary_across_fold_t"] = tci(fold_d)
                cmp_[f"COV-{f}"] = c
            cmp_["COV-frozenCOV"] = contrast(avg["COV"], fam_frozen("COV", mi))
            cmp_["HYB-frozenHYB"] = contrast(avg["HYB"], hyb0)
            cmp_["COV-frozenHYB"] = contrast(avg["COV"], hyb0)
            cmp_["frozenCOV-frozenHYB"] = contrast(fam_frozen("COV", mi), hyb0)
            for f in BELIEF_FREE:
                cmp_[f"{f}-frozenHYB"] = contrast(avg[f], hyb0)
                cmp_[f"frozenCOV-{f}"] = contrast(fam_frozen("COV", mi), avg[f])
            cmp_["COV200-bestBF"] = contrast(avg["COV200"], avg[bf_best])
            cmp_["COV-COV200"] = contrast(avg["COV"], avg["COV200"])
            M_["contrasts"] = cmp_
            sel = {}
            for f in fams_all:
                cnt = {}
                for c in chosen[m][f].ravel():
                    cnt[int(c)] = cnt.get(int(c), 0) + 1
                top = sorted(cnt.items(), key=lambda kv: -kv[1])
                sel[f] = {"distinct": len(cnt),
                          "top": [{"cell": list(grid_of[f][c]), "times": t} for c, t in top[:8]],
                          "primary": [list(grid_of[f][c]) for c in chosen[m][f][0]]}
            M_["selections"] = sel
            PR[m] = M_

        X = {}
        for mi, m in enumerate(MODELS):
            row = {}
            vals = {}
            for f in FAMS:
                g_ = gather(off[f][:, mi, :], chosen["DEU"][f], folds).mean(axis=0)
                vals[f] = g_
                row[f] = float(g_.mean())
            bf = max(BELIEF_FREE, key=lambda f: vals[f].mean())
            row["best_BF"] = bf
            row["COV-bestBF"] = contrast(vals["COV"], vals[bf])
            row["COV-UG"] = contrast(vals["COV"], vals["UG"])
            row["COV-NMS"] = contrast(vals["COV"], vals["NMS"])
            X[m] = row
        PR["cross_model_selected_DEU"] = X
        W_ = {}
        for wi, h in enumerate(WINS):
            row = {"fixed_cells": {}, "retuned": {}}
            for mode in ("fixed_cells", "retuned"):
                vals = {}
                for f in FAMS:
                    t_ = pw[f][:, 0, :, wi]
                    if mode == "fixed_cells":
                        g_ = gather(t_, chosen["DEU"][f], folds).mean(axis=0)
                    else:
                        ch = np.zeros((N_REP, K_FOLDS), dtype=np.int64)
                        for r in range(N_REP):
                            ch[r], _ = nested_one(t_, folds[r])
                        g_ = gather(t_, ch, folds).mean(axis=0)
                    vals[f] = g_
                    row[mode][f] = float(g_.mean())
                bf = max(BELIEF_FREE, key=lambda f: vals[f].mean())
                row[mode]["best_BF"] = bf
                row[mode]["COV-bestBF"] = contrast(vals["COV"], vals[bf])
                row[mode]["COV-UG"] = contrast(vals["COV"], vals["UG"])
                row[mode]["COV-NMS"] = contrast(vals["COV"], vals["NMS"])
            W_[f"+-{h}"] = row
        PR["windows_DEU"] = W_

        rng = np.random.default_rng(MATCH_SEED)
        dm = []
        bf_counts = {}
        for _b in range(MATCH_DRAWS):
            sub = {f: np.sort(rng.choice(len(GRIDS[f]), size=min(MATCH_SIZE, len(GRIDS[f])), replace=False))
                   for f in FAMS}
            av = {}
            for f in FAMS:
                oo = np.zeros((N_REP, n))
                for r in range(N_REP):
                    _c, oo[r] = nested_one(off[f][sub[f], 0, :], folds[r])
                av[f] = oo.mean(axis=0)
            bf = max(BELIEF_FREE, key=lambda f: av[f].mean())
            bf_counts[bf] = bf_counts.get(bf, 0) + 1
            dm.append(float(av["COV"].mean() - av[bf].mean()))
        PR["grid_size_matched_DEU"] = {"size": MATCH_SIZE, "draws": MATCH_DRAWS,
                                       "COV_minus_bestBF": [float(np.mean(dm)), float(np.min(dm)), float(np.max(dm))],
                                       "positive_draws": int(sum(1 for x in dm if x > 0)),
                                       "best_BF_counts": bf_counts}
        marg = {}
        for mi, m in enumerate(("DEU", "SAU_NEO")):
            hyb0 = fam_frozen("HYB", mi)
            mm = {}
            for j, name in enumerate(("tau", "sigma", "h", "g")):
                vals_ = sorted({c[j] for c in G_COV})
                mm[name] = {str(v): float(off["COV"][[i for i, c in enumerate(G_COV) if c[j] == v], mi, :].mean()
                                          / hyb0.mean() - 1.0) for v in vals_}
            marg[m] = mm
        PR["COV_marginals_rel_vs_frozen_HYB"] = marg

        sim = {}
        rows_c = {}
        for f in ("COV", "UG", "NMS", "HYB"):
            rows_c[f] = [rows_for(f, GRIDS[f][chosen["DEU"][f][0][folds[0][qi]]], pools[p][qi]) for qi in range(n)]
        for f in ("UG", "NMS", "HYB"):
            ident, same_vid, near6, top1_c, top1_o = [], [], [], [], []
            for qi in range(n):
                rc, ro = rows_c["COV"][qi], rows_c[f][qi]
                so = set(ro)
                ident.append(sum(1 for x in rc if x in so) / len(rc))
                vo = {v for v, _ in ro}
                same_vid.append(sum(1 for v, _ in rc if v in vo) / len(rc))
                byv = {}
                for v, fr in ro:
                    byv.setdefault(v, []).append(fr)
                near6.append(sum(1 for v, fr in rc if v in byv and min(abs(fr - g) for g in byv[v]) <= 6) / len(rc))
                v1 = pools[p][qi][0].video_id
                top1_c.append(sum(1 for v, _ in rc if v == v1) / len(rc))
                top1_o.append(sum(1 for v, _ in ro if v == v1) / len(ro))
            sc_c = oof["DEU"]["COV"][0]
            sc_o = oof["DEU"][f][0]
            both = (sc_c > 0) | (sc_o > 0)
            sim[f"COV_vs_{f}"] = {
                "identical_lines_share_mean": float(np.mean(ident)),
                "identical_lines_share_median": float(np.median(ident)),
                "same_video_lines_share_mean": float(np.mean(same_vid)),
                "within_6_frames_same_video_share_mean": float(np.mean(near6)),
                "COV_lines_in_rank1_video": float(np.mean(top1_c)),
                f"{f}_lines_in_rank1_video": float(np.mean(top1_o)),
                "spearman_per_query_scores": spearman(sc_c, sc_o),
                "spearman_on_queries_where_either_scores": spearman(sc_c[both], sc_o[both]),
                "n_either_scores": int(both.sum()),
                "queries_equal_score": int((np.abs(sc_c - sc_o) < 1e-12).sum()),
                "queries_COV_higher": int((sc_c - sc_o > 1e-12).sum()),
                "queries_other_higher": int((sc_o - sc_c > 1e-12).sum()),
            }
        sim["distinct_videos_per_query"] = {f: float(np.mean([len({v for v, _ in r}) for r in rows_c[f]]))
                                            for f in rows_c}
        sim["correct_video_in_100"] = {f: int(sum(any(v == gt_vids[qi] for v, _ in r)
                                                  for qi, r in enumerate(rows_c[f]))) for f in rows_c}
        gaps = []
        for r in rows_c["COV"]:
            v1 = r[0][0]
            fr = sorted(f_ for v, f_ in r if v == v1)
            if len(fr) > 2:
                gaps.extend(np.diff(fr).tolist())
        gaps = np.array(gaps)
        sim["COV_gap_in_first_video_frames"] = {
            "median": float(np.median(gaps)), "p10": float(np.percentile(gaps, 10)),
            "p90": float(np.percentile(gaps, 90)),
            "share_eq_15": float(np.mean(gaps == 15)), "share_le_20": float(np.mean(gaps <= 20))}
        PR["similarity_primary_partition_DEU"] = sim
        SR["pools"][p] = PR
    return SR


# ---------------------------------------------------------------------------------------------
# the held-out half: every family tuned once on TUNE, scored on TEST
# ---------------------------------------------------------------------------------------------


def analyse_test(tu, te, OFF, PW, idx):
    cov200 = np.array([idx["COV"][c] for c in GRID_COV200])
    tu = np.asarray(tu)
    te = np.asarray(te)
    out = {"n_tune": int(tu.size), "n_test": int(te.size), "pools": {}}
    for p in [p for p in POOLS if (p, "COV") in OFF]:
        PR = {}
        off = {f: OFF[(p, f)] for f in FAMS}
        pw = {f: PW[(p, f)] for f in FAMS}
        off["COV200"] = off["COV"][cov200]
        pw["COV200"] = pw["COV"][cov200]
        fams_all = FAMS + ("COV200",)
        grid_of = dict(GRIDS)
        grid_of["COV200"] = [G_COV[i] for i in cov200]
        fz = {"COV": idx["COV"][FROZEN["COV"]], "HYB": idx["HYB"][FROZEN["HYB"]]}
        for fam in BELIEF_FREE:
            fz[f"{fam}_E6"] = idx[fam][E6_FROZEN[p][fam]]
        sel = {}
        for mi, m in enumerate(MODELS):
            sel[m] = {f: int(np.argmax(off[f][:, mi, tu].mean(axis=1))) for f in fams_all}
        for mi, m in enumerate(MODELS):
            M_ = {}
            hyb0 = off["HYB"][fz["HYB"], mi, te]
            vals = {f: off[f][sel[m][f], mi, te] for f in fams_all}
            table = {}
            for f in fams_all:
                test_all = off[f][:, mi, te].mean(axis=1)
                table[f] = {"test_mean": float(vals[f].mean()),
                            "tune_mean_selected": float(off[f][sel[m][f], mi, tu].mean()),
                            "selected_cell": list(grid_of[f][sel[m][f]]),
                            "rel_vs_frozen_HYB": float(vals[f].mean() / hyb0.mean() - 1.0),
                            "test_oracle_cell": list(grid_of[f][int(np.argmax(test_all))]),
                            "test_oracle_mean": float(test_all.max())}
            frozen_vals = {}
            for key in fz:
                x = off[key.split("_")[0]][fz[key], mi, te]
                frozen_vals[key] = x
                table[f"frozen_{key}"] = {"mean": float(x.mean()),
                                          "rel_vs_frozen_HYB": float(x.mean() / hyb0.mean() - 1.0)}
            M_["families"] = table
            bf = max(BELIEF_FREE, key=lambda f: vals[f].mean())
            M_["best_belief_free"] = bf
            cmp_ = {}
            for f in ("HYB", "UG", "NMS", "FLAT"):
                cmp_[f"COV-{f}"] = contrast(vals["COV"], vals[f])
            cmp_["COV-frozenCOV"] = contrast(vals["COV"], frozen_vals["COV"])
            cmp_["HYB-frozenHYB"] = contrast(vals["HYB"], hyb0)
            cmp_["COV-frozenHYB"] = contrast(vals["COV"], hyb0)
            cmp_["frozenCOV-frozenHYB"] = contrast(frozen_vals["COV"], hyb0)
            for f in BELIEF_FREE:
                cmp_[f"{f}-frozenHYB"] = contrast(vals[f], hyb0)
                cmp_[f"frozenCOV-{f}"] = contrast(frozen_vals["COV"], vals[f])
            cmp_["COV200-bestBF"] = contrast(vals["COV200"], vals[bf])
            cmp_["COV-COV200"] = contrast(vals["COV"], vals["COV200"])
            cmp_["COV200-frozenHYB"] = contrast(vals["COV200"], hyb0)
            M_["contrasts"] = cmp_
            PR[m] = M_
        X = {}
        for mi, m in enumerate(MODELS):
            row = {}
            vals = {f: off[f][sel["DEU"][f], mi, te] for f in FAMS}
            for f in FAMS:
                row[f] = float(vals[f].mean())
            bf = max(BELIEF_FREE, key=lambda f: vals[f].mean())
            row["best_BF"] = bf
            row["COV-bestBF"] = contrast(vals["COV"], vals[bf])
            row["COV-UG"] = contrast(vals["COV"], vals["UG"])
            row["COV-NMS"] = contrast(vals["COV"], vals["NMS"])
            X[m] = row
        PR["cross_model_selected_DEU"] = X
        W_ = {}
        for wi, h in enumerate(WINS):
            row = {"fixed_cells": {}, "retuned": {}}
            for mode in ("fixed_cells", "retuned"):
                vals = {}
                for f in FAMS:
                    t_ = pw[f][:, 0, :, wi]
                    ci_ = sel["DEU"][f] if mode == "fixed_cells" else int(np.argmax(t_[:, tu].mean(axis=1)))
                    vals[f] = t_[ci_, te]
                    row[mode][f] = float(vals[f].mean())
                bf = max(BELIEF_FREE, key=lambda f: vals[f].mean())
                row[mode]["best_BF"] = bf
                row[mode]["COV-bestBF"] = contrast(vals["COV"], vals[bf])
                row[mode]["COV-UG"] = contrast(vals["COV"], vals["UG"])
                row[mode]["COV-NMS"] = contrast(vals["COV"], vals["NMS"])
            W_[f"+-{h}"] = row
        PR["windows_DEU"] = W_
        out["pools"][p] = PR
    return out


def verdict_of(SR):
    """The pre-registered rule of the fair comparison: PASS iff tuned COV beats the best
    belief-free family on the raw pool under DEU with a CI above 0 AND the sign holds under
    SAU_NEO; MIXED / FAIL otherwise (see the original script)."""
    def cond_a(pool):
        c = SR["pools"][pool]["DEU"]
        bf = c["best_belief_free"]
        k = c["contrasts"][f"COV-{bf}"]
        return bf, k, bool(k["diff"] > 0 and k["ci95"][0] > 0)

    def cond_b(pool):
        c = SR["pools"][pool]["SAU_NEO"]
        bf = c["best_belief_free"]
        k = c["contrasts"][f"COV-{bf}"]
        return bf, k, bool(k["diff"] > 0)

    bfa, ka, A = cond_a("raw")
    bfb, kb, B = cond_b("raw")
    _pa, _kpa, Ap = cond_a("prod")
    _pb, _kpb, Bp = cond_b("prod")
    if A and B:
        verdict = "PASS"
    elif (A and not B) or ((not A) and Ap and Bp):
        verdict = "MIXED"
    else:
        verdict = "FAIL"
        if ka["ci95"][1] < 0:
            verdict = "FAIL (REVERSED)"
    return {"verdict": verdict,
            "a_raw_DEU": {"best_BF": bfa, "diff": ka["diff"], "ci95": ka["ci95"],
                          "rel": ka["rel"], "perm_p": ka["perm_p"], "holds": A},
            "b_raw_SAU_NEO": {"best_BF": bfb, "diff": kb["diff"], "ci95": kb["ci95"],
                              "rel": kb["rel"], "perm_p": kb["perm_p"], "holds": B},
            "prod_a_holds": Ap, "prod_b_holds": Bp,
            "prod_a": {"best_BF": _pa, "diff": _kpa["diff"], "ci95": _kpa["ci95"]},
            "prod_b": {"best_BF": _pb, "diff": _kpb["diff"], "ci95": _kpb["ci95"]}}


# ---------------------------------------------------------------------------------------------
# data from the release
# ---------------------------------------------------------------------------------------------


def load_release(policy="C_EITHER_ANCHOR"):
    """-> clean items, primary positions, two-scene label (over the set), TUNE/TEST (set
    positions), pools {raw, prod} over the set, keyframe index."""
    from reproduce import cands_of, load_jsonl, load_keyframes
    items = load_jsonl(REL / "benchmark" / "items.jsonl")
    clean = [it for it in items if it["in_clean_set"]]
    raw_by = {r["item_id"]: r for r in load_jsonl(REL / "pools" / "b174_raw.jsonl")}
    sb_by = {r["item_id"]: r for r in load_jsonl(REL / "pools" / "scene_b.jsonl")}
    sp = [q for q, it in enumerate(clean) if it["policies"][policy]]
    two = np.array([bool(clean[q]["two_scene_labels"]["both_judges_image_evidence"]) for q in sp])
    split = [clean[q]["splits"]["OWN"][policy] for q in sp]
    tu = [i for i, s in enumerate(split) if s == "TUNE"]
    te = [i for i, s in enumerate(split) if s == "TEST"]
    raw = [cands_of(raw_by[clean[q]["item_id"]]) for q in sp]
    prod = [AL.build_production_pool(r, sb_by[clean[q]["item_id"]]) for r, q in zip(raw, sp)]
    return clean, sp, two, tu, te, {"raw": raw, "prod": prod}, load_keyframes()


def sweep(pools, gt_vids, truths, pool_names, workers, log=print):
    import multiprocessing as mp
    jobs = [(p, fam, ci) for p in pool_names for fam in FAMS for ci in range(len(GRIDS[fam]))]
    OFF = {(p, f): np.zeros((len(GRIDS[f]), len(MODELS), len(gt_vids))) for p in pool_names for f in FAMS}
    PW = {(p, f): np.zeros((len(GRIDS[f]), len(MODELS), len(gt_vids), len(WINS))) for p in pool_names for f in FAMS}
    t0 = time.time()
    done = 0
    sub_pools = {p: pools[p] for p in pool_names}
    if workers <= 1:
        _init(sub_pools, gt_vids, truths)
        it = map(_job, jobs)
        ctx = None
    else:
        ctx = mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(sub_pools, gt_vids, truths),
                                           maxtasksperchild=200)
        it = ctx.imap_unordered(_job, jobs, chunksize=2)
    try:
        for p, fam, ci, off, pw in it:
            OFF[(p, fam)][ci] = off
            PW[(p, fam)][ci] = pw
            done += 1
            if done % 100 == 0 or done == len(jobs):
                el = time.time() - t0
                log(f"   sweep {done}/{len(jobs)} cells, {el:.0f}s, ~{el / done * (len(jobs) - done):.0f}s left")
    finally:
        if ctx is not None:
            ctx.close()
            ctx.join()
    return OFF, PW


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--pools", default="raw,prod", help="raw,prod (default) or raw")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    pool_names = tuple(p for p in args.pools.split(",") if p)
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clean, sp, two, tu, te, pools, kf = load_release()
    print(f"primary set: {len(sp)} items, {int(two.sum())} two-scene; TUNE/TEST {len(tu)}/{len(te)}; "
          f"pools {pool_names}; {args.workers} workers", flush=True)
    truths_full = [P.draw_moments_model(P.SEED_ROOT, clean, kf, m) for m in MODELS]
    truths = [t[:, sp, :] for t in truths_full]
    gt_vids = [clean[q]["video_id"] for q in sp]
    # gate: the shipped coverage and ranking-based cells equal the allocators of the paper
    for p in pool_names:
        bad = sum(rows_for("COV", FROZEN["COV"], c) != AL.alloc_B(c) for c in pools[p])
        bad += sum(rows_for("HYB", FROZEN["HYB"], c) != AL.alloc_A(c) for c in pools[p])
        print(f"   gate {p}: frozen COV / HYB cells = allocators B / A on every item: {bad == 0}", flush=True)
        if bad:
            return 1
    OFF, PW = sweep(pools, gt_vids, truths, pool_names, args.workers,
                    log=lambda s: print(s, flush=True))
    idx = {fam: {cell: i for i, cell in enumerate(GRIDS[fam])} for fam in FAMS}
    res = {"ALL": jround(analyse_set(two, OFF, PW, idx, pools, gt_vids)),
           "TEST": jround(analyse_test(tu, te, OFF, PW, idx))}
    if set(pool_names) == set(POOLS):
        res["VERDICT_ALL"] = jround(verdict_of(res["ALL"]))
        res["VERDICT_TEST_confirmation"] = jround(verdict_of(
            {"pools": {p: {m: {"best_belief_free": res["TEST"]["pools"][p][m]["best_belief_free"],
                               "contrasts": res["TEST"]["pools"][p][m]["contrasts"]}
                           for m in ("DEU", "SAU_NEO")} for p in POOLS}}))
    (out / "fair_comparison.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    exp = json.loads((REL / "expected" / "fair_comparison.json").read_text(encoding="utf-8"))
    mism, n_leaf = T.compare_subset(res, exp, tol=1e-9)
    ok = not mism and n_leaf > 0
    print(f"  [{'PASS' if ok else 'FAIL'}] fair comparison = expected/fair_comparison.json: "
          f"{n_leaf} numbers compared, {len(mism)} mismatches" + (f", first: {mism[0]}" if mism else ""))
    a = res["ALL"]["pools"]["raw"]
    print("\nALL items, raw pool (out-of-fold official score; COV minus family [95% CI], sign-flip p):")
    for m in ("DEU", "SAU_NEO"):
        fam = a[m]["families"]
        print(f"  {m}: " + ", ".join(f"{f} {fam[f]['oof_mean']:.3f}" for f in FAMS))
        for f in ("HYB", "UG", "NMS", "FLAT"):
            c = a[m]["contrasts"][f"COV-{f}"]
            print(f"     COV-{f}: {c['diff']:+.3f} [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], p {c['perm_p']:.3f}")
    if "VERDICT_ALL" in res:
        print(f"verdict of the pre-registered rule (ALL): {res['VERDICT_ALL']['verdict']}")
    print(f"FAIR COMPARISON: {'PASS' if ok else 'FAIL'} in {time.time() - t0:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
