"""The scoring protocol of the paper: true-moment draws, the vectorised official
scorer, the TUNE/TEST split rule, the paired bootstrap and the sign-flip test.

Copied from the scripts that produced the paper's numbers:

  BUCKET, draw_moments ... scripts/experiment_phu_quet_luoi.py (BUCKET,
                           boc_khoanh_khac, cac_lan_boc)
  score_rows ............. scripts/e2_allocation_158.py::score_rows
  stratified_split ....... scripts/e2_allocation_158.py (split block) and
                           scripts/r1_thang_chinh_sach.py::policy_split
  boot_indices, paired ... scripts/e2_allocation_158.py
  signflip ............... scripts/r1_thang_chinh_sach.py::signflip
  draw_moments_model ..... scripts/cong_do_ben_mo_hinh_boc.py (the other three
                           moment models, used by the fair comparison)

The vectorised scorer is checked against the literal official scorer
(submission.final_score / submission.r_score_kis) by ``check_scorer_parity``.

Protocol constants (every number in the paper uses these):
  windows +-6, +-10, +-20 frames, averaged; 4 seed families x 48 draws;
  seed root 930000 (seed = root + 1000*family + draw); the true moment is
  drawn uniformly inside the keyframe cell of the anchor (DEU), not snapped
  to the keyframe; split seed 20260917; bootstrap 4,000 resamples, seed 4242;
  sign-flip test 10,000 flips, seed 20260924.
"""

from __future__ import annotations

import numpy as np

from submission import MAX_ROWS, RANK_THRESHOLDS, final_score, r_score_kis

SEED_ROOT = 930000
REPL_ROOT = 77000
FAMILIES = 4
DRAWS = 48
WINDOWS = (6, 10, 20)
SPLIT_SEED = 20260917
BOOT_SEED = 4242
N_BOOT = 4000
PERM_SEED = 20260924
N_PERM = 10_000
KS = tuple(RANK_THRESHOLDS)

#: value of the official Final Score when the first correct line is at rank r
#: (1-based); index 0 = no correct line
BUCKET = np.array(
    [0.0] + [sum(1 for k in RANK_THRESHOLDS if k >= r) / len(RANK_THRESHOLDS)
             for r in range(1, MAX_ROWS + 1)]
)


# ---------------------------------------------------------------------------
# true moments
# ---------------------------------------------------------------------------


def _draw_once(seed, items, kf):
    """One draw per item: uniform in the keyframe cell around the anchor."""
    rng = np.random.default_rng(seed)
    out = []
    for g in items:
        a = kf[g["video_id"]]
        i = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
        lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
        hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
        out.append(int(rng.integers(lo, max(lo + 1, hi))))
    return out


def draw_moments(root, items, kf, families=FAMILIES, draws=DRAWS):
    """int64 array (families, len(items), draws).

    The RNG stream runs over ``items`` in order, so the draws of one item
    depend on the list it is drawn in. The paper draws ONCE over the clean
    158-item list and slices every subset from it (common random numbers);
    ``redraw`` below re-runs the stream on a subset, which is the convention of
    the documented replication values 0.2048 / 0.2873 / 0.1233.
    """
    fam = []
    for s in range(families):
        d = [_draw_once(root + s * 1000 + t, items, kf) for t in range(draws)]
        fam.append(np.array(d, dtype=np.int64).T)  # (Q, draws)
    return np.stack(fam)


#: the four models of where the true moment lies inside the anchor's keyframe
#: cell [lo, hi): DEU uniform (the paper's default); SAU_NEO uniform on
#: [anchor, hi), i.e. after the cut that opens scene B; GAUSS12 normal with
#: sd 12 frames around the anchor, clipped to the cell; TAM_GIAC triangular
#: with its mode at the anchor. Copied from
#: scripts/cong_do_ben_mo_hinh_boc.py (boc_kieu, cac_lan_boc_kieu).
MOMENT_MODELS = ("DEU", "SAU_NEO", "GAUSS12", "TAM_GIAC")


def _cell(g, kf):
    a = kf[g["video_id"]]
    i = int(np.argmin(np.abs(a - int(g["frame_idx"]))))
    lo = (a[i] + a[i - 1]) // 2 if i > 0 else a[i] - 30
    hi = (a[i] + a[i + 1]) // 2 if i + 1 < len(a) else a[i] + 30
    return int(lo), int(max(lo + 1, hi)), int(a[i])


def _draw_once_model(seed, items, kf, model):
    rng = np.random.default_rng(seed)
    out = []
    for g in items:
        lo, hi, c = _cell(g, kf)
        if model == "DEU":
            x = int(rng.integers(lo, hi))
        elif model == "TAM_GIAC":
            x = int(np.clip(int(rng.triangular(lo, min(max(c, lo), hi - 1), hi)), lo, hi - 1))
        elif model == "GAUSS12":
            x = int(np.clip(int(c + rng.normal(0, 12)), lo, hi - 1))
        elif model == "SAU_NEO":
            a0 = min(max(c, lo), hi - 1)
            x = int(rng.integers(a0, max(a0 + 1, hi)))
        else:
            raise ValueError(model)
        out.append(x)
    return out


def draw_moments_model(root, items, kf, model, families=FAMILIES, draws=DRAWS):
    """Like ``draw_moments`` under one of MOMENT_MODELS (DEU gives identical draws)."""
    fam = []
    for s in range(families):
        d = [_draw_once_model(root + s * 1000 + t, items, kf, model) for t in range(draws)]
        fam.append(np.array(d, dtype=np.int64).T)
    return np.stack(fam)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score_rows(rows_of, items, truths):
    """bucket (Q, W, F): official score averaged over draws; rak (Q, W, K)."""
    F, Q, D = truths.shape
    bucket = np.zeros((Q, len(WINDOWS), F))
    rak = np.zeros((Q, len(WINDOWS), len(KS)))
    for q in range(Q):
        f = np.full(MAX_ROWS, -(10**9), dtype=np.int64)
        m = np.zeros(MAX_ROWS, dtype=bool)
        for i, (v, fr) in enumerate(rows_of[q][:MAX_ROWS]):
            f[i] = int(fr)
            m[i] = v == items[q]["video_id"]
        d = np.abs(f[None, None, :] - truths[:, q, :, None])  # (F, D, 100)
        for wi, h in enumerate(WINDOWS):
            hit = m[None, None, :] & (d <= h)
            co = hit.any(axis=2)
            rank = np.where(co, hit.argmax(axis=2) + 1, 0)
            bucket[q, wi, :] = BUCKET[rank].mean(axis=1)
            for ki, k in enumerate(KS):
                rak[q, wi, ki] = float(((rank > 0) & (rank <= k)).mean())
    return bucket, rak


def official_literal(rows_of, items, truths_fq):
    """The organisers' rule applied line by line with submission.py (slow)."""
    per_w = []
    for half in WINDOWS:
        tot, n = 0.0, 0
        for rows, g, ts in zip(rows_of, items, truths_fq):
            for t in ts:
                span = (int(t) - half, int(t) + half)
                tot += final_score([r_score_kis(v, f, g["video_id"], span) for v, f in rows])
                n += 1
        per_w.append(tot / n)
    return sum(per_w) / len(per_w)


def check_scorer_parity(rows_of, items, truths, n_items=12, n_draws=3):
    sub = list(range(min(n_items, len(items))))
    small = truths[:1, sub, :n_draws]
    ref = official_literal([rows_of[q] for q in sub], [items[q] for q in sub],
                           [small[0, j] for j in range(len(sub))])
    b, _ = score_rows([rows_of[q] for q in sub], [items[q] for q in sub], small)
    return abs(ref - float(b.mean()))


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def stratified_split(members, labels, seed=SPLIT_SEED):
    """Within each stratum (label False first, then True) permute with one RNG
    and send the first floor(n/2) to TUNE. ``members`` are positions in the
    clean list, ``labels`` a bool array indexed by position."""
    rng = np.random.default_rng(seed)
    mem = np.asarray(sorted(members), dtype=np.int64)
    lab = np.asarray(labels, dtype=bool)[mem]
    tune = []
    for mask in (~lab, lab):
        idx = mem[np.flatnonzero(mask)]
        idx = idx[rng.permutation(len(idx))]
        tune.extend(int(i) for i in idx[: len(idx) // 2])
    tune = sorted(tune)
    test = sorted(set(int(i) for i in mem) - set(tune))
    return tune, test


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def boot_indices(n):
    return np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n))


def paired(xb, xa, idx):
    """Paired query-level percentile bootstrap of mean(B) - mean(X) and B/X - 1."""
    mb, ma = xb[idx].mean(axis=1), xa[idx].mean(axis=1)
    d = mb - ma
    lo, hi = np.percentile(d, [2.5, 97.5])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(ma > 0, mb / ma - 1.0, np.nan)
    rlo, rhi = np.nanpercentile(rel, [2.5, 97.5])
    diff = xb - xa
    return {
        "n": int(len(xa)),
        "mean_other": float(xa.mean()),
        "mean_B": float(xb.mean()),
        "diff": float(xb.mean() - xa.mean()),
        "rel": float(xb.mean() / xa.mean() - 1.0) if xa.mean() > 0 else None,
        "ci95": [float(lo), float(hi)],
        "rel_ci95": [float(rlo), float(rhi)],
        "p_le_0": float((d <= 0).mean()),
        "improved": int((diff > 1e-9).sum()),
        "worsened": int((diff < -1e-9).sum()),
        "unchanged": int((np.abs(diff) <= 1e-9).sum()),
    }


def signflip(d, n_perm=N_PERM, seed=PERM_SEED):
    d = np.asarray(d, dtype=float)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    stat = (s * d[None, :]).mean(axis=1)
    return {
        "n": int(d.size), "mean_diff": obs, "n_perm": int(n_perm), "seed": int(seed),
        "p_two_sided": float((1 + int(np.sum(np.abs(stat) >= abs(obs) - 1e-15))) / (n_perm + 1)),
        "p_one_sided_greater": float((1 + int(np.sum(stat >= obs - 1e-15))) / (n_perm + 1)),
        "n_nonzero": int(np.sum(np.abs(d) > 1e-12)),
    }
