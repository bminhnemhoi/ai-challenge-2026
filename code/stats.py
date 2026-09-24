"""Statistics used by the paper beyond the paired percentile bootstrap and the
sign-flip test of ``protocol.py``. Every function is a copy of the one that
produced the paper's numbers; each keeps its own seed convention, because the
printed intervals depend on it.

  bca_contrast, perm_p_exact, holm .... scripts/e5_stats_chinh.py (contrast,
                                        _bca_bounds, perm_p, holm): the BCa
                                        interval and the Holm-adjusted p of the
                                        whole-set allocation gain
  boot_query, boot_ratio .............. scripts/r1b_thang_chinh_sach.py (the
                                        robustness ladder: two-scene gap and
                                        the keyframe-authored / segment ratio)
  perm_p_606, contrast_fair ........... scripts/e6_control_allocators.py::perm_p
                                        and scripts/e16_so_cong_bang.py::contrast
                                        (the fair comparison of tuned allocators)
  paired_boot_e1, sign_perm_e1 ........ scripts/e1_exact_chinh.py (exact optimum
                                        and oracles)

scipy is used only for the normal quantile in the BCa interval; without it the
same bisection fallback as in the original script is used (it agrees to about
1e-12 in the quantile, which can move a BCa bound by one bootstrap order
statistic in rare cases).
"""

from __future__ import annotations

import math
import zlib

import numpy as np

import protocol as P

try:
    from scipy.stats import norm as _norm

    def _ppf(p):
        return float(_norm.ppf(p))

    def _cdf(z):
        return float(_norm.cdf(z))

    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

    def _cdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _ppf(p):
        lo, hi = -12.0, 12.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if _cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# e5: BCa interval, exact/Monte-Carlo sign-flip p, Holm
# ---------------------------------------------------------------------------

PERM_EXACT_MAX = 20


def perm_p_exact(diff, n_perm=P.N_PERM, seed=20260924):
    """Two-sided paired sign-flip p over the nonzero differences; exact when at
    most 20 are nonzero, Monte Carlo (``n_perm`` flips) otherwise."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    obs = float(diff.mean())
    nz = diff[np.abs(diff) > 1e-12]
    m = len(nz)
    tol = 1e-12
    if m == 0:
        return {"p": 1.0, "exact": True, "n_nonzero": 0, "n_perm": 0, "obs": obs}
    if m <= PERM_EXACT_MAX:
        total = 1 << m
        ge = 0
        signs_base = np.array([1.0, -1.0])
        chunk = 1 << min(m, 16)
        for start in range(0, total, chunk):
            idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
            bits = ((idx[:, None] >> np.arange(m)[None, :]) & 1)
            stats = (signs_base[bits] * nz[None, :]).sum(axis=1) / n
            ge += int((np.abs(stats) >= abs(obs) - tol).sum())
        return {"p": ge / total, "exact": True, "n_nonzero": m, "n_perm": total, "obs": obs}
    rng = np.random.default_rng(seed)
    ge = 0
    done = 0
    while done < n_perm:
        b = min(2000, n_perm - done)
        s = rng.choice(np.array([-1.0, 1.0]), size=(b, m))
        stats = (s * nz[None, :]).sum(axis=1) / n
        ge += int((np.abs(stats) >= abs(obs) - tol).sum())
        done += b
    return {"p": (1 + ge) / (1 + n_perm), "exact": False, "n_nonzero": m, "n_perm": n_perm, "obs": obs}


def _bca_bounds(boot, theta, jack, alpha=0.05):
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    if len(boot) < 100:
        return [float("nan"), float("nan")], float("nan"), float("nan")
    frac = float((boot < theta).mean())
    frac = min(max(frac, 1.0 / (2 * len(boot))), 1.0 - 1.0 / (2 * len(boot)))
    z0 = _ppf(frac)
    jack = np.asarray(jack, dtype=float)
    u = jack.mean() - jack
    den = 6.0 * (float((u ** 2).sum()) ** 1.5)
    a = float((u ** 3).sum()) / den if den > 0 else 0.0
    out = []
    for z_a in (_ppf(alpha / 2), _ppf(1 - alpha / 2)):
        z = z0 + (z0 + z_a) / max(1e-12, (1.0 - a * (z0 + z_a)))
        out.append(float(np.percentile(boot, 100.0 * _cdf(z))))
    return [min(out), max(out)], z0, a


def slice_seed(key: str) -> int:
    """The per-contrast bootstrap seed: 4242 + crc32(key) mod 1000003."""
    return P.BOOT_SEED + int(zlib.crc32(key.encode("utf-8")) % 1000003)


def bca_contrast(xb, xa, key):
    """mean(B) - mean(X) and mean(B)/mean(X) - 1 with percentile and BCa
    intervals (4,000 resamples at the per-contrast seed) and the sign-flip p at
    the same seed. ``key`` is the contrast name, e.g.
    'C_EITHER_ANCHOR/R1B/ALL/all/raw/DEU/B-A'."""
    xb = np.asarray(xb, dtype=float)
    xa = np.asarray(xa, dtype=float)
    n = len(xa)
    d = xb - xa
    mb, ma = float(xb.mean()), float(xa.mean())
    theta = mb - ma
    rel = theta / ma if ma > 0 else float("nan")
    res = {"key": key, "n": n, "mean_B": mb, "mean_other": ma, "diff": theta, "rel": rel,
           "improved": int((d > 1e-9).sum()), "worsened": int((d < -1e-9).sum()),
           "unchanged": int((np.abs(d) <= 1e-9).sum())}
    jack_d = np.empty(n)
    jack_r = np.empty(n)
    for i in range(n):
        sel = np.ones(n, dtype=bool)
        sel[i] = False
        a2, b2 = float(xa[sel].mean()), float(xb[sel].mean())
        jack_d[i] = b2 - a2
        jack_r[i] = (b2 / a2 - 1.0) if a2 > 0 else np.nan
    sd = slice_seed(key)
    idx = np.random.default_rng(sd).integers(0, n, size=(P.N_BOOT, n))
    bb, bas = xb[idx].mean(axis=1), xa[idx].mean(axis=1)
    bd = bb - bas
    with np.errstate(divide="ignore", invalid="ignore"):
        br = np.where(bas > 0, bb / bas - 1.0, np.nan)
    pl, ph = np.percentile(bd, [2.5, 97.5])
    rpl, rph = np.nanpercentile(br, [2.5, 97.5])
    bca_d, _z0d, _ad = _bca_bounds(bd, theta, jack_d)
    bca_r, _z0r, _ar = _bca_bounds(br[np.isfinite(br)], rel, jack_r[np.isfinite(jack_r)])
    res["boot"] = {"seed": sd, "pct_ci": [float(pl), float(ph)], "pct_rel_ci": [float(rpl), float(rph)],
                   "bca_ci": bca_d, "bca_rel_ci": bca_r, "p_le_0": float((bd <= 0).mean())}
    res["perm"] = perm_p_exact(d, seed=sd)
    return res


def holm(pvals: dict):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


# ---------------------------------------------------------------------------
# r1b: query bootstrap of set functionals, ratio of two disjoint sets
# ---------------------------------------------------------------------------


def boot_ci(vals):
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def boot_query(per_q: dict, members, fns: dict, n_boot=P.N_BOOT, seed=P.BOOT_SEED):
    """Unpaired nonparametric query bootstrap of arbitrary functionals of a set."""
    members = np.asarray(members, dtype=np.int64)
    n = len(members)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    cols = {k: np.asarray(v)[members] for k, v in per_q.items()}
    acc = {k: np.empty(n_boot) for k in fns}
    for b in range(n_boot):
        take = idx[b]
        sub = {k: v[take] for k, v in cols.items()}
        for k, f in fns.items():
            acc[k][b] = f(sub)
    out = {}
    for k, f in fns.items():
        out[k] = {"point": float(f(cols)), "ci95": boot_ci(acc[k]), "p_le_0": float(np.mean(acc[k] <= 0.0))}
    return out


def boot_ratio(num, den, n_boot=P.N_BOOT, seed=P.BOOT_SEED):
    """Unpaired percentile bootstrap of mean(num) / mean(den) -- two disjoint sets."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    rng = np.random.default_rng(seed)
    a = num[rng.integers(0, num.size, size=(n_boot, num.size))].mean(axis=1)
    b = den[rng.integers(0, den.size, size=(n_boot, den.size))].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b > 0, a / b, np.nan)
    return {"ratio": float(num.mean() / den.mean()), "ci95": boot_ci(r),
            "p_le_1": float(np.nanmean(r <= 1.0)), "p_lt_1_5": float(np.nanmean(r < 1.5)),
            "n_num": int(num.size), "n_den": int(den.size)}


# ---------------------------------------------------------------------------
# e6 / e16: sign-flip at seed 606 and the contrast of the fair comparison
# ---------------------------------------------------------------------------

PERM_SEED_606 = 606


def perm_p_606(xb, xa, n_perm=P.N_PERM, seed=PERM_SEED_606):
    """Two-sided paired sign-flip permutation test on the per-query differences."""
    d = np.asarray(xb, dtype=float) - np.asarray(xa, dtype=float)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, size=(n_perm, d.size)) * 2 - 1
    null = np.abs((s * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-15) + 1) / (n_perm + 1))


def contrast_fair(xb, xa):
    """Paired bootstrap (protocol.paired, seed 4242) + sign-flip p (seed 606)."""
    bi = P.boot_indices(len(xb))
    p = P.paired(np.asarray(xb), np.asarray(xa), bi)
    p["perm_p"] = perm_p_606(xb, xa)
    return p


# ---------------------------------------------------------------------------
# e1: paired bootstrap and sign-flip of the exact-optimum and oracle contrasts
# ---------------------------------------------------------------------------


def sign_perm_e1(xb, xa, n_perm=10000):
    d = xb - xa
    rng = np.random.default_rng(P.BOOT_SEED)
    s = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(d)))
    null = (s * d).mean(axis=1)
    return float((np.abs(null) >= abs(d.mean()) - 1e-15).mean())


def paired_boot_e1(xb, xa):
    idx = np.random.default_rng(P.BOOT_SEED).integers(0, len(xa), size=(P.N_BOOT, len(xa)))
    d = xb[idx].mean(axis=1) - xa[idx].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(xa[idx].mean(axis=1) > 0, xb[idx].mean(axis=1) / xa[idx].mean(axis=1) - 1.0, np.nan)
    rlo, rhi = np.nanpercentile(rel, [2.5, 97.5])
    return {
        "n": int(len(xa)), "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
        "diff": float(xb.mean() - xa.mean()), "ci95": [float(lo), float(hi)],
        "rel": float(xb.mean() / xa.mean() - 1.0) if xa.mean() > 0 else None,
        "rel_ci95": [float(rlo), float(rhi)],
        "p_le_0": float((d <= 0).mean()), "p_ge_0": float((d >= 0).mean()),
        "improved": int(((xb - xa) > 1e-9).sum()), "worsened": int(((xb - xa) < -1e-9).sum()),
        "perm_p_two_sided": sign_perm_e1(xb, xa),
    }
