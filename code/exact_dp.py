"""E1-EXACT -- exact optimum of the SHIPPED coverage objective (DP + knapsack), and the
replication of the shipped greedy, as a library.  Nothing here writes to disk.

=============================================================================
PRE-REGISTRATION (written before any number of this experiment was computed;
                  quoted verbatim in soict2026/results/E1-EXACT_RESULTS.md)
=============================================================================
Object under test
-----------------
Cov(S) = the belief mass of the grid cells covered by the line set S, for the coverage
function the SHIPPED code actually computes (src/core/submission.py:455-502):
  * per video: truc = arange(lo, hi+1, g), lo = max(0, min candidate frame - 4*sigma),
    hi = max(lo, min(video_last_frame, max candidate frame + 4*sigma));
  * mass(t) = sum_i w_i exp(-(t-f_i)^2 / 2 sigma^2), w = softmax(score/tau) over the WHOLE
    pool, masses quantised by floor(m/1e-9 + 0.5)*1e-9;
  * a line at grid cell c covers cells [max(0, c-r), min(n_v-1, c+r)] with r = max(1, h//g);
  * shipped constants tau = 0.02, sigma = 30, h = 6, g = 5, budget B = 100.
OPT_k = max over line sets S with |S| <= k of Cov(S), over the union of all per-video grids.
G = the shipped greedy list (allocate_coverage_rows, ties and the already-used test included).

Decision rule (pre-registered, from PLAN.json .experiments[id=E1-EXACT])
-----------------------------------------------------------------------
PREDICTION: greedy realises at least 0.98 of OPT at every official cut-off
k in {1, 5, 20, 50, 100} and exactly 1.000 at k = 1; and the exact-DP allocator's official
score differs from greedy's by less than 0.002 absolute with a 95% CI containing zero.
If this holds, the Proposition and Corollary (main.tex:146-155) are deleted as a numbered
contribution and replaced by the measured gap plus a cited remark.

FALSIFICATION: if any cut-off shows greedy/OPT below 0.95 on 5% or more of items, or the
exact DP beats greedy on the official score by more than 0.005 absolute with a 95% CI
excluding zero, then the approximation-gap story is wrong: the paper ships the DP as the
allocator and keeps an optimality claim.  Either outcome is reportable; only the current
unmeasured bound is not.

Statistic
---------
Per item and per cut-off: ratio = Cov(G_k) / OPT_k (defined as 1.0 when OPT_k <= 1e-15).
Reported as min / median / mean / max and the full distribution, plus the count of items
below 1 - 1/e = 0.632.  Score comparison: the official score (mean of R@k over
k in {1,5,20,50,100}, windows +-6/10/20, 4 seed families x 48 non-snapping DEU draws at
seed root 930000, common random numbers over the clean 158) of the exact-DP allocator
against the shipped greedy, paired query-level bootstrap, 4000 resamples, seed 4242.

Gates that run before any ratio is read
---------------------------------------
V1 brute force: the per-video DP equals exhaustive enumeration on 400 randomised
   single-video instances and the DP+knapsack equals exhaustive enumeration on 120
   randomised multi-video instances (n <= 12 cells, k <= 4, r in {1,2}).  Any mismatch:
   STOP.
V2 greedy parity: the greedy replicated here emits, on every clean item and both pools,
   exactly the rows that data/cache_e2_v2/rows.json records for allocator B, up to the
   point where the shipped code hands over to the ranking-based tail filler.
V3 optimality sanity: OPT_k >= Cov(G_k) for every item and every k, and OPT is
   non-decreasing in k.
V4 scorer parity: the per-item scores recomputed here for allocator B equal
   data/cache_e2_v2/per_query_v2.json to < 1e-12.
=============================================================================

The DP (independent re-derivation of the one reviewer R4-T1 proposes).
Per video, cells 0..n-1, a line at centre c covers [lo(c), hi(c)] with lo(c)=max(0,c-r),
hi(c)=min(n-1,c+r).  Order the chosen centres increasingly and let
    A[j][c] = mass covered among cells 0..hi(c) by j lines whose largest centre is c.
Because c' < c implies lo(c') <= lo(c), the cells [lo(c), hi(c')] are already covered by
the predecessor itself, so the new mass is exactly W(max(lo(c), hi(c')+1), hi(c)) and
    A[1][c] = W(lo(c), hi(c)),
    A[j][c] = max_{c' < c} A[j-1][c'] + W(max(lo(c), hi(c')+1), hi(c)).
For c' <= c-(2r+1) the added mass is the constant W(lo(c), hi(c)), so a running prefix max
of A[j-1][.] serves; the remaining 2r predecessors are evaluated directly.  O(n*B*r) per
video.  Across videos the only coupling is the shared budget, so a (max,+) knapsack
K_v[b] = max_{0<=t<=b} K_{v-1}[b-t] + f_v(t) gives OPT_k = K_V[k] in O(V*B^2).  Both are
polynomial; there is no hardness barrier anywhere in the instance class.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Sequence, Tuple

import numpy as np

# shipped constants (src/core/submission.py CoveragePlan defaults, as used by alloc_B)
NHIET = 0.02      # tau
SIGMA = 30.0
NUA = 6           # h
LUOI = 5          # g
BUDGET = 100
QUANT = 1e-9      # _MASS_QUANTUM
KS = (1, 5, 20, 50, 100)
NEG = -1e18


def r_cells(nua: int = NUA, luoi: int = LUOI) -> int:
    """r = max(1, h // g), exactly submission.py:477 (`nua = max(1, plan.nua_cua_so // plan.luoi`)."""
    return max(1, nua // luoi)


# ---------------------------------------------------------------------------
# the belief, built bit-for-bit as src/core/submission.py:455-473 builds it
# ---------------------------------------------------------------------------
def build_blocks(cands, nhiet: float = NHIET, sigma: float = SIGMA, luoi: int = LUOI):
    """[(video_id, frame_idx, score, video_last_frame)] -> OrderedDict vid -> (truc, mass)."""
    diem = np.array([round(float(c[2]), 4) for c in cands], dtype=np.float64)
    w = np.exp((diem - diem.max()) / max(nhiet, 1e-9))
    w /= w.sum()

    theo_video: "collections.OrderedDict[str, List[Tuple[int, float, int]]]" = collections.OrderedDict()
    for c, wi in zip(cands, w):
        last = int(c[3]) if c[3] is not None else 1 << 31
        theo_video.setdefault(str(c[0]), []).append((int(c[1]), float(wi), last))

    khoi: "collections.OrderedDict[str, Tuple[np.ndarray, np.ndarray]]" = collections.OrderedDict()
    for vid, items in theo_video.items():
        last = max(x[2] for x in items)
        lo = max(0, min(x[0] for x in items) - 4 * int(sigma))
        hi = max(lo, min(last, max(x[0] for x in items) + 4 * int(sigma)))
        truc = np.arange(lo, hi + 1, luoi, dtype=np.int64)
        if truc.size == 0:
            continue
        mass = np.zeros(truc.size, dtype=np.float64)
        for f, wi, _ in items:                       # same accumulation order as production
            mass += wi * np.exp(-0.5 * ((truc - f) / sigma) ** 2)
        mass = np.floor(mass / QUANT + 0.5) * QUANT
        khoi[vid] = (truc, mass)
    return khoi


# ---------------------------------------------------------------------------
# the shipped greedy, replicated (rows + the coverage curve it realises)
# ---------------------------------------------------------------------------
def greedy(khoi, r: int, budget: int = BUDGET):
    """Returns (rows, cov[1..budget], marginals[1..budget], n_greedy).

    Identical in every branch to allocate_coverage_rows before its tail filler:
    per-video sliding-window sum, argmax taken BEFORE the already-used test, strict `>`
    against the running best (so ties go to the first video in insertion order), stop on
    non-positive gain.
    """
    chua = {v: m.copy() for v, (_t, m) in khoi.items()}
    rows: List[Tuple[str, int]] = []
    cov: List[float] = []
    marg: List[float] = []
    total = 0.0
    da_dung: set = set()
    while len(rows) < budget:
        tot_v, tot_i, tot_gia = None, -1, 0.0
        for vid, (truc, _m) in khoi.items():
            con = chua[vid]
            if con.size == 0:
                continue
            tich = np.cumsum(np.concatenate(([0.0], con)))
            lo = np.maximum(0, np.arange(con.size) - r)
            hi = np.minimum(con.size, np.arange(con.size) + r + 1)
            gia = tich[hi] - tich[lo]
            j = int(np.argmax(gia))
            if gia[j] > tot_gia and (vid, int(truc[j])) not in da_dung:
                tot_v, tot_i, tot_gia = vid, j, float(gia[j])
        if tot_v is None or tot_gia <= 0:
            break
        truc, _ = khoi[tot_v]
        f = int(truc[tot_i])
        rows.append((tot_v, f))
        da_dung.add((tot_v, f))
        total += tot_gia
        cov.append(total)
        marg.append(tot_gia)
        chua[tot_v][max(0, tot_i - r): tot_i + r + 1] = 0.0
    n_greedy = len(rows)
    while len(cov) < budget:                      # the curve is flat past exhaustion
        cov.append(total)
        marg.append(0.0)
    return rows, np.array(cov), np.array(marg), n_greedy


def cov_of_rows(khoi, rows: Sequence[Tuple[str, int]], r: int) -> np.ndarray:
    """Cumulative Cov of an arbitrary ORDERED list of (video, frame) grid lines."""
    idx_of = {v: {int(f): i for i, f in enumerate(truc)} for v, (truc, _m) in khoi.items()}
    chua = {v: m.copy() for v, (_t, m) in khoi.items()}
    out, total = [], 0.0
    for v, f in rows:
        if v in idx_of and int(f) in idx_of[v]:
            c = idx_of[v][int(f)]
            lo, hi = max(0, c - r), min(chua[v].size, c + r + 1)
            total += float(chua[v][lo:hi].sum())
            chua[v][lo:hi] = 0.0
        out.append(total)
    return np.array(out)


# ---------------------------------------------------------------------------
# exact per-video profile:  f[j] = max mass covered by <= j lines
# ---------------------------------------------------------------------------
def _kmax(n: int, r: int, budget: int) -> int:
    return int(min(budget, (n + 2 * r) // (2 * r + 1) + 1))


def video_profile(mass: np.ndarray, r: int, budget: int = BUDGET) -> np.ndarray:
    n = mass.size
    W = np.concatenate(([0.0], np.cumsum(mass)))
    c = np.arange(n)
    lo = np.maximum(0, c - r)
    hi = np.minimum(n - 1, c + r)
    full = W[hi + 1] - W[lo]
    best = np.zeros(budget + 1)
    km = _kmax(n, r, budget)
    A = full.copy()
    best[1] = float(A.max()) if n else 0.0
    for j in range(2, km + 1):
        P = np.maximum.accumulate(A)                     # prefix max of A[j-1]
        nxt = np.full(n, NEG)
        d0 = 2 * r + 1
        if n > d0:
            ok = c >= d0
            nxt[ok] = P[c[ok] - d0] + full[ok]
        for d in range(1, 2 * r + 1):                    # overlapping predecessors
            cc = c[c >= d]
            if cc.size == 0:
                continue
            hicp = np.minimum(n - 1, cc - d + r)
            a = np.minimum(np.maximum(lo[cc], hicp + 1), n)
            gain = np.where(a <= hi[cc], W[np.minimum(hi[cc] + 1, n)] - W[a], 0.0)
            nxt[cc] = np.maximum(nxt[cc], A[cc - d] + gain)
        A = nxt
        m = float(A.max())
        best[j] = m if m > NEG / 2 else best[j - 1]
    for j in range(km + 1, budget + 1):
        best[j] = best[km] if km >= 1 else 0.0
    return np.maximum.accumulate(best)


def video_profile_rows(mass: np.ndarray, r: int, t: int) -> List[int]:
    """The actual centres of an optimal set of <= t lines (backtracked)."""
    if t <= 0 or mass.size == 0:
        return []
    n = mass.size
    W = np.concatenate(([0.0], np.cumsum(mass)))
    c = np.arange(n)
    lo = np.maximum(0, c - r)
    hi = np.minimum(n - 1, c + r)
    full = W[hi + 1] - W[lo]
    km = min(t, _kmax(n, r, t))
    As = [None, full.copy()]
    preds = [None, np.full(n, -1, dtype=np.int64)]
    for j in range(2, km + 1):
        A = As[j - 1]
        P = np.maximum.accumulate(A)
        # index attaining the running prefix max (vectorised; ties -> the last such index)
        Parg = np.maximum.accumulate(np.where(A >= P, np.arange(n, dtype=np.int64), 0))
        nxt = np.full(n, NEG)
        pr = np.full(n, -1, dtype=np.int64)
        d0 = 2 * r + 1
        if n > d0:
            ok = c >= d0
            nxt[ok] = P[c[ok] - d0] + full[ok]
            pr[ok] = Parg[c[ok] - d0]
        for d in range(1, 2 * r + 1):
            cc = c[c >= d]
            if cc.size == 0:
                continue
            hicp = np.minimum(n - 1, cc - d + r)
            a = np.minimum(np.maximum(lo[cc], hicp + 1), n)
            gain = np.where(a <= hi[cc], W[np.minimum(hi[cc] + 1, n)] - W[a], 0.0)
            cand = A[cc - d] + gain
            better = cand > nxt[cc]
            sel = cc[better]
            nxt[sel] = cand[better]
            pr[sel] = sel - d
        As.append(nxt)
        preds.append(pr)
    bj, bc, bv = 0, -1, -1.0
    for j in range(1, km + 1):
        m = float(As[j].max())
        if m > NEG / 2 and m > bv + 1e-18:
            bv, bj, bc = m, j, int(np.argmax(As[j]))
    if bj == 0:
        return []
    out = []
    j, cc = bj, bc
    while j >= 1 and cc >= 0:
        out.append(int(cc))
        cc = int(preds[j][cc]) if j >= 2 else -1
        j -= 1
    return sorted(out)


# ---------------------------------------------------------------------------
# (max,+) knapsack over videos
# ---------------------------------------------------------------------------
def exact_opt(khoi, r: int, budget: int = BUDGET, want_rows: bool = False):
    """OPT[0..budget]; with want_rows also the backtracked optimal set at k=budget."""
    vids = [v for v, (_t, m) in khoi.items() if m.size and m.sum() > 0]
    profs = {v: video_profile(khoi[v][1], r, budget) for v in vids}
    K = np.full(budget + 1, NEG)
    K[0] = 0.0
    choices = []
    for v in vids:
        f = profs[v]
        M = np.full((budget + 1, budget + 1), NEG)
        for t in range(budget + 1):
            M[t:, t] = K[: budget + 1 - t] + f[t]
        ch = M.argmax(axis=1)
        K = M.max(axis=1)
        choices.append((v, ch))
    OPT = np.maximum.accumulate(np.where(K > NEG / 2, K, 0.0))
    if not want_rows:
        return OPT
    b = budget
    take: Dict[str, int] = {}
    for v, ch in reversed(choices):
        t = int(ch[b])
        if t > 0:
            take[v] = t
        b -= t
    rows: List[Tuple[str, int]] = []
    for v, t in take.items():
        truc, mass = khoi[v]
        for c in video_profile_rows(mass, r, t):
            rows.append((v, int(truc[c])))
    return OPT, rows


def order_by_greedy(khoi, rows: Sequence[Tuple[str, int]], r: int) -> List[Tuple[str, int]]:
    """Order a FIXED set of lines by largest uncovered mass (greedy inside the set)."""
    idx_of = {v: {int(f): i for i, f in enumerate(truc)} for v, (truc, _m) in khoi.items()}
    chua = {v: m.copy() for v, (_t, m) in khoi.items()}
    left = list(rows)
    out: List[Tuple[str, int]] = []
    while left:
        bi, bg = 0, -1.0
        for i, (v, f) in enumerate(left):
            if v not in idx_of or int(f) not in idx_of[v]:
                g = 0.0
            else:
                c = idx_of[v][int(f)]
                g = float(chua[v][max(0, c - r): c + r + 1].sum())
            if g > bg:
                bi, bg = i, g
        v, f = left.pop(bi)
        if v in idx_of and int(f) in idx_of[v]:
            c = idx_of[v][int(f)]
            chua[v][max(0, c - r): c + r + 1] = 0.0
        out.append((v, int(f)))
    return out


# ---------------------------------------------------------------------------
# instance-specific certificate (the free one, computable inside the greedy loop)
# ---------------------------------------------------------------------------
def certificate(cov: np.ndarray, marg: np.ndarray, k: int) -> float:
    """OPT_k <= min_{i<k} ( Cov(G_i) + k * delta_{i+1} ).  Cov(G_0)=0, delta_1 = marg[0]."""
    ub = float("inf")
    for i in range(0, min(k, len(marg))):
        prev = 0.0 if i == 0 else float(cov[i - 1])
        ub = min(ub, prev + k * float(marg[i]))
    return ub


# ---------------------------------------------------------------------------
# brute force, for the validation gate V1
# ---------------------------------------------------------------------------
def brute_single(mass: np.ndarray, r: int, k: int) -> float:
    import itertools
    n = mass.size
    best = 0.0
    for j in range(0, k + 1):
        for S in itertools.combinations(range(n), j):
            cov = np.zeros(n, dtype=bool)
            for c in S:
                cov[max(0, c - r): min(n, c + r + 1)] = True
            best = max(best, float(mass[cov].sum()))
    return best


def brute_multi(masses: Sequence[np.ndarray], r: int, k: int) -> float:
    import itertools
    ground = [(v, c) for v, m in enumerate(masses) for c in range(m.size)]
    best = 0.0
    for j in range(0, k + 1):
        for S in itertools.combinations(range(len(ground)), j):
            covs = [np.zeros(m.size, dtype=bool) for m in masses]
            for gi in S:
                v, c = ground[gi]
                covs[v][max(0, c - r): min(masses[v].size, c + r + 1)] = True
            best = max(best, float(sum(m[cv].sum() for m, cv in zip(masses, covs))))
    return best
