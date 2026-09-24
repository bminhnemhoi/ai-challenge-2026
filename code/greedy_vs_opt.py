"""How far the shipped greedy coverage allocator is from the exact optimum of
its own objective, and what an exact optimiser and two oracles would score.

A copy of the per-item loop and the summary blocks of scripts/e1_exact_chinh.py,
fed from the release. The exact optimum is computed by ``exact_dp.py``
(byte-identical to scripts/e1_exact_core.py): a per-video dynamic programme
plus a (max,+) knapsack over videos; its pre-registration is its docstring.

Per item and pool:
  Cov(G_k) / OPT_k at k in {1, 5, 20, 50, 100}, where G is the shipped greedy
  (checked line for line against the coverage allocator's lines) and OPT_k the
  exact optimum with at most k lines;
  E   = the exact optimum at k = 100, ordered greedily inside the optimal set;
  O1  = the greedy run on the true video alone (a perfect cross-video ranker);
  O2  = the greedy run on a perfectly calibrated belief (one Gaussian of the
        shipped sigma on the answer frame), continued as a ladder.
Scores: official rule, DEU moments at seed root 930000 (drawn over the 158
scored items and sliced), paired bootstrap 4,000 at seed 4242.
"""

from __future__ import annotations

import time

import numpy as np

import protocol as P
import stats as S
from exact_dp import (BUDGET, KS, LUOI, QUANT, SIGMA, build_blocks, cov_of_rows, exact_opt, greedy,
                      order_by_greedy, r_cells)

ALLOCS = ("E", "O1", "O2")


def oracle_block(anchor: int, last: int):
    lo = max(0, anchor - 4 * int(SIGMA))
    hi = max(lo, min(last, anchor + 4 * int(SIGMA)))
    truc = np.arange(lo, hi + 1, LUOI, dtype=np.int64)
    mass = np.exp(-0.5 * ((truc - anchor) / SIGMA) ** 2)
    mass = np.floor(mass / QUANT + 0.5) * QUANT
    return {"__oracle__": (truc, mass)}


def ladder_tail(rows, anchor: int, last: int, vid: str, r: int, budget: int = BUDGET):
    """Continue outward at the greedy spacing (2r+1 grid cells) until the budget is full."""
    out = list(rows)
    seen = set(out)
    step = (2 * r + 1) * LUOI
    j = 1
    while len(out) < budget and j < 10000:
        for f in (anchor - step * j, anchor + step * j):
            if 0 <= f <= last:
                key = (vid, int(f))
                if key not in seen:
                    seen.add(key)
                    out.append(key)
                    if len(out) >= budget:
                        return out
        j += 1
    return out


def per_item(cands, rows_b, g, last_of):
    """cands: list of Candidate; rows_b: the coverage allocator's 100 lines."""
    r = r_cells()
    cl = [(c.video_id, int(c.frame_idx), float(c.score), c.video_last_frame) for c in cands]
    khoi = build_blocks(cl)
    t0 = time.perf_counter()
    g_rows, gcov, _gmarg, ng = greedy(khoi, r)
    t_greedy = time.perf_counter() - t0
    t0 = time.perf_counter()
    OPT, opt_rows = exact_opt(khoi, r, BUDGET, want_rows=True)
    t_dp = time.perf_counter() - t0
    parity = g_rows == [(v, int(f)) for v, f in rows_b][: len(g_rows)]
    e_rows = order_by_greedy(khoi, opt_rows, r)
    if len(e_rows) < BUDGET:
        seen = set(e_rows)
        for k_ in g_rows:
            if k_ not in seen:
                seen.add(k_)
                e_rows.append(k_)
                if len(e_rows) >= BUDGET:
                    break
    e_rows = e_rows[:BUDGET]
    ecov = cov_of_rows(khoi, e_rows, r)
    tv, anc = g["video_id"], int(g["frame_idx"])
    last = int(last_of.get(tv, anc + 4 * int(SIGMA)))
    if tv in khoi:
        o1_rows, _c, _m, _n = greedy({tv: khoi[tv]}, r)
    else:
        o1_rows = []
    o2_raw, _c, _m, _n = greedy(oracle_block(anc, last), r)
    o2_rows = ladder_tail([(tv, int(f)) for _v, f in o2_raw], anc, last, tv, r)
    rec = {"n_videos": len(khoi), "n_greedy": ng, "n_opt_rows": len(opt_rows),
           "cov": {str(k): float(gcov[k - 1]) for k in KS},
           "opt": {str(k): float(OPT[k]) for k in KS},
           "cov_E": {str(k): float(ecov[k - 1]) for k in KS},
           "greedy_parity": bool(parity), "t_greedy": t_greedy, "t_dp": t_dp}
    return rec, {"E": e_rows, "O1": o1_rows, "O2": o2_rows}


def desc(a):
    a = np.asarray(a, dtype=float)
    q = lambda p: float(np.percentile(a, p))  # noqa: E731
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()),
            "p05": q(5), "p25": q(25), "p75": q(75), "p95": q(95)}


def ratio_block(rec, idx):
    out = {}
    for k in KS:
        v = np.array([rec[i]["cov"][str(k)] / rec[i]["opt"][str(k)]
                      if rec[i]["opt"][str(k)] > 1e-15 else 1.0 for i in idx])
        d = desc(v)
        d["exactly_1"] = int((v >= 1.0 - 1e-12).sum())
        d["below_0.99"] = int((v < 0.99).sum())
        d["below_0.98"] = int((v < 0.98).sum())
        d["below_0.95"] = int((v < 0.95).sum())
        d["below_1_minus_1_over_e"] = int((v < 1 - 1 / np.e - 1e-12).sum())
        out[str(k)] = d
    g = np.array([[rec[i]["cov"][str(k)] for k in KS] for i in idx]).mean(axis=1)
    e = np.array([[rec[i]["cov_E"][str(k)] for k in KS] for i in idx]).mean(axis=1)
    o = np.array([[rec[i]["opt"][str(k)] for k in KS] for i in idx]).mean(axis=1)
    sl = np.array([[1.0 - (rec[i]["cov"][str(k)] / rec[i]["opt"][str(k)]
                           if rec[i]["opt"][str(k)] > 1e-15 else 1.0) for k in KS] for i in idx])
    out["objective"] = {"mean_cov_G": float(g.mean()), "mean_cov_E": float(e.mean()),
                        "mean_opt": float(o.mean()), "pooled_ratio": float(g.sum() / o.sum()),
                        "headroom_any_list_pct": float(100 * (o.sum() / g.sum() - 1.0)),
                        "headroom_worst_item_pct": float(100 * ((o / g).max() - 1.0)),
                        "slack_mean_pct": float(100 * sl.mean()),
                        "slack_worst_pct": float(100 * sl.max())}
    return out


def score_block(per_item_scores, idx):
    ii = np.array(sorted(idx), dtype=int)
    blk = {"n": int(ii.size), "B": float(per_item_scores["B"][ii].mean())}
    for a in ALLOCS:
        blk[a] = float(per_item_scores[a][ii].mean())
        blk[f"{a}-B"] = S.paired_boot_e1(per_item_scores[a][ii], per_item_scores["B"][ii])
    tot = blk["O2"] - blk["B"]
    blk["decomposition"] = {
        "total_B_to_O2": tot,
        "optimiser_share": (blk["E"] - blk["B"]) / tot if tot else None,
        "video_ranking_share": (blk["O1"] - blk["B"]) / tot if tot else None,
        "localisation_share": (blk["O2"] - blk["O1"]) / tot if tot else None}
    return blk


def run(clean, pools, rows_b, members, tune, test, two, last_of, truths_deu, pool_names=("raw", "prod"),
        progress=None):
    """members/tune/test: clean positions of the set; two: bool array (stratum);
    returns (records keyed by pool then position, summary)."""
    members = sorted(int(q) for q in members)
    recs, summ = {}, {"ratios": {}, "score": {}, "gates": {}}
    slices = {"ALL": members, "TEST": sorted(test), "TUNE": sorted(tune)}
    for p in pool_names:
        recs[p] = {}
        rows_of = {a: {} for a in ALLOCS}
        t0 = time.time()
        for j, q in enumerate(members):
            rec, rr = per_item(pools[p][q], rows_b[p][q], clean[q], last_of)
            recs[p][q] = rec
            for a in ALLOCS:
                rows_of[a][q] = rr[a]
            if progress and (j + 1) % 25 == 0:
                progress(f"   {p}: {j + 1}/{len(members)} items, {time.time() - t0:.0f}s")
        summ["gates"][f"{p}_greedy_parity"] = int(sum(recs[p][q]["greedy_parity"] for q in members))
        ok_mono = all(all(b >= a - 1e-12 for a, b in zip([recs[p][q]["opt"][str(k)] for k in KS],
                                                          [recs[p][q]["opt"][str(k)] for k in KS][1:]))
                      and all(recs[p][q]["cov"][str(k)] <= recs[p][q]["opt"][str(k)] + 1e-12 for k in KS)
                      for q in members)
        summ["gates"][f"{p}_opt_monotone_and_above_greedy"] = bool(ok_mono)
        sc = {"B": np.zeros(len(clean))}
        for a in ALLOCS:
            sc[a] = np.zeros(len(clean))
        sub_items = [clean[q] for q in members]
        tr = truths_deu[:, members, :]
        b_b, _ = P.score_rows([rows_b[p][q] for q in members], sub_items, tr)
        sc["B"][members] = b_b.mean(axis=(1, 2))
        for a in ALLOCS:
            b_a, _ = P.score_rows([rows_of[a][q] for q in members], sub_items, tr)
            sc[a][members] = b_a.mean(axis=(1, 2))
        rec_list = {q: recs[p][q] for q in members}
        for sl, idx in slices.items():
            groups = {"all": list(idx), "one": [q for q in idx if not two[q]], "two": [q for q in idx if two[q]]}
            for gname, gi in groups.items():
                if not gi:
                    continue
                key = f"{p}/{sl}/{gname}"
                summ["ratios"][key] = ratio_block(rec_list, gi)
                summ["score"][key] = score_block(sc, gi)
    return recs, summ


def summary_markdown(summ):
    L = ["| pool | slice | k=1 | k=5 | k=20 | k=50 | k=100 | items below 0.98 | headroom, any list | worst item |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(k for k in summ["ratios"] if k.endswith("/all") and "/TUNE/" not in k):
        r = summ["ratios"][key]
        p, sl, _ = key.split("/")
        cells = [f"{r[str(k)]['mean']:.3f} / {r[str(k)]['median']:.3f} / {r[str(k)]['min']:.3f}" for k in KS]
        below = max(r[str(k)]["below_0.98"] for k in KS)
        L.append(f"| {p} | {sl} ({r['1']['n']}) | " + " | ".join(cells) +
                 f" | {below} | {r['objective']['headroom_any_list_pct']:.2f}% | "
                 f"{r['objective']['headroom_worst_item_pct']:.2f}% |")
    L.append("")
    L.append("Cells: greedy coverage / exact optimum per item, mean / median / min.")
    L.append("")
    L.append("| pool | slice | shipped greedy | exact optimum | exact - greedy [95% CI] | true video known | calibrated belief | ranking share | localisation share |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(k for k in summ["score"] if k.endswith("/all") and "/TUNE/" not in k):
        s = summ["score"][key]
        p, sl, _ = key.split("/")
        e = s["E-B"]
        d = s["decomposition"]
        L.append(f"| {p} | {sl} ({s['n']}) | {s['B']:.3f} | {s['E']:.3f} | {e['diff']:+.4f} "
                 f"[{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] | {s['O1']:.3f} | {s['O2']:.3f} | "
                 f"{100 * d['video_ranking_share']:.1f}% | {100 * d['localisation_share']:.1f}% |")
    return "\n".join(L)
