"""The robustness ladder: every headline of the paper under eleven audit
policies (Table 1 of the paper shows six of them).

A copy of the per-rung block of scripts/r1b_thang_chinh_sach.py, reduced to the
quantities the paper prints and fed from the release:

  n, videos, declared two-scene, two-scene under the rung's own label, the
  number of items that carry a Gemini-only verdict on some gate;
  the TUNE/TEST split (the rung's own stratified split, seed 20260917);
  coverage vs ranking-based allocation on the raw pool, DEU moments, on the
  held-out TEST half and on ALL items: paired percentile bootstrap (4,000,
  seed 4242) and sign-flip p (10,000, seed 20260924);
  the ratio of the keyframe-authored set's score (B60) to the rung's score,
  both at seed root 77000 (unpaired bootstrap, 4,000, seed 4242);
  the rung's score at seed root 77000 with the draws re-run on the rung;
  two-scene minus one-scene score (query bootstrap);
  the failure buckets at +-20 frames: hit, lost by placement, no candidate of
  the correct video near the target, wrong video.
"""

from __future__ import annotations

import numpy as np

import protocol as P
import stats as S

BUCKET_WINDOW = 20
BUCKET_NAMES = ["hit", "lost by placement", "no candidate near target", "wrong video"]
#: rung -> (policy list, stratum label); every split is the rung's own
RUNGS = {
    "P0_LEGACY": ("P0_LEGACY", "declared"),
    "P2_NEO": ("P2_NEO", "gemini"),
    "P3_QA": ("P3_QA", "gemini"),
    "P4_STRICT": ("P4_STRICT", "gemini"),
    "C_BOTH_ANCHOR": ("C_BOTH_ANCHOR", "both"),
    "C_EITHER_ANCHOR": ("C_EITHER_ANCHOR", "both"),
    "C_EITHER_ANCHOR_QA": ("C_EITHER_ANCHOR_QA", "both"),
    "C_EITHER_ANCHOR_QA_G3LENIENT": ("C_EITHER_ANCHOR_QA_G3LENIENT", "both"),
    "C_STRICT_BOTH": ("C_STRICT_BOTH", "both"),
    "C_STRICT_BOTH_G3LENIENT": ("C_STRICT_BOTH_G3LENIENT", "both"),
    "C_STRICT_EITHER": ("C_STRICT_EITHER", "both"),
}
#: the six columns of Table 1, in its order
TABLE1 = ("P0_LEGACY", "P2_NEO", "C_EITHER_ANCHOR", "C_BOTH_ANCHOR", "C_STRICT_EITHER", "C_STRICT_BOTH")


def bucket_shares(clean, rows_raw_b, raw_frames_true, truths_deu):
    """Per item: share of the 4x48 DEU draws in each failure bucket at +-20."""
    t_all = np.concatenate([truths_deu[f] for f in range(truths_deu.shape[0])], axis=1)
    share = np.zeros((len(clean), 4))
    near = []
    for q, g in enumerate(clean):
        v, fr = g["video_id"], int(g["frame_idx"])
        lf = np.array([f for vv, f in rows_raw_b[q] if vv == v], dtype=np.int64)
        cf = np.asarray(raw_frames_true[q], dtype=np.int64)
        t = t_all[q]
        hit = (np.abs(lf[None, :] - t[:, None]) <= BUCKET_WINDOW).any(axis=1) if lf.size \
            else np.zeros(t.size, bool)
        cnear = (np.abs(cf[None, :] - t[:, None]) <= BUCKET_WINDOW).any(axis=1) if cf.size \
            else np.zeros(t.size, bool)
        anyline = lf.size > 0
        b = np.zeros((t.size, 4))
        b[:, 0] = hit
        rest = ~hit
        b[:, 2] = rest & ~cnear
        b[:, 3] = rest & cnear & (not anyline)
        b[:, 1] = rest & cnear & anyline
        share[q] = b.mean(axis=0)
        near.append(int(np.abs(cf - fr).min()) if cf.size else None)
    return share, near


def _paired_block(perq_raw, idx):
    ii = np.asarray(idx, dtype=np.int64)
    out = {"n": int(ii.size), "alloc": {a: {"mean": float(perq_raw[a][ii].mean())} for a in "ABCD"},
           "paired": {}}
    bi = P.boot_indices(len(ii))
    for other in ("A", "C", "D"):
        p = P.paired(perq_raw["B"][ii], perq_raw[other][ii], bi)
        p["signflip"] = P.signflip(perq_raw["B"][ii] - perq_raw[other][ii])
        out["paired"][f"B-{other}"] = p
    return out


def run_ladder(clean, members, labels, fallback, perq_raw, perq_77000_crn, redraw_mean,
               b60_perq, share, near, vid100_raw_b):
    """members: {policy: clean positions}; labels: {"declared","gemini","both": bool arrays};
    fallback: {policy: items with a Gemini-only verdict on some gate} (C_* only);
    perq_raw: {alloc: per-item official score, raw pool, DEU, root 930000};
    perq_77000_crn: raw coverage per item at root 77000 (draws over the 158);
    redraw_mean(positions) -> (mean, family_sd) at root 77000 with draws re-run on them;
    b60_perq: B60 coverage per query at root 77000."""
    declared, gemini, both = labels["declared"], labels["gemini"], labels["both"]
    s = perq_raw["B"]
    out = {}
    for rung, (pol, lab_name) in RUNGS.items():
        mem = np.asarray(sorted(members[pol]), dtype=np.int64)
        lab = labels[lab_name]
        tune, test = P.stratified_split(mem, lab)
        node = {"policy_list": pol, "stratum_label": {"gemini": "evidence"}.get(lab_name, lab_name),
                "n": int(mem.size), "n_two": int(lab[mem].sum()), "n_one": int((~lab[mem]).sum()),
                "declared_two": int(declared[mem].sum()),
                "evidence_two_gemini": int(gemini[mem].sum()),
                "videos": len({clean[q]["video_id"] for q in mem}),
                "gpt_fallback_items": fallback.get(pol) if pol.startswith("C_") else None,
                "n_tune": len(tune), "n_test": len(test),
                "tune_two": int(lab[tune].sum()), "test_two": int(lab[test].sum()),
                "members": [int(q) for q in mem], "tune": tune, "test": test}
        if pol.startswith("C_"):
            node["evidence_two_both"] = int(both[mem].sum())
        node["slices"] = {"raw": {"DEU": {"ALL": {"all": _paired_block(perq_raw, mem)},
                                          "TEST": {"all": _paired_block(perq_raw, test)}}}}
        for sl, mm in (("ALL", mem), ("TEST", np.asarray(test, dtype=np.int64))):
            one, two = mm[~lab[mm]], mm[lab[mm]]
            if one.size and two.size:
                st = S.boot_query(
                    {"s": s, "two": lab.astype(float)}, mm,
                    {"gap": lambda d: (d["s"][d["two"] > 0.5].mean() - d["s"][d["two"] < 0.5].mean())
                     if (d["two"] > 0.5).any() and (d["two"] < 0.5).any() else np.nan,
                     "ratio": lambda d: (d["s"][d["two"] > 0.5].mean() / d["s"][d["two"] < 0.5].mean())
                     if (d["two"] > 0.5).any() and (d["two"] < 0.5).any()
                     and d["s"][d["two"] < 0.5].mean() > 0 else np.nan})
                node[f"two_scene_gap_{sl}"] = {
                    "one": float(s[one].mean()), "two": float(s[two].mean()),
                    "n_one": int(one.size), "n_two": int(two.size), "gap": st["gap"], "ratio": st["ratio"],
                    "rel": float(s[two].mean() / s[one].mean() - 1.0) if s[one].mean() > 0 else None}
        one_m = [int(q) for q in mem if not lab[q]]
        two_m = [int(q) for q in mem if lab[q]]
        rd_all, rd_sd = redraw_mean(list(int(q) for q in mem))
        node["set77000"] = {
            "n": int(mem.size),
            "redraw": {"mean": rd_all, "family_sd": rd_sd,
                       "one": redraw_mean(one_m)[0] if one_m else None,
                       "two": redraw_mean(two_m)[0] if two_m else None},
            "crn": {"mean": float(perq_77000_crn[mem].mean()),
                    "one": float(perq_77000_crn[one_m].mean()) if one_m else None,
                    "two": float(perq_77000_crn[two_m].mean()) if two_m else None},
            "correct_video_in_100": int(vid100_raw_b[mem].sum()),
            "correct_video_rate": float(vid100_raw_b[mem].mean()),
        }
        node["Q2_ratio"] = S.boot_ratio(b60_perq, perq_77000_crn[mem])
        node["buckets"] = {}
        for gname, idx in (("all", [int(q) for q in mem]), ("one", one_m), ("two", two_m)):
            if not idx:
                continue
            ii = np.asarray(idx, dtype=np.int64)
            sub = share[ii]
            node["buckets"][gname] = {
                "n": int(ii.size),
                "mean_share": {n: float(sub[:, j].mean()) for j, n in enumerate(BUCKET_NAMES)},
                "expected_items": {n: float(sub[:, j].sum()) for j, n in enumerate(BUCKET_NAMES)},
                "dominant_failure": max(BUCKET_NAMES[1:], key=lambda n: sub[:, BUCKET_NAMES.index(n)].mean()),
                "production_coverage": float(s[ii].mean()),
                "video_absent_from_pool": int(sum(1 for q in idx if near[q] is None)),
                "nearest_pool_candidate_median": float(np.median([near[q] for q in idx if near[q] is not None]))
                if any(near[q] is not None for q in idx) else None,
                "share_candidate_within_20": float(np.mean(
                    [(near[q] is not None and near[q] <= BUCKET_WINDOW) for q in idx])),
            }
        out[rung] = node
    return out


def verdict(rungs):
    """The pre-registered rule: on every rung, (a) TEST gain > 0 with CI > 0 and
    (b) ratio >= 1.5 with CI above 1.0."""
    per = {}
    for p, r in rungs.items():
        t = r["slices"]["raw"]["DEU"]["TEST"]["all"]["paired"]["B-A"]
        a = r["slices"]["raw"]["DEU"]["ALL"]["all"]["paired"]["B-A"]
        q2 = r["Q2_ratio"]
        per[p] = {"a_rel": t["rel"], "a_ci95": t["rel_ci95"], "a_pass": bool(t["rel"] > 0 and t["rel_ci95"][0] > 0),
                  "a_signflip_p": t["signflip"]["p_two_sided"],
                  "a_all_rel": a["rel"], "a_all_ci95": a["rel_ci95"], "a_all_signflip_p": a["signflip"]["p_two_sided"],
                  "b_ratio": q2["ratio"], "b_ci95": q2["ci95"],
                  "b_pass": bool(q2["ratio"] >= 1.5 and q2["ci95"][0] > 1.0),
                  "bprime_pass": bool(q2["ci95"][0] > 1.0), "n": r["n"], "n_test": r["n_test"]}
    return per


def table1_markdown(rungs):
    """Table 1 of the paper, from the numbers above (printed precision)."""
    head = ["No removal", "Gemini gate", "Either judge", "Both judges", "Strict, either", "Strict, both"]
    cols = [rungs[p] for p in TABLE1]

    def row(label, fn):
        return f"| {label} | " + " | ".join(fn(r) for r in cols) + " |"

    def pct(x):
        return f"{100 * x:+.1f}"

    def pct1(x):
        """1 dp, or 2 dp when a nonzero value would print as 0.0 (the paper's rule)."""
        return f"{x:.2f}" if x != 0 and abs(x) < 0.05 else f"{x:.1f}"

    def p3(x):
        return "<0.001" if x < 0.001 else f"{x:.3f}"

    L = ["| | " + " | ".join(head) + " |", "|---" * (len(head) + 1) + "|",
         row("Items / videos", lambda r: f"{r['n']}/{r['videos']}"),
         row("Two-scene, declared / label", lambda r: f"{r['declared_two']}/{r['n_two']}"),
         row("Ratio, keyframe-authored / set", lambda r: f"{r['Q2_ratio']['ratio']:.2f}"),
         row("  95% CI", lambda r: f"{r['Q2_ratio']['ci95'][0]:.2f}-{r['Q2_ratio']['ci95'][1]:.2f}"),
         row("Gain, held-out half (%)",
             lambda r: pct(r["slices"]["raw"]["DEU"]["TEST"]["all"]["paired"]["B-A"]["rel"])),
         row("  95% CI", lambda r: " to ".join(
             pct1(100 * x) for x in r["slices"]["raw"]["DEU"]["TEST"]["all"]["paired"]["B-A"]["rel_ci95"])),
         row("  sign-flip p", lambda r: p3(r["slices"]["raw"]["DEU"]["TEST"]["all"]["paired"]["B-A"]
                                          ["signflip"]["p_two_sided"])),
         row("Two-scene minus one-scene", lambda r: f"{r['two_scene_gap_ALL']['gap']['point']:+.3f}"),
         row("  upper 95% bound", lambda r: f"{r['two_scene_gap_ALL']['gap']['ci95'][1]:+.2f}"),
         row("No candidate / placement (two-scene)", lambda r: "{:.2f}/{:.2f}".format(
             r["buckets"]["two"]["mean_share"]["no candidate near target"],
             r["buckets"]["two"]["mean_share"]["lost by placement"]))]
    return "\n".join(L)
