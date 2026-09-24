"""Aggregation glue: which items enter which cell of the allocation tables.

The numerical functions (split rule, bootstrap, sign-flip test) are passed in
as ``fns`` so that the release builder can fill the same structure with the
ORIGINAL repository functions and ``reproduce.py`` with the copies in
``protocol.py``; the two results are then compared cell by cell.

Audit policies (``benchmark/items.jsonl`` -> ``policies``), all subsets of the
158 items outside shard c:

  Gemini judge only
    P0_LEGACY    all 158 (no removal)
    P2_NEO       remove items whose answer frame the Gemini judge rejects
    P3_QA        P2, minus items whose question-answer triple Gemini rejects
    P4_STRICT    items that pass every Gemini gate
    X0_AUDITED   P0 minus the 7 rejects of an earlier 64-item audit (historical)
  Two judges (Gemini and GPT-5.2; an item GPT-5.2 did not reach takes Gemini's
  verdict on that gate)
    C_BOTH_ANCHOR                 remove only if BOTH judges reject the answer frame
    C_EITHER_ANCHOR               remove if EITHER judge rejects it -- the PRIMARY set
    C_EITHER_ANCHOR_QA            ... and if either finds a Q&A defect (unresolved
                                  GPT answer comparisons counted as wrong)
    C_EITHER_ANCHOR_QA_G3LENIENT  the same, unresolved comparisons counted as correct
    C_STRICT_BOTH                 items that pass every gate under both judges
                                  (unresolved comparisons counted as wrong)
    C_STRICT_BOTH_G3LENIENT       the same, counted as correct
    C_STRICT_EITHER               items that pass every gate under at least one judge

Two-scene label (the stratum) of each policy:
  declared  the generator's quota (P0_LEGACY, X0_AUDITED)
  gemini    Gemini's image evidence for the earlier scene (P2, P3, P4)
  both      image evidence from BOTH judges (every C_* policy)

Split modes:
  E2   the split of the 158-item set (stratified by the DECLARED flag, seed
       20260917), restricted to the set: the split of draft 2 of the paper
  OWN  a split drawn on the set itself with the same rule and seed, stratified
       by the policy's own two-scene label. For C_EITHER_ANCHOR this is the
       split of the paper (TUNE 73 / TEST 75).
"""

from __future__ import annotations

import numpy as np

POOLS = ("raw", "prod")
ALLOCS = ("A", "B", "C", "D")
SLICES = ("ALL", "TUNE", "TEST")
GROUPS = ("all", "one", "two")
HEADLINE = ("C_EITHER_ANCHOR", "OWN")
#: name -> (two-scene label, split modes)
POLICIES = {
    "C_EITHER_ANCHOR": ("both", ("OWN",)),
    "P0_LEGACY": ("declared", ("E2",)),
    "P2_NEO": ("gemini", ("E2", "OWN")),
    "P3_QA": ("gemini", ("E2", "OWN")),
    "P4_STRICT": ("gemini", ("E2", "OWN")),
    "X0_AUDITED": ("declared", ("E2",)),
    "C_BOTH_ANCHOR": ("both", ("OWN",)),
    "C_EITHER_ANCHOR_QA": ("both", ("OWN",)),
    "C_EITHER_ANCHOR_QA_G3LENIENT": ("both", ("OWN",)),
    "C_STRICT_BOTH": ("both", ("OWN",)),
    "C_STRICT_BOTH_G3LENIENT": ("both", ("OWN",)),
    "C_STRICT_EITHER": ("both", ("OWN",)),
}
POLICY_ORDER = tuple(POLICIES)
GEMINI_POLICIES = ("P0_LEGACY", "P2_NEO", "P3_QA", "P4_STRICT", "X0_AUDITED")
CONSENSUS_POLICIES = tuple(p for p in POLICIES if p.startswith("C_"))


def node(perq, vid_in, rak, idx, fns):
    idx = np.asarray(idx, dtype=np.int64)
    out = {"n": int(len(idx)), "pools": {}}
    if len(idx) == 0:
        return out
    bi = fns["boot_indices"](len(idx))
    for p in POOLS:
        cell = {"alloc": {}, "paired": {}, "signflip": {}}
        for a in ALLOCS:
            cell["alloc"][a] = {
                "mean": float(perq[p][a][idx].mean()),
                "video_in_100": int(vid_in[p][a][idx].sum()),
                "r_at_k": [float(x) for x in rak[p][a][idx].mean(axis=0)],
            }
        for other in ("A", "C", "D"):
            cell["paired"][f"B-{other}"] = fns["paired"](perq[p]["B"][idx], perq[p][other][idx], bi)
        for other in ("A", "C"):
            cell["signflip"][f"B-{other}"] = fns["signflip"](perq[p]["B"][idx] - perq[p][other][idx])
        out["pools"][p] = cell
    return out


def build_all(perq, vid_in, rak, policies, labels, fns):
    """policies: {name: sorted positions in the clean list};
    labels: {"declared"|"gemini"|"both": bool array over the clean list}."""
    labels = {k: np.asarray(v, dtype=bool) for k, v in labels.items()}
    declared = labels["declared"]
    allpos = list(range(len(declared)))
    e2_tune, e2_test = fns["split"](allpos, declared)
    res = {"splits": {"E2": {"tune": e2_tune, "test": e2_test}}, "policies": {}}
    for name in POLICY_ORDER:
        if name not in policies:
            continue
        lab_name, modes = POLICIES[name]
        mem = sorted(int(i) for i in policies[name])
        pol = {"n": len(mem), "label": lab_name, "modes": {}}
        for mode in modes:
            if mode == "E2":
                keep = set(mem)
                tune = sorted(keep & set(e2_tune))
                test = sorted(keep & set(e2_test))
                lab = declared
            else:
                lab = labels[lab_name]
                tune, test = fns["split"](mem, lab)
            m = {"label": "declared" if mode == "E2" else lab_name,
                 "tune": tune, "test": test, "slices": {}}
            for sl, members in (("ALL", mem), ("TUNE", tune), ("TEST", test)):
                g = {"all": members,
                     "one": [q for q in members if not lab[q]],
                     "two": [q for q in members if lab[q]]}
                m["slices"][sl] = {gn: node(perq, vid_in, rak, gm, fns) for gn, gm in g.items()}
            pol["modes"][mode] = m
        res["policies"][name] = pol
    return res


def compare(a, b, tol=1e-9, path="", out=None):
    """Every numeric leaf of ``a`` and ``b`` within ``tol``; returns mismatches."""
    if out is None:
        out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}/{k}", "missing on one side"))
                continue
            compare(a[k], b[k], tol, f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} vs {len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, tol, f"{path}[{i}]", out)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) \
            or a is None or b is None:
        if a != b:
            out.append((path, f"{a!r} vs {b!r}"))
    else:
        fa, fb = float(a), float(b)
        if fa != fa and fb != fb:  # both NaN
            return out
        if not (abs(fa - fb) <= tol):
            out.append((path, f"{a!r} vs {b!r}"))
    return out


def compare_subset(mine, ref, tol=1e-9, path="", out=None, counter=None):
    """Every leaf of ``mine`` must exist in ``ref`` and agree within ``tol``;
    keys only in ``ref`` are ignored (used where the release reproduces a part
    of a larger result file). Returns (mismatches, number of leaves compared)."""
    if out is None:
        out, counter = [], [0]
    if isinstance(mine, dict):
        if not isinstance(ref, dict):
            out.append((path, "not a dict in the reference"))
            return out, counter[0]
        for k in sorted(mine):
            if k not in ref:
                out.append((f"{path}/{k}", "absent from the reference"))
                continue
            compare_subset(mine[k], ref[k], tol, f"{path}/{k}", out, counter)
    elif isinstance(mine, (list, tuple)):
        if not isinstance(ref, (list, tuple)) or len(mine) != len(ref):
            out.append((path, "list length differs"))
            return out, counter[0]
        for i, (x, y) in enumerate(zip(mine, ref)):
            compare_subset(x, y, tol, f"{path}[{i}]", out, counter)
    else:
        counter[0] += 1
        bad = compare(mine, ref, tol, path)
        out.extend(bad)
    return out, counter[0]
