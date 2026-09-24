"""Regenerate the allocation numbers of the paper from this release alone.

    python code/reproduce.py                 # everything except the fair comparison, ~6-8 min on one core
    python code/reproduce.py --no-exact      # skip the exact optimum (section 8), ~3 min
    python code/fair_comparison.py --workers 4   # the fair comparison (Sect. 5), about an hour on 4 cores

Needs Python >= 3.9 and numpy. scipy is used if installed (normal quantile of the
BCa interval); otherwise a numpy fallback is used. No video, no keyframe image, no
embedding, no network. What it does, in order:

 1. rebuilds the production candidate pool of every item from the raw pool and
    the per-item scene-B table, and checks it equals pools/b174_prod.jsonl;
 2. runs the four allocators (A ranking-based, B coverage-based, C top-100,
    D one ladder per video) on both pools of the 158 items outside shard c,
    and checks the lines against the SHA-256 of the lines the paper scored;
 3. draws the true moments (uniform in the anchor's keyframe cell, 4 x 48
    draws, seed root 930000), scores every line list with the official rule,
    and checks the vectorised scorer against the literal organiser formula;
 4. computes every cell of the allocation table for all twelve audit policies,
    their split modes, slices (ALL / TUNE / TEST) and groups (all / one- /
    two-scene): means, correct-video counts, R@k, paired bootstrap intervals,
    sign-flip p;
 5. recomputes the replication values (seed root 77000) and the keyframe-authored
    set's score (B60);
 6. the headline statistics of the primary set C_EITHER_ANCHOR: the relative gain
    of coverage-based over ranking-based allocation with its BCa interval and
    Holm-adjusted p, on all 148 items and on the held-out half;
 7. the robustness ladder (Table 1 and the five further policies);
 8. greedy against the exact optimum of the coverage objective, the exact
    optimiser's score and the two oracles (the headroom decomposition);
 9. compares all of it with expected/*.json, which the release builder took
    from the original repository computations the paper's numbers come from,
    and prints PASS/FAIL.

Outputs go to ./out/ (results.json and markdown tables).
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
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
from submission import Candidate  # noqa: E402

PRIMARY = "C_EITHER_ANCHOR"


def load_jsonl(p: Path):
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def rows_sha(rows) -> str:
    s = json.dumps([[[v, int(f)] for v, f in r] for r in rows], separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_keyframes():
    kf = {}
    with (REL / "keyframes" / "keyframe_index.tsv").open(encoding="utf-8") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        assert head == ["video_id", "n", "frame_idx"], head
        for line in fh:
            v, _n, f = line.rstrip("\n").split("\t")
            kf.setdefault(v, []).append(int(f))
    return {v: np.array(sorted(a), dtype=np.int64) for v, a in kf.items()}


def cands_of(rec):
    return [Candidate(str(v), int(f), float(s), None if last is None else int(last))
            for v, f, s, last in rec["candidates"]]


def labels_of(clean):
    return {"declared": np.array([it["declared_two_scene"] for it in clean]),
            "gemini": np.array([bool(it["two_scene_labels"]["gemini_image_evidence"]) for it in clean]),
            "both": np.array([bool(it["two_scene_labels"]["both_judges_image_evidence"]) for it in clean])}


def fmt_table(res, policy, mode, slice_="TEST"):
    node = res["tables"]["policies"][policy]["modes"][mode]["slices"][slice_]["all"]
    L = [f"### {policy} / split {mode} / {slice_} (n={node['n']})", "",
         "| Allocator | Raw pool | Production pool |", "|---|---|---|"]
    for a in ("C", "D", "A", "B"):
        L.append(f"| {AL.ALLOC_LABEL[a]} | {node['pools']['raw']['alloc'][a]['mean']:.4f} | "
                 f"{node['pools']['prod']['alloc'][a]['mean']:.4f} |")
    for other, lab in (("A", "ranking-based"), ("C", "top-100 candidates")):
        cells, rels, ps = [], [], []
        for p in ("raw", "prod"):
            q = node["pools"][p]["paired"][f"B-{other}"]
            cells.append(f"{q['diff']:+.3f} ({q['ci95'][0]:.3f} to {q['ci95'][1]:.3f})")
            rels.append(f"{100 * q['rel']:+.1f}% ({100 * q['rel_ci95'][0]:.1f} to {100 * q['rel_ci95'][1]:.1f})")
            ps.append(f"{node['pools'][p]['signflip'][f'B-{other}']['p_two_sided']:.4f}")
        L.append(f"| Delta coverage vs {lab} | {cells[0]} | {cells[1]} |")
        L.append(f"| relative | {rels[0]} | {rels[1]} |")
        L.append(f"| sign-flip p (two-sided) | {ps[0]} | {ps[1]} |")
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=PRIMARY, help="set printed as the headline table")
    ap.add_argument("--mode", default="OWN", help="its split mode, E2 or OWN")
    ap.add_argument("--no-exact", action="store_true", help="skip section 8 (exact optimum)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    expected = json.loads((REL / "expected" / "reference_values.json").read_text(encoding="utf-8"))
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)

    def compare(name, mine, ref, subset=False):
        if subset:
            mism, n = T.compare_subset(mine, ref, tol=1e-9)
        else:
            mism, n = T.compare(mine, ref, tol=1e-9), None
        check(name, not mism, (f"{n} numbers, " if n is not None else "") + f"{len(mism)} mismatches"
              + (f", first: {mism[0]}" if mism else ""))

    # ---- load ------------------------------------------------------------------------
    items = load_jsonl(REL / "benchmark" / "items.jsonl")
    b60 = load_jsonl(REL / "benchmark" / "b60.jsonl")
    raw_by = {r["item_id"]: r for r in load_jsonl(REL / "pools" / "b174_raw.jsonl")}
    prod_by = {r["item_id"]: r for r in load_jsonl(REL / "pools" / "b174_prod.jsonl")}
    sb_by = {r["item_id"]: r for r in load_jsonl(REL / "pools" / "scene_b.jsonl")}
    b60_raw = {r["item_id"]: r for r in load_jsonl(REL / "pools" / "b60_raw.jsonl")}
    kf = load_keyframes()
    last_of = {v: int(a.max()) for v, a in kf.items()}
    clean = [it for it in items if it["in_clean_set"]]
    print(f"release: {len(items)} items, {len(clean)} outside shard c, B60 {len(b60)}, "
          f"keyframe table {len(kf)} videos", flush=True)

    # ---- 1. production pools -----------------------------------------------------------
    print("1. production pools from raw pool + scene-B table", flush=True)
    raw, prod = {}, {}
    same = 0
    for it in items:
        iid = it["item_id"]
        raw[iid] = cands_of(raw_by[iid])
        prod[iid] = AL.build_production_pool(raw[iid], sb_by.get(iid))
        shipped = cands_of(prod_by[iid])
        same += int([(c.video_id, c.frame_idx, c.score, c.video_last_frame) for c in prod[iid]]
                    == [(c.video_id, c.frame_idx, c.score, c.video_last_frame) for c in shipped])
    check("production pools rebuilt = shipped", same == len(items), f"{same}/{len(items)}")

    # ---- 2. lines ----------------------------------------------------------------------
    print("2. allocating 100 lines per item (the coverage allocator takes ~0.5 s per item)", flush=True)
    rows = {p: {} for p in T.POOLS}
    for p, pools in (("raw", raw), ("prod", prod)):
        for a in T.ALLOCS:
            t1 = time.time()
            rows[p][a] = [AL.run(AL.ALLOC_FN[a], pools[it["item_id"]]) for it in clean]
            ok = rows_sha(rows[p][a]) == expected["rows_sha256"][p][a]
            check(f"lines {p}/{a} = lines the paper scored (sha256)", ok, f"{time.time() - t1:.0f}s")

    # ---- 3. score ----------------------------------------------------------------------
    print("3. true moments and official scoring", flush=True)
    truths = P.draw_moments(P.SEED_ROOT, clean, kf)
    gap = max(P.check_scorer_parity(rows[p][a], clean, truths) for p in T.POOLS for a in T.ALLOCS)
    check("vectorised scorer = literal official formula", gap < 1e-12, f"max |diff| {gap:.1e}")
    perq, vid_in, rak = {}, {}, {}
    for p in T.POOLS:
        perq[p], vid_in[p], rak[p] = {}, {}, {}
        for a in T.ALLOCS:
            b, r = P.score_rows(rows[p][a], clean, truths)
            perq[p][a] = b.mean(axis=(1, 2))
            rak[p][a] = r.mean(axis=1)
            vid_in[p][a] = np.array([any(v == g["video_id"] for v, _ in rr) for rr, g in zip(rows[p][a], clean)])
    worst = max(float(np.abs(perq[p][a] - np.array(expected["per_query"][p][a])).max())
                for p in T.POOLS for a in T.ALLOCS)
    check("per-item official scores = reference", worst < 1e-12, f"max |diff| {worst:.1e}")

    # ---- 4. tables ---------------------------------------------------------------------
    print("4. tables for every policy x split x slice x group", flush=True)
    pos = {it["item_id"]: q for q, it in enumerate(clean)}
    policies = {name: sorted(pos[it["item_id"]] for it in clean if it["policies"][name]) for name in T.POLICY_ORDER}
    labels = labels_of(clean)
    fns = {"split": P.stratified_split, "boot_indices": P.boot_indices, "paired": P.paired, "signflip": P.signflip}
    tables = T.build_all(perq, vid_in, rak, policies, labels, fns)
    shipped_e2 = sorted(pos[it["item_id"]] for it in clean if it["splits"]["E2"] == "TEST")
    shipped_own = sorted(pos[it["item_id"]] for it in clean if it["splits"]["OWN"].get(PRIMARY) == "TEST")
    check("splits = splits shipped in items.jsonl (E2 and the primary set's own)",
          tables["splits"]["E2"]["test"] == shipped_e2
          and tables["policies"][PRIMARY]["modes"]["OWN"]["test"] == shipped_own,
          f"primary TUNE/TEST {len(tables['policies'][PRIMARY]['modes']['OWN']['tune'])}/{len(shipped_own)}")

    # ---- 5. replication gate and B60 ---------------------------------------------------
    print("5. replication values (root 77000) and the keyframe-authored set", flush=True)

    def redraw_mean(members):
        sub = [clean[q] for q in members]
        b, _ = P.score_rows([rows["raw"]["B"][q] for q in members], sub, P.draw_moments(P.REPL_ROOT, sub, kf))
        return float(b.mean()), float(b.mean(axis=(0, 1)).std())

    allq = list(range(len(clean)))
    declared = labels["declared"]
    m_all, sd_all = redraw_mean(allq)
    repl = {"mean": m_all, "family_sd": sd_all,
            "one": redraw_mean([q for q in allq if not declared[q]])[0],
            "two": redraw_mean([q for q in allq if declared[q]])[0],
            "video": int(vid_in["raw"]["B"].sum())}
    b60_rows = {a: [AL.run(AL.ALLOC_FN[a], cands_of(b60_raw[g["item_id"]])) for g in b60] for a in ("A", "B")}
    t60 = P.draw_moments(P.REPL_ROOT, b60, kf)
    b60_res = {}
    for a in ("A", "B"):
        bb, _ = P.score_rows(b60_rows[a], b60, t60)
        b60_res[a] = {"mean": float(bb.mean()), "family_sd": float(bb.mean(axis=(0, 1)).std()),
                      "per_query": [float(x) for x in bb.mean(axis=(1, 2))],
                      "video_in_100": int(sum(any(v == g["video_id"] for v, _ in r) for r, g in zip(b60_rows[a], b60)))}
    res = {"tables": tables, "replication_77000": repl, "b60_77000": b60_res,
           "per_query": {p: {a: [float(x) for x in perq[p][a]] for a in T.ALLOCS} for p in T.POOLS},
           "rows_sha256": {p: {a: rows_sha(rows[p][a]) for a in T.ALLOCS} for p in T.POOLS}}
    for key in ("tables", "replication_77000", "b60_77000", "per_query", "rows_sha256"):
        compare(f"{key} = reference (every numeric leaf, tol 1e-9)", res[key], expected[key])

    # ---- 6. the headline statistics of the primary set --------------------------------
    print(f"6. primary set {PRIMARY}: gain with BCa interval and Holm-adjusted p", flush=True)
    import stats as S
    pmem = policies[PRIMARY]
    own = tables["policies"][PRIMARY]["modes"]["OWN"]
    prim = {"contrasts": {}, "holm": {}}
    for sl, mm in (("ALL", pmem), ("TEST", own["test"])):
        ii = np.asarray(mm, dtype=np.int64)
        fam = {}
        for p in T.POOLS:
            for other, cname in (("A", "B-A"), ("C", "B-C")):
                key = f"{PRIMARY}/R1B/{sl}/all/{p}/DEU/{cname}"
                c = S.bca_contrast(perq[p]["B"][ii], perq[p][other][ii], key)
                prim["contrasts"][key] = c
                fam[f"{p}/{cname}"] = c["perm"]["p"]
        prim["holm"][sl] = {"raw_p": fam, "holm_p": S.holm(fam)}
    exp_prim = json.loads((REL / "expected" / "primary_stats.json").read_text(encoding="utf-8"))
    compare("primary-set statistics = reference", prim, exp_prim, subset=True)
    res["primary_stats"] = prim

    # ---- 7. the robustness ladder -----------------------------------------------------
    print("7. robustness ladder (eleven audit policies)", flush=True)
    import ladder as LD
    truths77 = P.draw_moments(P.REPL_ROOT, clean, kf)
    b77, _ = P.score_rows(rows["raw"]["B"], clean, truths77)
    perq77 = b77.mean(axis=(1, 2))
    raw_frames_true = [[int(c.frame_idx) for c in raw[it["item_id"]] if c.video_id == it["video_id"]] for it in clean]
    share, near = LD.bucket_shares(clean, rows["raw"]["B"], raw_frames_true, truths)
    fallback = {name: sum(1 for q in policies[name] if clean[q]["audit"]["consensus_gemini_fallback_gates"])
                for name in T.CONSENSUS_POLICIES}
    rungs = LD.run_ladder(clean, policies, labels, fallback, perq["raw"], perq77, redraw_mean,
                          np.array(b60_res["B"]["per_query"]), share, near, vid_in["raw"]["B"])
    exp_lad = json.loads((REL / "expected" / "ladder.json").read_text(encoding="utf-8"))
    compare("ladder = reference", rungs, exp_lad["rungs"], subset=True)
    res["ladder"] = {"rungs": rungs, "verdict": LD.verdict(rungs)}

    # ---- 8. greedy against the exact optimum ------------------------------------------
    exact_md = ""
    if not args.no_exact:
        print("8. greedy vs exact optimum of the coverage objective, exact optimiser, oracles", flush=True)
        import greedy_vs_opt as GO
        pools_by_pos = {"raw": [raw[it["item_id"]] for it in clean], "prod": [prod[it["item_id"]] for it in clean]}
        recs, summ = GO.run(clean, pools_by_pos, {p: rows[p]["B"] for p in T.POOLS}, pmem, own["tune"], own["test"],
                            labels["both"], last_of, truths, progress=lambda s: print(s, flush=True))
        for p in T.POOLS:
            check(f"greedy replica = coverage lines ({p} pool) and OPT >= greedy, monotone",
                  summ["gates"][f"{p}_greedy_parity"] == len(pmem) and summ["gates"][f"{p}_opt_monotone_and_above_greedy"],
                  f"{summ['gates'][f'{p}_greedy_parity']}/{len(pmem)}")
        exp_ex = json.loads((REL / "expected" / "exact_dp.json").read_text(encoding="utf-8"))
        per_item = {p: {str(clean[q]["item_id"]): {k: recs[p][q][k] for k in ("cov", "opt", "cov_E")}
                        for q in pmem} for p in T.POOLS}
        compare("per-item greedy / exact optimum = reference", per_item, exp_ex["per_item"], subset=True)
        compare("greedy/OPT summary, exact optimiser and oracle scores = reference",
                {"ratios": summ["ratios"], "score": summ["score"]}, exp_ex["summary"], subset=True)
        res["exact_dp"] = {"summary": summ, "per_item": per_item}
        exact_md = GO.summary_markdown(summ)

    (out / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")

    # ---- report ------------------------------------------------------------------------
    pa = prim["contrasts"][f"{PRIMARY}/R1B/ALL/all/raw/DEU/B-A"]
    pt = prim["contrasts"][f"{PRIMARY}/R1B/TEST/all/raw/DEU/B-A"]
    lines = [
        "# Numbers regenerated from the release", "",
        f"Primary set {PRIMARY}: {len(pmem)} items, TUNE/TEST {len(own['tune'])}/{len(own['test'])}.", "",
        "## Coverage-based vs ranking-based allocation, raw pool", "",
        "| slice | ranking-based | coverage-based | difference | relative gain [BCa 95% CI] | Holm p |",
        "|---|---|---|---|---|---|",
        f"| ALL ({pa['n']}) | {pa['mean_other']:.4f} | {pa['mean_B']:.4f} | {pa['diff']:+.4f} | "
        f"{100 * pa['rel']:+.1f}% [{100 * pa['boot']['bca_rel_ci'][0]:.1f}, {100 * pa['boot']['bca_rel_ci'][1]:.1f}] | "
        f"{prim['holm']['ALL']['holm_p']['raw/B-A']:.4f} |",
        f"| TEST ({pt['n']}) | {pt['mean_other']:.4f} | {pt['mean_B']:.4f} | {pt['diff']:+.4f} | "
        f"{100 * pt['rel']:+.1f}% [{100 * pt['boot']['bca_rel_ci'][0]:.1f}, {100 * pt['boot']['bca_rel_ci'][1]:.1f}] | "
        f"{prim['holm']['TEST']['holm_p']['raw/B-A']:.4f} |", "",
        "## Table 1 (robustness ladder), six of the eleven policies", "",
        LD.table1_markdown(rungs), ""]
    if exact_md:
        lines += ["## Greedy against the exact optimum (primary set)", "", exact_md, ""]
    lines += ["## Allocation tables", "", fmt_table(res, args.policy, args.mode, "TEST"), fmt_table(res, args.policy, args.mode, "ALL")]
    for pol in T.POLICY_ORDER:
        for md_mode in T.POLICIES[pol][1]:
            for sl in ("TEST", "ALL"):
                lines.append(fmt_table(res, pol, md_mode, sl))
    lines.append(f"Replication (root 77000, redraw, raw coverage, 158 items): mean {repl['mean']:.4f} "
                 f"(sd {repl['family_sd']:.4f}), one-scene {repl['one']:.4f}, two-scene {repl['two']:.4f}, "
                 f"correct video {repl['video']}/{len(clean)}")
    lines.append(f"\nB60 (keyframe-authored, root 77000): coverage {b60_res['B']['mean']:.4f}, "
                 f"ranking-based {b60_res['A']['mean']:.4f}, correct video (coverage) {b60_res['B']['video_in_100']}/{len(b60)}")
    (out / "paper_numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines[:16]))
    if exact_md:
        print("\n" + exact_md)
    n_fail = sum(1 for _n, ok, _d in checks if not ok)
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed in {time.time() - t0:.0f}s "
          f"-> {out.resolve().name}/results.json, {out.resolve().name}/paper_numbers.md")
    print("REPRODUCTION: " + ("PASS" if n_fail == 0 else "FAIL"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
