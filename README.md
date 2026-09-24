# Release for "Test-Query Authorship and the Limits of Submission Allocation in Video Event Retrieval"

Version 2.1.0. This release accompanies the SOICT 2026 paper by Ngo Binh Minh, Le Xuan Khanh
and Ngo Lam Tien (Ton Duc Thang University). It holds everything the three authors produced for the
paper: the segment-authored benchmark with its generation and verification records, the verdicts of
two automatic image audits by judges of different model families, the audit-policy memberships and
splits, the keyframe-authored in-house queries, the candidate pools, every prompt, and the code that
regenerates the paper's allocation numbers. It contains no data that belongs to the competition
organisers (see "Not included").

The release is the branch `soict2026-release` and nothing else. It is an orphan branch: it shares no
history with the other branches of this repository, which hold the team's competition-time working
code and files, are not part of this release, and are not covered by its licences or its datasheet.

## Reproduce the paper's allocation numbers

You need Python 3.9 or newer and numpy. scipy is used when it is installed (for the normal quantile
of the BCa interval and for Spearman's rho) and a numpy fallback otherwise; both give the expected
values to the stated tolerance. Nothing else: no video, no keyframe image, no embedding, no GPU and
no network.

```
python -m venv venv && venv/bin/pip install numpy scipy      # Windows: venv\Scripts\pip
venv/bin/python code/reproduce.py                             # about 7 minutes on one CPU core
venv/bin/python code/fair_comparison.py --workers 4           # about an hour on 4 cores
```

`code/reproduce.py`:

1. rebuilds the production candidate pools from the raw pools and the scene-B table;
2. runs the four allocators on both pools of the 158 scored items and checks every line
   list against the SHA-256 of the lines the paper scored;
3. draws the true moments, scores every line list under the official rule, and checks the fast
   scorer against the literal formula in `code/submission.py`;
4. computes every cell of the allocation table for all twelve audit policies, their splits, slices
   and groups;
5. recomputes the seed-root-77000 values and the keyframe-authored set's score;
6. computes the headline gain on the primary set with its BCa interval and Holm-adjusted p;
7. computes the robustness ladder (Table 1 of the paper and five further policies);
8. computes the transfer check (the keyframe-authored set scored with the coverage cell that tuning
   selects on the primary set) and the share of the drop that needs no two-scene query, under four
   labels (see "Further checks" below);
9. compares the shipped greedy allocator with the exact optimum of its objective, scores the exact
   optimiser and the two oracles, and decomposes the headroom;
10. compares all of it with `expected/`, which holds the values the original repository
   computations produced (the values the paper prints), and prints `REPRODUCTION: PASS` or `FAIL`.

`code/fair_comparison.py` tunes five allocator families by the same nested cross-validation on the
primary set and compares every number with `expected/fair_comparison.json`; it prints
`FAIR COMPARISON: PASS` or `FAIL`. It is separate because it scores 1,448 grid cells per pool.

Every comparison uses a tolerance of 1e-9, and the line lists are compared by SHA-256.

## The primary set

The paper's primary set is `C_EITHER_ANCHOR`: 148 of the 158 scored items
(133 videos). An item is removed if either judge (Gemini flash-lite or GPT-5.2) rejects its
answer frame, and it counts as two-scene only if both judges find image evidence for the earlier
scene (59 items; 73 were declared two-scene by the generator). The split is
drawn on the set itself, stratified by that label (TUNE 73 / TEST 75,
`splits.OWN.C_EITHER_ANCHOR`). 8 primary items carry only Gemini's verdict on at
least one gate, because the GPT-5.2 run stopped before reaching them (see `DATASHEET.md`).

Official score (mean of R@k, k in {1, 5, 20, 50, 100}, windows ±6, 10 and 20 frames averaged, true
moment uniform in the answer frame's keyframe cell, 4 × 48 draws at seed root 930000):

| Allocator | ALL, raw pool | ALL, production pool | TEST, raw pool | TEST, production pool |
|---|---|---|---|---|
| Top-100 candidates, one line each | 0.1233 | 0.1233 | 0.1145 | 0.1145 |
| One frame ladder per video | 0.0907 | 0.1258 | 0.0843 | 0.1151 |
| Ranking-based (rounds 1-2) | 0.1759 | 0.1759 | 0.1664 | 0.1664 |
| Coverage-based (round 3) | 0.2058 | 0.2504 | 0.2004 | 0.2224 |

Coverage-based over ranking-based allocation on the raw pool: +17.0% on all 148 items
(difference +0.0299; BCa 95% CI 10.0 to 24.2; Holm-adjusted sign-flip
p <0.001), and +20.5% on the held-out half (BCa 9.1 to 31.7). The
gain holds on the raw pool with narrow windows and a true moment spread across the keyframe cell; the
paper discusses where it does not.

## Eleven audit policies

Every policy is a subset of the 158 scored items; membership is a flag in `policies`, the
split in `splits`. Gain: coverage over ranking-based allocation on the held-out half (raw pool,
percentile bootstrap 95% CI). Ratio: keyframe-authored set's score over the policy's score (seed root
77000, unpaired bootstrap 95% CI).

| Policy | Rule | Items/videos | Two-scene, declared/label | Gemini-only items | TUNE/TEST | Gain, TEST | Ratio |
|---|---|---|---|---|---|---|---|
| `P0_LEGACY` | no removal: the 158 items outside shard c | 158/143 | 79/67 (declared) | - | 78/80 | +17.2% [+7.2, +28.2] | 1.96 [1.47, 2.59] |
| `P2_NEO` | remove items whose answer frame Gemini rejects | 151/136 | 75/63 (Gemini) | - | 75/76 | +19.3% [+8.9, +29.9] | 1.92 [1.43, 2.56] |
| `P3_QA` | P2_NEO, minus items whose Q&A triple Gemini rejects | 122/111 | 62/50 (Gemini) | - | 61/61 | +16.2% [+6.1, +26.6] | 1.95 [1.45, 2.62] |
| `P4_STRICT` | items that pass every Gemini gate | 106/95 | 47/47 (Gemini) | - | 52/54 | +15.1% [+4.9, +26.0] | 1.90 [1.42, 2.57] |
| `C_BOTH_ANCHOR` | remove items whose answer frame BOTH judges reject | 154/139 | 77/61 (both judges) | 8 | 76/78 | +21.9% [+11.4, +32.4] | 1.95 [1.45, 2.63] |
| `C_EITHER_ANCHOR` | **primary**: remove items whose answer frame EITHER judge rejects | 148/133 | 73/59 (both judges) | 8 | 73/75 | +20.5% [+8.8, +30.8] | 1.94 [1.45, 2.58] |
| `C_EITHER_ANCHOR_QA` | C_EITHER_ANCHOR, minus items either judge finds a Q&A defect in (unresolved GPT answer comparisons counted as wrong) | 97/91 | 49/35 (both judges) | 6 | 48/49 | +12.4% [+1.3, +24.5] | 2.18 [1.57, 3.05] |
| `C_EITHER_ANCHOR_QA_G3LENIENT` | the same, unresolved comparisons counted as correct | 114/105 | 58/44 (both judges) | 6 | 57/57 | +15.2% [+6.1, +24.3] | 2.06 [1.52, 2.85] |
| `C_STRICT_BOTH` | items that pass every gate under both judges (unresolved counted as wrong) | 80/74 | 32/32 (both judges) | 5 | 40/40 | +12.3% [-0.0, +23.7] | 2.00 [1.44, 2.89] |
| `C_STRICT_BOTH_G3LENIENT` | the same, unresolved counted as correct | 96/87 | 41/41 (both judges) | 5 | 47/49 | +11.0% [+0.5, +20.8] | 1.89 [1.39, 2.61] |
| `C_STRICT_EITHER` | items that pass every gate under at least one judge | 126/113 | 55/52 (both judges) | 5 | 63/63 | +18.2% [+8.7, +27.8] | 1.97 [1.45, 2.69] |

`X0_AUDITED` (the 151 items left by an earlier 64-item audit) is kept for continuity with earlier
reports and is not part of the ladder.

## Greedy against the exact optimum

`code/exact_dp.py` solves the coverage objective of the shipped allocator exactly (a per-video
dynamic programme and a knapsack over videos; the file is the original, unchanged, with its
pre-registration). On the primary set, raw pool, greedy coverage divided by the optimum per item:

| k | mean | median | min | items below 0.98 |
|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 5 | 0.9950 | 0.9953 | 0.9835 | 0 |
| 20 | 0.9962 | 0.9965 | 0.9893 | 0 |
| 50 | 0.9962 | 0.9962 | 0.9909 | 0 |
| 100 | 0.9963 | 0.9963 | 0.9922 | 0 |

No list of any kind can raise the objective over greedy by more than 0.31% on average
(0.73% on the worst item). The exact optimiser scores 0.2063 against greedy's
0.2058 (difference +0.0005, 95% CI -0.0024 to +0.0037). A ranker that always put
the correct video first would score 0.426, and a perfectly placed belief 0.805; of that distance,
ranking the correct video first accounts for 36.8% and localising within it for
63.2%.

## Tuned against tuned

With every family tuned by the same nested cross-validation (raw pool, all items, out-of-fold
official score): coverage 0.2214, ranking-based 0.2049, uniform grid 0.1741,
peak-capped NMS 0.2040, flat-belief coverage 0.0659. Coverage minus NMS:
+0.0174 (95% CI -0.0062 to +0.0398, sign-flip p 0.143). The
pre-registered rule of this comparison returns FAIL: covering the belief is not shown to
beat equally tuned belief-free spacing.

## Further checks

These are reported briefly in the paper or not at all, for lack of space.

- **Transfer check** (`code/reproduce.py`, step 8). The coverage cell that nested cross-validation
  selects most often on the primary set (tau 0.015, sigma 90, h 6, g 5; 76 of 100 selections) scores
  0.320 on the keyframe-authored set, against 0.400 for the shipped cell (difference
  -0.080, paired 95% CI -0.113 to -0.048, seed root 77000): tuning does not transfer
  back.
- **Share of the drop, and which label** (`code/reproduce.py`, step 8). Items whose query the
  generator wrote as one scene score 0.290, so everything else that differs between the
  sets accounts for 57.0% of the drop from the keyframe-authored score to the primary
  set's (26.5 to 79.2, keyframe-authored score fixed); 59.0% and
  62.4% with the Gemini and GPT-5.2 text labels. The pre-registered analysis used the
  both-judge image-evidence label instead (70.1%, 46.5 to 87.3).
  The paper leads with the declared label because the claim is about how the query was written:
  14 primary items written as two-scene queries lack image evidence and so count as
  one-scene under the image label, yet score like two-scene items (0.129 against
  0.119). See `DATASHEET.md`, section 2.
- **Video-clustered intervals.** The primary set has 148 items in 133 videos.
  Resampling videos instead of items changes the width of the 95% interval of coverage against each
  of the three other allocators (all items, both pools) by a factor of 0.98 to
  1.06, and no interval comes to include zero (`data/cache_chinh/e5/primary/e5_stats.json`
  in the team's repository; not recomputed here).
- **Where the answer frame of a two-scene item sits.** The generator was told to anchor a two-scene
  query on the first frame of the later scene, the moment that scene begins
  (`prompts/01c_generation_anchor_rule_two_scene.txt`); a one-scene query is anchored on the frame it
  describes (`prompts/01d_generation_anchor_rule_one_scene.txt`).
- **Organiser-query labels** (`benchmark/labels_organiser.tsv`). With this table the organiser side
  of the paper's label comparison can be checked without the query text: 35 of 91
  organiser queries are two-scene under GPT-5.2 and 47 under Gemini, the two labellers agree
  on 79, and the surface counts give the organiser column of the paper's comparison. The
  classifier scores are the organiser half of the out-of-fold scores behind the paper's ROC-AUC of
  0.953; rerunning that classifier needs the organisers' query texts, which are not released.
  See `DATASHEET.md`, section 2.

## Contents

| Path | What it is |
|---|---|
| `benchmark/items.jsonl` | the 174 generated items: queries, scene clauses, Q&A triple, answer frame, generation record, anchor verification, both image audits per gate, two-scene labels, text labels, membership in all twelve audit policies, TUNE/TEST splits |
| `benchmark/b60.jsonl` | the 60 keyframe-authored in-house queries with their keyframe answers |
| `benchmark/labels_organiser.tsv` | for each of the organisers' 91 queries: round, file stem, task, SHA-256 of the file, both text labellers' two-scene labels, surface features, out-of-fold classifier score; no query text |
| `pools/b174_raw.jsonl` | for each item, the 400 retrieved candidates `[video_id, frame_idx, similarity, last frame of the video]` |
| `pools/scene_b.jsonl` | for each item, the scene-B top-100 keyframes and the scene-B similarity of every keyframe the permutation reads |
| `pools/b174_prod.jsonl` | the production pool that `code/allocators.py::build_production_pool` rebuilds from the two files above |
| `pools/b60_raw.jsonl` | the 400 retrieved candidates for each keyframe-authored query |
| `keyframes/keyframe_index.tsv` | `video_id, n, frame_idx` for the 188 videos the items touch (47304 rows) |
| `prompts/` | every prompt used to generate, verify, audit and label the items, verbatim, with thresholds |
| `code/submission.py` | the official scorer and the ranking-based and coverage allocators, byte-identical to the file that produced the paper's numbers |
| `code/allocators.py`, `protocol.py`, `alloc_tables.py`, `stats.py` | the four allocators, the scoring protocol, the table aggregation, the statistics |
| `code/ladder.py` | the robustness ladder |
| `code/exact_dp.py`, `code/greedy_vs_opt.py` | the exact optimum, the exact optimiser, the oracles |
| `code/fair_comparison.py` | the fair comparison of tuned allocator families |
| `code/reproduce.py` | runs everything except the fair comparison and checks it |
| `expected/` | the values every script must match |
| `DATASHEET.md` | datasheet for the benchmark, with the audits and the known limitations |

## Not included

This release does not contain the organisers' videos or keyframe images, embeddings, OCR, speech
transcripts, captions, the organisers' 91 query texts (`benchmark/labels_organiser.tsv` has our
labels of them and their SHA-256 hashes, not the texts), or any round submission file. Every
`video_id` refers to the AI Challenge HCMC 2026 collection; to look at a frame, request the
collection from the organisers under their terms. The candidate pools are the similarity scores our
system computed, and the keyframe table holds only frame numbers.

## Known limitations

No human annotated the benchmark: every label, verdict and audit is automatic, apart from one team
member looking at 16 answer frames of the first batch. The GPT-5.2 audit stopped before
reaching 9 items (i07, i08, i09, i10, i11, i12, i13, i14, i15) and 33 answer
comparisons; 165 of 174 items have every GPT-5.2 image judgement. The
candidate pools come from one retrieval system. See `DATASHEET.md`, section 8.

## Licence and citation

The data (everything outside `code/`) is licensed under CC BY 4.0 (`LICENSE-DATA`). The code is
licensed under MIT (`LICENSE-CODE`). Please cite the paper; see `CITATION.cff`.

## Integrity

`SHA256SUMS` lists the SHA-256 of every file. To check it: `sha256sum -c SHA256SUMS`.

## Changes

- **2.1.0** (25 Sep 2026). `DATASHEET.md` states the generator's style examples correctly (nine
  round-1 organiser queries; seven shown in a two-scene prompt, six in a one-scene prompt; version
  2.0.0 said eleven). New: `benchmark/labels_organiser.tsv`; the transfer check and the share of the
  drop in `code/reproduce.py` and `expected/transfer_and_share.json`; the "Further checks" section;
  `CITATION.cff` points at this branch.
- **2.0.0** (24 Sep 2026). First public version.
