# Datasheet: segment-authored KIS benchmark for the AI Challenge HCMC 2026 collection

Release 2.0.0. This datasheet follows the questions in Gebru et al., "Datasheets for Datasets"
(Communications of the ACM 64(12), 2021). Defects and gaps are listed as such, with counts.

## 1. Motivation

**Why was the dataset created?** The team tuned a video event retrieval system for the AI Challenge
HCMC 2026 on 60 test queries written while looking at single keyframes (B60, included here). Two text
labellers of different model families found no query in B60 that describes two consecutive scenes,
but they found that structure in 38% to 52% of the organisers' 91 queries of the 2026 rounds. B60
could therefore not measure anything about temporal structure. The generated set was built to fix
that: each query is written from a short video segment rather than one frame, with about half of the
queries assigned in advance to describe two consecutive scenes. The paper uses the set to show how
the way test queries are written changes what a test set shows, and to measure how the 100
submission lines should be allocated.

**Who created it and who funded it?** Ngo Binh Minh, Le Xuan Khanh and Ngo Lam Tien (Faculty of
Information Technology, Ton Duc Thang University), as a student team in the competition. There was no
external funding.

## 2. Composition

**What are the instances?** There are two sets.

* `benchmark/items.jsonl`: 174 generated items. Each item is a known-item search (KIS) query in
  Vietnamese with an English translation, a question-and-answer (Q&A) triple about the same moment,
  and one answer moment given as `video_id`, `frame_idx` and keyframe number `keyframe_n`. Items that
  were assigned the two-scene quota also carry the two scene clauses `scene_A` and `scene_B`; their
  answer moment is the first frame of scene B. For one-scene items it is the frame the query
  describes.
* `benchmark/b60.jsonl`: the 60 keyframe-authored in-house queries (B60), each with its
  keyframe answer and a Q&A triple.

**What is scored?** The paper scores the 158 items outside shard c (`in_clean_set`; see
section 8). Audit policies select subsets of these items (`policies`, one flag per policy):

| Policy | Rule |
|---|---|
| `P0_LEGACY` | every item outside shard c (no removal) |
| `P2_NEO` | P0, minus items whose answer frame fails the Gemini audit |
| `P3_QA` | P2, minus items whose Q&A triple fails the Gemini audit |
| `P4_STRICT` | items that pass every gate of the Gemini audit |
| `X0_AUDITED` | P0, minus the 7 rejects of an earlier audit that covered 64 items (historical) |
| `C_BOTH_ANCHOR` | P0, minus items whose answer frame both judges reject |
| `C_EITHER_ANCHOR` | **the paper's primary set**: P0, minus items whose answer frame either judge rejects |
| `C_EITHER_ANCHOR_QA` | C_EITHER_ANCHOR, minus items in which either judge finds a Q&A defect (an unresolved GPT-5.2 answer comparison counts as a defect) |
| `C_EITHER_ANCHOR_QA_G3LENIENT` | the same, an unresolved comparison counts as correct |
| `C_STRICT_BOTH` | items that pass every gate under both judges (unresolved comparisons count as defects) |
| `C_STRICT_BOTH_G3LENIENT` | the same, unresolved comparisons count as correct |
| `C_STRICT_EITHER` | items that pass every gate under at least one judge |

The `C_*` policies use both judges. Where GPT-5.2 has no verdict on a gate (see "Cross-family
audit" below) the item takes Gemini's verdict on that gate; `audit.consensus_gemini_fallback_gates`
lists those gates. 8 items of the primary set have such a fallback. The primary set
has 148 items in 133 videos. It was fixed on validity grounds before any result was
computed on it. Nestings: P4 ⊂ P3 ⊂ P2 ⊂ P0; C_STRICT_BOTH ⊂ C_EITHER_ANCHOR_QA ⊂ C_EITHER_ANCHOR ⊂
C_BOTH_ANCHOR ⊂ P0; C_EITHER_ANCHOR ⊂ P2.

**Is anything missing?** Yes. The videos, keyframe images, speech transcripts, OCR and captions are
the organisers' material and are not included. A reader needs the AI Challenge HCMC 2026 collection
(873 videos, 177,321 keyframes, Vietnamese television news and features) to look at any frame. The
organisers' 91 queries are not included, and neither are the team's translations of them.

**Label fields, and how far they can be trusted.** No human labelled any item for this release.
Every label is automatic, and each records which model family produced it (`two_scene_labels`):

* `declared`: the quota the generator was given before it saw the video. It is an instruction, not a
  measurement.
* `gemini_image_evidence`: true only when the Gemini image audit found scene A in a keyframe before
  the answer frame and not in the answer frame itself.
* `gpt_image_evidence`: the same judgement by GPT-5.2; null where GPT-5.2 did not reach the gate.
* `both_judges_image_evidence`: true only when both judges find the evidence (Gemini's verdict where
  GPT-5.2 has none). This is the stratum label of the `C_*` policies and of the paper.
* `text_labels_two_scene`: the judgement of two text-only labellers (Gemini, prompt version
  2; GPT-5.2 with the same prompt), which read the query text and not the images:
  174 items and 60 B60 queries carry a GPT label.

**Splits.** `splits.E2` is the TUNE/TEST split of the 158 scored items, stratified by the
declared flag: within each stratum the first floor(n/2) items of a seeded permutation go to TUNE
(seed 20260917). `splits.OWN.<policy>` applies the same rule to the policy's own items, stratified by
the policy's label (Gemini evidence for P2 to P4, both judges' evidence for the `C_*` policies).
`splits.OWN.C_EITHER_ANCHOR` is the split of the paper (TUNE 73 / TEST 75).
Nothing was tuned on the held-out halves: the shipped allocator parameters were fixed on B60.

**Relations between instances.** Among the 158 scored items, 15 videos
contribute two items each. Item ids are shard codes (`a00` is item 0 of generation shard a).

**Does the data identify people, or contain sensitive content?** The videos are broadcast news.
Queries describe the visible appearance of people (clothing, pose), on-screen text, and sometimes the
names of public figures or places shown on screen. The generator was told not to invent names. The
dataset does not target individuals.

## 3. Collection process

**Generation** (`prompts/01_*`, source `scripts/sinh_gt_doan_video.py`). Videos were sampled at random
within the collection's video bands (L21 to L30), with a fixed seed per shard. From each video the
generator received one segment of consecutive keyframes: 12 keyframes for the two-scene quota
and 9 for the one-scene quota, chosen before the segment was seen. The frames came in temporal
order at 512 px, with the speech transcript of that segment when one existed. The generator wrote
one query, chose the answer frame, and wrote a Q&A triple whose answer must be a concrete noun, at
temperature 0.35, without any retrieval score. The prompt contained 11 real organiser queries
as examples of style (five of the requested structure, four of the other, two contrast examples) and
told the generator to copy their style and not their content. Those example texts are not released;
`prompts/01_generation_prompt.txt` shows where they went (`{vi_du}`). When the model declared a
segment unusable, another segment of the same video was drawn (`generator.resampled_segments`). No
item was dropped because the retrieval system failed on it. Generator checkpoints: gemini-3.1-flash-lite 52, gemini-3.5-flash-lite 51, gemini-flash-lite-latest 45, gemini-2.5-flash-lite 26.

**Anchor verification** (`prompts/03_*`, source `scripts/kiem_neo_don_anh.py`, verifier gpt-5.2 at
512 px). One request sent one keyframe and the query (scene B for two-scene items) and asked for a 0
to 100 match score. The verifier scored ±2 keyframes around the answer frame, moved it when
a neighbour scored at least 15 points higher, and flagged the item when every frame scored
below 55. 27 of 174 answer frames were moved
(`anchor_verification.outcome = moved`; the original is kept in `generator_anchor_n` and
`generator_anchor_frame_idx`). One flagged item was quarantined; it is not among the 174.
Before this step, 3 answer frames of shard a had been corrected by a team member who looked
at the frames (`manual_fix_before_verification`); 16 items of shard a carry
`anchor_eye_checked`. This is the only human inspection in the pipeline, and it is not a systematic
annotation.

**Image audit, first judge** (`prompts/05_*` to `09_*`, source `scripts/kiem_gt_moi.py`). The judge was
gemini-3.5-flash-lite (fallbacks gemini-2.5-flash-lite, gemini-flash-lite-latest, gemini-3.1-flash-lite, gemini-2.5-flash, gemini-flash-latest), at temperature 0, eight images per request, without
access to the generator's self-reported confidence. Its gates:

1. **Answer frame** (G1). The answer frame is scored against the query (scene B for two-scene items).
   It fails below 50/100.
2. **Scene A** (G2). Up to 7 keyframes before the answer frame are scored against
   scene A. Scene A is absent if every score is below 50. The two-scene claim is fake if scene
   A also scores at least 70 on the answer frame while the answer frame scores at least
   70.
3. **Q&A** (G3). The question is put to the judge with only the answer frame visible, and the answer
   is compared with the item's answer in a separate text-only pass.
4. **Not generic** (G4). A text judgement of whether the query pins down one moment. It rejects an
   item only when at least 5 of 15 other frames of the same video also
   score at least 70.
5. **Detail check.** Contradicted details are recorded as a flag. This gate never rejects.

Coverage: 174 of 174 items; the 27 moved answer frames were re-audited on
their current frames (`verdict_on_current_anchor` is true for 174 items). Outcome on the
158 scored items: 106 pass, 52 reject, 0 undecided. Rejections that
concern retrieval: 23 (answer frame does not depict the query 7, fake
two-scene claim 9, scene A absent 3, generic description 4; one item can
carry more than one reason). Rejections that concern only the Q&A triple: 29 (not answerable
from the answer frame 14, wrong answer 15). The answer-frame scores are bimodal:
157 score 95 or more, 7 score 40 or less, 10 fall in between.

**Cross-family audit, second judge** (`audit.gpt`; source `scripts/e14_kiem_cheo_ho.py`). GPT-5.2
(gpt-5.2-2025-12-11) applied gates G1 to G4 with the same prompts, thresholds and 512-px keyframes,
one image per request, to all 174 items. The OpenAI credit ran out before the run finished,
so the audit is incomplete, and `audit.gpt.measurement` says how far it got for each item:

* `complete`: 132 items;
* `images_complete_answer_comparison_unresolved`: 33 items. Every image
  judgement was made, but the text-only comparison of GPT-5.2's answer with the item's answer was not.
  For these items `audit.gpt.gates.G3_qa` is null and the file gives both bounds:
  `verdict_if_unresolved_answer_counted_correct` and `..._counted_wrong`;
* `anchor_gate_only` and `not_measured`: 9 items (i07, i08, i09, i10, i11, i12, i13, i14, i15) that
  GPT-5.2 did not reach (1 of them has only the answer-frame score).

So 165 of 174 items have every GPT-5.2 image judgement, and
9 do not have a complete GPT audit. On the 158 scored items GPT-5.2's
verdict (unresolved comparisons left undecided) is: 83 pass, 38 reject,
37 undecided. The current answer frames were positioned by gpt-5.2 during anchor
verification, so this judge's answer-frame gate is not independent of the answer frame's placement;
its other gates are clean cross-family checks. Both audits keep the numeric scores and the pass/fail
outcome of every gate; every free-text rationale and each judge's own reading of the frame are
dropped, because those texts describe organiser video content.

**Candidate pools** (`pools/`). These are the team's retrieval scores, computed on 3 September 2026 by
the competition system: all keyframes ranked by a four-prompt SigLIP-2 ensemble over the Vietnamese
and English readings of the query, with peak preference and an object boost, top 400 kept. The
production pool adds, for 71 declared two-scene items (71 of them among the
scored items), the top 100 keyframes by similarity to the scene-B clause, appended after position
400, and permutes scores within the top three videos by scene-B similarity. `pools/scene_b.jsonl`
holds exactly what that step reads, so the production pool can be rebuilt without the similarity
matrix. Retrieval itself is not bit-exact across machines, which is why the pools are released.

## 4. Preprocessing, cleaning and labelling

* Items were never filtered on retrieval difficulty.
* Shard c (16 items) is kept in the file but excluded from every scored set; see section 8.
* The audits remove items only through the policies. `benchmark/items.jsonl` keeps every item and
  every verdict of both judges, so any policy can be recomputed.
* Q&A answers that were only a generic category word triggered one rewrite
  (`prompts/01e_generation_qa_repair_suffix.txt`). `generator.has_generator_warning` marks items whose
  answer stayed weak or unverified.

## 5. Uses

**Intended.** Offline evaluation of known-item search and of submission-line allocation on the AIC
2026 collection under the official R@k scoring, and a worked example of building a test set from
video segments and auditing it with judges of two model families.

**Not intended.** The set is not a sample of the organisers' queries. A character n-gram classifier
separates it from the 91 organiser queries with ROC-AUC 0.95; the organiser queries are longer
(median 63 words against 44) and contain more numerals. The set matches them in structure (the rate
of consecutive-scene descriptions), not in surface or in the channel that decides the answer. Scores
on this set are not estimates of scores on real competition queries. The set must not be used to
train models that are then evaluated on it, and it must not be used for any claim about individuals
shown in the videos.

**What could mislead a user?** Scores depend on the audit policy (see the table in `README.md`) and on
how the true moment is modelled inside a keyframe cell; the paper draws it uniformly within the cell.
Scoring against the keyframe itself flatters any keyframe-only system, especially on B60, where 56 of
the 60 answers are keyframes (the correct video is among the coverage allocator's 100 lines for
57 of the 60 queries).

## 6. Distribution

This release is distributed as a public archive under CC BY 4.0 for the data and MIT for the code. It
contains only material the three authors produced: generated query and Q&A text, answer-frame
numbers, audit scores and outcomes, retrieval scores, keyframe numbers, prompts and code. It contains
no video, image, embedding, OCR, speech or caption text, no organiser query text and no submission
file. `video_id` and frame numbers refer to the organisers' collection, which the organisers
distribute under their own terms. `keyframes/keyframe_index.tsv` lists the frame numbers of the
organisers' keyframes in the videos the items touch; it holds integers only and is needed to place
the true moment inside a keyframe cell.

## 7. Maintenance

The authors maintain the release. Errata and new audit results (for example a completed GPT-5.2 run)
will be published as new versions with a changelog; the version number is in `README.md` and
`CITATION.cff`. `SHA256SUMS` identifies the exact files of a version. The builder that produced this
release is `scripts/xuat_ban_phat_hanh_v2.py` in the team's repository; it is deterministic (two
builds produce identical checksums).

## 8. Known limitations and defects (read before using the set)

1. **No human annotation.** Answer frames, two-scene labels and Q&A answers are automatic judgements.
   The only human checks are the 3 manual answer-frame fixes and the look at the answer frames
   of shard a. Two automatic judges of different families bound model-specific errors; they do not
   show that human annotators would accept the same items.
2. **The GPT-5.2 audit is incomplete.** 9 items (i07, i08, i09, i10, i11, i12, i13, i14, i15) have no
   complete GPT-5.2 audit and 33 answer comparisons are unresolved (see section 3).
   Where GPT-5.2 has no verdict, the two-judge policies use Gemini's; 8 items of the
   primary set are affected. Completing the run can only remove items from the either-judge policies.
3. **The first judge belongs to the generator's model family** (both Gemini flash-lite). The
   cross-family audit is the check on a shared blind spot.
4. **The second judge placed the answer frames.** GPT-5.2 moved 27 answer frames during
   verification, so its answer-frame gate is not an independent check of those frames.
5. **Declared and evidenced two-scene structure differ.** Of the 158 scored items, 79 were
   declared two-scene; 67 have Gemini's image evidence; the primary set has 73 declared and
   59 with both judges' evidence. Use the image-evidence labels for claims about temporal
   structure.
6. **Shard c** assigned the two-scene quota by the same index parity that assigned the video band, so
   the quota and the band are confounded. Its 16 items are excluded from every scored set.
7. **Shard b** has no `scene_A` or `scene_B` fields; its two-scene flag is self-declared. The audit
   split its queries with the text labeller before scoring scene A.
8. **Shard d** items do not record a generator prompt version (`generator.prompt_version` is null).
   45 items were generated through an unpinned model alias (`generator.model` names the alias).
9. **One answer per item.** Adjacent keyframes are sometimes indistinguishable from the answer frame;
   a system can then find the right shot and still be scored as a miss.
10. **Some Q&A answers are channel logos or on-screen text** that is constant across a programme. They
    are concrete, but they do not localise.
11. **The generation prompt used organiser queries as style examples**, and the queries were written
    from the speech transcript as well as the frames. No run of twelve or more words is shared with
    any organiser text (queries, speech, on-screen text, titles and descriptions); the shorter shared
    runs are stock phrases of the query style.
12. **Candidate pools come from one retrieval system** (SigLIP-2). Allocation results on these pools
    say nothing about pools produced by other encoders.
