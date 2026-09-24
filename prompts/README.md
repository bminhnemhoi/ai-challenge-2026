# Prompts

Every file below was extracted from the source code with Python's `ast` module and was not
retyped. The prompts are in Vietnamese, because the queries and the collection are Vietnamese.
Placeholders in braces (`{n}`, `{q}`, `{mo_ta}`, ...) are filled in at call time.

- `prompts/01_generation_prompt.txt`
- `prompts/01a_generation_requirement_two_scene.txt`
- `prompts/01b_generation_requirement_one_scene.txt`
- `prompts/01c_generation_anchor_rule_two_scene.txt`
- `prompts/01d_generation_anchor_rule_one_scene.txt`
- `prompts/01e_generation_qa_repair_suffix.txt`
- `prompts/02_multi_frame_anchor_recheck_prompt.txt`
- `prompts/02a_multi_frame_anchor_rule_two_scene.txt`
- `prompts/02b_multi_frame_anchor_rule_one_scene.txt`
- `prompts/03_single_image_anchor_verification_prompt.txt`
- `prompts/04_two_scene_text_labeller_prompt.txt`
- `prompts/05_audit_frame_scoring_prompt.txt`
- `prompts/06_audit_qa_answer_from_anchor_prompt.txt`
- `prompts/07_audit_answer_equivalence_prompt.txt`
- `prompts/08_audit_detail_contradiction_prompt.txt`
- `prompts/08a_audit_detail_contradiction_two_scene_suffix.txt`
- `prompts/09_audit_localisability_prompt.txt`

## Where each prompt is used, and with which settings

| Step | Files | Model | Settings |
|---|---|---|---|
| Generate a query, anchor and Q&A from a video segment | `01_*` | Gemini flash-lite checkpoints (gemini-3.1-flash-lite 52, gemini-3.5-flash-lite 51, gemini-flash-lite-latest 45, gemini-2.5-flash-lite 26) | temperature 0.35; 12 keyframes (two-scene quota) or 9 (one-scene quota); JSON output |
| Re-check the anchor on the whole segment (superseded by step 3; kept for completeness) | `02_*` | Gemini flash-lite | multi-image request |
| Verify the anchor, one image per request | `03_*` | gpt-5.2, 512 px images | scan ±2 keyframes; move if a neighbour scores ≥ 15 points higher; flag if every frame < 55 |
| Two-scene text labeller | `04_*` | Gemini flash-lite; the same prompt is used for the GPT labeller | temperature 0; prompt version 2 |
| Image audit: score frames against a description | `05_*` | gemini-3.5-flash-lite (fallbacks gemini-2.5-flash-lite, gemini-flash-lite-latest, gemini-3.1-flash-lite, gemini-2.5-flash, gemini-flash-latest) | temperature 0; anchor fails < 50; scene A absent < 50; fake two-scene if scene A ≥ 70 on the anchor; "other frame matches" ≥ 70 |
| Image audit: answer the question from the anchor frame only | `06_*` | same | temperature 0 |
| Image audit: compare two answers | `07_*` | same | temperature 0, text only |
| Image audit: detail contradictions (a flag, never a gate) | `08_*` | same | temperature 0 |
| Image audit: does the query pin down one moment | `09_*` | same | temperature 0, text only; rejects only if ≥ 5 of 15 other frames match |

**Generation few-shot examples.** The generation prompt's `{vi_du}` slot was filled with real
organiser query texts as examples of style. They are the organisers' text and are not included here.
The generator also received the automatic speech transcript of the segment when one existed
(`{loi_thoai}`). That transcript is not included either.

**Cross-family audit.** The second judge (gpt-5.2-2025-12-11) applied the same four gates with the
same prompts (`05_*`, `06_*`, `07_*` and `09_*`), the same thresholds and the same 512-px keyframes,
sending one image per request instead of Gemini's batches of eight. It did not run the detail check
(`08_*`). Its verdicts are in `benchmark/items.jsonl` under `audit.gpt`.
