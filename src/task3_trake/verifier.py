"""L4 — VVR: VLM Verification Rerank (spec section 7).

Before submitting, the top candidate sequences are verified frame-by-frame by
a VLM ("does this image show <event>?").  Final score::

    final = 0.7 * normalize(CHRONOS_score) + 0.3 * mean(confidence_i)

The VLM client is injected (any callable object with a ``verify`` method), so
this module has no model/network dependency of its own.  :class:`MockVLM`
provides a deterministic stand-in for tests and offline benchmarks.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Protocol, Sequence

VERIFY_PROMPT = (
    'Ảnh này có thể hiện: "{event}" không? '
    'Trả về JSON {{"match": true/false, "confidence": 0-1, "reason": "<10 từ"}}'
)


class VLMClient(Protocol):
    def verify(self, image_ref: str, event_text: str) -> dict:
        """Return {"match": bool, "confidence": float, "reason": str}."""
        ...


class MockVLM:
    """Deterministic VLM stand-in.

    ``judge`` maps ``(image_ref, event_text) -> confidence in [0, 1]``;
    default answers 0.5 for everything.
    """

    def __init__(self, judge: Optional[Callable[[str, str], float]] = None) -> None:
        self.judge = judge or (lambda image_ref, event_text: 0.5)
        self.calls: List[tuple] = []

    def verify(self, image_ref: str, event_text: str) -> dict:
        self.calls.append((image_ref, event_text))
        conf = float(self.judge(image_ref, event_text))
        conf = min(max(conf, 0.0), 1.0)
        return {"match": conf >= 0.5, "confidence": conf, "reason": "mock"}


def _minmax_normalize(scores: Sequence[float]) -> List[float]:
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def verify_and_rerank(
    results: List[dict],
    event_texts: Sequence[str],
    vlm: VLMClient,
    topk_verify: int = 3,
    w_chronos: float = 0.7,
    w_vlm: float = 0.3,
) -> List[dict]:
    """Verify the top ``topk_verify`` candidates and rerank them in place.

    Each result must carry ``score`` (CHRONOS) and ``events`` — a list of
    per-event dicts with a ``thumb`` (image reference) field.  Verified
    results gain ``vlm_match`` / ``vlm_conf`` per event and a ``final_score``;
    only the verified head of the list is re-ordered, the tail keeps its
    original CHRONOS ordering below it.

    Each frame is paired with its own event's text via the record's ``idx``
    field, never by position and never by re-applying ``event_order``.  The
    engine emits ``events`` already sorted by event number while
    ``event_order`` remains the CHRONOLOGICAL permutation, so permuting one
    list and not the other transposes the pairs — and asking the VLM the wrong
    question deflates the confidence of a CORRECT swapped sequence, demoting
    it below a wrong one.
    """
    if not results or topk_verify <= 0:
        return results
    head = results[:topk_verify]
    tail = results[topk_verify:]

    norm = _minmax_normalize([r["score"] for r in head])
    for r, ns in zip(head, norm):
        texts = [
            event_texts[ev["idx"] - 1]
            if isinstance(ev.get("idx"), int) and 1 <= ev["idx"] <= len(event_texts)
            else ""
            for ev in r["events"]
        ]
        confs: List[float] = []
        for ev, text in zip(r["events"], texts):
            verdict = vlm.verify(ev.get("thumb", ""), text)
            ev["vlm_match"] = bool(verdict.get("match", False))
            ev["vlm_conf"] = float(verdict.get("confidence", 0.0))
            ev["vlm_reason"] = str(verdict.get("reason", ""))
            confs.append(ev["vlm_conf"])
        r["final_score"] = w_chronos * ns + w_vlm * (sum(confs) / len(confs) if confs else 0.0)

    head.sort(key=lambda r: -r["final_score"])
    return head + tail
