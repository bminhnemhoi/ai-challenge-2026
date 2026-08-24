"""Submission formatting — isolated on purpose (risk R4).

The 2026 preliminary-round submission format is UNCONFIRMED (frame index vs
timestamp, with/without an `answer` column).  Every format decision lives in
this one module so a format change on competition day is a one-line edit.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

#: currently assumed format; flip to "ts_ms" if the 2026 rules ask for timestamps
DEFAULT_VALUE_FIELD = "frame_idx"


def format_submission(
    video_id: str,
    events_meta: Sequence[dict],
    value_field: str = DEFAULT_VALUE_FIELD,
    answer: Optional[str] = None,
    sep: str = ",",
) -> str:
    """Build one submission line, e.g. ``L21_V001,1450,1802,2110,2455``.

    Parameters
    ----------
    events_meta : per-event metadata dicts holding at least ``frame_idx`` and
                  ``ts_ms`` (both are always kept in every record — risk R4).
    value_field : "frame_idx" | "ts_ms".
    answer      : optional textual answer appended as the last column (some
                  DRES variants require it).
    """
    if value_field not in ("frame_idx", "ts_ms"):
        raise ValueError(f"unsupported value_field {value_field!r}")
    values: List[str] = [str(int(ev[value_field])) for ev in events_meta]
    parts = [video_id, *values]
    if answer is not None:
        parts.append(str(answer))
    return sep.join(parts)


def format_csv(
    results: Sequence[dict],
    value_field: str = DEFAULT_VALUE_FIELD,
    strict: bool = False,
) -> str:
    """Render multiple candidate results as CSV lines (one per candidate).

    A record whose ``submission_line`` is ``None`` is SKIPPED, never re-derived
    from its events: the engine sets that field to ``None`` precisely when the
    frame numbers are keyframe ordinals rather than video frame indices, and
    re-deriving here would defeat the guard and emit a confidently wrong
    answer.  Pass ``strict=True`` to raise instead of skipping.

    A cached ``submission_line`` is reused only for the DEFAULT value field —
    the engine builds it with ``frame_idx``.  Asking for a different field
    re-derives the line from ``events``, which stays correct because the
    engine already emits them in event-number order.  Reusing the cache
    regardless would silently ignore ``value_field``, and that parameter is
    the whole point of isolating format changes here (risk R4).
    """
    lines: List[str] = []
    for r in results:
        cached = r.get("submission_line", ...)
        if cached is None:
            if strict:
                raise ValueError(
                    f"{r.get('video_id', '?')}: no submission line available "
                    f"({r.get('warning', 'frame indices unavailable')})"
                )
            continue
        if cached is not ... and value_field == DEFAULT_VALUE_FIELD:
            lines.append(cached)
        else:
            lines.append(
                format_submission(r["video_id"], r["events"], value_field=value_field)
            )
    return "\n".join(lines)
