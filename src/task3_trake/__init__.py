"""Task 3 — TRAKE / CHRONOS (Member 4).

CHRONOS solves the ordered-subsequence-selection problem at the heart of TRAKE
with an O(N*T) dynamic program over the whole keyframe corpus. See
``alignment.py`` for the recurrence and the reasoning behind each term.
"""

from .alignment import (
    NEG,
    adaptive_lambda,
    apply_anchors,
    chronos_align,
    chronos_align_ref,
    chronos_search,
    estimate_tempo,
    estimate_tempo_dp,
    score_scale,
    soft_order_align,
    strict_window_align,
    unordered_align,
)
from .decomposer import (
    DecomposedQuery,
    Event,
    LLMDecomposer,
    RuleBasedDecomposer,
    detect_first_occurrence,
)
from .scoring import SCORING_MODES, blend_ook_vector, fuse, zscore_rows
from .trake_engine import TRAKEEngine

__all__ = [
    "TRAKEEngine",
    "NEG",
    "adaptive_lambda",
    "apply_anchors",
    "chronos_align",
    "chronos_align_ref",
    "chronos_search",
    "estimate_tempo",
    "estimate_tempo_dp",
    "score_scale",
    "soft_order_align",
    "strict_window_align",
    "unordered_align",
    "DecomposedQuery",
    "Event",
    "LLMDecomposer",
    "RuleBasedDecomposer",
    "detect_first_occurrence",
    "SCORING_MODES",
    "blend_ook_vector",
    "fuse",
    "zscore_rows",
]
