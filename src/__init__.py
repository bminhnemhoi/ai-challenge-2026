"""
AI Challenge 2026 (AIC 2026) Master Source Package.
Subpackages:
- src.core: Shared Index Builders & Evaluators
- src.task1_kis: Textual Known Item Search (Member 1)
- src.task2_vqa: Visual Question Answering (Member 2 & 3)
- src.task3_trake: Temporal Alignment Engine (Member 4)
"""

from src.core import build_official_btc_index, KeyframeIndexBuilder, AIC2026Evaluator
from src.task1_kis import TextualKISRetriever
from src.task2_vqa import VisualQAEngine
from src.task3_trake import TRAKEEngine

__all__ = [
    "build_official_btc_index",
    "KeyframeIndexBuilder",
    "AIC2026Evaluator",
    "TextualKISRetriever",
    "VisualQAEngine",
    "TRAKEEngine"
]
