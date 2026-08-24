"""
AI Challenge 2026 (AIC 2026) Master Source Package.

Subpackages:
- src.core: shared index builders, the official scorer, submission packaging
- src.task1_kis: Textual Known Item Search (Member 1)
- src.task2_vqa: Visual Question Answering (Member 2 & 3)
- src.task3_trake: Temporal Alignment Engine — CHRONOS (Member 4)

Eager re-exports are deliberately kept to the ones that always import.  Pulling
every engine in at package-import time meant one optional dependency (opencv,
paddleocr, a Gemini key) could stop `import src` outright, and with it the two
tasks that never needed that dependency in the first place.  Heavy engines are
therefore exposed lazily through ``__getattr__`` (PEP 562): the name still
works, it just loads on first use and raises where it can be understood.
"""

from src.core import AIC2026Evaluator, KeyframeIndexBuilder, build_official_btc_index

__all__ = [
    "build_official_btc_index",
    "KeyframeIndexBuilder",
    "AIC2026Evaluator",
    "TextualKISRetriever",
    "VisualQAEngine",
    "TRAKEEngine",
]

_LAZY = {
    "TextualKISRetriever": ("src.task1_kis", "TextualKISRetriever"),
    "VisualQAEngine": ("src.task2_vqa", "VisualQAEngine"),
    "TRAKEEngine": ("src.task3_trake", "TRAKEEngine"),
}


def __getattr__(name: str):
    if name in _LAZY:
        module, attr = _LAZY[name]
        import importlib

        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'src' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
