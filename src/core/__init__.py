"""Core shared modules for AI Challenge 2026.

Dataset index builders, the official scorer, submission packaging, and the
retrieval helpers every task leans on.

Nothing is imported eagerly. `index_builder` needs torch and `gemini_engine`
needs the Gemini client, so re-exporting them at package-import time meant that
`import src` — which every script and every test does — dragged in torch and
transformers before a single line ran: about ten seconds of startup, and a CI
job that had to install the whole ML stack just to exercise pure-arithmetic
scoring tests. Names below still resolve exactly as before, they just load on
first use (PEP 562).
"""

from typing import TYPE_CHECKING

_LAZY = {
    "build_official_btc_index": (".btc_index_builder", "build_official_btc_index"),
    "KeyframeIndexBuilder": (".index_builder", "KeyframeIndexBuilder"),
    "AIC2026Evaluator": (".evaluator", "AIC2026Evaluator"),
    "GeminiAIOptimizer": (".gemini_engine", "GeminiAIOptimizer"),
}

__all__ = list(_LAZY)

if TYPE_CHECKING:  # so editors and type checkers still see the real symbols
    from .btc_index_builder import build_official_btc_index
    from .evaluator import AIC2026Evaluator
    from .gemini_engine import GeminiAIOptimizer
    from .index_builder import KeyframeIndexBuilder


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module, __name__), attr)
    raise AttributeError(f"module 'src.core' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
