"""Guards for long-query chunking.

SigLIP-2's text tower takes 64 tokens and silently drops the rest.  Measured on
the official round-1 query set, 13 of 24 descriptions exceed that, and what
falls off is the identifying detail — a describer sets the scene first and
states the specifics last.  On a padded ground-truth set, truncating scored
0.029 against 0.323 for chunking (scripts/experiment_long_query.py).

These tests use a stub tokenizer so they run without loading the 3.5 GB model.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.kis_engine import KISEngine  # noqa: E402


class _StubTokenizer:
    """One token per whitespace word, plus 2 for the special tokens."""

    def __call__(self, text, **kw):
        return {"input_ids": [0] * (len(str(text).split()) + 2)}


class _StubProcessor:
    tokenizer = _StubTokenizer()


@pytest.fixture
def engine():
    e = KISEngine.__new__(KISEngine)
    e.processor = _StubProcessor()
    return e


def test_short_text_is_left_alone(engine):
    text = "An orange tabby cat standing on a concrete yard."
    assert engine.chunk_text(text, max_tokens=58) == [text]


def test_long_text_is_split_on_sentence_boundaries(engine):
    sents = [f"Sentence number {i} describes another part of the scene." for i in range(12)]
    chunks = engine.chunk_text(" ".join(sents), max_tokens=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) + 2 <= 20
    # nothing is dropped
    joined = " ".join(chunks)
    for s in sents:
        assert s in joined


def test_the_tail_survives_which_is_the_whole_point(engine):
    """The identifying detail lives at the END of a contest description."""
    text = (
        "This is a scene from a Vietnamese television report recorded during the day. "
        "The camera is steady and the light is natural throughout the whole sequence. "
        "People are going about ordinary activities in the background of the shot. "
        "The second racer wears a red hat and the last racer wears a black hat."
    )
    chunks = engine.chunk_text(text, max_tokens=20)
    assert any("black hat" in c for c in chunks), "the distinguishing clause was lost"


def test_a_single_overlong_sentence_is_split_on_commas(engine):
    text = ", ".join(f"clause number {i} with several words" for i in range(10))
    chunks = engine.chunk_text(text, max_tokens=15)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) + 2 <= 15 or "," not in c


def test_empty_and_whitespace_are_safe(engine):
    assert engine.chunk_text("") == [""]
    assert engine.chunk_text("   ") == [""]


def test_newline_separated_events_split(engine):
    text = "E1: first moment happens here\nE2: second moment happens there\nE3: third one"
    chunks = engine.chunk_text(text, max_tokens=10)
    assert len(chunks) >= 2


def test_chunks_never_exceed_the_limit(engine):
    """The limit is what the encoder can actually see; exceeding it is the bug."""
    import random

    rng = random.Random(0)
    for _ in range(40):
        n = rng.randint(1, 60)
        text = ". ".join(" ".join("w" for _ in range(rng.randint(1, 15))) for _ in range(n))
        for c in engine.chunk_text(text, max_tokens=30):
            assert len(c.split()) + 2 <= 30 or " " not in c
