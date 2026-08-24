"""The spoken channel: what is SAID in a video, with timestamps.

Everything else in this system decides from one modality — what a keyframe looks
like. That fails in a specific and measurable way: query-p1-21 asks for research
at a university in Lausanne into insect flight for robotics, and the visual model
ranked a Vietnamese lifestyle clip first, ahead of the right kind of video by
0.003 (1.8%), because a news anchor at a desk looks like a news anchor at a desk.
A presenter saying "Lausanne" does not have that problem.

The AIC 2025 team that scored 79/88 ran three text channels beside the visual one
(MERVIN, arXiv:2605.16120 §3.1). This is the first of ours.

Two properties make it worth more than plain metadata:

* it is TIMESTAMPED, so a hit localises a moment, not just a video — which is
  exactly what TRAKE needs and what a title cannot give;
* it carries proper nouns. "Lausanne", "củ năng", "Nguyễn Trung Trực" are
  invisible to an image encoder and unambiguous in text.

Retrieval here is BM25 over word unigrams and bigrams rather than a second
embedding model. Vietnamese ASR output is noisy but its NOUNS are usually right,
and exact lexical overlap is precisely the signal a dense encoder throws away.
No model to download, no GPU, milliseconds per query.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: Vietnamese function words plus the boilerplate every one of these broadcasts
#: opens with. Left in, "chương trình" and "quý vị" match every news video.
STOP = {
    "và", "của", "có", "là", "được", "cho", "với", "các", "những", "này", "đó",
    "một", "trong", "khi", "đã", "sẽ", "cũng", "như", "để", "từ", "tại", "về",
    "người", "nhiều", "rất", "thì", "mà", "nên", "vì", "nếu", "hay", "hoặc",
    "chương", "trình", "quý", "vị", "kính", "chào", "xin", "mời", "theo", "dõi",
    "hôm", "nay", "ngày", "tháng", "năm", "sau", "đây", "tin", "tức", "thông",
    "the", "and", "for", "with", "that", "this", "you", "are", "was",
}


def normalise(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or "")).lower()


def tokenise(text: str) -> List[str]:
    """Unigrams plus bigrams.

    Vietnamese is written in syllables, so "củ năng" and "măng tây" are two
    tokens each and the unigrams alone would match any video mentioning "củ" or
    "tây". The bigram is the discriminative unit.
    """
    words = re.findall(r"[0-9a-zà-ỹ]+", normalise(text))
    words = [w for w in words if len(w) > 1]
    grams = [w for w in words if w not in STOP]
    grams += [
        f"{a}_{b}"
        for a, b in zip(words, words[1:])
        if not (a in STOP and b in STOP)
    ]
    return grams


class TranscriptIndex:
    """BM25 over per-video transcripts, with per-segment timestamps kept."""

    #: standard BM25 constants; k1 controls term-frequency saturation and b the
    #: length normalisation, which matters here because a 20-minute news
    #: bulletin has 30x the text of a 3-minute cooking clip
    K1 = 1.2
    B = 0.75

    def __init__(self) -> None:
        self.docs: Dict[str, List[str]] = {}
        self.segments: Dict[str, List[Tuple[float, str]]] = {}
        self.titles: Dict[str, str] = {}
        self._df: Counter = Counter()
        self._tf: Dict[str, Counter] = {}
        self._len: Dict[str, int] = {}
        self._avg = 1.0
        self._postings: Dict[str, List[str]] = defaultdict(list)

    # ------------------------------------------------------------------ load
    def add(self, video_id: str, segments: Sequence, title: str = "") -> None:
        segs = [
            (float(s.get("start", 0.0)), str(s.get("text", "")).strip())
            if isinstance(s, dict)
            else (float(s[0]), str(s[1]).strip())
            for s in segments
        ]
        segs = [(t, x) for t, x in segs if x]
        if not segs:
            return
        self.segments[video_id] = segs
        self.titles[video_id] = title
        # the title is worth repeating: it is human-written, unlike the ASR, and
        # for near-identical cooking or lion-dance videos it is the discriminator
        text = " ".join(x for _t, x in segs) + (" " + title) * 3
        self.docs[video_id] = tokenise(text)

    def load_dir(self, *dirs: Path) -> "TranscriptIndex":
        """Later directories fill gaps in earlier ones, never overwrite them."""
        for d in dirs:
            d = Path(d)
            if not d.is_dir():
                continue
            for p in sorted(d.glob("*.json")):
                vid = p.stem
                if vid in self.segments:
                    continue
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 - a truncated file is not fatal
                    continue
                if isinstance(raw, dict):
                    self.add(vid, raw.get("segments") or [], raw.get("title", ""))
                elif isinstance(raw, list):
                    self.add(vid, raw)
        return self.build()

    def build(self) -> "TranscriptIndex":
        self._df = Counter()
        self._tf = {}
        self._len = {}
        self._postings = defaultdict(list)
        for vid, toks in self.docs.items():
            tf = Counter(toks)
            self._tf[vid] = tf
            self._len[vid] = len(toks)
            for term in tf:
                self._df[term] += 1
                self._postings[term].append(vid)
        self._avg = (sum(self._len.values()) / len(self._len)) if self._len else 1.0
        return self

    @property
    def n_videos(self) -> int:
        return len(self.docs)

    # ----------------------------------------------------------------- score
    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self._df.get(term, 0)
        if not df:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def idf_of(self, term: str) -> float:
        """How distinctive a word is. Public so callers can require real evidence."""
        return self._idf(term)

    def score_videos(self, query: str, restrict: Optional[Iterable[str]] = None) -> Dict[str, float]:
        """BM25 score per video. Only videos containing a query term are touched."""
        terms = Counter(tokenise(query))
        if not terms:
            return {}
        allow = set(restrict) if restrict is not None else None
        out: Dict[str, float] = defaultdict(float)
        for term, qtf in terms.items():
            idf = self._idf(term)
            if idf <= 0:
                continue
            for vid in self._postings.get(term, ()):
                if allow is not None and vid not in allow:
                    continue
                f = self._tf[vid][term]
                dl = self._len[vid]
                out[vid] += (
                    qtf * idf * f * (self.K1 + 1) / (f + self.K1 * (1 - self.B + self.B * dl / self._avg))
                )
        return dict(out)

    def best_segment(self, query: str, video_id: str, window: int = 5):
        """(start_seconds, text) of the passage in one video that best matches.

        Scored over a sliding window of consecutive cues, because a single ASR
        cue is about three seconds of speech and a query's nouns are usually
        spread across a sentence or two.
        """
        segs = self.segments.get(video_id)
        if not segs:
            return None
        terms = Counter(tokenise(query))
        if not terms:
            return None
        best, best_s = None, 0.0
        for i in range(len(segs)):
            chunk = segs[i : i + window]
            tf = Counter(tokenise(" ".join(x for _t, x in chunk)))
            s = sum(self._idf(t) * min(tf.get(t, 0), 3) * q for t, q in terms.items())
            if s <= best_s:
                continue
            # The window is ~15 seconds of speech. Returning its START would put
            # the player up to 15 seconds before the words, which is a different
            # scene in a news bulletin — so the timestamp comes from the cue that
            # actually carries a query term, and the window only supplies context.
            at = chunk[0][0]
            for t, x in chunk:
                low = normalise(x)
                if any(term.replace("_", " ") in low for term in terms):
                    at = t
                    break
            best_s, best = s, (at, " ".join(x for _t, x in chunk))
        return best
