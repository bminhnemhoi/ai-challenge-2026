"""Use the BTC object detections to choose the video — never to choose the frame.

The organisers ship an OpenImages detection per keyframe, and half the round-1
queries name a thing the detector knows: "three cyclists", "a red car", "four
children". The obvious move is to boost any frame whose detections match the
query. Measured, that is worthless or harmful:

    (60 ground-truth queries, official formula, non-snapped answer key,
     64 re-draws, windows 6/10/20/40 -- scripts/experiment_objects_rerank.py)

    baseline                                  0.374   video R@1 26/60
    per-frame bonus, best weight              0.375   (+0.4%, noise)
    per-frame count match, any weight         0.374   (+0.0%, completely inert)
    per-frame bonus, weight 0.05              0.346   (-7.4%)
    PER-VIDEO bonus, weight 0.01              0.386   (+3.3%)

The split is the whole point, and it is the same trap that produced this
project's original 5.8: a per-frame bonus raised video R@1 from 26 to 30 while
LOWERING the contest score, because the frame holding the matching object is
not the frame nearest the answer instant. Promoting it displaces a frame of the
same video that was closer to the truth.

So the bonus is computed once per video -- the best match among that video's
candidate frames -- and added to all of its frames equally. Videos get
reordered; the frames inside a video keep the order the embedding gave them,
which is the order that knows about timing.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

#: the confidence cut the organisers' own baseline uses
OBJ_CONF = 0.4

#: measured optimum; the curve is flat from 0.005 to 0.02 and turns sharply
#: negative by 0.05, so this sits in the middle of the plateau
DEFAULT_WEIGHT = 0.01

#: classes present in so many keyframes that they carry no information
#: ("Person" alone fires on 57% of the corpus)
UNINFORMATIVE = {
    "clothing", "human face", "human body", "human head", "human arm", "human leg",
    "human hand", "human nose", "human hair", "human eye", "human mouth", "human ear",
    "footwear", "sports equipment",
}

_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


class ObjectIndex:
    """Per-frame detector classes, loaded once and reused across queries."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        #: (video_id, frame_stem) -> Counter of class -> count
        self.by_frame: Dict[tuple, Counter] = {}
        self.vocab: Set[str] = set()

    # ------------------------------------------------------------------ load
    def load_for(self, keys: Iterable[tuple]) -> "ObjectIndex":
        """Read detections for the given (video_id, frame_stem) pairs only.

        Member names follow ``objects/<video_id>/<stem>.json`` without
        exception, so each frame is one dictionary lookup into the archive's
        central directory. Scanning ``namelist()`` instead — 178,195 entries —
        once per query turned a 40-second run into 86 seconds for a boost worth
        3%, which is not a trade worth making inside a three-hour window.
        """
        keys = {k for k in keys if k not in self.by_frame}
        if not keys:
            return self

        unpacked = self.data_dir / "objects"
        if unpacked.is_dir():
            hit = False
            for vid, stem in keys:
                f = unpacked / vid / f"{stem}.json"
                if f.exists():
                    self._absorb((vid, stem), f.read_bytes())
                    hit = True
            if hit:
                self.vocab = {c for ctr in self.by_frame.values() for c in ctr}
                return self

        zf = self._zip()
        if zf is None:
            return self
        for vid, stem in keys:
            try:
                self._absorb((vid, stem), zf.read(f"objects/{vid}/{stem}.json"))
            except KeyError:
                self.by_frame[(vid, stem)] = Counter()  # no detections for this frame
        self.vocab = {c for ctr in self.by_frame.values() for c in ctr}
        return self

    def _zip(self) -> Optional[zipfile.ZipFile]:
        """The archive, opened once and kept open across queries."""
        if not hasattr(self, "_zf"):
            z = self.data_dir / "objects-aic25-b1.zip"
            self._zf = zipfile.ZipFile(z) if z.exists() else None
        return self._zf

    def close(self) -> None:
        zf = getattr(self, "_zf", None)
        if zf is not None:
            zf.close()
        self._zf = None

    def _absorb(self, key: tuple, payload: bytes) -> None:
        try:
            j = json.loads(payload)
        except Exception:  # noqa: BLE001 - a corrupt detection file is not fatal
            return
        c: Counter = Counter()
        for cls, sc in zip(
            j.get("detection_class_entities", []), j.get("detection_scores", [])
        ):
            cls = str(cls).strip().lower()
            try:
                ok = float(sc) > OBJ_CONF
            except (TypeError, ValueError):
                ok = False
            if ok and cls and cls not in UNINFORMATIVE:
                c[cls] += 1
        self.by_frame[key] = c

    @property
    def available(self) -> bool:
        return bool(self.by_frame)

    # ----------------------------------------------------------------- terms
    def query_terms(self, text: str) -> Dict[str, Optional[int]]:
        """Detector classes named in an English query, and any count asked for."""
        if not text or not self.vocab:
            return {}
        low = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())) + " "
        out: Dict[str, Optional[int]] = {}
        for cls in self.vocab:
            if f" {cls} " not in low and f" {cls}s " not in low:
                continue
            want = None
            m = re.search(
                rf"(\d+|{'|'.join(_NUMBERS)})\s+(?:\w+\s+){{0,2}}{re.escape(cls)}s?\b", low
            )
            if m:
                tok = m.group(1)
                want = int(tok) if tok.isdigit() else _NUMBERS[tok]
            out[cls] = want
        return out

    # ----------------------------------------------------------------- boost
    def rerank(
        self,
        hits: Sequence,
        query_en: Optional[str],
        stem_of,
        weight: float = DEFAULT_WEIGHT,
    ) -> List:
        """Reorder VIDEOS by detection match, leaving each video's frames alone.

        ``stem_of(hit)`` returns the frame filename stem, which is how the
        detection files are named. Returns the hits unchanged when the query
        names nothing the detector knows, which is most of the time.
        """
        terms = self.query_terms(query_en or "")
        if not terms or not self.by_frame:
            return list(hits)

        best: Dict[str, float] = defaultdict(float)
        for h in hits:
            ctr = self.by_frame.get((h.video_id, stem_of(h)))
            if not ctr:
                continue
            b = sum(weight for cls in terms if ctr.get(cls, 0))
            if b > best[h.video_id]:
                best[h.video_id] = b

        if not best:
            return list(hits)

        # The boost is written back into ``score``, not just used for sorting.
        # Reordering while leaving the old scores in place left every downstream
        # confidence measure reading numbers that no longer described the list:
        # the margin between rank 1 and the best other video came out NEGATIVE on
        # three round-1 queries, and the "needs a human" flag silently stopped
        # firing on exactly the queries the boost had just changed its mind about.
        import dataclasses

        order = [
            (h.score + best[h.video_id], i, dataclasses.replace(h, score=h.score + best[h.video_id]))
            for i, h in enumerate(hits)
        ]
        # stable on the original position, so each video's internal frame order
        # is exactly what the embedding gave — that order knows about timing
        order.sort(key=lambda t: (-t[0], t[1]))
        return [t[2] for t in order]
