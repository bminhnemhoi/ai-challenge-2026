"""Read the text that is burned into the picture.

Three of the round-1 queries cannot be answered any other way. query-p1-19 asks
which two lines of verse appear in the shot — the frames ARE pages of poetry.
query-p1-22 asks for a dish name written on a recipe card. Half the news frames
carry a lower-third naming a place or a programme. An image embedding reduces
all of that to "a page with writing on it".

It also catches wrong answers that nothing else can. OCR on the frame currently
submitted for query-p1-19 reads "Trích Văn bia THOẠI NGỌC HẦU" and a passage
about a mountain — the query is about Nguyễn Trung Trực. The video is simply
wrong, and no similarity score was ever going to say so.

Cost on this hardware: about 4 seconds per 1280x720 frame on one CPU core, so a
round's shortlist (a few hundred frames) is a background pass of a few minutes,
not something to run over all 177,321 keyframes in-round. Everything is cached to
disk by (video_id, frame), so the second run is instant.
"""

from __future__ import annotations

import io
import json
import re
import threading
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CDN = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main"
UA = {"User-Agent": "Mozilla/5.0"}

#: below this the recogniser is usually hallucinating on texture, not reading
MIN_CONF = 0.35


def normalise(s: str) -> str:
    return unicodedata.normalize("NFC", str(s or "")).lower()


class OCRIndex:
    """Burned-in text per keyframe, cached on disk.

    ``data/ocr/<video_id>.json`` maps a frame index to
    ``[[text, confidence], ...]``, one file per video so a partial run is never
    lost and a second round reuses everything the first one read.
    """

    def __init__(self, data_dir: str | Path = "data", langs: Sequence[str] = ("vi", "en")) -> None:
        self.dir = Path(data_dir) / "ocr"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.langs = list(langs)
        self._cache: Dict[str, Dict[str, list]] = {}
        self._dirty: set = set()
        self._reader = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ cache
    def _video(self, video_id: str) -> Dict[str, list]:
        if video_id not in self._cache:
            p = self.dir / f"{video_id}.json"
            try:
                self._cache[video_id] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a missing or truncated cache is not fatal
                self._cache[video_id] = {}
        return self._cache[video_id]

    def flush(self) -> None:
        for vid in sorted(self._dirty):
            (self.dir / f"{vid}.json").write_text(
                json.dumps(self._cache[vid], ensure_ascii=False), encoding="utf-8"
            )
        self._dirty.clear()

    def get(self, video_id: str, frame_idx: int) -> Optional[List[Tuple[str, float]]]:
        v = self._video(video_id).get(str(int(frame_idx)))
        return [(t, float(c)) for t, c in v] if v is not None else None

    def text_of(self, video_id: str, frame_idx: int, min_conf: float = MIN_CONF) -> str:
        got = self.get(video_id, frame_idx)
        if not got:
            return ""
        return " ".join(t for t, c in got if c >= min_conf)

    @property
    def n_frames(self) -> int:
        return sum(len(v) for v in self._cache.values())

    # -------------------------------------------------------------------- run
    def _get_reader(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self.langs, gpu=False, verbose=False)
        return self._reader

    def read_frames(
        self,
        items: Iterable[Tuple[str, int, str]],
        progress=None,
        download_workers: int = 8,
        colours: "ColourIndex | None" = None,
        detections=None,
    ) -> int:
        """Read (video_id, frame_idx, frame_filename) triples that are not cached.

        Downloads run in parallel because the network dominates for small images;
        recognition is serialised because one EasyOCR reader is not thread-safe
        and a second one would double the memory for no gain on a CPU.
        """
        import numpy as np
        from PIL import Image

        todo = [
            (v, int(f), fn)
            for v, f, fn in items
            if str(int(f)) not in self._video(v)
        ]
        if not todo:
            return 0

        def grab(job):
            v, f, fn = job
            try:
                raw = urllib.request.urlopen(
                    urllib.request.Request(f"{CDN}/{v}/{fn}", headers=UA), timeout=40
                ).read()
                return v, f, Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:  # noqa: BLE001 - one dead thumbnail must not stop the pass
                return v, f, None

        reader = self._get_reader()
        done = 0
        with ThreadPoolExecutor(max_workers=download_workers) as ex:
            for v, f, img in ex.map(grab, todo):
                out: list = []
                if img is not None:
                    try:
                        res = reader.readtext(np.array(img), detail=1, paragraph=False)
                        out = [[str(t), round(float(c), 3)] for _b, t, c in res if str(t).strip()]
                    except Exception:  # noqa: BLE001
                        out = []
                self._video(v)[str(f)] = out
                self._dirty.add(v)
                # colour comes free once the image is in memory, and the
                # download is what costs
                if colours is not None and img is not None:
                    det = detections(v, f) if detections else None
                    colours.put(v, f, img, det)
                done += 1
                if progress and done % 20 == 0:
                    progress(done, len(todo))
                if done % 100 == 0:
                    self.flush()
        self.flush()
        if progress:
            progress(done, len(todo))
        return done

    # ------------------------------------------------------------------ score
    def find(self, terms: Iterable[str], video_id: str) -> List[Tuple[int, str, int]]:
        """(frame_idx, text, how many terms matched) for frames of one video.

        Matching is on the normalised substring, because Vietnamese OCR gets the
        diacritics right far more often than it gets word boundaries right.
        """
        wanted = [normalise(t) for t in terms if len(str(t)) > 2]
        if not wanted:
            return []
        out = []
        for k, v in self._video(video_id).items():
            text = " ".join(t for t, c in v if float(c) >= MIN_CONF)
            low = normalise(text)
            n = sum(1 for w in wanted if w in low)
            if n:
                out.append((int(k), text, n))
        out.sort(key=lambda r: (-r[2], r[0]))
        return out


def query_phrases(text: str, max_words: int = 4) -> List[str]:
    """Phrases from a query worth looking for in on-screen text.

    Proper nouns and quoted strings first — they are what a lower-third or a
    recipe card actually says, and what an embedding cannot represent.
    """
    out: List[str] = []
    out += re.findall(r'"([^"]{3,60})"', text)
    out += re.findall(r"“([^”]{3,60})”", text)
    # runs of Capitalised Words: place names, programme names, people
    for m in re.finditer(r"\b([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){1,%d})" % (max_words - 1), text):
        out.append(m.group(1))
    seen, uniq = set(), []
    for p in out:
        k = normalise(p)
        if k not in seen and len(k) > 3:
            seen.add(k)
            uniq.append(p.strip())
    return uniq


class ColourIndex:
    """Dominant colours of the main subject per keyframe, cached like the OCR.

    Half the round-1 queries name a colour and the shortlist ignores it: the
    lion-dance query asks for a yellow-black-white lion and the top frames show
    red ones. Measured on the DETECTED OBJECT rather than the frame, because a
    lion-dance stage is red whatever colour the lion is.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.dir = Path(data_dir) / "colours"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[str, list]] = {}
        self._dirty: set = set()

    def _video(self, video_id: str) -> Dict[str, list]:
        if video_id not in self._cache:
            p = self.dir / f"{video_id}.json"
            try:
                self._cache[video_id] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self._cache[video_id] = {}
        return self._cache[video_id]

    def flush(self) -> None:
        for vid in sorted(self._dirty):
            (self.dir / f"{vid}.json").write_text(
                json.dumps(self._cache[vid], ensure_ascii=False), encoding="utf-8"
            )
        self._dirty.clear()

    def put(self, video_id: str, frame_idx: int, img, detections=None) -> None:
        from src.core.colours import subject_colours

        try:
            got = subject_colours(img, detections)
        except Exception:  # noqa: BLE001 - a colour reading must never break a pass
            got = []
        self._video(video_id)[str(int(frame_idx))] = [[n, round(s, 3)] for n, s in got]
        self._dirty.add(video_id)

    def get(self, video_id: str, frame_idx: int):
        v = self._video(video_id).get(str(int(frame_idx)))
        return [(n, float(s)) for n, s in v] if v is not None else None

    def names(self, video_id: str, frame_idx: int, min_share: float = 0.12) -> List[str]:
        got = self.get(video_id, frame_idx) or []
        return [n for n, s in got if s >= min_share]

    @property
    def n_frames(self) -> int:
        return sum(len(v) for v in self._cache.values())
