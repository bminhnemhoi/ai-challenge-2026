"""A vision-language model that actually looks at the frame.

SigLIP-2 scores a query against a keyframe with one dot product. That is what
makes 177,321 frames searchable instantly, and it is also why the system cannot
tell a yellow lion from a red one: colour, count, action and ordinality all
collapse into a single 1152-dimensional vector.

Measured on this project's own data, that failure is not subtle. For the
lion-dance query, 3 of 4 events score BELOW the noise floor — the embedding's
best keyframe is no more prominent than a random one. Asked the same question,
gemini-3.5-flash-lite named the lion's colour in every one of the six candidate
videos correctly and scored the yellow-black-white one at 100 against 0-30 for
the rest, in 4.5 seconds for six images.

So this is a reranker, not a retriever: SigLIP-2 finds the shortlist, the VLM
reads it. Everything is cached by (model, query hash, video, frame), because the
same frame gets re-judged every time a round is rebuilt and a contest afternoon
has three hours in it.

Cost, measured: about 1,100 input tokens per image at 512px plus the prompt.
A 24-query round at 24 candidates each is roughly 600 images, ~700k tokens,
which on flash-lite list pricing is a few cents.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CDN = "https://huggingface.co/datasets/BaeBaeBoo1010/aic2026-keyframes/resolve/main"
UA = {"User-Agent": "Mozilla/5.0"}

#: flash-lite is the working default: it answered the lion-colour question
#: correctly at a fraction of the price, and the pro tiers add latency and
#: 503s under load without adding accuracy on a task this concrete
DEFAULT_MODEL = "gemini-3.5-flash-lite"
# The free tier meters 500 requests per DAY per model, so the fallback chain is
# not only a reliability device — it is the day's budget. Each distinct name
# that resolves carries its own 500. Verified against the live API on 21 Aug
# 2026: gemini-2.0-flash and gemini-2.0-flash-lite now 404 and are dropped;
# gemini-3.1-flash-lite and gemini-flash-latest answer and are added.
FALLBACK_MODELS = (
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
)

#: how many frames go in one request. Larger batches are cheaper per frame but
#: the model starts losing track of which image is which past a dozen or so.
BATCH = 8

#: 512px keeps text legible and diacritics readable while costing ~1.1k tokens
MAX_SIDE = 512

#: A per-minute 429 clears within the minute; these waits straddle it so a later
#: attempt lands in a fresh window instead of the one that just filled.
RETRY_WAIT = (8.0, 22.0, 40.0)


def _is_daily_quota(msg: str) -> bool:
    """True only for the 429 that will still be 429 an hour from now.

    Google spends the same code and the same RESOURCE_EXHAUSTED status on the
    per-minute limit and on the per-day one. Guessing wrong is expensive in
    both directions: call a per-minute limit fatal and a working model is struck
    off for the afternoon; call a per-day limit transient and the run sleeps
    through the round. The quota id carried in the message is the only thing
    that separates them.
    """
    low = msg.lower().replace(" ", "").replace("_", "")
    return "perday" in low or "dailylimit" in low


def load_env(path: str | Path = ".env") -> None:
    """Read KEY=value lines so a key never has to live in a committed file."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class VLMJudge:
    """Score candidate keyframes against a query by actually looking at them."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "vlm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        load_env(Path(data_dir).parent / ".env")
        load_env(".env")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = None
        self._cache: Dict[str, dict] = {}
        self._dirty: set = set()
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.errors: List[str] = []
        self.exhausted: set = set()
        self._turn = -1
        self.frame_dir = self.data_dir / "frames"
        self.fetch_failures = 0
        self.last_fetch_error = ""

    @property
    def ready(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    # ------------------------------------------------------------------ cache
    def _key(self, query: str) -> str:
        h = hashlib.sha1(f"{self.model}\n{query.strip()}".encode("utf-8")).hexdigest()[:16]
        return h

    def _bucket(self, qkey: str) -> dict:
        if qkey not in self._cache:
            p = self.cache_dir / f"{qkey}.json"
            try:
                self._cache[qkey] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self._cache[qkey] = {}
        return self._cache[qkey]

    def flush(self) -> None:
        for qkey in sorted(self._dirty):
            (self.cache_dir / f"{qkey}.json").write_text(
                json.dumps(self._cache[qkey], ensure_ascii=False), encoding="utf-8"
            )
        self._dirty.clear()

    # ------------------------------------------------------------------ image
    def _fetch(self, video_id: str, filename: str) -> Optional[bytes]:
        """The 512px thumbnail, from disk if it has ever been fetched before.

        Two things learned the hard way live here. A single DNS blip used to
        take out every frame of a run at once, and because a dead thumbnail
        returns None quietly, the round finished with an empty verdict and no
        complaint — so failures are retried, then counted where cost_note can
        say so. And the same keyframes get re-judged every time a question is
        sharpened, so the downscaled bytes are worth keeping: the thumbnail is
        ~35 KB against ~250 KB on the wire, and re-asking is the normal case,
        not the exception.
        """
        from PIL import Image

        thumb = self.frame_dir / video_id / f"{Path(filename).stem}.jpg"
        try:
            if thumb.is_file():
                return thumb.read_bytes()
        except OSError:
            pass

        last = None
        for attempt in range(3):
            try:
                raw = urllib.request.urlopen(
                    urllib.request.Request(f"{CDN}/{video_id}/{filename}", headers=UA), timeout=40
                ).read()
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                im.thumbnail((MAX_SIDE, MAX_SIDE))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=85)
                blob = buf.getvalue()
                try:
                    thumb.parent.mkdir(parents=True, exist_ok=True)
                    thumb.write_bytes(blob)
                except OSError:  # a full disk must not fail the round
                    pass
                return blob
            except Exception as exc:  # noqa: BLE001 - one dead thumbnail must not stop a round
                last = exc
                time.sleep(1.5 * (attempt + 1))
        self.fetch_failures += 1
        self.last_fetch_error = f"{type(last).__name__}: {str(last)[:80]}"
        return None

    # ------------------------------------------------------------------- ask
    _PROMPT = (
        "Bạn đang giúp tìm một khoảnh khắc trong kho video tiếng Việt.\n\n"
        "CÂU HỎI TÌM KIẾM:\n{query}\n\n"
        "Dưới đây là {n} khung hình, đánh số 1..{n}. Với TỪNG khung hình hãy chấm "
        "điểm 0-100 mức độ khung hình đó ĐÚNG là cảnh mà câu hỏi mô tả.\n\n"
        "Chấm nghiêm khắc theo các chi tiết cụ thể trong câu hỏi:\n"
        "- MÀU SẮC nêu trong câu hỏi phải khớp (áo vàng ≠ áo đỏ)\n"
        "- SỐ LƯỢNG phải khớp (bốn em nhỏ ≠ hai em nhỏ)\n"
        "- HÀNH ĐỘNG/TƯ THẾ phải đúng khoảnh khắc được mô tả\n"
        "- CHỮ hiện trên hình, nếu câu hỏi có nhắc, phải khớp\n"
        "Cảnh cùng chủ đề nhưng sai chi tiết thì cho 20-40, không cho điểm cao.\n\n"
        "Trả về DUY NHẤT một mảng JSON, không giải thích:\n"
        '[{{"i":1,"d":85,"vi":"lý do rất ngắn"}}, ...]'
    )

    def _model_order(self) -> List[str]:
        """The models to try, rotated so consecutive batches hit different ones.

        The free tier meters requests per minute PER MODEL. Asking one model
        until it refuses meant a sweep spent most of its wall-clock asleep in
        backoff — 7 minutes a query — while four other models sat idle with
        full windows. Spreading the batches multiplies the usable rate by the
        number of models, and the preferred model still leads its own turn, so
        the mix stays weighted towards the one chosen for the round.

        Rotation is per call and deliberately not random: a rebuild has to
        reproduce the same submission, and randomness there would be a bug.
        """
        chain = [self.model] + [m for m in FALLBACK_MODELS if m != self.model]
        # Only peers rotate. Scores from one sweep get compared against each
        # other to find a peak, so the frames in a profile want a judge that
        # marks to the same scale; a heavier model mixed in would move a frame
        # up or down for reasons that have nothing to do with the picture.
        # The lite models mark alike, so they share the load, and the heavier
        # names stay where they were — a last resort, not part of the round.
        peers = [m for m in chain if "lite" in m and m not in self.exhausted]
        if len(peers) < 2:
            return chain
        self._turn = (self._turn + 1) % len(peers)
        k = self._turn
        rotated = peers[k:] + peers[:k]
        return rotated + [m for m in chain if m not in rotated]

    def _ask_batch(self, query: str, images: Sequence[bytes]) -> List[Tuple[int, str]]:
        from google.genai import types

        client = self._get_client()
        prompt = self._PROMPT.format(query=query.strip()[:1500], n=len(images))
        parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in images]
        last_err = None
        for model in self._model_order():
            if model in self.exhausted:
                continue
            for attempt in range(3):
                try:
                    r = client.models.generate_content(
                        model=model,
                        contents=[*parts, prompt],
                        config=types.GenerateContentConfig(
                            temperature=0.0, max_output_tokens=3000
                        ),
                    )
                    self.calls += 1
                    u = getattr(r, "usage_metadata", None)
                    if u:
                        self.tokens_in += u.prompt_token_count or 0
                        self.tokens_out += u.candidates_token_count or 0
                    return self._parse(r.text or "", len(images))
                except Exception as exc:  # noqa: BLE001
                    last_err = f"{type(exc).__name__}: {str(exc)[:90]}"
                    msg = str(exc)
                    # Two different 429s wear the same status code and they need
                    # opposite handling. The per-MINUTE limit clears on its own
                    # in under a minute, so waiting is exactly right. The daily
                    # quota does not clear today, so waiting on it is how a dead
                    # run turns into a 40-minute hang — that one has to move on
                    # and be remembered. Only the message text separates them.
                    if _is_daily_quota(msg):
                        self.exhausted.add(model)
                        break
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        time.sleep(RETRY_WAIT[min(attempt, len(RETRY_WAIT) - 1)])
                        continue
                    if "503" in msg or "UNAVAILABLE" in msg:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    break
        self.errors.append(last_err or "unknown")
        return []

    @staticmethod
    def _parse(text: str, n: int) -> List[Tuple[int, str]]:
        m = re.search(r"\[.*\]", text, re.S)
        if not m:
            return []
        try:
            rows = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return []
        out = []
        for row in rows:
            try:
                i = int(row["i"])
                d = float(row.get("d", row.get("diem", 0)))
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= i <= n:
                out.append((i, max(0.0, min(100.0, d)) / 100.0, str(row.get("vi", ""))[:90]))
        return [(i, s, v) for i, s, v in out]

    # ------------------------------------------------------------------- API
    def score(
        self,
        query: str,
        candidates: Sequence[Tuple[str, int, str]],
        progress=None,
        workers: int = 4,
    ) -> Dict[Tuple[str, int], Tuple[float, str]]:
        """{(video, frame): (0..1 score, short reason)} for the given candidates.

        ``candidates`` are (video_id, frame_idx, frame_filename). Cached entries
        are returned without a call.
        """
        qkey = self._key(query)
        bucket = self._bucket(qkey)
        out: Dict[Tuple[str, int], Tuple[float, str]] = {}
        todo = []
        for v, f, fn in candidates:
            hit = bucket.get(f"{v}:{int(f)}")
            if hit is not None:
                out[(v, int(f))] = (float(hit[0]), str(hit[1]))
            else:
                todo.append((v, int(f), fn))
        if not todo or not self.ready:
            return out

        # fetch in parallel, judge in batches
        with ThreadPoolExecutor(max_workers=8) as ex:
            blobs = list(ex.map(lambda j: self._fetch(j[0], j[2]), todo))
        pairs = [(t, b) for t, b in zip(todo, blobs) if b]

        batches = [pairs[i : i + BATCH] for i in range(0, len(pairs), BATCH)]

        def run(batch):
            imgs = [b for _t, b in batch]
            return batch, self._ask_batch(query, imgs)

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for batch, scored in ex.map(run, batches):
                for i, s, why in scored:
                    (v, f, _fn), _blob = batch[i - 1]
                    out[(v, f)] = (s, why)
                    bucket[f"{v}:{f}"] = [s, why]
                self._dirty.add(qkey)
                done += len(batch)
                if progress:
                    progress(done, len(pairs))
        self.flush()
        return out

    def cost_note(self) -> str:
        # list prices for the flash-lite tier as published Aug 2026
        usd = self.tokens_in / 1e6 * 0.10 + self.tokens_out / 1e6 * 0.40
        note = (f"{self.calls} lần gọi, {self.tokens_in:,} token vào, "
                f"{self.tokens_out:,} token ra  ≈ ${usd:.3f}")
        # A round that judged nothing must never read like a round that judged
        # and found nothing: an exhausted quota once produced an empty verdict
        # that the pipeline happily packaged as a finished submission.
        if self.exhausted:
            note += f"\n  !! HET QUOTA: {', '.join(sorted(self.exhausted))}"
        if self.errors:
            note += f"\n  !! {len(self.errors)} lô lỗi, gần nhất: {self.errors[-1]}"
        if self.fetch_failures:
            note += (f"\n  !! {self.fetch_failures} khung hình KHÔNG TẢI ĐƯỢC"
                     f" ({self.last_fetch_error}) — chúng bị bỏ qua, không phải bị chấm 0.")
        if not self.calls and (self.errors or self.fetch_failures):
            note += "\n  !! KHONG CHAM DUOC KHUNG HINH NAO — dung coi ket qua nay la da xet."
        return note

    @property
    def usable(self) -> bool:
        """False once every model this judge can reach has hit its daily quota.

        Callers check this to stop early with a clear message instead of
        grinding through the whole round producing empty verdicts.
        """
        return self.ready and bool(
            {self.model, *FALLBACK_MODELS} - self.exhausted
        )
