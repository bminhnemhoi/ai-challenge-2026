"""Lean, contest-grade retrieval over the SigLIP-2 keyframe index.

`src.task1_kis.retriever.TextualKISRetriever` is the exploration surface: ~1150
lines with several alternative scoring paths, a spaCy branch, and reranking
hooks.  This module is the opposite — the smallest thing that produces a
submission, so that on contest day there is one code path, it is measured, and
it is covered by tests.

What it does, and why:

* **4-prompt ensemble** over (English, Vietnamese, "a high quality video
  keyframe of ...", "a photo of ...").  Measured on the 60-sample ground truth
  this beats English-only by 6.6 points of video R@1; the two prefix prompts
  are near-free and add a little.
* **Flat frame ranking, no per-video cap.**  The shipped retriever keeps at
  most 2 frames per video and forces a 10-second gap between them.  That is
  tuned for a video-level metric, but the official score needs a frame inside
  an interval, and R@k is a maximum — so extra frames from an already-listed
  video are free insurance.  Removing the cap raised the measured official
  score at every window width tested.
* **No temporal smoothing, no per-video standardisation.**  Both were tried
  and both made the measured score worse (see scripts/experiment_retrieval.py);
  they are recorded here so nobody re-adds them on intuition.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

DEFAULT_MODEL = "google/siglip2-so400m-patch14-384"

#: weights over (english, vietnamese, keyframe-prefix, photo-prefix)
PROMPT_WEIGHTS = (0.45, 0.35, 0.10, 0.10)


@dataclass
class Hit:
    video_id: str
    frame_idx: int
    score: float
    n: int
    pts_time: float
    video_last_frame: int


class KISEngine:
    """Text -> ranked keyframes.  Load once, query many times."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        skip_first_n: int = 2,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.device = device
        #: the first couple of keyframes of a video are usually a channel
        #: bumper or a black frame, never the answer
        self.skip_first_n = skip_first_n
        self._loaded = False
        self.model = None
        self.processor = None
        self._tcache: Optional[Dict[str, str]] = None
        self._tcache_path: Optional[Path] = None
        self._warned_translate = False
        self._gemini = None

    # ------------------------------------------------------------------ load
    def load(self) -> "KISEngine":
        if self._loaded:
            return self
        meta_path = self.data_dir / "metadata.json"
        emb_path = self.data_dir / "embeddings_siglip2_384.npy"
        for p in (meta_path, emb_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} is missing — run `python scripts/download_data.py` first"
                )

        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        self.embeddings = np.load(emb_path, mmap_mode="r")
        if self.embeddings.shape[0] != len(self.metadata):
            raise ValueError(
                f"index/metadata mismatch: {self.embeddings.shape[0]} vectors "
                f"vs {len(self.metadata)} metadata rows"
            )

        self.video_id = np.array([m["video_id"] for m in self.metadata], dtype=object)
        self.frame_idx = np.array([m["frame_idx"] for m in self.metadata], dtype=np.int64)
        self.n_in_video = np.array([m.get("n", 1) for m in self.metadata], dtype=np.int64)
        self.pts_time = np.array([m.get("pts_time", 0.0) for m in self.metadata], dtype=np.float32)

        self.last_frame: Dict[str, int] = {}
        for v, f in zip(self.video_id, self.frame_idx):
            if f > self.last_frame.get(v, -1):
                self.last_frame[v] = int(f)

        self.valid = self.n_in_video > self.skip_first_n
        blank = self.data_dir / "blank_frame_indices.json"
        if blank.exists():
            try:
                raw = json.loads(blank.read_text(encoding="utf-8"))
                idx = raw if isinstance(raw, list) else raw.get("blank_frame_indices", [])
                idx = np.array([i for i in idx if 0 <= i < len(self.metadata)], dtype=np.int64)
                if idx.size:
                    self.valid[idx] = False
            except Exception:
                pass

        self._load_model()
        self._loaded = True
        return self

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, dtype=torch.float32).to(dev).eval()

    # --------------------------------------------------------------- encode
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalised SigLIP-2 text embeddings, one row per input."""
        torch = self._torch
        inp = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        ).to(self.device)
        with torch.inference_mode():
            f = self.model.get_text_features(**inp)
        if not isinstance(f, torch.Tensor):  # transformers 5.x returns ModelOutput
            for attr in ("pooler_output", "text_embeds", "last_hidden_state"):
                v = getattr(f, attr, None)
                if isinstance(v, torch.Tensor):
                    f = v
                    break
            else:
                f = f[0]
        if f.ndim == 3:
            f = f.mean(dim=1)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------ translate
    def translate(self, text: str) -> str:
        """Vietnamese -> English, cached on disk, never fatal.

        Worth the dependency: on the 60-sample ground truth, feeding SigLIP-2 a
        real English rendering instead of the raw Vietnamese moves video R@1
        from 35.0% to 43.3%.  SigLIP-2 is multilingual, but not equally good in
        both directions, and the corpus captions it was trained on are English.

        Cached to ``data/translation_cache.json`` so a repeated query costs
        nothing and so the contest never blocks on a network call.  If
        translation is unavailable the original text is returned and retrieval
        degrades to the Vietnamese-only path rather than failing.
        """
        text = (text or "").strip()
        if not text or not any(ord(c) > 127 for c in text):
            return text  # already ASCII: assume English

        if self._tcache is None:
            self._tcache_path = self.data_dir / "translation_cache.json"
            try:
                self._tcache = json.loads(self._tcache_path.read_text(encoding="utf-8"))
            except Exception:
                self._tcache = {}
        if text in self._tcache:
            return self._tcache[text]

        out = None
        for backend in self._translate_backends():
            try:
                got = backend(text)
            except Exception:  # noqa: BLE001 - try the next backend
                continue
            if got and got.strip() and got.strip() != text:
                out = got.strip()
                break
        if out is None:
            if not self._warned_translate:
                print(
                    "  ! automatic translation unavailable; falling back to the Vietnamese text.\n"
                    "    Measured cost: video R@1 drops from ~43% to ~35%. To recover it, drop a\n"
                    "    hand-written English rendering next to each query as <name>.en.txt."
                )
                self._warned_translate = True
            return text

        self._tcache[text] = out
        try:
            self._tcache_path.write_text(
                json.dumps(self._tcache, ensure_ascii=False, indent=0), encoding="utf-8"
            )
        except Exception:
            pass
        return out

    def _translate_backends(self):
        """Translation backends, most reliable first.

        The free Google endpoint that ``deep_translator`` scrapes rate-limits
        hard and returns TranslationNotFound under load — which is exactly when
        a contest round would hit it.  Gemini is tried first when a key is
        configured because it is an authenticated API, and it also produces
        better renderings of Vietnamese scene descriptions.
        """

        def gemini(text: str) -> Optional[str]:
            from src.core.gemini_engine import GeminiAIOptimizer

            if self._gemini is None:
                self._gemini = GeminiAIOptimizer()
            if not getattr(self._gemini, "is_ready", False):
                return None
            # the attribute is _client; `client` does not exist on the engine
            client = getattr(self._gemini, "_client", None) or self._gemini.client
            resp = client.models.generate_content(
                model=self._gemini.model_name,
                contents=(
                    "Translate this Vietnamese video scene description into natural English. "
                    "Reply with the translation only.\n\n" + text
                ),
            )
            return getattr(resp, "text", None)

        def google_free(text: str) -> Optional[str]:
            from deep_translator import GoogleTranslator

            return GoogleTranslator(source="vi", target="en").translate(text)

        def mymemory(text: str) -> Optional[str]:
            from deep_translator import MyMemoryTranslator

            return MyMemoryTranslator(source="vi-VN", target="en-US").translate(text)

        return (gemini, google_free, mymemory)

    def chunk_text(self, text: str, max_tokens: int = 58) -> List[str]:
        """Split a long description into pieces the encoder can actually see.

        SigLIP-2's text tower takes 64 tokens and silently drops the rest.
        Contest descriptions routinely run past that — measured on the round-1
        set, 13 of 24 queries exceeded it — and what falls off the end is
        exactly the distinguishing detail, because a describer states the scene
        first and the identifying specifics last ("...the second racer wears a
        red hat and the last a black hat", "...this bird is commonly seen in
        southern Vietnam").

        Sentences are packed greedily so each chunk stays under the limit and
        no clause is cut in half.
        """
        tok = self.processor.tokenizer
        text = (text or "").strip()
        sentences = re.split(r"(?<=[.!?;])\s+|\n+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text]

        chunks: List[str] = []
        cur = ""
        for s in sentences:
            cand = f"{cur} {s}".strip()
            if cur and len(tok(cand)["input_ids"]) > max_tokens:
                chunks.append(cur)
                cur = s
            else:
                cur = cand
        if cur:
            chunks.append(cur)

        # a single sentence can still be too long; split it on commas
        out: List[str] = []
        for c in chunks:
            if len(tok(c)["input_ids"]) <= max_tokens:
                out.append(c)
                continue
            piece = ""
            for part in c.split(","):
                cand = f"{piece}, {part}".strip(", ")
                if piece and len(tok(cand)["input_ids"]) > max_tokens:
                    out.append(piece)
                    piece = part.strip()
                else:
                    piece = cand
            if piece:
                out.append(piece)
        return out or [text]

    def query_vector(self, query: str, query_en: Optional[str] = None) -> np.ndarray:
        """The 4-prompt ensemble vector for one query.

        ``query`` is whatever the organisers gave us (Vietnamese in practice);
        ``query_en`` overrides the automatic translation when a human has
        already written a better English rendering.
        """
        vi = query
        en = query_en if query_en is not None else self.translate(query)
        prompts = [
            en,
            vi,
            f"a high quality video keyframe of {en}",
            f"a photo of {en}",
        ]
        V = self.encode_texts(prompts)
        w = np.array(PROMPT_WEIGHTS, dtype=np.float32).reshape(-1, 1)
        v = (V * w).sum(axis=0)
        return v / max(float(np.linalg.norm(v)), 1e-6)

    # --------------------------------------------------------------- search
    def similarities(self, vec: np.ndarray, chunk: int = 200_000) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        T = self.embeddings.shape[0]
        out = np.empty(T, dtype=np.float32)
        for a in range(0, T, chunk):
            b = min(a + chunk, T)
            out[a:b] = np.asarray(self.embeddings[a:b], dtype=np.float32) @ vec
        return out

    def query_similarities(
        self,
        query: str,
        query_en: Optional[str] = None,
        combine: str = "mean",
    ) -> np.ndarray:
        """Similarity of every keyframe to the query, chunked if it is long.

        A description longer than the encoder's window is scored chunk by
        chunk and the per-chunk similarities are combined, instead of throwing
        the tail away.  ``mean`` rewards a frame that satisfies the whole
        description; ``max`` would reward matching any single clause, which is
        how a bookshelf ends up beating the scene the clause belonged to.
        """
        if not self._loaded:
            self.load()
        en = query_en if query_en is not None else self.translate(query)
        chunks = self.chunk_text(en)
        if len(chunks) <= 1:
            return self.similarities(self.query_vector(query, en))

        # the first chunk keeps the Vietnamese branch (it carries the scene);
        # later chunks are detail clauses and are encoded in English only
        acc = self.similarities(self.query_vector(query, chunks[0]))
        parts = [acc]
        for c in chunks[1:]:
            parts.append(self.similarities(self.query_vector(c, c)))
        stacked = np.stack(parts, axis=0)
        return stacked.mean(axis=0) if combine == "mean" else stacked.max(axis=0)

    def search(
        self,
        query: str,
        query_en: Optional[str] = None,
        top_n: int = 150,
        restrict_video: Optional[str] = None,
        combine: str = "mean",
    ) -> List[Hit]:
        """Top ``top_n`` keyframes, best first, with no per-video cap."""
        if not self._loaded:
            self.load()
        sims = self.query_similarities(query, query_en, combine=combine)
        mask = self.valid
        if restrict_video is not None:
            mask = mask & (self.video_id == restrict_video)
        s = np.where(mask, sims, -np.inf)
        n = int(min(top_n, np.isfinite(s).sum()))
        if n <= 0:
            return []
        top = np.argpartition(-s, n - 1)[:n]
        top = top[np.argsort(-s[top])]
        return [
            Hit(
                video_id=str(self.video_id[i]),
                frame_idx=int(self.frame_idx[i]),
                score=float(s[i]),
                n=int(self.n_in_video[i]),
                pts_time=float(self.pts_time[i]),
                video_last_frame=self.last_frame[str(self.video_id[i])],
            )
            for i in top
        ]

    def rank_videos(self, query: str, query_en: Optional[str] = None, top_n: int = 20):
        """Videos ranked by their single best-matching keyframe.

        Used by TRAKE, where the video is all-or-nothing and only one video is
        worth submitting.
        """
        if not self._loaded:
            self.load()
        sims = np.where(self.valid, self.similarities(self.query_vector(query, query_en)), -np.inf)
        best: Dict[str, float] = {}
        for i in np.argsort(-sims)[: max(top_n * 200, 20000)]:
            v = str(self.video_id[i])
            if v not in best:
                best[v] = float(sims[i])
        return sorted(best.items(), key=lambda kv: -kv[1])[:top_n]
