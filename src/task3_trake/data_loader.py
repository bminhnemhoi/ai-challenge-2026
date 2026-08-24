"""L0 access — loads embeddings.npy, videos.json, keyframes table.

Enforces the three hard requirements negotiated with the core team
(spec section 3.1):

* R1 — globally contiguous keyframe ids: every video owns ``[s_v, e_v]``,
  the union covers ``[0, T-1]`` with no gaps and no overlaps.
  :func:`validate_contiguity` checks this; :class:`IndexRemapper` repairs a
  broken index from the raw keyframe table (risk R2).
* R2 — embeddings as one dense L2-normalized ``(T, D)`` matrix (memory-mapped).
* R3 — keyframe density is the core team's job; this loader only reports stats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

PathLike = Union[str, Path]


@dataclass
class KeyframeTable:
    """Columnar keyframe metadata, indexed by gid (row == gid)."""

    video_id: np.ndarray  # (T,) object/str
    frame_idx: np.ndarray  # (T,) int
    ts_ms: np.ndarray  # (T,) int
    shot_id: np.ndarray  # (T,) int
    pos_in_shot: np.ndarray  # (T,) float

    def __len__(self) -> int:
        return len(self.frame_idx)

    @classmethod
    def load(cls, path: PathLike) -> "KeyframeTable":
        """Load from .npz (canonical), .parquet (if pyarrow present) or .jsonl."""
        path = Path(path)
        if path.suffix == ".npz":
            z = np.load(path, allow_pickle=True)
            return cls(
                video_id=z["video_id"],
                frame_idx=z["frame_idx"].astype(np.int64),
                ts_ms=z["ts_ms"].astype(np.int64),
                shot_id=z["shot_id"].astype(np.int64),
                pos_in_shot=z["pos_in_shot"].astype(np.float32),
            )
        if path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("pyarrow is required to read parquet keyframes") from exc
            tbl = pq.read_table(path)
            cols = {name: tbl.column(name).to_numpy(zero_copy_only=False) for name in tbl.column_names}
            order = np.argsort(cols["gid"], kind="stable") if "gid" in cols else np.arange(len(tbl))
            return cls(
                video_id=cols["video_id"][order],
                frame_idx=cols["frame_idx"][order].astype(np.int64),
                ts_ms=cols["ts_ms"][order].astype(np.int64),
                shot_id=cols["shot_id"][order].astype(np.int64),
                pos_in_shot=cols["pos_in_shot"][order].astype(np.float32),
            )
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            rows.sort(key=lambda r: r["gid"])
            return cls(
                video_id=np.array([r["video_id"] for r in rows], dtype=object),
                frame_idx=np.array([r["frame_idx"] for r in rows], dtype=np.int64),
                ts_ms=np.array([r["ts_ms"] for r in rows], dtype=np.int64),
                shot_id=np.array([r.get("shot_id", -1) for r in rows], dtype=np.int64),
                pos_in_shot=np.array([r.get("pos_in_shot", 0.0) for r in rows], dtype=np.float32),
            )
        raise ValueError(f"unsupported keyframe table format: {path.suffix}")


def validate_contiguity(videos: Dict[str, dict], T: int) -> List[str]:
    """Check requirement R1.  Returns a list of problems (empty == valid)."""
    problems: List[str] = []
    ranges = sorted(((int(m["s_v"]), int(m["e_v"]), v) for v, m in videos.items()))
    if not ranges:
        return ["no videos defined"]
    if ranges[0][0] != 0:
        problems.append(f"first range starts at {ranges[0][0]}, expected 0")
    prev_end, prev_vid = None, None
    for s, e, v in ranges:
        if e < s:
            problems.append(f"{v}: e_v={e} < s_v={s}")
        if prev_end is not None:
            if s <= prev_end:
                problems.append(f"{v} overlaps {prev_vid}: [{s},{e}] vs end {prev_end}")
            elif s > prev_end + 1:
                problems.append(f"gap of {s - prev_end - 1} gids between {prev_vid} and {v}")
        prev_end, prev_vid = e, v
    if prev_end is not None and prev_end != T - 1:
        problems.append(f"last range ends at {prev_end}, expected T-1={T - 1}")
    return problems


class IndexRemapper:
    """Rebuild a contiguous global index from a raw keyframe table (risk R2).

    Sorts keyframes by (first-appearance video order, frame_idx), producing:

    * ``order``      — permutation: new_gid -> old_row
    * ``old_to_new`` — inverse mapping: old_row -> new_gid
    * ``videos``     — fresh contiguous ``{video_id: {s_v, e_v}}``
    """

    def __init__(self, order: np.ndarray, videos: Dict[str, dict]) -> None:
        self.order = order
        self.old_to_new = np.empty_like(order)
        self.old_to_new[order] = np.arange(len(order))
        self.videos = videos

    @classmethod
    def from_keyframes(cls, table: KeyframeTable) -> "IndexRemapper":
        vids = np.asarray(table.video_id, dtype=object)
        # stable sort keeps first-appearance ordering of videos deterministic
        first_seen: Dict[str, int] = {}
        for v in vids:
            if v not in first_seen:
                first_seen[v] = len(first_seen)
        video_rank = np.array([first_seen[v] for v in vids], dtype=np.int64)
        order = np.lexsort((table.frame_idx, video_rank))
        videos: Dict[str, dict] = {}
        sorted_vids = vids[order]
        for new_gid, v in enumerate(sorted_vids):
            if v not in videos:
                videos[v] = {"s_v": new_gid, "e_v": new_gid}
            else:
                videos[v]["e_v"] = new_gid
        return cls(order, videos)

    def apply_to_table(self, table: KeyframeTable) -> KeyframeTable:
        o = self.order
        return KeyframeTable(
            video_id=np.asarray(table.video_id, dtype=object)[o],
            frame_idx=table.frame_idx[o],
            ts_ms=table.ts_ms[o],
            shot_id=table.shot_id[o],
            pos_in_shot=table.pos_in_shot[o],
        )

    def apply_to_embeddings(self, E: np.ndarray) -> np.ndarray:
        return np.asarray(E)[self.order]


class TrakeData:
    """Bundle of corpus artifacts used by the engine.

    Attributes
    ----------
    embeddings : (T, D) float16/float32 array or memmap, L2-normalized.
    videos     : {video_id: {"s_v", "e_v", "fps", "n_frames"}}.
    keyframes  : optional :class:`KeyframeTable` for gid -> metadata lookup.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        videos: Dict[str, dict],
        keyframes: Optional[KeyframeTable] = None,
        strict: bool = True,
    ) -> None:
        self.embeddings = embeddings
        self.videos = videos
        self.keyframes = keyframes
        problems = validate_contiguity(videos, embeddings.shape[0])
        if problems:
            if strict:
                raise ValueError(
                    "corpus index violates R1 (contiguity): "
                    + "; ".join(problems[:5])
                    + " — use IndexRemapper or strict=False"
                )
            self.index_problems = problems
        else:
            self.index_problems = []
        if keyframes is not None and len(keyframes) != embeddings.shape[0]:
            raise ValueError(
                f"keyframe table has {len(keyframes)} rows but embeddings have T={embeddings.shape[0]}"
            )

    @property
    def T(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    @classmethod
    def load(
        cls,
        data_dir: PathLike,
        mmap: bool = True,
        strict: bool = True,
        upcast: bool = False,
    ) -> "TrakeData":
        """Load ``embeddings.npy`` + ``videos.json`` (+ keyframes.{npz,parquet,jsonl}).

        ``upcast=True`` converts the embeddings to a RESIDENT float32 array at
        load time (T=1.2M, D=1024 -> ~4.9 GB RAM).  This removes the per-query
        float16->float32 conversion from the GEMM hot path (~2.5 s -> ~0.25 s
        on CPU) and is the recommended competition-day setting on a machine
        with enough RAM; keep the default (memory-mapped float16) otherwise.
        """
        d = Path(data_dir)
        E = np.load(d / "embeddings.npy", mmap_mode="r" if mmap and not upcast else None)
        if upcast:
            E = np.ascontiguousarray(E, dtype=np.float32)
        videos = json.loads((d / "videos.json").read_text(encoding="utf-8"))
        table: Optional[KeyframeTable] = None
        for name in ("keyframes.npz", "keyframes.parquet", "keyframes.jsonl"):
            if (d / name).exists():
                table = KeyframeTable.load(d / name)
                break
        return cls(E, videos, table, strict=strict)

    def video_of_gid(self, gid: int) -> Optional[str]:
        if self.keyframes is not None:
            return str(self.keyframes.video_id[gid])
        for v, m in self.videos.items():
            if m["s_v"] <= gid <= m["e_v"]:
                return v
        return None

    def lookup(self, gid: int) -> dict:
        """Full metadata for one keyframe (used to build API responses).

        Without a keyframe table there is no way to recover the real video
        frame index: keyframes are sparsely sampled, so the keyframe ORDINAL
        returned in degraded mode is off by roughly the sampling stride.  Such
        records are flagged ``degraded=True`` and callers must not build a
        submission line from them — a silently wrong frame number is a
        guaranteed wrong answer at submission time.
        """
        gid = int(gid)
        if not 0 <= gid < self.T:
            raise IndexError(f"gid {gid} out of range [0, {self.T})")
        if self.keyframes is not None:
            k = self.keyframes
            return {
                "gid": gid,
                "video_id": str(k.video_id[gid]),
                "frame_idx": int(k.frame_idx[gid]),
                "ts_ms": int(k.ts_ms[gid]),
                "shot_id": int(k.shot_id[gid]),
                "pos_in_shot": float(k.pos_in_shot[gid]),
                "degraded": False,
            }
        vid = self.video_of_gid(gid)
        meta = self.videos.get(vid, {}) if vid else {}
        offset = gid - int(meta.get("s_v", 0)) if meta else gid
        return {
            "gid": gid,
            "video_id": vid or "?",
            "frame_idx": offset,  # keyframe ordinal, NOT a video frame index
            "ts_ms": -1,
            "shot_id": -1,
            "pos_in_shot": 0.0,
            "degraded": True,
        }

    def norm_stats(self, sample: int = 1000, seed: int = 0) -> dict:
        """Requirement R2: verify the embeddings really are L2-normalized.

        ``visual_scores`` renormalizes the query vectors but never the corpus,
        so un-normalized rows would silently bias every score toward
        long-vector keyframes.  Samples rows rather than scanning 1.2M.
        """
        rng = np.random.default_rng(seed)
        n = min(sample, self.T)
        idx = rng.choice(self.T, size=n, replace=False) if n < self.T else np.arange(self.T)
        norms = np.linalg.norm(np.asarray(self.embeddings[np.sort(idx)], dtype=np.float32), axis=1)
        max_dev = float(np.abs(norms - 1.0).max())
        return {
            "sampled": int(n),
            "min_norm": float(norms.min()),
            "max_norm": float(norms.max()),
            "max_deviation": max_dev,
            "l2_normalized": max_dev <= 0.05,
        }

    def density_stats(self, min_per_shot: int = 4) -> dict:
        """Requirement R3: report keyframes per shot (spec asks for >= 4).

        Shots are keyed by ``(video_id, shot_id)`` — a bare shot id is only
        unique within its video — and rows carrying the ``-1`` sentinel (no
        shot segmentation) are counted separately rather than collapsed into
        one giant pseudo-shot, which would report R3 as satisfied for a corpus
        that has no shot information at all.
        """
        if self.keyframes is None:
            return {"available": False, "meets_r3": False}
        shots = self.keyframes.shot_id
        labelled = shots >= 0
        n_unlabelled = int((~labelled).sum())
        if not labelled.any():
            return {
                "available": True,
                "shot_ids": False,
                "unlabelled": n_unlabelled,
                "meets_r3": False,
            }
        keys = np.array(
            [f"{v}#{s}" for v, s in zip(self.keyframes.video_id[labelled], shots[labelled])],
            dtype=object,
        )
        _, counts = np.unique(keys, return_counts=True)
        return {
            "available": True,
            "shot_ids": True,
            "n_shots": int(counts.size),
            "unlabelled": n_unlabelled,
            "min_per_shot": int(counts.min()),
            "mean_per_shot": float(counts.mean()),
            "shots_below_min": int((counts < min_per_shot).sum()),
            "meets_r3": bool((counts >= min_per_shot).all()) and n_unlabelled == 0,
        }
