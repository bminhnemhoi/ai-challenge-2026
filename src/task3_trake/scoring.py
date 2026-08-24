"""L2 — MCS: Multi-Channel Scoring.

Fuses Visual + OCR + ASR signals into one score matrix::

    S[i, t] = alpha * Z(S_vis)[i, t] + beta * Z(S_ocr)[i, t] + gamma * Z(S_asr)[i, t]

where ``Z`` is a PER-ROW z-score (spec section 5.1).  Row-wise normalization is
mandatory: events differ in "difficulty" (familiar concepts get uniformly
higher cosine similarity), and without it the DP sacrifices hard events to
optimize easy ones.

Two corrections to the naive formula, both load-bearing at corpus scale:

* **Sparse-channel capping.**  An OCR/ASR row holding k hits among T zeros has
  std ~ sqrt(k/T), so a single hit z-scores to ~sqrt(T/k): at T = 1.2M one hit
  reaches z ~ 1094, which at beta = 0.15 contributes 164 against a visual
  ceiling of ~3.5.  One Elasticsearch hit would then override the entire
  visual channel.  Sparse channels are therefore clipped to +/-``z_cap``.
* **Per-row weight renormalization.**  A channel matrix is built whenever ANY
  event has a hint, so events without one carry an all-zero row.  Weights are
  renormalized per ROW over the channels that actually carry signal there, so
  an event's amplitude never depends on whether OTHER events had hints.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-6

#: mode -> (alpha_visual, beta_ocr, gamma_asr), spec section 5.2
SCORING_MODES: Dict[str, Dict[str, float]] = {
    "visual": {"visual": 1.0, "ocr": 0.0, "asr": 0.0},
    "balanced": {"visual": 0.70, "ocr": 0.15, "asr": 0.15},
    "text_heavy": {"visual": 0.40, "ocr": 0.40, "asr": 0.20},
    "speech": {"visual": 0.50, "ocr": 0.10, "asr": 0.40},
}

#: z-score ceiling applied to sparse (lexical) channels, see module docstring
SPARSE_Z_CAP = 6.0
#: a channel row is "sparse" below this nonzero fraction
SPARSE_NNZ_FRACTION = 0.01


def zscore_rows(S: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Z-score each row independently: ``(S - mean_row) / (std_row + eps)``.

    A constant row maps to all-zeros (the eps guard prevents division blowup),
    which is the correct neutral value for a channel carrying no signal.
    Statistics are computed in float64 so a 1.2M-wide float32 row does not
    lose precision in the mean.
    """
    A = np.asarray(S, dtype=np.float64)
    mean = A.mean(axis=1, keepdims=True)
    std = A.std(axis=1, keepdims=True)
    return ((A - mean) / (std + eps)).astype(np.float32)


def visual_scores(
    U: np.ndarray,
    E: np.ndarray,
    chunk_size: int = 200_000,
    renormalize: bool = True,
) -> np.ndarray:
    """Dense cosine scores ``S_vis = U @ E.T``.

    A float32 array takes a single BLAS call with no copy.  A float16 array
    (or memmap) must be upcast, which is done in chunks.

    Parameters
    ----------
    U : (N, D) query vectors (one per event).  Re-L2-normalized defensively
        unless ``renormalize=False``.
    E : (T, D) corpus embeddings, assumed L2-normalized (requirement R2).
    chunk_size : rows of E per upcast+matmul chunk, for float16 input only.

    Latency note (measured at T = 1.2M, D = 1024 on CPU): NumPy's float16 ->
    float32 conversion has no SIMD path and dominates this call — roughly
    8 s at ``chunk_size`` 400k, 20 s at 50k, versus ~0.25 s when E is already
    resident float32.  Smaller chunks are strictly slower, and preallocating
    the destination buffer does not help, so the chunk size trades latency for
    peak memory rather than buying anything back.  **Convert once at load
    time** (``TrakeData.load(..., upcast=True)``, ~4.9 GB at that size) and
    keep the chunked path for memory-constrained machines only.
    """
    U = np.asarray(U, dtype=np.float32)
    if U.ndim != 2:
        raise ValueError(f"U must be (N, D), got {U.shape}")
    if renormalize:
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        U = U / np.maximum(norms, EPS)
    N, D = U.shape
    T = E.shape[0]
    if E.shape[1] != D:
        raise ValueError(f"dim mismatch: U has D={D}, E has D={E.shape[1]}")

    # a resident float32 array needs no chunking or copying at all
    if isinstance(E, np.ndarray) and E.dtype == np.float32 and not isinstance(E, np.memmap):
        return U @ E.T

    S = np.empty((N, T), dtype=np.float32)
    for start in range(0, T, chunk_size):
        stop = min(start + chunk_size, T)
        S[:, start:stop] = U @ np.asarray(E[start:stop], dtype=np.float32).T
    return S


def sparse_to_dense(hits: Dict[int, float], T: int) -> np.ndarray:
    """Expand sparse retrieval hits ``{gid: score}`` into a dense (T,) row."""
    row = np.zeros(T, dtype=np.float32)
    if hits:
        gids = np.fromiter(hits.keys(), dtype=np.int64, count=len(hits))
        vals = np.fromiter(hits.values(), dtype=np.float32, count=len(hits))
        if gids.size and (gids.min() < 0 or gids.max() >= T):
            raise ValueError("hit gid out of range")
        row[gids] = vals
    return row


def channel_matrix(hits_per_event: Sequence[Optional[Dict[int, float]]], T: int) -> np.ndarray:
    """Stack per-event sparse hits into an (N, T) channel matrix.

    ``None`` entries (events without an OCR/ASR hint) become zero rows.
    """
    rows = [sparse_to_dense(h or {}, T) for h in hits_per_event]
    return np.stack(rows, axis=0)


def blend_ook_vector(
    u_text: np.ndarray,
    u_image: np.ndarray,
    w_text: float = 0.6,
) -> np.ndarray:
    """Spec 4.4 branch B: anchor an out-of-knowledge event with an exemplar image.

    ``u_i = normalize(0.6 * u_text_rewritten + 0.4 * u_image_exemplar)``.
    Accepts a single exemplar vector or a (K, D) stack, which is averaged
    first.  Both inputs must come from the SAME encoder as the corpus.

    Both operands are L2-normalized before mixing, and so is the mean of a
    multi-exemplar stack.  The stated 0.6/0.4 split only *is* 0.6/0.4 when the
    operands are unit vectors, and the mean of K distinct unit vectors always
    has norm below 1 — without this the image branch is silently
    under-weighted in exactly the multi-exemplar case it was added for.
    """

    def _unit(v: np.ndarray) -> np.ndarray:
        return v / max(float(np.linalg.norm(v)), EPS)

    t = _unit(np.asarray(u_text, dtype=np.float32).reshape(-1))
    img = np.asarray(u_image, dtype=np.float32)
    if img.ndim == 2:
        img = np.stack([_unit(r) for r in img]).mean(axis=0)
    img = _unit(img.reshape(-1))
    if t.shape != img.shape:
        raise ValueError(f"dim mismatch: text {t.shape} vs image {img.shape}")
    return _unit(w_text * t + (1.0 - w_text) * img)


def _prepare_channel(
    M: np.ndarray,
    normalize: bool,
    z_cap: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(Z, active)`` for one channel.

    ``active[i]`` is True when row i actually carries signal (nonconstant), so
    weight renormalization can be done per row.  Sparse rows are z-capped.
    """
    A = np.asarray(M, dtype=np.float32)
    active = A.std(axis=1) > EPS
    if not normalize:
        return A, active
    Z = zscore_rows(A)
    nnz_frac = np.count_nonzero(A, axis=1) / max(A.shape[1], 1)
    sparse = nnz_frac < SPARSE_NNZ_FRACTION
    if sparse.any() and z_cap > 0:
        # Rescale, do NOT clip.  In a genuinely sparse row EVERY hit z-scores
        # far above the cap, so a hard clip saturates all of them to exactly
        # z_cap and throws away the retrieval scores the searcher returned —
        # the channel degenerates into a binary hit/no-hit flag that rewards
        # an Elasticsearch noise hit exactly as much as the true match.
        # A per-row rescale bounds the amplitude just as well and keeps the
        # ranking among hits intact.
        peak = np.abs(Z[sparse]).max(axis=1, keepdims=True)
        Z[sparse] *= np.where(peak > z_cap, z_cap / np.maximum(peak, EPS), 1.0)
    return Z, active


def fuse(
    S_vis: np.ndarray,
    S_ocr: Optional[np.ndarray] = None,
    S_asr: Optional[np.ndarray] = None,
    mode: str = "visual",
    weights: Optional[Dict[str, float]] = None,
    event_weights: Optional[Sequence[float]] = None,
    normalize: bool = True,
    z_cap: float = SPARSE_Z_CAP,
) -> np.ndarray:
    """Fuse channels with per-row z-scoring (spec section 5.1).

    Parameters
    ----------
    mode          : one of :data:`SCORING_MODES`; ignored if ``weights`` given.
    weights       : explicit {"visual": a, "ocr": b, "asr": g} override.
    event_weights : per-event multiplier from EDL (e.g. 1.2 for OOK events).
                    Boosting (w > 1) applies to the POSITIVE side only —
                    a plain multiplication would also deepen the event's
                    negative z-scores, making a hard event's poor frames more
                    repulsive and demoting the very video the weight was meant
                    to promote.  De-weighting (w < 1) instead scales the row
                    SYMMETRICALLY, so an event the operator asked to matter
                    less also loses its power to veto a video (at w = 0 the
                    event drops out of the objective entirely, rather than
                    surviving as a pure penalty term).
    normalize     : apply row z-score to each channel (True in production;
                    False only for already-normalized inputs / ablations).
    z_cap         : ceiling for sparse lexical channels; 0 disables capping.

    Missing channels (None, zero weight) are dropped, and the remaining
    weights are renormalized PER ROW over the channels carrying signal in that
    row — so `balanced` without ASR data degrades to visual+OCR, and an event
    with no OCR hint keeps full visual amplitude even when other events have
    hints.
    """
    if weights is None:
        if mode not in SCORING_MODES:
            raise ValueError(f"unknown scoring mode {mode!r}; choose from {list(SCORING_MODES)}")
        weights = SCORING_MODES[mode]

    channels: List[tuple] = [("visual", S_vis), ("ocr", S_ocr), ("asr", S_asr)]
    active = [(name, M) for name, M in channels if M is not None and weights.get(name, 0.0) > 0.0]
    if not active:
        raise ValueError("no active channel: check inputs and weights")

    N, T = np.asarray(active[0][1]).shape
    acc = np.zeros((N, T), dtype=np.float32)
    row_w = np.zeros((N, 1), dtype=np.float32)

    for name, M in active:
        Z, row_active = _prepare_channel(M, normalize, z_cap)
        if Z.shape != (N, T):
            raise ValueError(f"channel {name} has shape {Z.shape}, expected {(N, T)}")
        w = np.float32(weights[name])
        mask = row_active.reshape(-1, 1).astype(np.float32)
        acc += w * Z * mask
        row_w += w * mask

    # a row with no active channel (all-constant everywhere) stays zero
    out = np.divide(acc, row_w, out=np.zeros_like(acc), where=row_w > 0)

    if event_weights is not None:
        w = np.asarray(event_weights, dtype=np.float32).reshape(-1, 1)
        if w.shape[0] != out.shape[0]:
            raise ValueError(f"event_weights length {w.shape[0]} != N={out.shape[0]}")
        if np.any(w < 0):
            raise ValueError("event_weights must be non-negative")
        shrink = np.minimum(w, 1.0)  # w < 1: scale the whole row
        boost = np.maximum(w - 1.0, 0.0)  # w > 1: lift rewards only
        out = shrink * out + boost * np.maximum(out, 0.0)
    return out
