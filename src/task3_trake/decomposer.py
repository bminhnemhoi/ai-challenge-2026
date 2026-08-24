"""L1 — EDL: Event Decomposition Layer.

Turns a raw Vietnamese query (read aloud by the judges: long, fuzzy, possibly
containing out-of-knowledge entities) into a machine-usable structure in under
2 seconds:

    {context, first_occurrence, events: [{idx, text, ook, ook_term,
                                          ocr_hint, asr_hint, weight}]}

Two decomposers are provided:

* :class:`LLMDecomposer` — production path.  Takes any callable
  ``llm_fn(system_prompt, user_text) -> str`` so the actual client (OpenAI,
  Anthropic, local) is injected, runs it with a hard timeout, parses/validates
  the JSON and falls back to the rule-based path on ANY failure (risk R3).
* :class:`RuleBasedDecomposer` — offline fallback.  Splits on ``E1:``/``E2:``
  markers, semicolons or Vietnamese sequence connectors.  Event text stays in
  Vietnamese (degraded but functional — the encoder may be multilingual).
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

SYSTEM_PROMPT = """\
Bạn là bộ phân rã truy vấn cho hệ thống truy xuất video.
Nhiệm vụ: tách mô tả thành bối cảnh chung và danh sách sự kiện có thứ tự.

QUY TẮC:
1. "context" mô tả toàn cảnh video (loại video, địa điểm, chủ đề).
2. Mỗi "event" phải là mô tả THỊ GIÁC CỤ THỂ của một khung hình đơn lẻ:
   nêu vật thể, màu sắc, tư thế, quan hệ không gian. Không dùng từ trừu tượng.
3. Viết event bằng TIẾNG ANH, thì hiện tại, dạng caption ảnh.
4. Nếu event nhắc tới thực thể riêng có khả năng model không biết
   (tên nhân vật, thương hiệu, meme, sản phẩm mới), đặt "ook": true
   và điền "ook_term" là từ khóa để tìm ảnh mẫu bên ngoài.
5. Nếu đề hỏi "thời điểm đầu tiên" / "lần đầu", đặt "first_occurrence": true.
6. Nếu event có chữ hiện trên màn hình, điền "ocr_hint".
7. Nếu event gắn với lời nói, điền "asr_hint" (giữ nguyên tiếng Việt).

Chỉ trả về JSON, không giải thích, không markdown fence.

SCHEMA:
{
  "context": "...",
  "first_occurrence": true,
  "events": [
    {"idx":1, "text":"...", "ook":false, "ook_term":null,
     "ocr_hint":null, "asr_hint":null, "weight":1.0}
  ]
}
"""

_FIRST_RE = re.compile(
    r"thời\s*điểm\s*đầu\s*tiên|lần\s*đầu|xuất\s*hiện\s*đầu\s*tiên|first\s*(time|occurrence|appear)",
    re.IGNORECASE,
)
# event markers: "E1:", "E 2 -", "E3 –" (queries pasted from formatted docs
# carry en/em dashes, not the ASCII hyphen)
_E_MARK_RE = re.compile(r"\bE\s*(\d+)\s*[:=.\-–—]\s*", re.IGNORECASE)

# Separators are ranked.  Commas are ONLY used as a last resort: the spec's own
# TRAKE-01 event ("nhân vật có sừng hươu và tai nhọn, trên sừng đậu hai con
# bướm") contains a descriptive comma, and splitting on it shreds one event
# into several bogus ones.
_SEQ_STRONG_RE = re.compile(
    r"\s*(?:;|\bsau\s+đó\b|\brồi\s+(?:đến|tới)?\b|\btiếp\s+theo\b|\bcuối\s+cùng\b)\s*",
    re.IGNORECASE,
)
_SEQ_WEAK_RE = re.compile(
    r"\s*(?:;|,|\bsau\s+đó\b|\brồi\s+(?:đến|tới)?\b|\btiếp\s+theo\b|\bcuối\s+cùng\b)\s*",
    re.IGNORECASE,
)


@dataclass
class Event:
    idx: int
    text: str
    ook: bool = False
    ook_term: Optional[str] = None
    ocr_hint: Optional[str] = None
    asr_hint: Optional[str] = None
    weight: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecomposedQuery:
    context: str
    first_occurrence: bool
    events: List[Event] = field(default_factory=list)
    source: str = "rule"  # "llm" | "rule" | "manual"

    def to_dict(self) -> dict:
        return {
            "context": self.context,
            "first_occurrence": self.first_occurrence,
            "events": [e.to_dict() for e in self.events],
            "source": self.source,
        }


def _clean(text: str) -> str:
    return text.strip().strip(".,;:").strip()


def _split_events(text: str) -> List[str]:
    """Split a listing into events, preferring strong separators over commas."""
    segs = [s for s in (_clean(x) for x in _SEQ_STRONG_RE.split(text)) if s]
    if len(segs) >= 2:
        return segs
    return [s for s in (_clean(x) for x in _SEQ_WEAK_RE.split(text)) if s]


def detect_first_occurrence(text: str) -> bool:
    """True when the query explicitly asks for the FIRST occurrence (spec O1)."""
    return bool(_FIRST_RE.search(text or ""))


class RuleBasedDecomposer:
    """Offline fallback: no network, deterministic, always returns >= 1 event.

    ``asr_hint`` is deliberately left empty here.  Spec rule 7 fills it only
    when an event is tied to speech; auto-filling it with the visual
    description would make :class:`~task3_trake.trake_engine.TrakeEngine`
    activate the ASR channel on EVERY rule-mode query and search transcripts
    for phrases nobody says.
    """

    def decompose(self, raw_query: str) -> DecomposedQuery:
        raw = (raw_query or "").strip()
        first = detect_first_occurrence(raw)

        # --- path 1: explicit E1:/E2: markers -------------------------------
        parts = _E_MARK_RE.split(raw)
        if len(parts) >= 3:
            context = _clean(parts[0])
            events: List[Event] = []
            # parts = [prefix, "1", "text1", "2", "text2", ...]
            for k in range(1, len(parts) - 1, 2):
                text = _clean(parts[k + 1])
                if text:
                    events.append(Event(idx=len(events) + 1, text=text))
            # a single match is more likely an accident ("ECG: ...") than a
            # real event list, so fall through to the connector-based paths
            if len(events) >= 2:
                return DecomposedQuery(context or raw, first, events, source="rule")

        # --- path 2: "context: e1; e2; e3" ---------------------------------
        if ":" in raw:
            head, _, tail = raw.rpartition(":")
            segs = _split_events(tail)
            if len(segs) >= 2:
                events = [Event(idx=i + 1, text=s) for i, s in enumerate(segs)]
                return DecomposedQuery(_clean(head) or raw, first, events, source="rule")

        # --- path 3: sequence connectors in flowing prose -------------------
        segs = _split_events(raw)
        if len(segs) >= 2:
            events = [Event(idx=i + 1, text=s) for i, s in enumerate(segs)]
            return DecomposedQuery(raw, first, events, source="rule")

        # --- last resort: whole query is one event --------------------------
        return DecomposedQuery(raw, first, [Event(idx=1, text=raw)], source="rule")


def _find_object(text: str, start: int) -> Optional[str]:
    """Return the brace-balanced substring beginning at ``start``, or None."""
    depth = 0
    in_str = False
    esc = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    return None


def parse_llm_json(text: str) -> dict:
    """Extract and parse the first VALID JSON object from an LLM reply.

    Tolerates markdown fences, leading/trailing chatter, trailing commas, and
    prose that itself contains braces (e.g. "trả về theo schema {context,
    events}:") — in that case the first balanced span fails to parse and the
    scan continues at the next '{' rather than giving up, because falling back
    to the rule-based decomposer costs a good English decomposition.

    Raises ValueError if no valid JSON object can be found.
    """
    if not text:
        raise ValueError("empty LLM reply")
    cleaned = re.sub(r"```(?:json)?", "", text).strip()

    start = cleaned.find("{")
    while start >= 0:
        candidate = _find_object(cleaned, start)
        if candidate is not None:
            for attempt in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
                try:
                    parsed = json.loads(attempt)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
        start = cleaned.find("{", start + 1)
    raise ValueError("no valid JSON object in LLM reply")


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class LLMDecomposer:
    """Production decomposer with hard timeout and rule-based fallback.

    Parameters
    ----------
    llm_fn    : ``(system_prompt, user_text) -> str`` — the injected client.
    timeout_s : hard wall-clock budget (spec/config: 2.0 s).
    fallback  : decomposer used on timeout / parse failure / bad schema.
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str], str],
        timeout_s: float = 2.0,
        fallback: Optional[RuleBasedDecomposer] = None,
    ) -> None:
        self.llm_fn = llm_fn
        self.timeout_s = timeout_s
        self.fallback = fallback or RuleBasedDecomposer()

    def decompose(self, raw_query: str) -> DecomposedQuery:
        # A fresh single-worker pool per call: a hung llm_fn can never occupy a
        # slot needed by the NEXT query.  shutdown(wait=False) returns
        # immediately and the abandoned daemon-like worker dies with the
        # process — during a competition, one dead API call must not disable
        # the LLM path for every query that follows (risk R3).
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(self.llm_fn, SYSTEM_PROMPT, raw_query)
            reply = future.result(timeout=self.timeout_s)
            data = parse_llm_json(reply)
            return self._validate(data, raw_query)
        except FuturesTimeout:
            future.cancel()
            return self.fallback.decompose(raw_query)
        except Exception:
            return self.fallback.decompose(raw_query)
        finally:
            pool.shutdown(wait=False)

    def _validate(self, data: dict, raw_query: str) -> DecomposedQuery:
        events_raw = data.get("events")
        if not isinstance(events_raw, list) or not events_raw:
            raise ValueError("LLM JSON: 'events' missing or empty")
        events: List[Event] = []
        for k, ev in enumerate(events_raw):
            # 'text' is the only field worth failing over; idx and weight are
            # auxiliary (idx is re-derived below, weight defaults to 1.0), so a
            # single malformed optional value must not throw away a whole set
            # of correctly translated English event descriptions.
            if not isinstance(ev, dict) or not str(ev.get("text", "")).strip():
                raise ValueError(f"LLM JSON: event {k} lacks 'text'")
            events.append(
                Event(
                    idx=_to_int(ev.get("idx"), k + 1),
                    text=str(ev["text"]).strip(),
                    ook=bool(ev.get("ook", False)),
                    ook_term=ev.get("ook_term") or None,
                    ocr_hint=ev.get("ocr_hint") or None,
                    asr_hint=ev.get("asr_hint") or None,
                    weight=_to_float(ev.get("weight"), 1.0),
                )
            )
        events.sort(key=lambda e: e.idx)
        for i, e in enumerate(events):
            e.idx = i + 1
        # OR, not override: the regex only fires on an explicit "thời điểm đầu
        # tiên" in the query, so it is positive evidence the LLM cannot veto.
        first = bool(data.get("first_occurrence")) or detect_first_occurrence(raw_query)
        context = str(data.get("context", "")).strip() or raw_query
        return DecomposedQuery(context, first, events, source="llm")
