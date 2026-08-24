"""What colour is the thing in the shot?

"Đoạn video múa lân một con lân màu vàng đen trắng" — the query names the colour,
and the shortlist came back with red lions. Colour is stated in half the round-1
queries ("áo sơ mi hồng", "xe màu đỏ mận", "áo xanh dương") and an image
embedding blurs it into everything else in the frame.

The measurement is done on the DETECTED OBJECT, not the whole picture. A lion
dance frame is mostly red stage and gold banners whatever colour the lion is, so
a global histogram says "red and gold" for every candidate and settles nothing.
The organisers ship a bounding box per detection, so the crop is free.

Reported to the operator, never scored. Everything in this project that was
folded into the ranking on plausibility rather than measurement has cost points:
video metadata (-3.3 points of video R@1), per-frame object bonuses, transcripts
at video level and at frame level. Colour has not been measured either, so it
goes where the unmeasured things go — on screen, next to the picture, where a
human can check it in a glance.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

#: Vietnamese colour words -> hue range in degrees, plus saturation/value gates.
#: Black, white and grey are decided by value/saturation alone, so they carry a
#: hue of None.
COLOURS: Dict[str, Tuple[Optional[Tuple[float, float]], float, float]] = {
    # name:            (hue range,        min saturation, min value)
    "đỏ": ((345.0, 12.0), 0.35, 0.25),
    "cam": ((12.0, 40.0), 0.40, 0.35),
    "vàng": ((40.0, 68.0), 0.32, 0.40),
    "xanh lá": ((68.0, 165.0), 0.22, 0.20),
    "xanh dương": ((185.0, 255.0), 0.22, 0.18),
    "tím": ((255.0, 300.0), 0.22, 0.18),
    "hồng": ((300.0, 345.0), 0.18, 0.45),
    "nâu": ((12.0, 45.0), 0.30, 0.12),
}

#: an object class whose colour tells you nothing about the query's subject
BACKGROUND_CLASSES = {
    "tree", "plant", "sky", "building", "house", "wall", "floor", "table",
    "clothing", "human face", "human head", "human hair", "human body",
}

_VN_COLOUR_WORDS = {
    "đỏ": "đỏ", "cam": "cam", "vàng": "vàng", "lục": "xanh lá", "tím": "tím",
    "hồng": "hồng", "nâu": "nâu", "đen": "đen", "trắng": "trắng", "xám": "xám",
    "bạc": "xám", "vang": "vàng",
}


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", str(s or "")).lower()


def colours_in_query(text: str) -> List[str]:
    """Colour names the query asks for, in Vietnamese.

    "xanh dương" and "xanh lá" are checked before bare "xanh", which on its own
    covers both blue and green in Vietnamese and so cannot discriminate.
    """
    low = _norm(text)
    found: List[str] = []
    for phrase, name in (("xanh dương", "xanh dương"), ("xanh lá", "xanh lá"),
                         ("xanh biển", "xanh dương"), ("xanh lơ", "xanh dương")):
        if phrase in low and name not in found:
            found.append(name)
    for word, name in _VN_COLOUR_WORDS.items():
        if word in low and name not in found:
            found.append(name)
    return found


def _classify(r: int, g: int, b: int) -> Optional[str]:
    import colorsys

    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    hue = h * 360.0
    if v < 0.18:
        return "đen"
    if s < 0.12:
        return "trắng" if v > 0.75 else ("xám" if v > 0.30 else "đen")
    for name, (span, min_s, min_v) in COLOURS.items():
        if span is None or s < min_s or v < min_v:
            continue
        lo, hi = span
        inside = (lo <= hue < hi) if lo < hi else (hue >= lo or hue < hi)
        if inside:
            # brown is orange that is too dark to read as orange
            if name == "cam" and v < 0.45:
                return "nâu"
            return name
    return None


def dominant(img, box: Optional[Sequence[float]] = None, top: int = 3) -> List[Tuple[str, float]]:
    """[(colour name, share of pixels)] for the whole image or one box.

    ``box`` is the organisers' normalised [ymin, xmin, ymax, xmax].
    """
    from collections import Counter

    w, h = img.size
    if box is not None:
        y0, x0, y1, x1 = (float(v) for v in box)
        crop = img.crop((int(x0 * w), int(y0 * h), max(int(x1 * w), int(x0 * w) + 1),
                         max(int(y1 * h), int(y0 * h) + 1)))
    else:
        crop = img
    # a small thumbnail is enough for a colour histogram and keeps this at a
    # millisecond, which matters when it runs over every candidate frame
    crop = crop.convert("RGB").resize((48, 48))
    c: Counter = Counter()
    for r, g, b in crop.getdata():
        name = _classify(r, g, b)
        if name:
            c[name] += 1
    total = sum(c.values()) or 1
    return [(n, k / total) for n, k in c.most_common(top)]


def subject_colours(img, detections: Optional[dict], top: int = 3) -> List[Tuple[str, float]]:
    """Colours of the largest foreground detection, falling back to the frame.

    "Largest foreground" is a deliberately blunt proxy for "the thing the query
    is about": in a lion-dance frame the lion is the biggest non-background box,
    and in a portrait it is the person.
    """
    best_box, best_area = None, 0.0
    if detections:
        ents = detections.get("detection_class_entities") or []
        boxes = detections.get("detection_boxes") or []
        scores = detections.get("detection_scores") or []
        for ent, box, sc in zip(ents, boxes, scores):
            try:
                if float(sc) < 0.3:
                    continue
                y0, x0, y1, x1 = (float(v) for v in box)
            except (TypeError, ValueError):
                continue
            if _norm(ent) in BACKGROUND_CLASSES:
                continue
            area = max(0.0, y1 - y0) * max(0.0, x1 - x0)
            # ignore boxes that are basically the whole frame: those are the
            # background wearing a label
            if area > best_area and area < 0.75:
                best_area, best_box = area, [y0, x0, y1, x1]
    return dominant(img, best_box, top=top)
