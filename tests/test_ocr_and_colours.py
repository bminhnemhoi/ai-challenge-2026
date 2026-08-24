"""Reading the picture: text burned into the frame, and the subject's colour.

Both are shown to the operator and neither is scored, for the reason this whole
project keeps rediscovering: every signal folded into the ranking on plausibility
rather than measurement has cost points. What they earn their place with is
catching answers nothing else can catch — OCR on the frame submitted for
query-p1-19 reads "Trích Văn bia THOẠI NGỌC HẦU" while the query is about
Nguyễn Trung Trực, which means that answer is on the wrong video.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.colours import colours_in_query, dominant, subject_colours
from src.core.ocr import ColourIndex, OCRIndex, query_phrases

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- colours


def _solid(rgb, size=(64, 64)):
    from PIL import Image

    return Image.new("RGB", size, rgb)


@pytest.mark.parametrize(
    "rgb,expected",
    [
        ((220, 30, 30), "đỏ"),
        ((240, 200, 30), "vàng"),
        ((30, 140, 60), "xanh lá"),
        ((40, 90, 210), "xanh dương"),
        ((250, 250, 250), "trắng"),
        ((8, 8, 8), "đen"),
        ((128, 128, 128), "xám"),
    ],
)
def test_a_flat_colour_is_named_correctly(rgb, expected):
    got = dominant(_solid(rgb), top=1)
    assert got and got[0][0] == expected, f"{rgb} -> {got}"


def test_the_query_colour_words_are_found():
    assert colours_in_query("một con lân màu vàng đen trắng") == ["vàng", "đen", "trắng"]
    assert "hồng" in colours_in_query("hai người đàn ông mặc áo sơ mi hồng")


def test_bare_xanh_is_not_guessed_as_blue_or_green():
    """Vietnamese "xanh" covers both; only the qualified phrase discriminates."""
    assert colours_in_query("áo xanh dương") == ["xanh dương"]
    assert colours_in_query("lá xanh lá") == ["xanh lá"]
    assert colours_in_query("áo xanh") == []


def test_colour_is_measured_on_the_subject_not_the_stage():
    """A lion-dance frame is mostly red stage whatever colour the lion is.

    The organisers ship a bounding box per detection, so the crop is free — and
    without it the histogram says "red" for every candidate and settles nothing.
    """
    from PIL import Image

    img = Image.new("RGB", (100, 100), (200, 20, 20))          # red stage
    for x in range(60, 90):
        for y in range(60, 90):
            img.putpixel((x, y), (240, 205, 40))               # a yellow lion
    whole = dominant(img, top=1)[0][0]
    assert whole == "đỏ", "the frame really is mostly red"

    det = {
        "detection_class_entities": ["Toy"],
        "detection_boxes": [["0.60", "0.60", "0.90", "0.90"]],
        "detection_scores": ["0.9"],
    }
    assert subject_colours(img, det, top=1)[0][0] == "vàng"


def test_background_classes_are_never_the_subject():
    from PIL import Image

    img = Image.new("RGB", (100, 100), (10, 10, 10))
    for x in range(0, 50):
        for y in range(0, 50):
            img.putpixel((x, y), (30, 150, 60))                # a big green tree
    for x in range(60, 80):
        for y in range(60, 80):
            img.putpixel((x, y), (240, 205, 40))               # a smaller yellow thing
    det = {
        "detection_class_entities": ["Tree", "Toy"],
        "detection_boxes": [["0.0", "0.0", "0.5", "0.5"], ["0.6", "0.6", "0.8", "0.8"]],
        "detection_scores": ["0.9", "0.8"],
    }
    assert subject_colours(img, det, top=1)[0][0] == "vàng"


def test_a_box_covering_the_whole_frame_is_ignored():
    """That is the background wearing a label, not a subject."""
    from PIL import Image

    img = Image.new("RGB", (100, 100), (200, 20, 20))
    det = {
        "detection_class_entities": ["Toy"],
        "detection_boxes": [["0.0", "0.0", "1.0", "1.0"]],
        "detection_scores": ["0.9"],
    }
    assert subject_colours(img, det, top=1)[0][0] == "đỏ"      # fell back to the frame


# ------------------------------------------------------------------------ OCR


def test_query_phrases_pulls_proper_nouns_and_quotes():
    got = query_phrases('Các em được trao bảng với nội dung "Trao kinh phí hỗ trợ" tại Kiên Giang')
    assert "Trao kinh phí hỗ trợ" in got
    assert any("Kiên Giang" in g for g in got)


def test_ocr_cache_round_trips(tmp_path):
    o = OCRIndex(tmp_path)
    o._video("L21_V001")["100"] = [["NẤM RƠM CẮT ĐÔI", 0.91], ["mờ", 0.10]]
    o._dirty.add("L21_V001")
    o.flush()

    again = OCRIndex(tmp_path)
    assert again.text_of("L21_V001", 100) == "NẤM RƠM CẮT ĐÔI", "low-confidence noise must be dropped"
    assert again.get("L21_V001", 999) is None
    assert (tmp_path / "ocr" / "L21_V001.json").exists()


def test_ocr_find_ranks_by_how_many_terms_matched(tmp_path):
    o = OCRIndex(tmp_path)
    v = o._video("V")
    v["10"] = [["Chương trình Trao kinh phí hỗ trợ trẻ em mồ côi", 0.9]]
    v["20"] = [["Trao kinh phí", 0.9]]
    v["30"] = [["Bản tin thời sự", 0.9]]
    hits = o.find(["trao kinh phí", "trẻ em mồ côi"], "V")
    assert [h[0] for h in hits] == [10, 20]


def test_colour_cache_round_trips(tmp_path):
    c = ColourIndex(tmp_path)
    c.put("V", 5, _solid((240, 205, 40)))
    c.flush()
    again = ColourIndex(tmp_path)
    assert "vàng" in again.names("V", 5)
    assert again.get("V", 999) is None


@pytest.mark.skipif(
    not (ROOT / "data" / "ocr").is_dir()
    or not any((ROOT / "data" / "ocr").glob("*.json")),
    reason="no OCR cache on this machine (run scripts/run_ocr.py)",
)
def test_the_real_cache_actually_read_vietnamese_text():
    o = OCRIndex(ROOT / "data")
    files = sorted((ROOT / "data" / "ocr").glob("*.json"))
    total = withtext = 0
    for f in files:
        for k, v in json.loads(f.read_text(encoding="utf-8")).items():
            total += 1
            if len(" ".join(t for t, c in v if c >= 0.35)) > 12:
                withtext += 1
    assert total > 50
    # news lower-thirds and recipe cards mean a large share of frames carry text;
    # if this collapses, the recogniser or the language pack has changed
    assert withtext / total > 0.2, f"only {withtext}/{total} frames had readable text"
