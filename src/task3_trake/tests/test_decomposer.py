"""Tests for L1 — EDL (query decomposition)."""

import json
import time

import pytest

from src.task3_trake.decomposer import (
    DecomposedQuery,
    LLMDecomposer,
    RuleBasedDecomposer,
    detect_first_occurrence,
    parse_llm_json,
)

COOKING = (
    "Trong một video nấu bò xào, xác định thời điểm đầu tiên mỗi nguyên liệu sau "
    "chạm vào chảo: E1: dầu ăn, E2: thịt bò, E3: hành, E4: bông điên điển."
)


# ------------------------------------------------------------- rule-based


def test_rule_based_e_markers():
    dq = RuleBasedDecomposer().decompose(COOKING)
    assert dq.first_occurrence is True
    assert len(dq.events) == 4
    assert dq.events[0].text == "dầu ăn"
    assert dq.events[1].text == "thịt bò"
    assert dq.events[2].text == "hành"
    assert dq.events[3].text == "bông điên điển"
    assert [e.idx for e in dq.events] == [1, 2, 3, 4]
    assert "nấu bò xào" in dq.context
    assert dq.source == "rule"


def test_rule_based_colon_semicolon():
    dq = RuleBasedDecomposer().decompose(
        "Lễ hội hóa trang: nhân vật da như đá; nhân vật có sừng hươu và tai nhọn"
    )
    assert len(dq.events) == 2
    assert dq.events[0].text.startswith("nhân vật da")
    assert dq.first_occurrence is False


def test_rule_based_keeps_descriptive_commas():
    """The spec's own TRAKE-01 event contains a comma inside one description.

    Regression: comma ranked equal to ';' shredded it into three bogus events.
    """
    dq = RuleBasedDecomposer().decompose(
        "Tại một lễ hội đông người hóa trang: nhân vật da như đá; "
        "nhân vật có sừng hươu và tai nhọn, trên sừng đậu hai con bướm"
    )
    assert len(dq.events) == 2
    assert "hai con bướm" in dq.events[1].text


def test_rule_based_comma_still_used_without_strong_separator():
    dq = RuleBasedDecomposer().decompose("các cảnh quay: con mèo, con chó, con chim")
    assert len(dq.events) == 3


def test_rule_based_dash_markers():
    """Queries pasted from formatted documents use en/em dashes, not hyphens."""
    dq = RuleBasedDecomposer().decompose(
        "Video nấu ăn, xác định các mốc: E1 – dầu ăn, E2 – thịt bò, E3 — hành"
    )
    assert [e.text for e in dq.events] == ["dầu ăn", "thịt bò", "hành"]


def test_rule_based_single_accidental_marker_falls_through():
    """One stray 'E1:' style match must not become a one-event decomposition."""
    dq = RuleBasedDecomposer().decompose("Biểu đồ E1: nhịp tim tăng; bác sĩ ghi chép")
    assert len(dq.events) == 2


def test_rule_based_leaves_asr_hint_empty():
    """Auto-filling asr_hint would fire transcript searches on visual queries."""
    dq = RuleBasedDecomposer().decompose(COOKING)
    assert all(e.asr_hint is None for e in dq.events)
    assert all(e.ocr_hint is None for e in dq.events)


def test_detect_first_occurrence_helper():
    assert detect_first_occurrence("thời điểm đầu tiên xuất hiện")
    assert not detect_first_occurrence("các cảnh trong video")
    assert not detect_first_occurrence("")


def test_rule_based_sequence_connectors():
    dq = RuleBasedDecomposer().decompose(
        "người đàn ông mở cửa xe sau đó bước vào xe rồi đến đóng cửa"
    )
    assert len(dq.events) == 3


def test_rule_based_single_event_fallback():
    dq = RuleBasedDecomposer().decompose("một người phụ nữ mặc áo đỏ")
    assert len(dq.events) == 1
    assert dq.events[0].text == "một người phụ nữ mặc áo đỏ"


def test_rule_based_first_occurrence_variants():
    d = RuleBasedDecomposer()
    assert d.decompose("lần đầu xuất hiện: a; b").first_occurrence
    assert d.decompose("thời điểm đầu tiên: a; b").first_occurrence
    assert not d.decompose("các cảnh: a; b").first_occurrence


# ------------------------------------------------------------ JSON parsing


def test_parse_llm_json_plain():
    data = parse_llm_json('{"context": "x", "events": [{"idx":1,"text":"a"}]}')
    assert data["context"] == "x"


def test_parse_llm_json_fenced_and_chatty():
    reply = 'Sure! Here you go:\n```json\n{"context": "c", "events": []}\n```\nHope it helps.'
    assert parse_llm_json(reply)["context"] == "c"


def test_parse_llm_json_nested_braces_in_strings():
    reply = '{"context": "a {weird} string", "events": [{"idx":1,"text":"b}"}]}'
    data = parse_llm_json(reply)
    assert data["events"][0]["text"] == "b}"


@pytest.mark.parametrize("bad", ["", "no json here", "{unbalanced", "[1,2,3]"])
def test_parse_llm_json_failures(bad):
    with pytest.raises(ValueError):
        parse_llm_json(bad)


def test_parse_llm_json_trailing_comma():
    reply = '{"context": "c", "events": [{"idx":1,"text":"a"},]}'
    assert parse_llm_json(reply)["events"][0]["text"] == "a"


def test_parse_llm_json_skips_brace_bearing_chatter():
    """Prose mentioning the schema in braces must not abort the scan."""
    reply = 'Tôi trả về JSON theo schema {context, events}:\n{"context":"c","events":[{"idx":1,"text":"a"}]}'
    assert parse_llm_json(reply)["context"] == "c"


# ----------------------------------------------------------------- LLM path


def _good_reply(system, user):
    return json.dumps(
        {
            "context": "a Vietnamese cooking video",
            "first_occurrence": True,
            "events": [
                {"idx": 1, "text": "cooking oil being poured into a hot wok"},
                {"idx": 2, "text": "raw sliced beef dropped into a wok"},
                {
                    "idx": 3,
                    "text": "yellow sesbania flowers added into a wok",
                    "ook": True,
                    "ook_term": "bông điên điển sesbania flower",
                    "weight": 1.2,
                },
            ],
        }
    )


def test_llm_decomposer_happy_path():
    dq = LLMDecomposer(_good_reply).decompose(COOKING)
    assert dq.source == "llm"
    assert len(dq.events) == 3
    assert dq.first_occurrence is True
    assert dq.events[2].ook is True
    assert dq.events[2].weight == pytest.approx(1.2)


def test_llm_decomposer_falls_back_on_garbage():
    dq = LLMDecomposer(lambda s, u: "I cannot help with that.").decompose(COOKING)
    assert dq.source == "rule"
    assert len(dq.events) == 4  # rule-based still parses E1..E4


def test_llm_decomposer_falls_back_on_empty_events():
    dq = LLMDecomposer(lambda s, u: '{"context": "c", "events": []}').decompose(COOKING)
    assert dq.source == "rule"


def test_llm_decomposer_falls_back_on_exception():
    def boom(s, u):
        raise RuntimeError("api down")

    dq = LLMDecomposer(boom).decompose(COOKING)
    assert dq.source == "rule"


def test_llm_decomposer_timeout():
    def slow(s, u):
        time.sleep(2.0)
        return _good_reply(s, u)

    t0 = time.perf_counter()
    dq = LLMDecomposer(slow, timeout_s=0.2).decompose(COOKING)
    elapsed = time.perf_counter() - t0
    assert dq.source == "rule"
    assert elapsed < 1.5  # did not wait for the slow call


def test_llm_decomposer_tolerates_bad_optional_fields():
    """One malformed idx/weight must not discard a good English decomposition."""
    reply = (
        '{"context":"c","events":[{"idx":"one","text":"a","weight":null},'
        '{"idx":2,"text":"b","weight":"high"}]}'
    )
    dq = LLMDecomposer(lambda s, u: reply).decompose(COOKING)
    assert dq.source == "llm"
    assert [e.text for e in dq.events] == ["a", "b"]
    assert all(e.weight == pytest.approx(1.0) for e in dq.events)


def test_llm_first_occurrence_cannot_be_vetoed_by_llm():
    """The regex is positive evidence; an LLM 'false' must not disable mu."""
    reply = '{"context":"c","first_occurrence":false,"events":[{"idx":1,"text":"a"}]}'
    dq = LLMDecomposer(lambda s, u: reply).decompose(COOKING)  # COOKING says "đầu tiên"
    assert dq.first_occurrence is True


def test_llm_repeated_hangs_do_not_disable_the_llm_path():
    """Regression: a shared 2-worker pool died permanently after two hangs."""
    state = {"hang": True}

    def flaky(s, u):
        if state["hang"]:
            time.sleep(30)
        return _good_reply(s, u)

    d = LLMDecomposer(flaky, timeout_s=0.15)
    for _ in range(3):
        assert d.decompose(COOKING).source == "rule"
    state["hang"] = False
    t0 = time.perf_counter()
    dq = d.decompose(COOKING)
    assert dq.source == "llm"
    assert time.perf_counter() - t0 < 1.0  # not queued behind the hung workers


def test_llm_decomposer_reindexes_events():
    reply = '{"context":"c","events":[{"idx":7,"text":"b"},{"idx":2,"text":"a"}]}'
    dq = LLMDecomposer(lambda s, u: reply).decompose("x truoc; y sau")
    assert [e.idx for e in dq.events] == [1, 2]
    assert dq.events[0].text == "a"  # sorted by original idx


def test_to_dict_roundtrip():
    dq = RuleBasedDecomposer().decompose(COOKING)
    d = dq.to_dict()
    assert d["first_occurrence"] is True
    assert len(d["events"]) == 4
    assert isinstance(DecomposedQuery(**{
        "context": d["context"],
        "first_occurrence": d["first_occurrence"],
        "events": [],
    }), DecomposedQuery)
