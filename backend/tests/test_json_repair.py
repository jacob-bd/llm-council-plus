"""Tests for JSON extraction and repair."""
from backend.json_repair import extract_json_block, repair_json


def test_extract_json_from_markdown():
    text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
    assert extract_json_block(text) == {"key": "value"}


def test_extract_json_fallback_braces():
    text = 'Preamble {"key": "value"} postamble'
    assert extract_json_block(text) == {"key": "value"}


def test_repair_trailing_comma():
    text = '{"a": 1, "b": 2,}'
    assert repair_json(text) == {"a": 1, "b": 2}


def test_repair_returns_none_on_garbage():
    assert repair_json("not json at all") is None


def test_extract_nested_json():
    text = 'Result: {"outer": {"inner": [1, 2]}}'
    result = extract_json_block(text)
    assert result == {"outer": {"inner": [1, 2]}}


def test_extract_json_array():
    text = 'Here: [{"a": 1}, {"b": 2}]'
    result = extract_json_block(text)
    assert result == [{"a": 1}, {"b": 2}]


def test_extract_empty_text():
    assert extract_json_block("") is None
    assert extract_json_block(None) is None


def test_repair_empty_text():
    assert repair_json("") is None
    assert repair_json(None) is None
