"""Tests for Stage 2 ranking parse helpers."""

from backend.council import parse_ranking_from_text


def test_parse_ranking_filters_hallucinated_labels():
    text = """Response A is strong.
Response B is weaker.

FINAL RANKING:
1. Response C
2. Response A
3. Response B"""

    parsed = parse_ranking_from_text(
        text,
        expected_count=2,
        valid_labels=["Response A", "Response B"],
    )

    assert parsed == ["Response A", "Response B"]


def test_parse_ranking_deduplicates_labels():
    text = """FINAL RANKING:
1. Response A
2. Response A
3. Response B"""

    parsed = parse_ranking_from_text(
        text,
        expected_count=2,
        valid_labels=["Response A", "Response B"],
    )

    assert parsed == ["Response A", "Response B"]
