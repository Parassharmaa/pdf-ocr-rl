"""Tests for the composite reward function."""

import pytest

from pdf_ocr_rl.reward.composite import (
    composite_reward,
    heading_accuracy,
    normalized_edit_distance,
    reading_order_reward,
    structural_validity_reward,
)


class TestNormalizedEditDistance:
    def test_identical(self):
        assert normalized_edit_distance("hello", "hello") == 1.0

    def test_empty_both(self):
        assert normalized_edit_distance("", "") == 1.0

    def test_empty_reference(self):
        assert normalized_edit_distance("hello", "") == 0.0

    def test_completely_different(self):
        score = normalized_edit_distance("abc", "xyz")
        assert score < 0.5

    def test_similar(self):
        score = normalized_edit_distance("hello world", "hello worlds")
        assert score > 0.8


class TestStructuralValidity:
    def test_valid_markdown(self):
        md = "# Heading\n\nSome text.\n\n## Sub\n\n- item 1\n- item 2"
        score = structural_validity_reward(md)
        assert score == 1.0

    def test_unbalanced_code_fence(self):
        md = "# Heading\n\n```python\nprint('hello')\n"
        score = structural_validity_reward(md)
        assert score < 1.0

    def test_valid_table(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        score = structural_validity_reward(md)
        assert score == 1.0

    def test_unbalanced_bold(self):
        md = "This is **bold but not closed"
        score = structural_validity_reward(md)
        assert score < 1.0


class TestHeadingAccuracy:
    def test_matching_headings(self):
        pred = "# Title\n\n## Section 1\n\n## Section 2"
        ref = "# Title\n\n## Section 1\n\n## Section 2"
        score = heading_accuracy(pred, ref)
        assert score == 1.0

    def test_no_headings_both(self):
        score = heading_accuracy("just text", "just text")
        assert score == 1.0

    def test_missing_headings(self):
        pred = "# Title"
        ref = "# Title\n\n## Section 1\n\n## Section 2"
        score = heading_accuracy(pred, ref)
        assert score < 1.0

    def test_wrong_levels(self):
        pred = "## Title\n\n### Section"
        ref = "# Title\n\n## Section"
        score = heading_accuracy(pred, ref)
        assert score < 1.0


class TestReadingOrder:
    def test_correct_order(self):
        pred = "Introduction to the topic.\n\nDetails about the subject.\n\nConclusion and summary."
        ref = "Introduction to the topic.\n\nDetails about the subject.\n\nConclusion and summary."
        score = reading_order_reward(pred, ref)
        assert score >= 0.8

    def test_empty_reference(self):
        score = reading_order_reward("some text", "")
        assert score == 1.0


class TestCompositeReward:
    def test_perfect_match(self):
        md = "# Title\n\n## Section\n\nSome content here with enough text to match.\n\nAnother paragraph with substantial content for matching.\n\n- item 1\n- item 2"
        score = composite_reward(md, md)
        assert score > 0.9

    def test_partial_match(self):
        pred = "# Title\n\nSome content."
        ref = "# Title\n\n## Section\n\nSome content here.\n\n- item 1"
        score = composite_reward(pred, ref)
        assert 0.3 < score < 0.9

    def test_custom_weights(self):
        md = "# Title\n\nContent"
        weights = {"edit_distance": 1.0, "structural": 0.0, "heading": 0.0, "reading_order": 0.0}
        score = composite_reward(md, md, weights=weights)
        assert score == 1.0
