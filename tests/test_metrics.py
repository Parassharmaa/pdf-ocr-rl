"""Tests for evaluation metrics."""

from pdf_ocr_rl.eval.metrics import (
    compute_code_block_accuracy,
    compute_edit_distance,
    compute_heading_accuracy,
    compute_table_accuracy,
    evaluate_batch,
    evaluate_sample,
)


class TestEditDistance:
    def test_identical(self):
        assert compute_edit_distance("test", "test") == 1.0

    def test_different(self):
        score = compute_edit_distance("abc", "xyz")
        assert score < 0.5


class TestHeadingAccuracy:
    def test_perfect(self):
        md = "# Title\n## Sub"
        result = compute_heading_accuracy(md, md)
        assert result["heading_f1"] == 1.0

    def test_no_headings(self):
        result = compute_heading_accuracy("text", "text")
        assert result["heading_f1"] == 1.0


class TestTableAccuracy:
    def test_matching_table(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = compute_table_accuracy(md, md)
        assert result["table_count_match"] == 1.0
        assert result["table_cell_accuracy"] == 1.0

    def test_no_tables(self):
        result = compute_table_accuracy("text", "text")
        assert result["table_count_match"] == 1.0


class TestCodeBlockAccuracy:
    def test_matching_blocks(self):
        md = "```python\nprint('hello')\n```"
        result = compute_code_block_accuracy(md, md)
        assert result["code_block_count_match"] == 1.0
        assert result["code_block_similarity"] == 1.0


class TestEvaluateSample:
    def test_perfect_match(self):
        md = "# Title\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```py\nx=1\n```"
        result = evaluate_sample(md, md)
        assert result["edit_distance"] == 1.0
        assert result["heading_f1"] == 1.0


class TestEvaluateBatch:
    def test_batch(self):
        preds = ["# Hello\nworld", "# Test\ncontent"]
        refs = ["# Hello\nworld", "# Test\ncontent"]
        result = evaluate_batch(preds, refs)
        assert result["edit_distance"] == 1.0
