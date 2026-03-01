"""Tests for PDF rendering and page-level markdown splitting."""

from pdf_ocr_rl.data.render_pdf import split_markdown_by_pages


def test_single_page():
    """Single page returns the full content."""
    md = "# Hello\n\nWorld"
    chunks = split_markdown_by_pages(md, 1)
    assert len(chunks) == 1
    assert chunks[0]["text"] == md
    assert chunks[0]["start_char"] == 0
    assert chunks[0]["end_char"] == len(md)


def test_multi_page_splits_at_paragraph_boundary():
    """Multi-page split prefers paragraph boundaries (\\n\\n)."""
    paragraphs = [f"Paragraph {i}.\nSome details here." for i in range(6)]
    md = "\n\n".join(paragraphs)
    chunks = split_markdown_by_pages(md, 3)

    assert len(chunks) == 3
    # Reassembled text should equal original
    reassembled = "".join(c["text"] for c in chunks)
    assert reassembled == md
    # Offsets should be contiguous
    assert chunks[0]["start_char"] == 0
    for i in range(1, 3):
        assert chunks[i]["start_char"] == chunks[i - 1]["end_char"]
    assert chunks[-1]["end_char"] == len(md)


def test_two_pages():
    """Two pages split roughly in half."""
    md = "# Part One\n\nContent A.\n\n# Part Two\n\nContent B."
    chunks = split_markdown_by_pages(md, 2)
    assert len(chunks) == 2
    assert "".join(c["text"] for c in chunks) == md


def test_no_paragraph_boundaries():
    """Falls back to newlines or raw split when no \\n\\n exists."""
    md = "line1\nline2\nline3\nline4\nline5\nline6"
    chunks = split_markdown_by_pages(md, 2)
    assert len(chunks) == 2
    assert "".join(c["text"] for c in chunks) == md


def test_empty_content():
    """Edge case: empty string."""
    chunks = split_markdown_by_pages("", 1)
    assert len(chunks) == 1
    assert chunks[0]["text"] == ""


def test_chunks_cover_full_content():
    """Offsets cover the entire content without gaps or overlaps."""
    md = "A" * 1000 + "\n\n" + "B" * 1000 + "\n\n" + "C" * 1000
    for num_pages in [2, 3, 5]:
        chunks = split_markdown_by_pages(md, num_pages)
        assert len(chunks) == num_pages
        assert chunks[0]["start_char"] == 0
        assert chunks[-1]["end_char"] == len(md)
        for i in range(1, len(chunks)):
            assert chunks[i]["start_char"] == chunks[i - 1]["end_char"]
