"""Composite reward function for GRPO training on PDF-to-markdown."""

import re

from Levenshtein import ratio as levenshtein_ratio


def normalized_edit_distance(predicted: str, reference: str) -> float:
    """Levenshtein similarity ratio between predicted and reference markdown."""
    if not reference:
        return 1.0 if not predicted else 0.0
    return levenshtein_ratio(predicted, reference)


def structural_validity_reward(markdown: str) -> float:
    """Check markdown structural validity: balanced code blocks, valid tables, proper headings."""
    score = 1.0
    penalties = 0

    # Check balanced code fences
    code_fences = markdown.count("```")
    if code_fences % 2 != 0:
        penalties += 1

    # Check heading structure (shouldn't skip levels drastically)
    headings = re.findall(r"^(#{1,6})\s", markdown, re.MULTILINE)
    if headings:
        levels = [len(h) for h in headings]
        for i in range(1, len(levels)):
            if levels[i] > levels[i - 1] + 1:
                penalties += 0.5

    # Check table structure (consistent column counts per table)
    table_rows = re.findall(r"^\|(.+)\|$", markdown, re.MULTILINE)
    if table_rows:
        col_counts = [row.count("|") + 1 for row in table_rows]
        # Group consecutive table rows
        tables = []
        current_table = [col_counts[0]]
        for i in range(1, len(col_counts)):
            current_table.append(col_counts[i])
        tables.append(current_table)
        for table in tables:
            if len(set(table)) > 1:
                penalties += 0.5

    # Check for orphaned formatting markers
    bold_markers = markdown.count("**")
    if bold_markers % 2 != 0:
        penalties += 0.3
    italic_single = len(re.findall(r"(?<!\*)\*(?!\*)", markdown))
    if italic_single % 2 != 0:
        penalties += 0.3

    score = max(0.0, score - penalties * 0.15)
    return score


def heading_accuracy(predicted: str, reference: str) -> float:
    """Compare heading structure between predicted and reference."""
    pred_headings = re.findall(r"^(#{1,6})\s+(.+)$", predicted, re.MULTILINE)
    ref_headings = re.findall(r"^(#{1,6})\s+(.+)$", reference, re.MULTILINE)

    if not ref_headings:
        return 1.0 if not pred_headings else 0.5

    # Compare heading count
    count_score = 1.0 - min(1.0, abs(len(pred_headings) - len(ref_headings)) / max(len(ref_headings), 1))

    # Compare heading levels
    pred_levels = [len(h[0]) for h in pred_headings]
    ref_levels = [len(h[0]) for h in ref_headings]
    min_len = min(len(pred_levels), len(ref_levels))
    if min_len > 0:
        level_matches = sum(1 for i in range(min_len) if pred_levels[i] == ref_levels[i])
        level_score = level_matches / max(len(ref_levels), 1)
    else:
        level_score = 0.0

    return 0.5 * count_score + 0.5 * level_score


def reading_order_reward(predicted: str, reference: str) -> float:
    """Check if content blocks appear in the same order.

    Extracts significant text chunks and checks their relative ordering.
    """
    def extract_chunks(text: str) -> list[str]:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Filter to non-trivial lines (>10 chars, not just formatting)
        chunks = []
        for line in lines:
            cleaned = re.sub(r"[#*|`\-=]", "", line).strip()
            if len(cleaned) > 10:
                chunks.append(cleaned[:50])  # first 50 chars as identifier
        return chunks

    pred_chunks = extract_chunks(predicted)
    ref_chunks = extract_chunks(reference)

    if not ref_chunks:
        return 1.0

    # Find matching chunks and check order preservation
    matches = []
    for i, ref_chunk in enumerate(ref_chunks):
        best_match = -1
        best_sim = 0.0
        for j, pred_chunk in enumerate(pred_chunks):
            sim = levenshtein_ratio(ref_chunk, pred_chunk)
            if sim > 0.6 and sim > best_sim:
                best_sim = sim
                best_match = j
        if best_match >= 0:
            matches.append((i, best_match))

    if len(matches) < 2:
        return 0.5

    # Check if matched positions are monotonically increasing
    in_order = sum(
        1
        for i in range(1, len(matches))
        if matches[i][1] > matches[i - 1][1]
    )
    return in_order / (len(matches) - 1)


def composite_reward(predicted: str, reference: str, weights: dict | None = None) -> float:
    """Compute composite reward combining all sub-rewards.

    Args:
        predicted: Model-generated markdown
        reference: Ground-truth markdown
        weights: Optional dict with keys 'edit_distance', 'structural',
                 'heading', 'reading_order'

    Returns:
        Float reward in [0, 1]
    """
    if weights is None:
        weights = {
            "edit_distance": 0.4,
            "structural": 0.15,
            "heading": 0.2,
            "reading_order": 0.25,
        }

    scores = {
        "edit_distance": normalized_edit_distance(predicted, reference),
        "structural": structural_validity_reward(predicted),
        "heading": heading_accuracy(predicted, reference),
        "reading_order": reading_order_reward(predicted, reference),
    }

    total = sum(weights[k] * scores[k] for k in weights)
    return total
