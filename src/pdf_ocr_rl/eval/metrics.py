"""Evaluation metrics for PDF-to-markdown conversion quality."""

import re

from Levenshtein import ratio as levenshtein_ratio


def compute_edit_distance(predicted: str, reference: str) -> float:
    """Normalized edit distance (Levenshtein similarity ratio)."""
    return levenshtein_ratio(predicted, reference)


def compute_heading_accuracy(predicted: str, reference: str) -> dict:
    """Evaluate heading extraction accuracy."""
    pred_headings = re.findall(r"^(#{1,6})\s+(.+)$", predicted, re.MULTILINE)
    ref_headings = re.findall(r"^(#{1,6})\s+(.+)$", reference, re.MULTILINE)

    if not ref_headings:
        return {"heading_precision": 1.0, "heading_recall": 1.0, "heading_f1": 1.0}

    # Match headings by text similarity
    matched = 0
    for ref_level, ref_text in ref_headings:
        for pred_level, pred_text in pred_headings:
            if levenshtein_ratio(ref_text.lower(), pred_text.lower()) > 0.7:
                matched += 1
                break

    precision = matched / len(pred_headings) if pred_headings else 0.0
    recall = matched / len(ref_headings)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"heading_precision": precision, "heading_recall": recall, "heading_f1": f1}


def compute_table_accuracy(predicted: str, reference: str) -> dict:
    """Evaluate table extraction accuracy."""
    def extract_tables(text: str) -> list[list[list[str]]]:
        tables = []
        current_table = []
        for line in text.split("\n"):
            if "|" in line and line.strip().startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if not all(re.match(r"^[-:]+$", c) for c in cells):
                    current_table.append(cells)
            elif current_table:
                tables.append(current_table)
                current_table = []
        if current_table:
            tables.append(current_table)
        return tables

    pred_tables = extract_tables(predicted)
    ref_tables = extract_tables(reference)

    if not ref_tables:
        return {"table_count_match": 1.0 if not pred_tables else 0.0, "table_cell_accuracy": 1.0}

    count_match = 1.0 if len(pred_tables) == len(ref_tables) else max(0, 1 - abs(len(pred_tables) - len(ref_tables)) / len(ref_tables))

    # Cell-level accuracy for matched tables
    cell_scores = []
    for i in range(min(len(pred_tables), len(ref_tables))):
        pred_t = pred_tables[i]
        ref_t = ref_tables[i]
        total_cells = sum(len(row) for row in ref_t)
        matched_cells = 0
        for r in range(min(len(pred_t), len(ref_t))):
            for c in range(min(len(pred_t[r]), len(ref_t[r]))):
                if levenshtein_ratio(pred_t[r][c], ref_t[r][c]) > 0.7:
                    matched_cells += 1
        cell_scores.append(matched_cells / total_cells if total_cells > 0 else 0.0)

    cell_accuracy = sum(cell_scores) / len(cell_scores) if cell_scores else 0.0

    return {"table_count_match": count_match, "table_cell_accuracy": cell_accuracy}


def compute_code_block_accuracy(predicted: str, reference: str) -> dict:
    """Evaluate code block extraction accuracy."""
    def extract_code_blocks(text: str) -> list[str]:
        blocks = re.findall(r"```[\w]*\n(.*?)```", text, re.DOTALL)
        return [b.strip() for b in blocks]

    pred_blocks = extract_code_blocks(predicted)
    ref_blocks = extract_code_blocks(reference)

    if not ref_blocks:
        return {"code_block_count_match": 1.0 if not pred_blocks else 0.0, "code_block_similarity": 1.0}

    count_match = 1.0 if len(pred_blocks) == len(ref_blocks) else max(0, 1 - abs(len(pred_blocks) - len(ref_blocks)) / len(ref_blocks))

    # Content similarity for matched blocks
    similarities = []
    for i in range(min(len(pred_blocks), len(ref_blocks))):
        similarities.append(levenshtein_ratio(pred_blocks[i], ref_blocks[i]))

    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

    return {"code_block_count_match": count_match, "code_block_similarity": avg_sim}


def compute_word_metrics(predicted: str, reference: str) -> dict:
    """Word-level precision, recall, F1 — more forgiving than char-level edit distance."""
    pred_words = set(predicted.lower().split())
    ref_words = set(reference.lower().split())
    if not ref_words:
        return {"word_precision": 1.0, "word_recall": 1.0, "word_f1": 1.0}
    if not pred_words:
        return {"word_precision": 0.0, "word_recall": 0.0, "word_f1": 0.0}
    common = pred_words & ref_words
    precision = len(common) / len(pred_words)
    recall = len(common) / len(ref_words)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"word_precision": precision, "word_recall": recall, "word_f1": f1}


def evaluate_sample(predicted: str, reference: str) -> dict:
    """Run all metrics on a single sample."""
    results = {
        "edit_distance": compute_edit_distance(predicted, reference),
    }
    results.update(compute_heading_accuracy(predicted, reference))
    results.update(compute_table_accuracy(predicted, reference))
    results.update(compute_code_block_accuracy(predicted, reference))
    results.update(compute_word_metrics(predicted, reference))
    return results


def evaluate_batch(predictions: list[str], references: list[str]) -> dict:
    """Evaluate a batch and return aggregated metrics."""
    all_metrics = [evaluate_sample(p, r) for p, r in zip(predictions, references)]

    # Average each metric
    keys = all_metrics[0].keys()
    avg = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in keys}
    return avg
