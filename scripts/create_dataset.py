"""Create the PDF-markdown dataset for training.

Usage:
    python scripts/create_dataset.py [--github-token TOKEN]

Steps:
1. Collect markdown files from GitHub repos, Wikipedia, arXiv LaTeX, and other sources
2. Render them to PDF images
3. Optionally fetch pre-made pairs from HuggingFace (olmOCR-mix)
4. Create paired dataset for training
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create PDF-markdown training dataset")
    parser.add_argument("--github-token", type=str, default=None, help="GitHub API token for higher rate limits")
    parser.add_argument("--max-per-repo", type=int, default=30, help="Max markdown files per repo")
    parser.add_argument("--max-arxiv", type=int, default=100, help="Max arXiv papers to fetch")
    parser.add_argument("--max-olmocr", type=int, default=0, help="Max olmOCR pre-made pairs (0=skip)")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for PDF rendering")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    token = args.github_token or os.environ.get("GITHUB_TOKEN")

    # Step 1: Collect markdown
    print("=" * 60)
    print("Step 1: Collecting markdown files")
    print("=" * 60)

    from pdf_ocr_rl.data.collect_markdown import collect_all

    raw_dir = "data/raw/markdown"
    docs = collect_all(raw_dir, max_per_repo=args.max_per_repo, max_arxiv=args.max_arxiv, github_token=token)
    print(f"\nCollected {len(docs)} markdown documents\n")

    # Step 2: Render to PDF images
    print("=" * 60)
    print("Step 2: Rendering markdown to PDF images")
    print("=" * 60)

    from pdf_ocr_rl.data.render_pdf import render_dataset

    pairs = render_dataset(raw_dir, args.output_dir, dpi=args.dpi)
    print(f"\nCreated {len(pairs)} image-markdown pairs\n")

    # Step 3: Optionally fetch pre-made pairs from HuggingFace
    if args.max_olmocr > 0:
        print("=" * 60)
        print("Step 3: Fetching pre-made pairs from olmOCR-mix")
        print("=" * 60)

        from pdf_ocr_rl.data.collect_markdown import fetch_olmocr_pairs

        olmocr_pairs = fetch_olmocr_pairs(max_samples=args.max_olmocr, output_dir=args.output_dir)

        if olmocr_pairs:
            # Append olmOCR pairs to existing dataset_meta.json
            meta_path = Path(args.output_dir) / "dataset_meta.json"
            existing_meta = json.loads(meta_path.read_text()) if meta_path.exists() else []
            existing_meta.extend(olmocr_pairs)
            meta_path.write_text(json.dumps(existing_meta, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nAdded {len(olmocr_pairs)} olmOCR pairs (total: {len(existing_meta)})\n")

    total_meta = Path(args.output_dir) / "dataset_meta.json"
    total = len(json.loads(total_meta.read_text())) if total_meta.exists() else 0

    print("=" * 60)
    print("Dataset creation complete!")
    print(f"Output: {args.output_dir}")
    print(f"Total pairs: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
