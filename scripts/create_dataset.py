"""Create the PDF-markdown dataset for training.

Usage:
    python scripts/create_dataset.py [--github-token TOKEN]

Steps:
1. Collect markdown files from GitHub repos and other sources
2. Render them to PDF images
3. Create paired dataset for training
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Create PDF-markdown training dataset")
    parser.add_argument("--github-token", type=str, default=None, help="GitHub API token for higher rate limits")
    parser.add_argument("--max-per-repo", type=int, default=15, help="Max markdown files per repo")
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
    docs = collect_all(raw_dir, max_per_repo=args.max_per_repo, github_token=token)
    print(f"\nCollected {len(docs)} markdown documents\n")

    # Step 2: Render to PDF images
    print("=" * 60)
    print("Step 2: Rendering markdown to PDF images")
    print("=" * 60)

    from pdf_ocr_rl.data.render_pdf import render_dataset

    pairs = render_dataset(raw_dir, args.output_dir, dpi=args.dpi)
    print(f"\nCreated {len(pairs)} image-markdown pairs\n")

    print("=" * 60)
    print("Dataset creation complete!")
    print(f"Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
