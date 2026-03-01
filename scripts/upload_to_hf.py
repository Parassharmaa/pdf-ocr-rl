"""Upload PDF-markdown dataset to HuggingFace Hub.

Usage:
    python scripts/upload_to_hf.py                        # upload full dataset
    python scripts/upload_to_hf.py --data-dir data/processed --split train
"""

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ID = "blazeofchi/pdf-ocr-rl-dataset"


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break
    if not token:
        print("Error: HF_TOKEN not found in environment or .env file")
        sys.exit(1)
    return token


def load_records(data_dir: str, split: str = "train") -> list[dict]:
    """Load image-markdown pairs with stratified train/test split."""
    meta_path = Path(data_dir) / "dataset_meta.json"
    if not meta_path.exists():
        print(f"No dataset metadata at {meta_path}")
        return []

    meta = json.loads(meta_path.read_text())

    # Stratified split: 90% train / 10% test per language
    by_lang: dict[str, list] = {}
    for entry in meta:
        lang = entry.get("language", "en")
        by_lang.setdefault(lang, []).append(entry)

    split_entries = []
    for lang, entries in by_lang.items():
        split_idx = int(len(entries) * 0.9)
        if split == "train":
            split_entries.extend(entries[:split_idx])
        else:
            split_entries.extend(entries[split_idx:])

    records = []
    for entry in split_entries:
        img_path = entry["image_path"]
        md_source = entry["source"]
        if not Path(img_path).exists() or not Path(md_source).exists():
            continue

        md_full = Path(md_source).read_text(encoding="utf-8")
        start = entry.get("page_start_char", 0)
        end = entry.get("page_end_char", len(md_full))
        source_stem = Path(md_source).stem

        records.append({
            "image": img_path,
            "markdown": md_full[start:end],
            "language": entry.get("language", "en"),
            "source": entry.get("repo", Path(md_source).parent.name),
            "doc_id": f"{entry.get('language', 'en')}_{source_stem}",
            "page_num": entry.get("page_index", 0),
        })

    return records


def upload_dataset(data_dir: str, repo_id: str = REPO_ID):
    """Upload the full dataset to HuggingFace Hub."""
    from datasets import Dataset, DatasetDict, Features, Image, Value

    token = get_hf_token()

    features = Features({
        "image": Image(),
        "markdown": Value("string"),
        "language": Value("string"),
        "source": Value("string"),
        "doc_id": Value("string"),
        "page_num": Value("int32"),
    })

    splits = {}
    for split in ["train", "test"]:
        records = load_records(data_dir, split=split)
        if records:
            splits[split] = Dataset.from_list(records, features=features)
            print(f"{split}: {len(records)} records")

    if not splits:
        print("No records found")
        return

    ds_dict = DatasetDict(splits)
    ds_dict.push_to_hub(repo_id, token=token, commit_message="Update dataset")
    print(f"\nUploaded: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace Hub")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--repo-id", type=str, default=REPO_ID)
    args = parser.parse_args()
    upload_dataset(args.data_dir, args.repo_id)


if __name__ == "__main__":
    main()
