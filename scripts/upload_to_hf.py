"""Upload PDF-markdown dataset to HuggingFace Hub.

Supports incremental updates via shard-based uploads.
New shards are added without touching existing ones.

Usage:
    python scripts/upload_to_hf.py                          # initial upload
    python scripts/upload_to_hf.py --shard-index 3          # add new shard
    python scripts/upload_to_hf.py --init-repo              # create repo + README only
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ID = "Parassharmaa/pdf-ocr-rl-dataset"
SHARD_MAX_SIZE_MB = 300

README_TEMPLATE = """---
license: cc-by-4.0
language:
  - en
  - ja
task_categories:
  - image-to-text
  - document-question-answering
tags:
  - pdf
  - ocr
  - markdown
  - grpo
  - rl
configs:
  - config_name: default
    data_files:
      - split: train
        path: "data/train/*.parquet"
      - split: test
        path: "data/test/*.parquet"
---

# PDF-OCR-RL Dataset

PDF page images paired with ground-truth markdown for training VLMs on document understanding.

## Overview

- **Task:** PDF page image → Markdown text conversion
- **Languages:** English (EN), Japanese (JA)
- **Sources:** GitHub repos, LaTeX papers (arXiv), Wikipedia
- **Format:** Parquet with embedded PNG images

## Usage

```python
from datasets import load_dataset

ds = load_dataset("Parassharmaa/pdf-ocr-rl-dataset")
sample = ds["train"][0]
sample["image"].show()
print(sample["markdown"][:200])
```

## Schema

| Column | Type | Description |
|--------|------|-------------|
| image | Image | Rendered PDF page as PNG |
| markdown | string | Ground-truth markdown text for this page |
| language | string | Language code (en / ja) |
| source | string | Source identifier (repo name, arxiv ID, etc.) |
| doc_id | string | Stable document identifier |
| page_num | int32 | Page index within source document |

## Incremental Updates

New data is added as additional Parquet shards (`shard-NNNNN.parquet`).
Existing shards are never modified, ensuring reproducibility.
"""


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


def get_next_shard_index(repo_id: str, split: str = "train") -> int:
    """Find the next available shard index by listing existing files."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        shards = [f for f in files if f.startswith(f"data/{split}/shard-") and f.endswith(".parquet")]
        return len(shards)
    except Exception:
        return 0


def init_repo(repo_id: str):
    """Create the HF dataset repo and upload the README."""
    from huggingface_hub import HfApi

    token = get_hf_token()
    api = HfApi(token=token)

    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(README_TEMPLATE)
        readme_path = f.name

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Initialize dataset repository",
    )
    os.unlink(readme_path)
    print(f"Repository initialized: https://huggingface.co/datasets/{repo_id}")


def load_dataset_records(data_dir: str, split: str = "train") -> list[dict]:
    """Load image-markdown pairs from processed dataset directory."""
    data_path = Path(data_dir)
    meta_path = data_path / "dataset_meta.json"

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

        # Create stable doc_id from source file
        source_stem = Path(md_source).stem
        doc_id = f"{entry.get('language', 'en')}_{source_stem}"

        records.append({
            "image": img_path,
            "markdown": md_full[start:end],
            "language": entry.get("language", "en"),
            "source": entry.get("repo", Path(md_source).parent.name),
            "doc_id": doc_id,
            "page_num": entry.get("page_index", 0),
        })

    return records


def upload_shard(records: list[dict], shard_index: int, split: str = "train", repo_id: str = REPO_ID):
    """Upload a single shard of records to HuggingFace."""
    from datasets import ClassLabel, Dataset, Features, Image, Value
    from huggingface_hub import HfApi

    token = get_hf_token()
    api = HfApi(token=token)

    features = Features({
        "image": Image(),
        "markdown": Value("string"),
        "language": Value("string"),
        "source": Value("string"),
        "doc_id": Value("string"),
        "page_num": Value("int32"),
    })

    ds = Dataset.from_list(records, features=features)

    import tempfile

    local_path = os.path.join(tempfile.gettempdir(), f"shard-{shard_index:05d}.parquet")
    ds.to_parquet(local_path)

    remote_path = f"data/{split}/shard-{shard_index:05d}.parquet"
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {split} shard {shard_index:05d} ({len(records)} rows)",
    )

    size_mb = os.path.getsize(local_path) / 1024 / 1024
    os.unlink(local_path)
    print(f"Uploaded {remote_path}: {len(records)} rows, {size_mb:.1f} MB")


def upload_dataset(data_dir: str, repo_id: str = REPO_ID, shard_index: int | None = None):
    """Upload the full dataset as shards."""
    token = get_hf_token()

    # Ensure repo exists
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

    for split in ["train", "test"]:
        records = load_dataset_records(data_dir, split=split)
        if not records:
            print(f"No {split} records found")
            continue

        print(f"\n{split.upper()}: {len(records)} records")

        start_idx = shard_index if shard_index is not None else get_next_shard_index(repo_id, split)

        # Split into shards of ~SHARD_MAX_SIZE_MB
        # Rough estimate: ~500KB per record with image
        records_per_shard = max(100, SHARD_MAX_SIZE_MB * 2)  # ~600 records per 300MB shard
        num_shards = max(1, (len(records) + records_per_shard - 1) // records_per_shard)

        for i in range(num_shards):
            chunk = records[i * records_per_shard : (i + 1) * records_per_shard]
            if chunk:
                upload_shard(chunk, start_idx + i, split=split, repo_id=repo_id)

    # Upload README
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(README_TEMPLATE)
        readme_path = f.name

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update README",
    )
    os.unlink(readme_path)

    print(f"\nDataset uploaded: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace Hub")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--repo-id", type=str, default=REPO_ID)
    parser.add_argument("--shard-index", type=int, default=None, help="Start shard index (auto-detect if not set)")
    parser.add_argument("--init-repo", action="store_true", help="Only create repo + README")
    args = parser.parse_args()

    if args.init_repo:
        init_repo(args.repo_id)
    else:
        upload_dataset(args.data_dir, args.repo_id, args.shard_index)


if __name__ == "__main__":
    main()
