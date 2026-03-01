"""HuggingFace-compatible dataset for PDF-to-markdown training.

Supports two data sources:
1. Local files via dataset_meta.json (from scripts/create_dataset.py)
2. HuggingFace Hub dataset: blazeofchi/pdf-ocr-rl-dataset

HuggingFace dataset schema:
    - image (Image): rendered PDF page
    - markdown (str): page-level markdown content
    - language (str): "en" or "ja"
    - source (str): origin repo/document
    - doc_id (str): unique document identifier
    - page_num (int): page index within the document
"""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

HF_DATASET_REPO = "blazeofchi/pdf-ocr-rl-dataset"


def load_hf_dataset(split: str = "train", max_samples: int | None = None):
    """Load the dataset from HuggingFace Hub.

    Args:
        split: "train" or "test"
        max_samples: cap the number of samples (None = all)

    Returns:
        HuggingFace Dataset with columns: image, markdown, language, source, doc_id, page_num
    """
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_REPO, split=split)
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    return ds


class PDFMarkdownDataset(Dataset):
    """Dataset of PDF image -> markdown pairs for VLM training.

    Loads from local data_dir if available, otherwise falls back to HuggingFace Hub.
    """

    def __init__(self, data_dir: str | None = None, split: str = "train", max_samples: int | None = None):
        self.pairs = []
        self._hf_dataset = None

        # Try local data first
        if data_dir and Path(data_dir).exists():
            self._load_local(data_dir, split)
        else:
            self._load_from_hub(split)

        if max_samples:
            if self._hf_dataset is not None:
                if len(self._hf_dataset) > max_samples:
                    self._hf_dataset = self._hf_dataset.select(range(max_samples))
            else:
                self.pairs = self.pairs[:max_samples]

    def _load_local(self, data_dir: str, split: str):
        meta_path = Path(data_dir) / "dataset_meta.json"
        if not meta_path.exists():
            self._load_from_hub(split)
            return

        meta = json.loads(meta_path.read_text())
        for entry in meta:
            img_path = entry["image_path"]
            md_source = entry["source"]
            if Path(img_path).exists() and Path(md_source).exists():
                self.pairs.append({
                    "image_path": img_path,
                    "markdown_path": md_source,
                    "language": entry.get("language", "en"),
                    "page_start_char": entry.get("page_start_char", 0),
                    "page_end_char": entry.get("page_end_char"),
                })

        # Stratified split: 90% train / 10% test per language
        by_lang: dict[str, list] = {}
        for pair in self.pairs:
            by_lang.setdefault(pair["language"], []).append(pair)

        split_pairs = []
        for lang, entries in by_lang.items():
            split_idx = int(len(entries) * 0.9)
            if split == "train":
                split_pairs.extend(entries[:split_idx])
            else:
                split_pairs.extend(entries[split_idx:])
        self.pairs = split_pairs

    def _load_from_hub(self, split: str):
        self._hf_dataset = load_hf_dataset(split=split)
        print(f"Loaded {len(self._hf_dataset)} samples from HuggingFace Hub ({HF_DATASET_REPO})")

    def __len__(self):
        if self._hf_dataset is not None:
            return len(self._hf_dataset)
        return len(self.pairs)

    def __getitem__(self, idx):
        if self._hf_dataset is not None:
            row = self._hf_dataset[idx]
            return {
                "image": row["image"].convert("RGB") if isinstance(row["image"], Image.Image) else row["image"],
                "markdown": row["markdown"],
                "language": row["language"],
            }

        pair = self.pairs[idx]
        image = Image.open(pair["image_path"]).convert("RGB")
        md_full = Path(pair["markdown_path"]).read_text(encoding="utf-8")
        start = pair.get("page_start_char", 0)
        end = pair.get("page_end_char") or len(md_full)
        markdown = md_full[start:end]
        return {
            "image": image,
            "markdown": markdown,
            "language": pair["language"],
        }


def format_prompt(language: str = "en") -> str:
    """Format the system/user prompt for PDF-to-markdown conversion."""
    if language == "ja":
        return (
            "この画像はPDFドキュメントのページです。"
            "画像の内容を正確にMarkdown形式に変換してください。"
            "見出し、表、コードブロック、リストなどの書式を正しく再現してください。"
        )
    return (
        "This image is a page from a PDF document. "
        "Convert the content of this image accurately to Markdown format. "
        "Preserve headings, tables, code blocks, lists, and other formatting faithfully."
    )
