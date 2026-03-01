"""HuggingFace-compatible dataset for PDF-to-markdown training."""

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class PDFMarkdownDataset(Dataset):
    """Dataset of PDF image -> markdown pairs for VLM training."""

    def __init__(self, data_dir: str, split: str = "train", max_samples: int | None = None):
        self.data_dir = Path(data_dir)
        self.pairs = []

        meta_path = self.data_dir / "dataset_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            # Load corresponding markdown content
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

        if max_samples:
            self.pairs = self.pairs[:max_samples]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
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


def create_hf_dataset(data_dir: str, output_path: str | None = None):
    """Create a HuggingFace Dataset from rendered PDF-markdown pairs."""
    try:
        from datasets import Dataset as HFDataset, Features, Image as HFImage, Value

        dataset_path = Path(data_dir)
        meta_path = dataset_path / "dataset_meta.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"No dataset metadata found at {meta_path}")

        meta = json.loads(meta_path.read_text())

        records = []
        for entry in meta:
            img_path = entry["image_path"]
            md_source = entry["source"]
            if Path(img_path).exists() and Path(md_source).exists():
                md_full = Path(md_source).read_text(encoding="utf-8")
                start = entry.get("page_start_char", 0)
                end = entry.get("page_end_char", len(md_full))
                records.append({
                    "image": img_path,
                    "markdown": md_full[start:end],
                    "language": entry.get("language", "en"),
                })

        features = Features({
            "image": HFImage(),
            "markdown": Value("string"),
            "language": Value("string"),
        })

        ds = HFDataset.from_list(records, features=features)

        if output_path:
            ds.save_to_disk(output_path)
            print(f"Saved dataset to {output_path}")

        return ds

    except ImportError:
        print("datasets library not available, returning raw pairs")
        return None


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
