"""Evaluate and compare base vs fine-tuned model on PDF-to-markdown.

Usage:
    python scripts/evaluate.py --model-path results/grpo_run1/final_model
    python scripts/evaluate.py --base-only  # evaluate base model only
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml
from PIL import Image


def load_test_data(data_dir: str, max_samples: int = 50) -> list[dict]:
    """Load stratified test split of the dataset (10% per language).

    Data sources (in priority order):
    1. Local data_dir with dataset_meta.json
    2. HuggingFace Hub: blazeofchi/pdf-ocr-rl-dataset (test split)
    """
    meta_path = Path(data_dir) / "dataset_meta.json"

    if not meta_path.exists():
        print("Local dataset not found, loading test split from HuggingFace Hub...")
        from pdf_ocr_rl.data.dataset import load_hf_dataset
        hf_ds = load_hf_dataset(split="test", max_samples=max_samples)
        samples = []
        for row in hf_ds:
            img = row["image"]
            if hasattr(img, "convert"):
                img = img.convert("RGB")
            # Save image to temp file for run_inference compatibility
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            samples.append({
                "image_path": tmp.name,
                "markdown": row["markdown"],
                "language": row.get("language", "en"),
            })
            if len(samples) >= max_samples:
                break
        print(f"Loaded {len(samples)} test samples from HuggingFace Hub")
        return samples

    meta = json.loads(meta_path.read_text())

    # Stratified split: take 10% from each language group
    by_lang: dict[str, list] = {}
    for entry in meta:
        lang = entry.get("language", "en")
        by_lang.setdefault(lang, []).append(entry)

    test_meta = []
    for lang, entries in by_lang.items():
        split_idx = int(len(entries) * 0.9)
        test_meta.extend(entries[split_idx:])

    samples = []
    for entry in test_meta[:max_samples]:
        img_path = entry["image_path"]
        md_source = entry["source"]
        if not Path(img_path).exists() or not Path(md_source).exists():
            continue

        md_full = Path(md_source).read_text(encoding="utf-8")
        start = entry.get("page_start_char", 0)
        end = entry.get("page_end_char", len(md_full))
        md_content = md_full[start:end]

        samples.append({
            "image_path": img_path,
            "markdown": md_content,
            "language": entry.get("language", "en"),
        })
    return samples


def run_inference(model, tokenizer, processor, image_path: str, language: str = "en") -> str:
    """Run inference on a single image."""
    prompt = (
        "この画像はPDFドキュメントのページです。画像の内容を正確にMarkdown形式に変換してください。"
        if language == "ja"
        else "This image is a page from a PDF document. Convert the content accurately to Markdown format."
    )

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Use processor.apply_chat_template to get text with image tokens
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    # Use processor directly (not tokenizer) to avoid Unsloth patch conflicts
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated.strip()


def evaluate_model(model, tokenizer, processor, test_data: list[dict], model_name: str) -> dict:
    """Evaluate a model on test data and return metrics."""
    from pdf_ocr_rl.eval.metrics import evaluate_sample

    results = {"model": model_name, "samples": [], "metrics_by_language": {}}
    all_metrics = []

    print(f"\nEvaluating {model_name} on {len(test_data)} samples...")
    start_time = time.time()

    for i, sample in enumerate(test_data):
        try:
            predicted = run_inference(model, tokenizer, processor, sample["image_path"], sample["language"])
            metrics = evaluate_sample(predicted, sample["markdown"])
            metrics["language"] = sample["language"]
            all_metrics.append(metrics)

            results["samples"].append({
                "image": sample["image_path"],
                "language": sample["language"],
                "metrics": metrics,
                "predicted_length": len(predicted),
                "reference_length": len(sample["markdown"]),
            })
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            avg = sum(m.get("composite", 0) for m in all_metrics) / len(all_metrics) if all_metrics else 0
            print(f"  Processed {i + 1}/{len(test_data)} ({elapsed:.1f}s, avg composite: {avg:.4f})...")

    elapsed = time.time() - start_time
    results["total_time_seconds"] = elapsed
    results["avg_time_per_sample"] = elapsed / max(len(all_metrics), 1)

    if not all_metrics:
        print("  No samples evaluated successfully!")
        return results

    # Aggregate metrics
    metric_keys = [k for k in all_metrics[0] if k != "language"]

    # Overall
    results["overall"] = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in metric_keys}

    # By language
    for lang in ["en", "ja"]:
        lang_metrics = [m for m in all_metrics if m["language"] == lang]
        if lang_metrics:
            results["metrics_by_language"][lang] = {
                k: sum(m[k] for m in lang_metrics) / len(lang_metrics) for k in metric_keys
            }
            results["metrics_by_language"][lang]["count"] = len(lang_metrics)

    return results


def print_comparison(base_results: dict | None, finetuned_results: dict | None):
    """Print a comparison table of base vs fine-tuned model."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS COMPARISON")
    print("=" * 80)

    def _print_metrics(label: str, base: dict | None, ft: dict | None):
        print(f"\n--- {label} ---")
        print(f"{'Metric':<30} ", end="")
        if base:
            print(f"{'Base':>10} ", end="")
        if ft:
            print(f"{'Fine-tuned':>10} ", end="")
        if base and ft:
            print(f"{'Delta':>10}", end="")
        print()
        print("-" * 70)

        keys = (base or ft).keys()
        for key in keys:
            if key == "count":
                continue
            print(f"{key:<30} ", end="")
            bv = base.get(key) if base else None
            fv = ft.get(key) if ft else None
            if bv is not None:
                print(f"{bv:>10.4f} ", end="")
            if fv is not None:
                print(f"{fv:>10.4f} ", end="")
            if bv is not None and fv is not None:
                delta = fv - bv
                sign = "+" if delta >= 0 else ""
                print(f"{sign}{delta:>9.4f}", end="")
            print()

    bo = base_results.get("overall") if base_results else None
    fo = finetuned_results.get("overall") if finetuned_results else None
    _print_metrics("Overall", bo, fo)

    for lang in ["en", "ja"]:
        bl = base_results.get("metrics_by_language", {}).get(lang) if base_results else None
        fl = finetuned_results.get("metrics_by_language", {}).get(lang) if finetuned_results else None
        if bl or fl:
            _print_metrics(f"Language: {lang.upper()}", bl, fl)


def main():
    parser = argparse.ArgumentParser(description="Evaluate PDF-to-markdown models")
    parser.add_argument("--model-path", type=str, help="Path to fine-tuned model")
    parser.add_argument("--base-only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="results/evaluation")
    parser.add_argument("--config", type=str, default="configs/grpo_train.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    base_model_name = config["model"]["name"]

    # Load test data
    test_data = load_test_data(args.data_dir, args.max_samples)
    print(f"Loaded {len(test_data)} test samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor
    from unsloth import FastVisionModel

    # Load processor separately (unpatched) for inference
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)

    base_results = None
    finetuned_results = None

    # Evaluate base model
    print("\n--- Evaluating BASE model ---")
    base_model, base_tokenizer = FastVisionModel.from_pretrained(
        base_model_name,
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(base_model)
    base_results = evaluate_model(base_model, base_tokenizer, processor, test_data, f"Base ({base_model_name})")
    (output_dir / "base_results.json").write_text(json.dumps(base_results, indent=2, default=str))

    del base_model, base_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate fine-tuned model
    if not args.base_only and args.model_path:
        print("\n--- Evaluating FINE-TUNED model ---")
        ft_model, ft_tokenizer = FastVisionModel.from_pretrained(
            args.model_path,
            max_seq_length=config["model"]["max_seq_length"],
            load_in_4bit=True,
        )
        FastVisionModel.for_inference(ft_model)
        finetuned_results = evaluate_model(ft_model, ft_tokenizer, processor, test_data, f"Fine-tuned ({args.model_path})")
        (output_dir / "finetuned_results.json").write_text(json.dumps(finetuned_results, indent=2, default=str))

    print_comparison(base_results, finetuned_results)

    comparison = {"base": base_results, "finetuned": finetuned_results}
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str))
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
