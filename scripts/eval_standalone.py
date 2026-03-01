"""Standalone evaluation using transformers + peft (no unsloth dependency).

Usage:
    python scripts/eval_standalone.py --model-path results/qwen3vl2b_v2/final_model --max-samples 20
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from pdf_ocr_rl.data.dataset import format_prompt, strip_thinking
from pdf_ocr_rl.eval.metrics import evaluate_sample


def load_test_data(max_samples: int = 20) -> list[dict]:
    """Load test data from HuggingFace Hub."""
    from pdf_ocr_rl.data.dataset import load_hf_dataset
    hf_ds = load_hf_dataset(split="test", max_samples=max_samples)
    samples = []
    for row in hf_ds:
        img = row["image"]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        samples.append({
            "image": img,
            "markdown": row["markdown"],
            "language": row.get("language", "en"),
        })
    print(f"Loaded {len(samples)} test samples from HuggingFace Hub")
    return samples


def run_inference(model, processor, image, language="en"):
    """Run inference on a single image."""
    prompt = format_prompt(language)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    generated = processor.tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return strip_thinking(generated)


def evaluate_model_standalone(model, processor, test_data, model_name):
    """Evaluate model and return metrics."""
    results = {"model": model_name, "samples": [], "metrics_by_language": {}}
    all_metrics = []

    print(f"\nEvaluating {model_name} on {len(test_data)} samples...")
    start_time = time.time()

    for i, sample in enumerate(test_data):
        try:
            predicted = run_inference(model, processor, sample["image"], sample["language"])
            metrics = evaluate_sample(predicted, sample["markdown"])
            metrics["language"] = sample["language"]
            all_metrics.append(metrics)
            results["samples"].append({
                "language": sample["language"],
                "metrics": metrics,
                "predicted_length": len(predicted),
                "reference_length": len(sample["markdown"]),
            })
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            import traceback; traceback.print_exc()
            continue

        if (i + 1) % 5 == 0:
            elapsed = time.time() - start_time
            avg_ed = sum(m.get("edit_distance", 0) for m in all_metrics) / len(all_metrics)
            avg_wf1 = sum(m.get("word_f1", 0) for m in all_metrics) / len(all_metrics)
            print(f"  [{i+1}/{len(test_data)}] {elapsed:.0f}s | edit_dist={avg_ed:.4f} word_f1={avg_wf1:.4f}")

    elapsed = time.time() - start_time
    results["total_time_seconds"] = elapsed

    if not all_metrics:
        print("  No samples evaluated!")
        return results

    metric_keys = [k for k in all_metrics[0] if k != "language"]
    results["overall"] = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in metric_keys}

    for lang in ["en", "ja"]:
        lang_m = [m for m in all_metrics if m["language"] == lang]
        if lang_m:
            results["metrics_by_language"][lang] = {
                k: sum(m[k] for m in lang_m) / len(lang_m) for k in metric_keys
            }
            results["metrics_by_language"][lang]["count"] = len(lang_m)

    return results


def print_comparison(base_r, ft_r):
    """Print comparison table."""
    print("\n" + "=" * 90)
    print("EVALUATION COMPARISON (with word-level metrics)")
    print("=" * 90)

    def _print(label, b, f):
        print(f"\n--- {label} ---")
        hdr = f"{'Metric':<30}"
        if b: hdr += f"{'Base':>12}"
        if f: hdr += f"{'Fine-tuned':>12}"
        if b and f: hdr += f"{'Delta':>12}"
        print(hdr)
        print("-" * 70)
        keys = (b or f).keys()
        for k in keys:
            if k == "count": continue
            line = f"{k:<30}"
            bv = b.get(k) if b else None
            fv = f.get(k) if f else None
            if bv is not None: line += f"{bv:>12.4f}"
            if fv is not None: line += f"{fv:>12.4f}"
            if bv is not None and fv is not None:
                d = fv - bv
                sign = "+" if d >= 0 else ""
                line += f"  {sign}{d:.4f}"
            print(line)

    _print("Overall", base_r.get("overall"), ft_r.get("overall") if ft_r else None)
    for lang in ["en", "ja"]:
        bl = base_r.get("metrics_by_language", {}).get(lang)
        fl = ft_r.get("metrics_by_language", {}).get(lang) if ft_r else None
        if bl or fl:
            _print(f"Language: {lang.upper()}", bl, fl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="results/eval_wordmetrics")
    args = parser.parse_args()

    test_data = load_test_data(args.max_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    # Evaluate base model (bf16 - A40 has 48GB, no need for quantization)
    print("\n--- Loading BASE model (bf16) ---")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    base_model.eval()
    base_results = evaluate_model_standalone(base_model, processor, test_data, f"Base ({args.base_model})")
    (output_dir / "base_results.json").write_text(json.dumps(base_results, indent=2, default=str))

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    ft_results = None
    if args.model_path:
        print(f"\n--- Loading FINE-TUNED model from {args.model_path} ---")
        ft_base = Qwen3VLForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
        ft_model = PeftModel.from_pretrained(ft_base, args.model_path)
        ft_model.eval()
        ft_results = evaluate_model_standalone(ft_model, processor, test_data, f"Fine-tuned ({args.model_path})")
        (output_dir / "finetuned_results.json").write_text(json.dumps(ft_results, indent=2, default=str))

    print_comparison(base_results, ft_results)

    comparison = {"base": base_results, "finetuned": ft_results}
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str))
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
