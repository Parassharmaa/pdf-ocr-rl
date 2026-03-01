"""SFT (Supervised Fine-Tuning) script for PDF-to-markdown VLM.

Usage:
    python scripts/train_sft.py --config configs/grpo_qwen3vl2b.yaml

SFT warm-up teaches the model the exact image→markdown mapping before
GRPO refinement. This is needed because GRPO alone produces near-zero
gradients when the base model already generates similar-quality outputs.
"""

import argparse
import json
import random
from pathlib import Path

import yaml
from PIL import Image

from pdf_ocr_rl.data.dataset import format_prompt


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_sft_dataset(config: dict):
    """Build dataset in Unsloth Vision SFT format.

    Format: list of conversation dicts with image content.
    {
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "markdown output"}]}
        ],
        "images": [PIL.Image]
    }
    """
    from datasets import Dataset

    data_dir = Path(config["data"]["train_dir"])
    meta_path = data_dir / "dataset_meta.json"
    languages = config["data"].get("languages", ["en", "ja"])
    max_samples = config["data"].get("max_train_samples", 500)

    use_hub = not meta_path.exists()

    if use_hub:
        print("Local dataset not found, loading from HuggingFace Hub...")
        from pdf_ocr_rl.data.dataset import load_hf_dataset
        hf_ds = load_hf_dataset(split="train")

        records = []
        for row in hf_ds:
            lang = row.get("language", "en")
            if lang not in languages:
                continue
            prompt_text = format_prompt(lang)
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": row["markdown"]}]},
            ]
            img = row["image"].convert("RGB") if hasattr(row["image"], "convert") else row["image"]
            records.append({"messages": messages, "images": [img]})
            if len(records) >= max_samples:
                break
    else:
        meta = json.loads(meta_path.read_text())
        random.seed(42)
        random.shuffle(meta)

        records = []
        for entry in meta:
            img_path = entry["image_path"]
            md_source = entry["source"]
            lang = entry.get("language", "en")

            if lang not in languages:
                continue
            if not Path(img_path).exists() or not Path(md_source).exists():
                continue

            md_full = Path(md_source).read_text(encoding="utf-8")
            start = entry.get("page_start_char", 0)
            end = entry.get("page_end_char", len(md_full))
            md_content = md_full[start:end]
            prompt_text = format_prompt(lang)
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": md_content}]},
            ]
            records.append({"messages": messages, "images": [Image.open(img_path).convert("RGB")]})

            if len(records) >= max_samples:
                break

    ds = Dataset.from_list(records)
    print(f"Loaded {len(ds)} SFT samples" + (" (from HuggingFace Hub)" if use_hub else " (from local)"))
    return ds


def main():
    parser = argparse.ArgumentParser(description="SFT training for PDF-to-markdown")
    parser.add_argument("--config", type=str, default="configs/grpo_qwen3vl2b.yaml")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps from config")
    args = parser.parse_args()

    config = load_config(args.config)
    print("=" * 60)
    print("SFT Training: PDF-to-Markdown VLM Fine-tuning")
    print("=" * 60)
    print(f"Model: {config['model']['name']}")
    print(f"LoRA r={config['lora']['r']}, alpha={config['lora']['alpha']}")
    print()

    # Load model with Unsloth
    print("Loading model with Unsloth...")
    from unsloth import FastVisionModel

    model, tokenizer = FastVisionModel.from_pretrained(
        config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=config["model"]["load_in_4bit"],
        dtype=None,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=0,
        bias="none",
        random_state=42,
        use_rslora=False,
    )
    print("Model loaded successfully!")

    # Load dataset
    print("\nLoading dataset...")
    dataset = build_sft_dataset(config)

    # Set up SFT trainer
    print("\nSetting up SFT trainer...")
    from trl import SFTConfig, SFTTrainer
    from unsloth import is_bf16_supported

    sft_config = config.get("sft", {})
    max_steps = args.max_steps or sft_config.get("max_steps", 100)
    output_dir = config["output"]["dir"].replace("grpo_", "sft_")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        max_steps=max_steps,
        learning_rate=2e-5,
        bf16=is_bf16_supported(),
        fp16=not is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=max_steps,  # Save only at end
        save_total_limit=1,
        report_to="none",
        gradient_checkpointing=True,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=config["model"]["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    print(f"\nStarting SFT training ({max_steps} steps)...")
    print(f"Output: {output_dir}")
    trainer.train()

    # Save model
    save_path = str(Path(output_dir) / "final_model")
    print(f"\nSaving SFT model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("SFT model saved!")

    results = {
        "model": config["model"]["name"],
        "lora_r": config["lora"]["r"],
        "sft_steps": max_steps,
        "output_dir": output_dir,
    }
    results_path = Path(output_dir) / "training_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSFT training complete! Results saved to {results_path}")


if __name__ == "__main__":
    main()
