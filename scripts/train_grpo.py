"""GRPO training script for PDF-to-markdown VLM fine-tuning.

Usage:
    python scripts/train_grpo.py --config configs/grpo_train.yaml

Designed to run on RunPod with a single RTX 3090 (24GB).
Uses Unsloth + TRL for memory-efficient GRPO training.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataset(config: dict):
    """Load the image-markdown dataset for GRPO training."""
    from datasets import Dataset, Features, Image, Value

    data_dir = Path(config["data"]["train_dir"])
    meta_path = data_dir / "dataset_meta.json"

    if not meta_path.exists():
        print(f"Error: No dataset found at {meta_path}")
        print("Run: python scripts/create_dataset.py first")
        sys.exit(1)

    meta = json.loads(meta_path.read_text())

    records = []
    languages = config["data"].get("languages", ["en", "ja"])
    max_samples = config["data"].get("max_train_samples", 500)

    for entry in meta:
        img_path = entry["image_path"]
        md_source = entry["source"]
        lang = entry.get("language", "en")

        if lang not in languages:
            continue
        if not Path(img_path).exists() or not Path(md_source).exists():
            continue

        md_content = Path(md_source).read_text(encoding="utf-8")

        # Format as conversation for Qwen2.5-VL
        prompt = _format_prompt(lang)

        records.append({
            "image": img_path,
            "prompt": prompt,
            "reference": md_content,
            "language": lang,
        })

        if len(records) >= max_samples:
            break

    features = Features({
        "image": Image(),
        "prompt": Value("string"),
        "reference": Value("string"),
        "language": Value("string"),
    })

    ds = Dataset.from_list(records, features=features)
    print(f"Loaded {len(ds)} training samples")
    return ds


def _format_prompt(language: str) -> str:
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


def create_reward_fn(config: dict):
    """Create the composite reward function for GRPO."""
    from pdf_ocr_rl.reward.composite import composite_reward

    weights = config.get("reward", {}).get("weights", None)

    def reward_fn(completions, reference, **kwargs):
        """Compute rewards for a batch of completions.

        Args:
            completions: List of generated markdown strings
            reference: Ground-truth markdown string

        Returns:
            List of float rewards
        """
        rewards = []
        for completion in completions:
            # Extract text from completion (strip any special tokens)
            text = completion
            if isinstance(completion, dict):
                text = completion.get("content", completion.get("text", str(completion)))

            r = composite_reward(text, reference, weights=weights)
            rewards.append(r)
        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="GRPO training for PDF-to-markdown")
    parser.add_argument("--config", type=str, default="configs/grpo_train.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print("=" * 60)
    print("GRPO Training: PDF-to-Markdown VLM Fine-tuning")
    print("=" * 60)
    print(f"Model: {config['model']['name']}")
    print(f"LoRA r={config['lora']['r']}, alpha={config['lora']['alpha']}")
    print(f"GRPO generations: {config['grpo']['num_generations']}")
    print(f"Max steps: {config['grpo']['max_steps']}")
    print()

    # Load model
    print("Loading model with Unsloth...")
    from pdf_ocr_rl.models.loader import load_model_for_training

    model, tokenizer = load_model_for_training(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        lora_r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        load_in_4bit=config["model"]["load_in_4bit"],
    )
    print("Model loaded successfully!")

    # Load dataset
    print("\nLoading dataset...")
    dataset = build_dataset(config)

    # Create reward function
    reward_fn = create_reward_fn(config)

    # Set up GRPO trainer
    print("\nSetting up GRPO trainer...")
    from trl import GRPOConfig, GRPOTrainer

    output_dir = config["output"]["dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_generations=config["grpo"]["num_generations"],
        max_completion_length=config["grpo"]["max_completion_length"],
        learning_rate=config["grpo"]["learning_rate"],
        num_train_epochs=config["grpo"]["num_train_epochs"],
        per_device_train_batch_size=config["grpo"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["grpo"]["gradient_accumulation_steps"],
        warmup_ratio=config["grpo"]["warmup_ratio"],
        weight_decay=config["grpo"]["weight_decay"],
        max_grad_norm=config["grpo"]["max_grad_norm"],
        bf16=config["grpo"]["bf16"],
        optim=config["grpo"]["optim"],
        logging_steps=config["grpo"]["logging_steps"],
        save_steps=config["grpo"]["save_steps"],
        max_steps=config["grpo"]["max_steps"],
        report_to="none",
    )

    # Format dataset for GRPO trainer
    def format_for_grpo(example):
        """Format a dataset example for the GRPO trainer."""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": example["prompt"]},
                ],
            }
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)

        return {
            "prompt": text,
            "image_inputs": image_inputs,
            "reference": example["reference"],
        }

    formatted_dataset = dataset.map(format_for_grpo, remove_columns=dataset.column_names)

    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=formatted_dataset,
        reward_funcs=reward_fn,
        tokenizer=tokenizer,
    )

    print("\nStarting GRPO training...")
    print(f"Output: {output_dir}")
    trainer.train()

    # Save model
    if config["output"].get("save_model", True):
        save_path = str(Path(output_dir) / "final_model")
        print(f"\nSaving model to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("Model saved!")

    # Save training results
    results = {
        "model": config["model"]["name"],
        "lora_r": config["lora"]["r"],
        "grpo_generations": config["grpo"]["num_generations"],
        "max_steps": config["grpo"]["max_steps"],
        "output_dir": output_dir,
    }
    results_path = Path(output_dir) / "training_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nTraining complete! Results saved to {results_path}")


if __name__ == "__main__":
    main()
