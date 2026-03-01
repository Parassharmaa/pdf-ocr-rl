"""GRPO training script for PDF-to-markdown VLM fine-tuning.

Usage:
    python scripts/train_grpo.py --config configs/grpo_train.yaml

Designed to run on RunPod with a single RTX 3090 (24GB).
Uses Unsloth + TRL for memory-efficient GRPO training.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import yaml
from PIL import Image

from pdf_ocr_rl.data.dataset import format_prompt, strip_thinking


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataset(config: dict):
    """Load the image-markdown dataset in Unsloth Vision GRPO format.

    Data sources (in priority order):
    1. Local data_dir with dataset_meta.json (from scripts/create_dataset.py)
    2. HuggingFace Hub: blazeofchi/pdf-ocr-rl-dataset

    Format expected by TRL GRPOTrainer:
    {
        "prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}],
        "image": [PIL.Image],
        "answer": "ground truth markdown"
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
            prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
            img = row["image"].convert("RGB") if hasattr(row["image"], "convert") else row["image"]
            records.append({"prompt": prompt, "image": img, "answer": row["markdown"], "language": lang})
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
            prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
            records.append({"prompt": prompt, "image": Image.open(img_path).convert("RGB"), "answer": md_content, "language": lang})

            if len(records) >= max_samples:
                break

    ds = Dataset.from_list(records)
    print(f"Loaded {len(ds)} training samples" + (" (from HuggingFace Hub)" if use_hub else " (from local)"))
    return ds


def create_reward_fn(config: dict):
    """Create the composite reward function for GRPO.

    The reward function signature for TRL GRPOTrainer:
        reward_fn(completions, **kwargs) -> list[float]

    where `completions` is a list of generated strings, and
    kwargs contains additional fields from the dataset (like `answer`).
    """
    from pdf_ocr_rl.reward.composite import composite_reward

    weights = config.get("reward", {}).get("weights", None)

    def reward_fn(completions, answer, **kwargs):
        """Compute rewards for GRPO completions.

        Args:
            completions: List of generated markdown strings
            answer: List of ground-truth markdown strings (from dataset)

        Returns:
            List of float rewards
        """
        rewards = []
        for i, completion in enumerate(completions):
            text = completion
            if isinstance(completion, dict):
                text = completion.get("content", completion.get("text", str(completion)))

            # Strip <think>...</think> blocks (Qwen3 thinking mode)
            text = strip_thinking(text)

            # Get the reference for this sample
            ref = answer[i] if isinstance(answer, list) else answer
            r = composite_reward(text, ref, weights=weights)
            rewards.append(r)
        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="GRPO training for PDF-to-markdown")
    parser.add_argument("--config", type=str, default="configs/grpo_train.yaml")
    parser.add_argument("--from-sft", type=str, default=None, help="Path to SFT checkpoint to continue from")
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

    # Load model with Unsloth
    from unsloth import FastVisionModel

    if args.from_sft:
        print(f"Loading SFT checkpoint from {args.from_sft}...")
        model, tokenizer = FastVisionModel.from_pretrained(
            args.from_sft,
            max_seq_length=config["model"]["max_seq_length"],
            load_in_4bit=config["model"]["load_in_4bit"],
            dtype=None,
        )
    else:
        print("Loading base model with Unsloth...")
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
        max_prompt_length=1024,
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
        save_total_limit=config["grpo"].get("save_total_limit", 2),
        max_steps=config["grpo"]["max_steps"],
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
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
