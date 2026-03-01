"""SFT + GRPO training without unsloth dependency.

Usage:
    python scripts/train_v3.py --config configs/grpo_qwen3vl2b_v3.yaml
"""

import argparse
import gc
import json
import re
import random
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

from pdf_ocr_rl.data.dataset import format_prompt, strip_thinking, load_hf_dataset
from pdf_ocr_rl.reward.composite import composite_reward


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_grpo_dataset(config):
    """Build GRPO dataset from HuggingFace Hub."""
    languages = config["data"].get("languages", ["en", "ja"])
    max_samples = config["data"].get("max_train_samples", 500)

    hf_ds = load_hf_dataset(split="train")
    records = []
    for row in hf_ds:
        lang = row.get("language", "en")
        if lang not in languages:
            continue
        img = row["image"].convert("RGB") if hasattr(row["image"], "convert") else row["image"]
        prompt_text = format_prompt(lang)
        prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
        records.append({
            "prompt": prompt,
            "image": img,
            "answer": row["markdown"],
            "language": lang,
        })
        if len(records) >= max_samples:
            break

    random.seed(42)
    random.shuffle(records)
    ds = Dataset.from_list(records)
    print(f"GRPO dataset: {len(ds)} samples")
    return ds


def create_reward_fn(config):
    """Create composite reward function for GRPO."""
    weights = config.get("reward", {}).get("weights", None)

    def reward_fn(completions, answer, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            text = completion
            if isinstance(completion, list):
                # trl 0.24+ may pass list of message dicts
                text = " ".join(
                    item.get("content", item.get("text", str(item)))
                    for item in completion if isinstance(item, dict)
                ) if completion else ""
            elif isinstance(completion, dict):
                text = completion.get("content", completion.get("text", str(completion)))
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            text = strip_thinking(text)
            ref = answer[i] if isinstance(answer, list) else answer
            r = composite_reward(text, ref, weights=weights)
            rewards.append(r)
        return rewards

    return reward_fn


def sft_train(config, processor, base_model_name):
    """Run SFT phase using manual training loop with processor."""
    from torch.utils.data import DataLoader

    sft_config = config.get("sft", {})
    max_steps = sft_config.get("max_steps", 100)
    lr = sft_config.get("learning_rate", 2e-5)
    output_dir = config["output"]["dir"].replace("qwen3vl2b", "sft_qwen3vl2b")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading base model for SFT...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_name, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Build dataset
    languages = config["data"].get("languages", ["en", "ja"])
    max_samples = config["data"].get("max_train_samples", 500)
    hf_ds = load_hf_dataset(split="train")

    samples = []
    for row in hf_ds:
        lang = row.get("language", "en")
        if lang not in languages:
            continue
        img = row["image"].convert("RGB") if hasattr(row["image"], "convert") else row["image"]
        samples.append({"image": img, "markdown": row["markdown"], "language": lang})
        if len(samples) >= max_samples:
            break

    random.seed(42)
    random.shuffle(samples)
    print(f"SFT dataset: {len(samples)} samples")

    # Manual training loop
    model.train()
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Simple cosine schedule
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)

    grad_accum = 4
    step = 0
    total_loss = 0
    optimizer.zero_grad()

    print(f"\nStarting SFT ({max_steps} steps, grad_accum={grad_accum})...")
    start = time.time()

    from qwen_vl_utils import process_vision_info

    while step < max_steps:
        for sample in samples:
            if step >= max_steps:
                break

            img = sample["image"]
            lang = sample["language"]
            prompt_text = format_prompt(lang)

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": sample["markdown"]},
                ]},
            ]

            text = processor.apply_chat_template(messages, tokenize=False)
            image_inputs, _ = process_vision_info(messages)

            inputs = processor(
                text=[text], images=image_inputs,
                return_tensors="pt", padding=True,
                
                
            ).to(model.device)

            # Create labels (mask user tokens, only train on assistant response)
            labels = inputs["input_ids"].clone()
            # Simple approach: use the full sequence as labels
            # The model will learn to generate the full conversation
            inputs["labels"] = labels

            outputs = model(**inputs)
            loss = outputs.loss / grad_accum
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % grad_accum == 0 or step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            if step % 5 == 0:
                avg_loss = total_loss / 5
                elapsed = time.time() - start
                print(f"  Step {step}/{max_steps} | loss={avg_loss:.4f} | {elapsed:.0f}s | lr={scheduler.get_last_lr()[0]:.2e}")
                total_loss = 0

    elapsed = time.time() - start
    print(f"SFT complete in {elapsed:.0f}s")

    save_path = str(Path(output_dir) / "final_model")
    model.save_pretrained(save_path)
    processor.tokenizer.save_pretrained(save_path)
    print(f"SFT model saved to {save_path}")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_qwen3vl2b_v3.yaml")
    parser.add_argument("--skip-sft", action="store_true")
    parser.add_argument("--sft-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    base_model = config["model"]["name"]
    output_dir = config["output"]["dir"]

    print("=" * 60)
    print("Training Pipeline: SFT -> GRPO")
    print("=" * 60)
    print(f"Base model: {base_model}")
    print(f"Output: {output_dir}")
    print(f"Reward weights: {config.get('reward', {}).get('weights', {})}")
    print()

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    # ======== PHASE 1: SFT ========
    sft_dir = output_dir.replace("qwen3vl2b", "sft_qwen3vl2b")
    sft_model_path = str(Path(sft_dir) / "final_model")

    if not args.skip_sft:
        print("\n" + "=" * 40)
        print("PHASE 1: SFT Warm-up")
        print("=" * 40)
        sft_model_path = sft_train(config, processor, base_model)

    if args.sft_only:
        print("\n--sft-only flag set, skipping GRPO.")
        return

    # ======== PHASE 2: GRPO ========
    print("\n" + "=" * 40)
    print("PHASE 2: GRPO Refinement")
    print("=" * 40)

    grpo_base = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    if Path(sft_model_path).exists():
        print(f"Loading SFT adapter from {sft_model_path}...")
        model = PeftModel.from_pretrained(grpo_base, sft_model_path, is_trainable=True)
    else:
        print("No SFT checkpoint found, applying fresh LoRA...")
        lora_cfg = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules="all-linear",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(grpo_base, lora_cfg)

    grpo_dataset = build_grpo_dataset(config)
    reward_fn = create_reward_fn(config)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    grpo_cfg = GRPOConfig(
        output_dir=output_dir,
        num_generations=config["grpo"]["num_generations"],
        max_completion_length=config["grpo"]["max_completion_length"],
        max_prompt_length=4096,
        learning_rate=config["grpo"]["learning_rate"],
        num_train_epochs=config["grpo"]["num_train_epochs"],
        per_device_train_batch_size=config["grpo"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["grpo"]["gradient_accumulation_steps"],
        warmup_ratio=config["grpo"]["warmup_ratio"],
        weight_decay=config["grpo"]["weight_decay"],
        max_grad_norm=config["grpo"]["max_grad_norm"],
        bf16=True,
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
        args=grpo_cfg,
        train_dataset=grpo_dataset,
        processing_class=processor,
    )

    print(f"\nStarting GRPO ({config['grpo']['max_steps']} steps)...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"GRPO complete in {elapsed:.0f}s")

    save_path = str(Path(output_dir) / "final_model")
    model.save_pretrained(save_path)
    processor.tokenizer.save_pretrained(save_path)
    print(f"Final model saved to {save_path}")

    results = {
        "config": args.config,
        "base_model": base_model,
        "sft_steps": config.get("sft", {}).get("max_steps", 100),
        "grpo_steps": config["grpo"]["max_steps"],
        "reward_weights": config.get("reward", {}).get("weights", {}),
    }
    Path(output_dir, "training_results.json").write_text(json.dumps(results, indent=2))
    print("Done!")


if __name__ == "__main__":
    main()
