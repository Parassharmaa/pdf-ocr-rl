# PDF-OCR-RL

Fine-tuning vision-language models for PDF-to-markdown conversion using **SFT + GRPO** (Group Relative Policy Optimization).

## Models

| Model | HuggingFace | Description |
|-------|-------------|-------------|
| **SFT + GRPO (best)** | [blazeofchi/pdf-ocr-rl-qwen3vl2b-sft-grpo](https://huggingface.co/blazeofchi/pdf-ocr-rl-qwen3vl2b-sft-grpo) | Two-stage: SFT warm-up → GRPO refinement |
| SFT only | [blazeofchi/pdf-ocr-rl-qwen3vl2b-sft-only](https://huggingface.co/blazeofchi/pdf-ocr-rl-qwen3vl2b-sft-only) | Intermediate SFT checkpoint |

**Dataset**: [blazeofchi/pdf-ocr-rl-dataset](https://huggingface.co/datasets/blazeofchi/pdf-ocr-rl-dataset) (500 train / 20 test)

## Results

Best model (SFT+GRPO v2) evaluated on 20 held-out test samples, bf16 on NVIDIA A40:

| Metric | Base Model | Fine-tuned | Delta |
|--------|-----------|------------|-------|
| Heading Precision | 0.855 | **0.930** | **+7.5%** |
| Heading F1 | 0.840 | **0.894** | **+5.4%** |
| Code Block Similarity | 0.578 | **0.757** | **+18.0%** |
| Code Block Count Match | 0.333 | **0.515** | **+18.2%** |
| Word Precision | 0.756 | **0.790** | **+3.5%** |
| Word F1 | 0.715 | **0.731** | **+1.6%** |
| Edit Distance | 0.753 | 0.735 | -1.8% |

## Training Pipeline

```
Qwen3-VL-2B-Instruct
    │
    ▼
┌─────────────────────┐
│  Stage 1: SFT       │  100 steps, LR=2e-5
│  (image → markdown)  │  Loss: 1.295 → 0.78
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 2: GRPO      │  100 steps, LR=5e-6
│  (reward optimization)│  Reward: 0.66 → 0.74
└─────────┬───────────┘
          │
          ▼
    Fine-tuned Model
```

**Reward function** (composite):
- `edit_distance` (0.4) — character-level Levenshtein similarity
- `reading_order` (0.25) — correct ordering of content blocks
- `heading` (0.2) — heading detection precision/recall
- `structural` (0.15) — markdown structure validity

## Experiment Configurations

We ran 4 configurations to find the best approach:

| # | Configuration | edit_dist | heading_f1 | code_sim | word_f1 | Verdict |
|---|--------------|-----------|-----------|----------|---------|---------|
| 1 | GRPO only (no SFT) | 0.753 | 0.840 | 0.578 | 0.715 | No improvement — near-zero gradients |
| 2 | **SFT + GRPO v2 (Unsloth)** | 0.735 | **0.894** | **0.757** | **0.731** | **Best model** |
| 3 | SFT + GRPO v3 (manual loop) | 0.749 | 0.785 | 0.568 | 0.716 | Worse — manual loop less effective |
| 4 | Extended SFT 200 steps | 0.761 | 0.807 | 0.551 | 0.717 | Marginal — more SFT doesn't help structure |

## Key Learnings

### 1. GRPO needs SFT warm-up

GRPO alone produces near-zero gradients (grad_norm ≈ 4.7e-6) on vision-language models. The base model generates outputs too similar to each other (reward_std = 0.017), so GRPO can't differentiate good from bad completions. SFT first diversifies the outputs, producing **400,000x stronger gradients** for GRPO to work with.

### 2. Unsloth's vision data collator is critical

Manual training loops that train on the full conversation (including user/system tokens) are significantly less effective. Unsloth's `UnslothVisionDataCollator` handles completion-only masking correctly, producing models that are structurally much better. Our manual v3 training showed **-5.5% heading F1 regression** vs. the Unsloth model's **+5.4% improvement**.

### 3. Character-level edit distance is too strict

Levenshtein ratio penalizes any reformatting even when the content is semantically correct. A model that adds proper headings or code blocks will score *lower* on edit distance despite being *better*. **Word-level F1** is more meaningful for document conversion evaluation.

### 4. Vision models need special GRPO config

- `max_prompt_length` must be ≥4096 (default 1024 truncates image tokens)
- `PeftModel.from_pretrained()` needs `is_trainable=True` for continued training
- `gradient_checkpointing_enable({"use_reentrant": False})` is required — default breaks LoRA gradient flow
- `GRPOTrainer` needs `processing_class=processor` (not `tokenizer`) for vision models

### 5. Unsloth environment is fragile

Never `pip install -e .` if your pyproject.toml might upgrade torch — it cascades and breaks unsloth/triton compatibility. Install project deps separately, then unsloth. Use standalone eval scripts with transformers+peft when the unsloth env breaks.

### 6. Training cost is very low

The entire experiment (4 training runs + evaluation) cost **~$4 total** on RunPod A40 ($0.79/hr). Individual runs: ~$1-2 each, under 2 hours.

## Project Structure

```
pdf-ocr-rl/
├── src/pdf_ocr_rl/
│   ├── data/           # Dataset creation, PDF rendering
│   └── eval/           # Evaluation metrics (edit distance, heading, table, code, word F1)
├── scripts/
│   ├── create_dataset.py    # Render markdown → PDF → images
│   ├── train_grpo.py        # SFT + GRPO training (Unsloth)
│   ├── train_v3.py          # SFT + GRPO training (manual, no Unsloth)
│   ├── eval_standalone.py   # Evaluation without Unsloth dependency
│   ├── evaluate.py          # Original evaluation script
│   ├── runpod_launch.py     # RunPod GPU provisioning
│   └── upload_to_hf.py      # Upload dataset to HuggingFace
├── configs/
│   ├── grpo_qwen3vl2b.yaml      # v2 config (best)
│   └── grpo_qwen3vl2b_v3.yaml   # v3 config
└── results/
    ├── RESULTS.md          # Detailed results documentation
    └── eval_final/         # Raw evaluation JSONs
```

## Quick Start

```bash
# Install dependencies
uv sync

# Create dataset (renders markdown → PDF → images)
uv run python scripts/create_dataset.py

# Train on RunPod (provisions GPU, runs training)
uv run python scripts/runpod_launch.py

# Evaluate locally
uv run python scripts/eval_standalone.py
```

## Usage

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch

base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "unsloth/Qwen3-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    base_model, "blazeofchi/pdf-ocr-rl-qwen3vl2b-sft-grpo"
)
processor = AutoProcessor.from_pretrained(
    "blazeofchi/pdf-ocr-rl-qwen3vl2b-sft-grpo"
)

image = Image.open("page.png")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "Convert this PDF page to well-structured markdown."}
]}]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
print(processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## License

Apache 2.0
