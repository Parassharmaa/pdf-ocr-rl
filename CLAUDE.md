# PDF-OCR-RL — RL Fine-Tuning for PDF-to-Markdown Conversion

GRPO-based reinforcement learning fine-tuning of a small multimodal VLM (Qwen2.5-VL-3B)
for accurate PDF-to-markdown conversion across multiple languages (EN, JA).

## Tech Stack
- Python 3.10+ managed with `uv`
- PyTorch 2.x, `transformers`, `unsloth`, `trl`
- Qwen2.5-VL-3B-Instruct as base model
- GRPO (Group Relative Policy Optimization) via TRL + Unsloth
- RunPod for GPU compute (budget: $10)

## Commands
- `uv sync` — install dependencies
- `uv run pytest` — run tests
- `uv run pytest tests/test_foo.py -v` — run specific test
- `uv run python scripts/<script>.py` — run a script

## Conventions
- Source code lives in `src/pdf_ocr_rl/`
- Use layered git commits — one logical change per commit
- Config-driven experiments: hyperparameters in `configs/`, not hardcoded
- Tests mirror source structure: `src/pdf_ocr_rl/models/foo.py` → `tests/test_foo.py`
- **Never** include `Co-Authored-By` or any Claude/AI email in git commits
- Run long experiments via **tmux** sessions (not blocking the terminal)

## Architecture
- **Base model:** Qwen2.5-VL-3B-Instruct (4-bit quantized via Unsloth)
- **Fine-tuning:** QLoRA (r=16, alpha=32) + GRPO
- **Reward function:** Composite (edit distance + structural validity + reading order)
- **Dataset:** Synthetic PDF-markdown pairs from open sources (EN + JA)

## Training Pipeline
1. **Dataset creation:** Collect markdown → render to PDF → create image-markdown pairs
2. **SFT warm-up:** Supervised fine-tuning on high-quality pairs
3. **GRPO training:** RL with composite reward function
4. **Evaluation:** Compare base vs fine-tuned on EN and JA test sets

## Project Structure
```
src/pdf_ocr_rl/
├── models/       # Model loading, LoRA config, training setup
├── data/         # Dataset creation, markdown collection, PDF rendering
├── eval/         # Evaluation metrics, comparison pipeline
├── reward/       # GRPO reward functions (edit distance, structural, etc.)
configs/          # YAML experiment configs
scripts/          # Training, evaluation, dataset creation entry points
tests/            # pytest tests mirroring src structure
results/          # Training results and evaluation outputs
```

## Key References
- [olmOCR 2](https://arxiv.org/abs/2510.19817) — Unit test rewards for GRPO
- [Infinity-Parser](https://arxiv.org/abs/2506.03197) — Multi-aspect reward
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO algorithm
- [Unsloth Vision RL](https://unsloth.ai/blog/vision-rl) — Training framework
