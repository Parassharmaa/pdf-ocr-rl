#!/bin/bash
# Full training pipeline for RunPod GPU pod
# Usage: Clone the repo, then run this script
# bash scripts/runpod_train.sh
set -e

echo "=============================================="
echo "PDF-OCR-RL: Full Training Pipeline"
echo "=============================================="
echo "Start time: $(date)"

cd /workspace

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo ""
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

# 2. Install system dependencies for PDF rendering
echo ""
echo "=== Installing system dependencies ==="
apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 \
    libffi-dev libcairo2 fonts-noto-cjk fonts-noto \
    poppler-utils 2>/dev/null
echo "System deps installed."

# 3. Clone project if not present
if [ ! -d "/workspace/pdf-ocr-rl" ]; then
    echo ""
    echo "=== Cloning project ==="
    cd /workspace
    git clone https://github.com/Parassharmaa/pdf-ocr-rl.git
fi

# 4. Setup project
echo ""
echo "=== Setting up project ==="
cd /workspace/pdf-ocr-rl
uv sync --extra train --extra data
echo "Dependencies installed."

# 5. Create dataset
echo ""
echo "=== Creating dataset ==="
if [ ! -f "data/processed/dataset_meta.json" ]; then
    uv run python scripts/create_dataset.py --max-per-repo 10
    echo "Dataset created."
else
    echo "Dataset already exists, skipping."
fi

# 6. Run GRPO training
echo ""
echo "=== Starting GRPO Training ==="
echo "Training start: $(date)"
uv run python scripts/train_grpo.py --config configs/grpo_train.yaml 2>&1 | tee results/training_log.txt
echo "Training end: $(date)"
echo "Training complete!"

# 7. Run evaluation
echo ""
echo "=== Running Evaluation ==="
uv run python scripts/evaluate.py \
    --model-path results/grpo_run1/final_model \
    --data-dir data/processed \
    --output-dir results/evaluation 2>&1 | tee results/evaluation_log.txt
echo "Evaluation complete!"

# 8. Push results back to git
echo ""
echo "=== Pushing results ==="
cd /workspace/pdf-ocr-rl
git add results/ -f
git commit -m "Add training and evaluation results from RunPod" || true
git push origin main || echo "Push failed - results saved locally in results/"

echo ""
echo "=============================================="
echo "ALL DONE! $(date)"
echo "Results are in results/evaluation/"
echo "=============================================="
echo ""
echo "IMPORTANT: Stop the pod to save budget!"
