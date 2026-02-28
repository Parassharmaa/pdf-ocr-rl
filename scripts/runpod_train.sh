#!/bin/bash
# Full training pipeline for RunPod GPU pod
# Run this after SSH-ing into the pod
set -e

echo "=============================================="
echo "PDF-OCR-RL: Full Training Pipeline"
echo "=============================================="

cd /workspace

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Install system dependencies
echo ""
echo "=== Installing system dependencies ==="
apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 \
    libffi-dev libcairo2 fonts-noto-cjk fonts-noto \
    poppler-utils git 2>/dev/null
echo "System deps installed."

# 3. Setup project
echo ""
echo "=== Setting up project ==="
cd /workspace/pdf-ocr-rl
uv sync --extra train --extra data
echo "Dependencies installed."

# 4. Create dataset
echo ""
echo "=== Creating dataset ==="
uv run python scripts/create_dataset.py --max-per-repo 10
echo "Dataset created."

# 5. Run GRPO training
echo ""
echo "=== Starting GRPO Training ==="
echo "This will take approximately 4-8 hours on RTX 3090"
uv run python scripts/train_grpo.py --config configs/grpo_train.yaml
echo "Training complete!"

# 6. Run evaluation
echo ""
echo "=== Running Evaluation ==="
uv run python scripts/evaluate.py \
    --model-path results/grpo_run1/final_model \
    --data-dir data/processed \
    --output-dir results/evaluation
echo "Evaluation complete!"

echo ""
echo "=============================================="
echo "ALL DONE! Results are in results/evaluation/"
echo "=============================================="
echo ""
echo "IMPORTANT: Remember to stop/terminate the pod to save budget!"
echo "  python scripts/runpod_launch.py --action stop --pod-id YOUR_POD_ID"
