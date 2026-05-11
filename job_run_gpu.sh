#!/bin/bash

#SBATCH --mail-user=mohammed.ali@students.unibe.ch
#SBATCH --mail-user=mario.gonzalezdelcamino@students.unibe.ch
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --job-name=dl_project
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1

mkdir -p logs

CONTAINER="docker://pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime"

# Install Python dependencies once into the user environment
singularity exec --nv "$CONTAINER" \
    pip install --user -q -r requirements.txt

# Add project root to PYTHONPATH so `from src.xxx import ...` works
export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "=== Step 1: Preprocessing ==="
singularity exec --nv "$CONTAINER" \
    python -m src.preprocess

echo "=== Step 2: Training ==="
singularity exec --nv "$CONTAINER" \
    python -m src.train

echo "=== Step 3: Explainability ==="
singularity exec --nv "$CONTAINER" \
    python -m src.explainability

echo "=== Step 4: Adaptive Decision ==="
singularity exec --nv "$CONTAINER" \
    python -m src.adaptive_decision

echo "=== DONE ==="
