#!/bin/bash
#SBATCH -J gemma_tum_full_p99
#SBATCH -w gpu-nvidia-h200-3
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

# Gemma tum_full trained → tum_p99 test (all 8 checkpoints)
for CHECKPOINT in $(ls -d gemma-3-1b-it-tum_full-qlora_run1/checkpoint-* | sort -V); do
    python evaluation/run_inference.py \
        --model google/gemma-3-1b-it --adapter "$CHECKPOINT" \
        --dataset tum_p99 --run "trainedon_tum_full_run1_$(basename $CHECKPOINT)"
done
