#!/bin/bash
#SBATCH -J gemma_tum_full_resume
#SBATCH -w gpu-nvidia-h200-3
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

# Gemma tum_full trained → tum_full test (missing checkpoint-7992 only)
python evaluation/run_inference.py \
    --model google/gemma-3-1b-it \
    --adapter gemma-3-1b-it-tum_full-qlora_run1/checkpoint-7992 \
    --dataset tum_full --run "trainedon_tum_full_run1_checkpoint-7992"
