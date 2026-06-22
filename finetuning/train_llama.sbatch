#!/bin/bash
#SBATCH -J tekgen
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

RUN=run1
DATASET=tekgen

python finetuning/finetune.py --model meta-llama/Llama-3.2-1B-Instruct --dataset $DATASET --run $RUN
