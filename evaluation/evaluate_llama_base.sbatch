#!/bin/bash
#SBATCH -J llama_base
#SBATCH -w gpu-nvidia-h200-3
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

# Llama base, rebel test set
python evaluation/run_inference.py --model meta-llama/Llama-3.2-1B-Instruct --dataset rebel --run run1

# Llama base, lagrange test set
python evaluation/run_inference.py --model meta-llama/Llama-3.2-1B-Instruct --dataset lagrange --run run1
