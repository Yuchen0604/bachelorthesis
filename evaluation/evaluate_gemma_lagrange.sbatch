#!/bin/bash
#SBATCH -J evaluate
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
RUN=run2
DATASET=lagrange


source "$VENV_PATH"

# Gemma base
python evaluation/run_inference.py --model google/gemma-3-1b-it --dataset $DATASET --run $RUN

# Qwen base
python evaluation/run_inference.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset $DATASET --run $RUN