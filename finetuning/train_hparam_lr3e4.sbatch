#!/bin/bash
#SBATCH -J 3e4
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

python finetuning/finetune.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset tum_p99 --run hparam_lr3e4 --lr 3e-4
