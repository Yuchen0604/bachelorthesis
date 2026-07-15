#!/bin/bash
#SBATCH -J tum_p99_r32
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

RUN=hparam_r32
DATASET=tum_p99

# LLaMA
python finetuning/finetune.py --model meta-llama/Llama-3.2-1B-Instruct --dataset $DATASET --run $RUN --lora_r 32

# Qwen
python finetuning/finetune.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset $DATASET --run $RUN --lora_r 32

# Gemma
python finetuning/finetune.py --model google/gemma-3-1b-it --dataset $DATASET --run $RUN --lora_r 32
