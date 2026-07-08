#!/bin/bash
#SBATCH -J 2_1e4_inf
#SBATCH -w gpu-nvidia-h200-3
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1


ADAPTER=Qwen2.5-1.5B-Instruct-tum_p99-qlora_hparam_lr1e4

for CHECKPOINT in $(ls -d ${ADAPTER}/checkpoint-* | sort -V); do
    for DATASET in rebel lagrange tum_full tum_p99; do
        python evaluation/run_inference.py \
            --model Qwen/Qwen2.5-1.5B-Instruct \
            --adapter "$CHECKPOINT" \
            --dataset $DATASET \
            --run "trainedon_tum_p99_hparam_lr1e4_$(basename $CHECKPOINT)"
    done
done
