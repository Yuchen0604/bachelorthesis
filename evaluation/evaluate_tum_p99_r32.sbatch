#!/bin/bash
#SBATCH -J eval_r32
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

run_inference() {
    MODEL_ID=$1
    ADAPTER_BASE=$2
    TEST_DATASET=$3
    for CHECKPOINT in $(ls -d ${ADAPTER_BASE}/checkpoint-* | sort -V); do
        python evaluation/run_inference.py \
            --model "$MODEL_ID" --adapter "$CHECKPOINT" \
            --dataset $TEST_DATASET --run "trainedon_tum_p99_hparam_r32_$(basename $CHECKPOINT)"
    done
}

# r32 Gemma → tum_p99 
run_inference google/gemma-3-1b-it  gemma-3-1b-it-tum_p99-qlora_hparam_r32  tum_p99
