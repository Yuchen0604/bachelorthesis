#!/bin/bash
#SBATCH -J b_new
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
            --dataset $TEST_DATASET --run "trainedon_tum_p99_new_run1_$(basename $CHECKPOINT)"
    done
}

# TUM-p99-new Llama → tum_full, tum_p99
run_inference meta-llama/Llama-3.2-1B-Instruct  Llama-3.2-1B-Instruct-tum_p99_new-qlora_run1  tum_full
run_inference meta-llama/Llama-3.2-1B-Instruct  Llama-3.2-1B-Instruct-tum_p99_new-qlora_run1  tum_p99

# TUM-p99-new Qwen → tum_full, tum_p99
run_inference Qwen/Qwen2.5-1.5B-Instruct  Qwen2.5-1.5B-Instruct-tum_p99_new-qlora_run1  tum_full
run_inference Qwen/Qwen2.5-1.5B-Instruct  Qwen2.5-1.5B-Instruct-tum_p99_new-qlora_run1  tum_p99

# TUM-p99-new Gemma → tum_full, tum_p99
run_inference google/gemma-3-1b-it  gemma-3-1b-it-tum_p99_new-qlora_run1  tum_full
run_inference google/gemma-3-1b-it  gemma-3-1b-it-tum_p99_new-qlora_run1  tum_p99
