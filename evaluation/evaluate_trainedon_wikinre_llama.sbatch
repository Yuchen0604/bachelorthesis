#!/bin/bash
#SBATCH -J train_llama
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
ADAPTER_BASE=Llama-3.2-1B-Instruct-wikinre-qlora_run1

run_inference() {
    TEST_DATASET=$1
    for CHECKPOINT in $(ls -d ${ADAPTER_BASE}/checkpoint-* | sort -V); do
        python evaluation/run_inference.py \
            --model "$MODEL_ID" --adapter "$CHECKPOINT" \
            --dataset $TEST_DATASET --run "trainedon_wikinre_run1_$(basename $CHECKPOINT)"
    done
}

# WikiNRE-trained Llama, tested on the other four test sets
run_inference rebel
run_inference lagrange
run_inference tum_full
run_inference tum_p99
