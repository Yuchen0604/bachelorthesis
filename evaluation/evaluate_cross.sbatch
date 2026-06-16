#!/bin/bash
#SBATCH -J cross_eval
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

RUN=run2

run_inference() {
    MODEL_ID=$1
    ADAPTER_BASE=$2
    TEST_DATASET=$3
    TRAIN_DATASET=$4
    for CHECKPOINT in $(ls -d ${ADAPTER_BASE}/checkpoint-* | sort -V); do
        python evaluation/run_inference.py \
            --model "$MODEL_ID" --adapter "$CHECKPOINT" \
            --dataset $TEST_DATASET --run "trainedon_${TRAIN_DATASET}_${RUN}_$(basename $CHECKPOINT)"
    done
}

# Rebel models → Lagrange test set
run_inference meta-llama/Llama-3.2-1B-Instruct  Llama-3.2-1B-Instruct-rebel-qlora_${RUN}    lagrange rebel
run_inference Qwen/Qwen2.5-1.5B-Instruct         Qwen2.5-1.5B-Instruct-rebel-qlora_${RUN}    lagrange rebel
run_inference google/gemma-3-1b-it               gemma-3-1b-it-rebel-qlora_${RUN}            lagrange rebel

# Lagrange models → Rebel test set
run_inference meta-llama/Llama-3.2-1B-Instruct  Llama-3.2-1B-Instruct-lagrange-qlora_${RUN} rebel lagrange
run_inference Qwen/Qwen2.5-1.5B-Instruct         Qwen2.5-1.5B-Instruct-lagrange-qlora_${RUN} rebel lagrange
run_inference google/gemma-3-1b-it               gemma-3-1b-it-lagrange-qlora_${RUN}         rebel lagrange
