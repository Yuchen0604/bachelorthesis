#!/bin/bash
#SBATCH -J train_eval
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
DATASET=lagrange

#Inference on all epoch checkpoints
run_inference() {
    MODEL_ID=$1
    ADAPTER_BASE=$2
    for CHECKPOINT in $(ls -d ${ADAPTER_BASE}/checkpoint-* | sort -V); do
        python evaluation/run_inference.py \
            --model "$MODEL_ID" --adapter "$CHECKPOINT" \
            --dataset $DATASET --run "${RUN}_$(basename $CHECKPOINT)"
    done
}

run_inference meta-llama/Llama-3.2-1B-Instruct  Llama-3.2-1B-Instruct-${DATASET}-qlora_${RUN}
run_inference Qwen/Qwen2.5-1.5B-Instruct         Qwen2.5-1.5B-Instruct-${DATASET}-qlora_${RUN}
run_inference google/gemma-3-1b-it               gemma-3-1b-it-${DATASET}-qlora_${RUN}
