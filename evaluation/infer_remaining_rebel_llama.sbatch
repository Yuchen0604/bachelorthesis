#!/bin/bash
#SBATCH -J rest_ll
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
MODEL_NAME=$(basename "$MODEL_ID")

ADAPTER_BASE_REBEL3=Llama-3.2-1B-Instruct-rebel-qlora_run3
ADAPTER_BASE_REBELFULL=Llama-3.2-1B-Instruct-rebel-full-qlora_run1

# Skips a checkpoint if its prediction file already exists, so this is safe to resubmit.
run_inference() {
    ADAPTER_BASE=$1
    RUN_LABEL=$2
    TEST_DATASET=$3
    for CHECKPOINT in $(ls -d ${ADAPTER_BASE}/checkpoint-* | sort -V); do
        RUN_TAG="${RUN_LABEL}_$(basename $CHECKPOINT)"
        OUT_FILE="predictions/${MODEL_NAME}_${TEST_DATASET}_finetuned_${RUN_TAG}.jsonl"
        if [ -f "$OUT_FILE" ]; then
            echo "Skipping $OUT_FILE (already exists)"
            continue
        fi
        python evaluation/run_inference.py \
            --model "$MODEL_ID" --adapter "$CHECKPOINT" \
            --dataset $TEST_DATASET --run "$RUN_TAG"
    done
}


run_inference "$ADAPTER_BASE_REBEL3" "trainedon_rebel_run3" wikinre
run_inference "$ADAPTER_BASE_REBEL3" "trainedon_rebel_run3" tum_p99


run_inference "$ADAPTER_BASE_REBELFULL" "trainedon_rebel-full_run1" wikinre
