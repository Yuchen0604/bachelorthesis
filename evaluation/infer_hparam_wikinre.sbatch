#!/bin/bash
#SBATCH -J hp_wiki
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

TEST_DATASET=wikinre

# Skips a checkpoint if its prediction file already exists, so this is safe to resubmit.
run_inference() {
    MODEL_ID=$1
    ADAPTER_BASE=$2
    RUN_LABEL=$3
    MODEL_NAME=$(basename "$MODEL_ID")
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

# --- tum_p99_hparam_r32 (all three architectures) ---
run_inference meta-llama/Llama-3.2-1B-Instruct Llama-3.2-1B-Instruct-tum_p99-qlora_hparam_r32 trainedon_tum_p99_hparam_r32
run_inference Qwen/Qwen2.5-1.5B-Instruct       Qwen2.5-1.5B-Instruct-tum_p99-qlora_hparam_r32  trainedon_tum_p99_hparam_r32
run_inference google/gemma-3-1b-it             gemma-3-1b-it-tum_p99-qlora_hparam_r32          trainedon_tum_p99_hparam_r32

# --- tum_p99_different_hparam (Qwen-only lr sweep) ---
run_inference Qwen/Qwen2.5-1.5B-Instruct Qwen2.5-1.5B-Instruct-tum_p99-qlora_hparam_lr1e4 trainedon_tum_p99_hparam_lr1e4
run_inference Qwen/Qwen2.5-1.5B-Instruct Qwen2.5-1.5B-Instruct-tum_p99-qlora_hparam_lr3e4 trainedon_tum_p99_hparam_lr3e4
