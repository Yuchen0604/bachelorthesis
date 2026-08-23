#!/bin/bash
#SBATCH -J newr
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

TEST_DATASET=rebel


run_single_checkpoint() {
    MODEL_ID=$1
    ADAPTER_BASE=$2
    RUN_LABEL=$3
    STEP=$4
    MODEL_NAME=$(basename "$MODEL_ID")
    CHECKPOINT="${ADAPTER_BASE}/checkpoint-${STEP}"
    RUN_TAG="${RUN_LABEL}_newrebeltest_checkpoint-${STEP}"
    OUT_FILE="predictions/${MODEL_NAME}_${TEST_DATASET}_finetuned_${RUN_TAG}.jsonl"
    if [ -f "$OUT_FILE" ]; then
        echo "Skipping $OUT_FILE (already exists)"
        return
    fi
    python evaluation/run_inference.py \
        --model "$MODEL_ID" --adapter "$CHECKPOINT" \
        --dataset $TEST_DATASET --run "$RUN_TAG"
}

LLAMA_ID=meta-llama/Llama-3.2-1B-Instruct
QWEN_ID=Qwen/Qwen2.5-1.5B-Instruct
GEMMA_ID=google/gemma-3-1b-it

# --- lagrange (run2) ---
run_single_checkpoint "$GEMMA_ID" gemma-3-1b-it-lagrange-qlora_run2               trainedon_lagrange_run2 6000
run_single_checkpoint "$QWEN_ID"  Qwen2.5-1.5B-Instruct-lagrange-qlora_run2       trainedon_lagrange_run2 6000
run_single_checkpoint "$LLAMA_ID" Llama-3.2-1B-Instruct-lagrange-qlora_run2       trainedon_lagrange_run2 7000

# --- tekgen (run1) ---
run_single_checkpoint "$GEMMA_ID" gemma-3-1b-it-tekgen-qlora_run1                 trainedon_tekgen_run1   3000
run_single_checkpoint "$QWEN_ID"  Qwen2.5-1.5B-Instruct-tekgen-qlora_run1         trainedon_tekgen_run1   5000
run_single_checkpoint "$LLAMA_ID" Llama-3.2-1B-Instruct-tekgen-qlora_run1         trainedon_tekgen_run1   3000

# --- tum_p99 (run1) ---
run_single_checkpoint "$GEMMA_ID" gemma-3-1b-it-tum_p99-qlora_run1                trainedon_tum_p99_run1  5994
run_single_checkpoint "$QWEN_ID"  Qwen2.5-1.5B-Instruct-tum_p99-qlora_run1        trainedon_tum_p99_run1  3996
run_single_checkpoint "$LLAMA_ID" Llama-3.2-1B-Instruct-tum_p99-qlora_run1        trainedon_tum_p99_run1  5994

# --- tum_p99_new (run1) ---
run_single_checkpoint "$GEMMA_ID" gemma-3-1b-it-tum_p99_new-qlora_run1            trainedon_tum_p99_new_run1 52296
run_single_checkpoint "$QWEN_ID"  Qwen2.5-1.5B-Instruct-tum_p99_new-qlora_run1    trainedon_tum_p99_new_run1 61012
run_single_checkpoint "$LLAMA_ID" Llama-3.2-1B-Instruct-tum_p99_new-qlora_run1    trainedon_tum_p99_new_run1 52296

# --- wikinre (run1) ---
run_single_checkpoint "$GEMMA_ID" gemma-3-1b-it-wikinre-qlora_run1                trainedon_wikinre_run1  4000
run_single_checkpoint "$QWEN_ID"  Qwen2.5-1.5B-Instruct-wikinre-qlora_run1        trainedon_wikinre_run1  6000
run_single_checkpoint "$LLAMA_ID" Llama-3.2-1B-Instruct-wikinre-qlora_run1        trainedon_wikinre_run1  5000

# --- tum_p99_hparam_r32 ---
run_single_checkpoint "$GEMMA_ID" gemma-3-1b-it-tum_p99-qlora_hparam_r32          trainedon_tum_p99_hparam_r32 4995
run_single_checkpoint "$QWEN_ID"  Qwen2.5-1.5B-Instruct-tum_p99-qlora_hparam_r32  trainedon_tum_p99_hparam_r32 4995
run_single_checkpoint "$LLAMA_ID" Llama-3.2-1B-Instruct-tum_p99-qlora_hparam_r32  trainedon_tum_p99_hparam_r32 4995

# --- tum_p99 different-hparam lr sweep (Qwen only) ---
run_single_checkpoint "$QWEN_ID" Qwen2.5-1.5B-Instruct-tum_p99-qlora_hparam_lr1e4 trainedon_tum_p99_hparam_lr1e4 5994
run_single_checkpoint "$QWEN_ID" Qwen2.5-1.5B-Instruct-tum_p99-qlora_hparam_lr3e4 trainedon_tum_p99_hparam_lr3e4 5994
