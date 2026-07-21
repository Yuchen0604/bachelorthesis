#!/bin/bash
#SBATCH -J eval_wikinre_gemma
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

MODEL_ID=google/gemma-3-1b-it
MODEL_NAME=$(basename "$MODEL_ID")

run_inference() {
    ADAPTER_BASE=$1
    TRAIN_LABEL=$2
    for CHECKPOINT in $(ls -d ${ADAPTER_BASE}/checkpoint-* | sort -V); do
        python evaluation/run_inference.py \
            --model "$MODEL_ID" --adapter "$CHECKPOINT" \
            --dataset wikinre --run "trainedon_${TRAIN_LABEL}_$(basename $CHECKPOINT)"
    done
}

# All existing Gemma training runs, tested on WikiNRE
run_inference ${MODEL_NAME}-rebel-qlora_run2        rebel_run2
run_inference ${MODEL_NAME}-lagrange-qlora_run2     lagrange_run2
run_inference ${MODEL_NAME}-tekgen-qlora_run1       tekgen_run1
run_inference ${MODEL_NAME}-tum_full-qlora_run1     tum_full_run1
run_inference ${MODEL_NAME}-tum_p99-qlora_run1      tum_p99_run1
run_inference ${MODEL_NAME}-tum_p99_new-qlora_run1  tum_p99_new_run1
