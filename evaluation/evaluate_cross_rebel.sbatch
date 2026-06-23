#!/bin/bash
#SBATCH -J cross_rebel
#SBATCH -w gpu-nvidia-h200-3
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

RUN=run1

run_inference() {
    MODEL_ID=$1
    ADAPTER_BASE=$2
    TEST_DATASET=$3
    for CHECKPOINT in $(ls -d ${ADAPTER_BASE}/checkpoint-* | sort -V); do
        python evaluation/run_inference.py \
            --model "$MODEL_ID" --adapter "$CHECKPOINT" \
            --dataset $TEST_DATASET --run "trainedon_tekgen_${RUN}_$(basename $CHECKPOINT)"
    done
}

# TekGen Gemma → Rebel test set
run_inference google/gemma-3-1b-it  gemma-3-1b-it-tekgen-qlora_${RUN}  rebel