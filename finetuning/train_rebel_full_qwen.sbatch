#!/bin/bash
#SBATCH -J full_qwen
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
source "$VENV_PATH"

export HF_HUB_OFFLINE=1

RUN=run1
DATASET=rebel-full
MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
MODEL_NAME=Qwen2.5-1.5B-Instruct
ADAPTER_BASE=${MODEL_NAME}-${DATASET}-qlora_${RUN}


python finetuning/finetune.py --model $MODEL_ID --dataset $DATASET --run $RUN

run_inference() {
    TEST_DATASET=$1
    if [ "$TEST_DATASET" == "$DATASET" ]; then
        RUN_LABEL=$RUN                        
    else
        RUN_LABEL=trainedon_${DATASET}_${RUN}  
    fi
    for CHECKPOINT in $(ls -d ${ADAPTER_BASE}/checkpoint-* | sort -V); do
        python evaluation/run_inference.py \
            --model "$MODEL_ID" --adapter "$CHECKPOINT" \
            --dataset $TEST_DATASET --run "${RUN_LABEL}_$(basename $CHECKPOINT)"
    done
}

run_inference rebel-full
run_inference rebel
run_inference lagrange
run_inference tum_full
run_inference tum_p99
run_inference wikinre
