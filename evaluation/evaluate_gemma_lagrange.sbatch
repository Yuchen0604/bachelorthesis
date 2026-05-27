#!/bin/bash
#SBATCH -J evaluate
#SBATCH -w gpu-nvidia-h100
#SBATCH --gres=gpu:nvidia:1
#SBATCH -t 24:00:00

PROJECT_DIR="/storage/home/$USER/bachelorthesis"
VENV_PATH="$PROJECT_DIR/venv_cluster/bin/activate"

module load cuda/12.6.3

cd "$PROJECT_DIR"
RUN=run1
DATASET=lagrange

source "$VENV_PATH"

#Gemma
python evaluation/run_inference.py --model google/gemma-3-1b-it --dataset $DATASET --run $RUN
python evaluation/evaluate.py --predictions predictions/gemma-3-1b-it_${DATASET}_base_${RUN}.jsonl
python evaluation/run_inference.py --model google/gemma-3-1b-it --dataset $DATASET --adapter gemma-3-1b-it-lagrange-qlora --run $RUN
python evaluation/evaluate.py --predictions predictions/gemma-3-1b-it_${DATASET}_finetuned_${RUN}.jsonl

