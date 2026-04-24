#!/bin/bash
#SBATCH --account=prj0000000224
#SBATCH --job-name=simko_qwen7b
#SBATCH --partition=h200n
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1024G
#SBATCH --gres=gpu:h200:8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/users/astar/cfar/stuziyang/SimKO

source /apps/miniforge/24.11.3/etc/profile.d/conda.sh
conda activate verl

export PYTHONNOUSERSITE=1
unset PYTHONPATH

export RAY_DEDUP_LOGS=0
export WANDB_MODE=online
export WANDB_DIR="$PWD/wandb"
export WANDB_CACHE_DIR="$PWD/.cache/wandb"
export HF_HOME="${HF_HOME:-/home/users/astar/cfar/stuziyang/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$HF_HOME"
mkdir -p /scratch/prj0000000224/models/MATH-Qwen2.5-math-7B-SimKO

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unset}"
nvidia-smi

bash run_qwen2.5-math-7b_SimKO.sh \
  trainer.default_local_dir=/scratch/prj0000000224/models/MATH-Qwen2.5-math-7B-SimKO \
  "$@"
