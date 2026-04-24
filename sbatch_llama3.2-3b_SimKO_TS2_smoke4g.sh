#!/bin/bash
#SBATCH --account=prj0000000224
#SBATCH --job-name=llama3b_ts2_smoke
#SBATCH --partition=h200n
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:h200:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/users/astar/cfar/stuziyang/SimKO

source /apps/miniforge/24.11.3/etc/profile.d/conda.sh
conda activate verl

export PYTHONNOUSERSITE=1
unset PYTHONPATH

export RAY_DEDUP_LOGS=0
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$PWD/wandb"
export WANDB_CACHE_DIR="$PWD/.cache/wandb"
export HF_HOME="${HF_HOME:-/scratch/prj0000000224/hf_cache}"
export TOKENIZERS_PARALLELISM=false

mkdir -p logs "$WANDB_DIR" "$WANDB_CACHE_DIR"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unset}"
echo "HF_HOME=${HF_HOME}"
nvidia-smi

bash run_llama3.2-3b_SimKO_TS2_smoke4g.sh "$@"
