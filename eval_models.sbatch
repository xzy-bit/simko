#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH --job-name=rlvr-eval
#SBATCH --partition=h200n
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/users/astar/cfar/stuziyang/RLVR-Decomposed}"
CONDA_ENV="${CONDA_ENV:-verl}"

module load miniforge/24.11.3-2
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

export HF_HOME="${HF_HOME:-/scratch/prj0000000224/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/prj0000000224/.cache}"

mkdir -p /scratch/prj0000000224/eval_outputs
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

cd "$REPO_DIR"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Repo: $REPO_DIR"
echo "Conda env: $CONDA_ENV"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Started at: $(date)"

NUM_GPUS="${NUM_GPUS:-2}" ./eval_models.sh

echo "Finished at: $(date)"
