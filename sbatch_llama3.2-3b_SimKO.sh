#!/bin/bash
#SBATCH --account=prj0000000224
#SBATCH --job-name=simko_llama3b
#SBATCH --partition=h200n
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
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
export HF_HOME="${HF_HOME:-/scratch/prj0000000224/hf_cache}"
export TOKENIZERS_PARALLELISM=false

mkdir -p logs "$WANDB_DIR" "$WANDB_CACHE_DIR"
mkdir -p /scratch/prj0000000224/models/MATH-Llama-3.2-3B-SimKO-Rerun

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unset}"
echo "HF_HOME=${HF_HOME}"
nvidia-smi

bash run_llama3.2-3b_SimKO.sh \
  trainer.default_local_dir=/scratch/prj0000000224/models/MATH-Llama-3.2-3B-SimKO-Rerun \
  trainer.experiment_name="MATH-Llama-3.2-3B-SimKO-Rerun" \
  trainer.remove_previous_ckpt_in_save=True \
  "$@"
