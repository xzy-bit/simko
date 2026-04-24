#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH --job-name=merge-3b-ts2-hf
#SBATCH --partition=testqueue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/prj0000000224/eval_outputs_full/%x-%j.out
#SBATCH --error=/scratch/prj0000000224/eval_outputs_full/%x-%j.err

set -euo pipefail

REPO_DIR="/home/users/astar/cfar/stuziyang/RLVR-Decomposed"
CONDA_ENV="${CONDA_ENV:-verl}"
ACTOR_DIR="${ACTOR_DIR:-/scratch/prj0000000224/models/MATH-Llama-3.2-3B-SimKO-TS2/global_step_160/actor}"
HF_DIR="$ACTOR_DIR/huggingface"

module load miniforge/24.11.3-2
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/scratch/prj0000000224/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/prj0000000224/hf_cache}"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME"

cd "$REPO_DIR"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Actor dir: $ACTOR_DIR"
echo "HF dir: $HF_DIR"
echo "Started at: $(date)"

if [[ ! -d "$ACTOR_DIR" ]]; then
  echo "Missing actor directory: $ACTOR_DIR" >&2
  exit 1
fi

if [[ ! -d "$HF_DIR" ]]; then
  LATEST_BROKEN="$(find "$ACTOR_DIR" -maxdepth 1 -type d -name 'huggingface.broken-*' | sort | tail -n 1)"
  if [[ -z "$LATEST_BROKEN" ]]; then
    echo "Missing $HF_DIR and no huggingface.broken-* backup found for config/tokenizer files." >&2
    exit 1
  fi
  echo "Restoring config/tokenizer files from: $LATEST_BROKEN"
  mkdir -p "$HF_DIR"
  find "$LATEST_BROKEN" -maxdepth 1 -type f \
    ! -name "*.safetensors" \
    ! -name "pytorch_model*.bin" \
    ! -name "model*.bin" \
    ! -name "model.safetensors.index.json" \
    -exec cp -p {} "$HF_DIR/" \;
else
  BACKUP_DIR="${HF_DIR}.bad-weights-${SLURM_JOB_ID:-manual}-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$BACKUP_DIR"
  echo "Backing up existing HF weight files to: $BACKUP_DIR"
  find "$HF_DIR" -maxdepth 1 -type f \( \
    -name "*.safetensors" \
    -o -name "pytorch_model*.bin" \
    -o -name "model*.bin" \
    -o -name "model.safetensors.index.json" \
  \) -exec mv {} "$BACKUP_DIR/" \;
fi

python scripts/model_merger.py --local_dir "$ACTOR_DIR"

echo "Merged files:"
find "$HF_DIR" -maxdepth 1 -type f -printf "%f %s bytes\n" | sort

echo "Finished at: $(date)"
