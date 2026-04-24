#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH --job-name=eval-3b-ts2-1gpu
#SBATCH --partition=h200n
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/prj0000000224/eval_outputs_full/rlvr-eval-%x-%j.out
#SBATCH --error=/scratch/prj0000000224/eval_outputs_full/rlvr-eval-%x-%j.err

set -euo pipefail

cd /home/users/astar/cfar/stuziyang/SimKO

export MODEL_SELECT="${MODEL_SELECT:-llama3b_ts2}"
export DATASETS="${DATASETS:-TianHongZXY/AIME2025,TianHongZXY/amc23}"
export STEP="${STEP:-160}"
export NUM_GPUS="${NUM_GPUS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export MAX_TOKENS="${MAX_TOKENS:-4096}"
export RUN_METRICS="${RUN_METRICS:-1}"
export RUN_TAG="${RUN_TAG:-eval-llama3b-ts2-old-1gpu-${SLURM_JOB_ID:-manual}}"

bash /home/users/astar/cfar/stuziyang/SimKO/eval_finished_model.sbatch "$@"
