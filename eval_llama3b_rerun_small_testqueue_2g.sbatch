#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH --job-name=eval-3b-rerun-2g
#SBATCH --partition=testqueue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/prj0000000224/eval_outputs_full/rlvr-eval-%x-%j.out
#SBATCH --error=/scratch/prj0000000224/eval_outputs_full/rlvr-eval-%x-%j.err

set -euo pipefail

cd /home/users/astar/cfar/stuziyang/SimKO

export MODEL_SELECT="${MODEL_SELECT:-llama3b_simko_rerun}"
export DATASETS="${DATASETS:-TianHongZXY/AIME2025,TianHongZXY/amc23}"
export STEP="${STEP:-160}"
export NUM_GPUS="${NUM_GPUS:-2}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export MAX_TOKENS="${MAX_TOKENS:-4096}"
export NUM_GENERATION="${NUM_GENERATION:-256}"
export RUN_METRICS="${RUN_METRICS:-1}"
export RUN_TAG="${RUN_TAG:-eval-${MODEL_SELECT}-2g-testqueue-${SLURM_JOB_ID:-manual}}"

bash /home/users/astar/cfar/stuziyang/SimKO/eval_finished_model.sbatch "$@"
