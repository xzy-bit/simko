#!/bin/bash
#SBATCH --account=prj0000000224
#SBATCH --job-name=simko_ts2_smoke
#SBATCH --partition=h200n
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
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
mkdir -p /scratch/prj0000000224/models/MATH-Qwen2.5-math-7B-SimKO-TS2-smoke

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unset}"
echo "HF_HOME=${HF_HOME}"
nvidia-smi

bash run_qwen2.5-math-7b_SimKO.sh \
  trainer.default_local_dir=/scratch/prj0000000224/models/MATH-Qwen2.5-math-7B-SimKO-TS2-smoke \
  trainer.experiment_name="MATH-Qwen2.5-7B-SimKO-TS2-Smoke4G" \
  actor_rollout_ref.actor.simko_ts2=True \
  actor_rollout_ref.actor.top_k=9 \
  data.train_batch_size=16 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=1 \
  trainer.n_gpus_per_node=4 \
  trainer.logger=['console'] \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=1 \
  trainer.total_epochs=1 \
  "$@"
