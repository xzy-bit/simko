#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH --job-name=cmp-aime-mix
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=cmp_diversity_%j.out

module load miniforge/24.11.3-2
eval "$(conda shell.bash hook)"
conda activate gem
cd /home/users/astar/cfar/stuziyang/SimKO
python scripts/compare_math_diversity.py \
  --spec grpo=/scratch/prj0000000224/eval_outputs_full/eval-llama3b-grpo-step160-55495/MATH-Llama-3.2-3B-GRPO-step160/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl \
  --spec simko=/scratch/prj0000000224/eval_outputs_full/eval-llama3b-simko-rerun-2g-testqueue-20260418-095356/MATH-Llama-3.2-3B-SimKO-Rerun-step160/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl \
  --spec ts2=/scratch/prj0000000224/eval_outputs_full/eval-llama3b-ts2-rerun-2g-testqueue-20260418-095401/MATH-Llama-3.2-3B-SimKO-TS2-Rerun-step160/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl \
  --spec mix001=/scratch/prj0000000224/eval_outputs_full/eval-llama3b-ts2-nosquare-mix001-step160/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix001-step160_rerun/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl \
  --spec mix005=/scratch/prj0000000224/eval_outputs_full/eval-llama3b-ts2-nosquare-mix005-step160/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix005-step160_rerun/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl \
  --spec mix01=/scratch/prj0000000224/eval_outputs_full/eval-llama3b-ts2-nosquare-mix01-step160/MATH-Llama-3.2-3B-SimKO-TS2-nosquare-mix01-step160_rerun/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl \
  --output_csv /scratch/prj0000000224/eval_outputs_full/compare_aime_mix_diversity_${SLURM_JOB_ID}.csv

