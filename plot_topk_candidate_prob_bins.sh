#!/bin/bash
# Submit with:
#   sbatch plot_topk_candidate_prob_bins.sh
#
# Override target model or resources with:
#   sbatch --partition=testqueue --gres=gpu:1 --time=01:00:00 \
#     --export=ALL,\
#LABEL=grpo,\
#MODEL_DIR=/scratch/prj0000000224/models/MATH-Llama-3.2-3B-GRPO/global_step_160/actor/huggingface,\
#EVAL_JSONL=/scratch/prj0000000224/eval_outputs_full/eval-llama3b-grpo-step160-55495/MATH-Llama-3.2-3B-GRPO-step160/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl \
#     plot_topk_candidate_prob_bins.sh

#SBATCH -A prj0000000224
#SBATCH --job-name=plot-topk-bins
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

module load miniforge/24.11.3-2
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-verl}"

REPO_DIR="${REPO_DIR:-/home/users/astar/cfar/stuziyang/SimKO}"
LABEL="${LABEL:-grpo}"
MODEL_DIR="${MODEL_DIR:-/scratch/prj0000000224/models/MATH-Llama-3.2-3B-GRPO/global_step_160/actor/huggingface}"
EVAL_JSONL="${EVAL_JSONL:-/scratch/prj0000000224/eval_outputs_full/eval-llama3b-grpo-step160-55495/MATH-Llama-3.2-3B-GRPO-step160/AIME2025-test-temp_0.6-top_p_0.95-top_k_-1.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/prj0000000224/eval_outputs_full/topk_candidate_prob_bins}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_DIR/${LABEL}_top6_prob_bins.json}"
OUTPUT_PNG="${OUTPUT_PNG:-$OUTPUT_DIR/${LABEL}_top6_prob_bins.png}"
TOP_K="${TOP_K:-6}"
NUM_BINS="${NUM_BINS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
DTYPE="${DTYPE:-bfloat16}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Label: $LABEL"
echo "Model dir: $MODEL_DIR"
echo "Eval jsonl: $EVAL_JSONL"
echo "Output json: $OUTPUT_JSON"
echo "Output png: $OUTPUT_PNG"
echo "Started at: $(date)"

cmd=(
  python scripts/plot_topk_candidate_prob_bins.py
  --label "$LABEL"
  --model_dir "$MODEL_DIR"
  --eval_jsonl "$EVAL_JSONL"
  --output_json "$OUTPUT_JSON"
  --output_png "$OUTPUT_PNG"
  --top_k "$TOP_K"
  --num_bins "$NUM_BINS"
  --batch_size "$BATCH_SIZE"
  --dtype "$DTYPE"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  cmd+=(--max_samples "$MAX_SAMPLES")
fi

"${cmd[@]}"

echo "Finished at: $(date)"
