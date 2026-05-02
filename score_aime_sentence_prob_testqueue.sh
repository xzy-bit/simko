#!/bin/bash
# Submit with:
#   sbatch score_aime_sentence_prob_testqueue.sh
# Override resources or target model with:
#   sbatch --partition=testqueue --gres=gpu:1 --time=01:00:00 \
#     --export=ALL,MODEL_PRESET=grpo,JOB_NAME=score-grpo \
#     score_aime_sentence_prob_testqueue.sh

#SBATCH -A prj0000000224
#SBATCH --job-name=score-aime-prob
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

set -euo pipefail

module load miniforge/24.11.3-2
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-verl}"

REPO_DIR="${REPO_DIR:-/home/users/astar/cfar/stuziyang/SimKO}"
MODEL_PRESET="${MODEL_PRESET:-grpo}"
MODEL_SPEC="${MODEL_SPEC:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/prj0000000224/eval_outputs_full/aime_sentence_prob}"
OUTPUT_JSON="${OUTPUT_JSON:-$OUTPUT_DIR/${MODEL_PRESET}.json}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Model preset: $MODEL_PRESET"
echo "Model spec: ${MODEL_SPEC:-<none>}"
echo "Output json: $OUTPUT_JSON"
echo "Started at: $(date)"

if [[ -n "$MODEL_SPEC" ]]; then
  python scripts/score_aime_sentence_prob_json.py \
    --spec "$MODEL_SPEC" \
    --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --output_json "$OUTPUT_JSON"
else
  python scripts/score_aime_sentence_prob_json.py \
    --preset "$MODEL_PRESET" \
    --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --output_json "$OUTPUT_JSON"
fi

echo "Finished at: $(date)"
