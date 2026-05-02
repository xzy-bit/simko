#!/bin/bash
# Submit with:
#   sbatch plot_aime_sentence_prob.sh
# Or override inputs/output:
#   sbatch --export=ALL,\
#SCORE_JSON_1=/scratch/prj0000000224/eval_outputs_full/aime_sentence_prob/grpo.json,\
#SCORE_JSON_2=/scratch/prj0000000224/eval_outputs_full/aime_sentence_prob/simko_rerun.json,\
#SCORE_JSON_3=/scratch/prj0000000224/eval_outputs_full/aime_sentence_prob/ts2_rerun.json,\
#OUTPUT_PNG=/scratch/prj0000000224/eval_outputs_full/aime_sentence_prob/aime_triplet.png \
#     plot_aime_sentence_prob.sh

#SBATCH -A prj0000000224
#SBATCH --job-name=plot-aime-prob
#SBATCH --partition=testqueue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/prj0000000224/eval_outputs_full/%x-%j.out
#SBATCH --error=/scratch/prj0000000224/eval_outputs_full/%x-%j.err

set -euo pipefail

module load miniforge/24.11.3-2
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-verl}"

REPO_DIR="${REPO_DIR:-/home/users/astar/cfar/stuziyang/SimKO}"
SCORE_JSON_1="${SCORE_JSON_1:-/scratch/prj0000000224/eval_outputs_full/aime_sentence_prob/grpo.json}"
SCORE_JSON_2="${SCORE_JSON_2:-/scratch/prj0000000224/eval_outputs_full/aime_sentence_prob/simko_rerun.json}"
SCORE_JSON_3="${SCORE_JSON_3:-/scratch/prj0000000224/eval_outputs_full/aime_sentence_prob/ts2_rerun.json}"
OUTPUT_PNG="${OUTPUT_PNG:-/scratch/prj0000000224/eval_outputs_full/aime_sentence_prob/aime_sentence_prob_triplet.png}"
TITLE="${TITLE:-AIME Sentence Prob}"

cd "$REPO_DIR"
mkdir -p "$(dirname "$OUTPUT_PNG")"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Output png: $OUTPUT_PNG"
echo "Started at: $(date)"

python scripts/plot_aime_sentence_prob_json.py \
  --score_json "$SCORE_JSON_1" \
  --score_json "$SCORE_JSON_2" \
  --score_json "$SCORE_JSON_3" \
  --output_png "$OUTPUT_PNG" \
  --title "$TITLE"

echo "Finished at: $(date)"
