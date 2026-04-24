#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH --job-name=rlvr-math-resume
#SBATCH --partition=h200n
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/prj0000000224/eval_outputs_full/math-resume-%j.out
#SBATCH --error=/scratch/prj0000000224/eval_outputs_full/math-resume-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/users/astar/cfar/stuziyang/RLVR-Decomposed}"
CONDA_ENV="${CONDA_ENV:-verl}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/prj0000000224/models}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/prj0000000224/eval_outputs_full}"
STEP="${STEP:-140}"
MODEL_SELECT="${MODEL_SELECT:-GRPO}"

DATASET="${DATASET:-TianHongZXY/MATH}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
NUM_GPUS="${NUM_GPUS:-4}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:--1}"
NUM_GENERATION="${NUM_GENERATION:-256}"
RUN_METRICS="${RUN_METRICS:-1}"

case "$MODEL_SELECT" in
  GRPO)
    MODEL_NAME="MATH-Qwen2.5-math-7B-GRPO"
    RUN_TAG="${RUN_TAG:-math-GRPO-step${STEP}}"
    ;;
  SimKO)
    MODEL_NAME="MATH-Qwen2.5-math-7B-SimKO"
    RUN_TAG="${RUN_TAG:-math-SimKO-step${STEP}}"
    ;;
  SimKO-TS2)
    MODEL_NAME="MATH-Qwen2.5-math-7B-SimKO-TS2"
    RUN_TAG="${RUN_TAG:-math-SimKO-TS2-step${STEP}}"
    ;;
  *)
    echo "Unsupported MODEL_SELECT=$MODEL_SELECT. Use GRPO, SimKO, or SimKO-TS2." >&2
    exit 1
    ;;
esac

OUTPUT_ROOT="$OUTPUT_BASE/$RUN_TAG"
OUTPUT_DIR="$OUTPUT_ROOT/${MODEL_NAME}-step${STEP}"
OUTPUT_FILE="$OUTPUT_DIR/MATH-${SPLIT}-temp_${TEMPERATURE}-top_p_${TOP_P}-top_k_${TOP_K}.jsonl"
ACTOR_DIR="$MODEL_ROOT/$MODEL_NAME/global_step_$STEP/actor"
HF_DIR="$ACTOR_DIR/huggingface"

module load miniforge/24.11.3-2
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

export HF_HOME="${HF_HOME:-/scratch/prj0000000224/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/prj0000000224/hf_cache}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$OUTPUT_DIR"

cd "$REPO_DIR"

has_hf_weights() {
  find "$1" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "pytorch_model*.bin" -o -name "model*.bin" \) | grep -q .
}

if [[ ! -d "$ACTOR_DIR" ]]; then
  echo "Missing actor checkpoint: $ACTOR_DIR" >&2
  exit 1
fi

if [[ ! -d "$HF_DIR" ]] || ! has_hf_weights "$HF_DIR"; then
  echo "HF weights not found in $HF_DIR; merging from $ACTOR_DIR"
  python scripts/model_merger.py --local_dir "$ACTOR_DIR"
fi

if [[ ! -s "$OUTPUT_FILE" ]]; then
  RESUME_ID=0
else
  LAST_IDS="$(tail -n 1 "$OUTPUT_FILE" | sed -n 's/.*"question_id": \([0-9][0-9]*\), "generation_id": \([0-9][0-9]*\).*/\1 \2/p')"
  if [[ -z "$LAST_IDS" ]]; then
    echo "Could not parse final line in $OUTPUT_FILE" >&2
    exit 1
  fi

  read -r LAST_QID LAST_GID <<< "$LAST_IDS"
  if [[ "$LAST_GID" -eq $((NUM_GENERATION - 1)) ]]; then
    RESUME_ID=$((LAST_QID + 1))
  else
    RESUME_ID="$LAST_QID"
    COMPLETE_LINES=$((RESUME_ID * NUM_GENERATION))
    TMP_FILE="${OUTPUT_FILE}.complete"
    echo "Last question is partial: question_id=$LAST_QID generation_id=$LAST_GID"
    echo "Truncating to $COMPLETE_LINES complete lines before resuming from question_id=$RESUME_ID"
    head -n "$COMPLETE_LINES" "$OUTPUT_FILE" > "$TMP_FILE"
    mv "$TMP_FILE" "$OUTPUT_FILE"
  fi
fi

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Model: $MODEL_SELECT"
echo "Checkpoint: $ACTOR_DIR"
echo "Output file: $OUTPUT_FILE"
echo "Resume id: $RESUME_ID"
echo "Started at: $(date)"

if [[ "$RESUME_ID" -lt 5000 ]]; then
  python eval.py \
    --model_name="$HF_DIR" \
    --datasets="$DATASET" \
    --split="$SPLIT" \
    --output_dir="$OUTPUT_DIR" \
    --batch_size="$BATCH_SIZE" \
    --max_tokens="$MAX_TOKENS" \
    --num_gpus="$NUM_GPUS" \
    --temperature="$TEMPERATURE" \
    --top_p="$TOP_P" \
    --top_k="$TOP_K" \
    --num_generation="$NUM_GENERATION" \
    --resume_id="$RESUME_ID"
else
  echo "All examples already appear complete; skipping generation."
fi

if [[ "$RUN_METRICS" == "1" ]]; then
  python calculate_metrics.py --file_path "$OUTPUT_FILE"
fi

echo "Finished at: $(date)"
