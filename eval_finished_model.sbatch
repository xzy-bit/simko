#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH --job-name=rlvr-eval
#SBATCH --partition=h200n
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/prj0000000224/eval_outputs_full/rlvr-eval-%x-%j.out
#SBATCH --error=/scratch/prj0000000224/eval_outputs_full/rlvr-eval-%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/users/astar/cfar/stuziyang/RLVR-Decomposed}"
CONDA_ENV="${CONDA_ENV:-verl}"
MODEL_ROOT="${MODEL_ROOT:-/scratch/prj0000000224/models}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/prj0000000224/eval_outputs_full}"

MODEL_SELECT="${MODEL_SELECT:-llama3b_grpo}"
DATASETS="${DATASETS:-TianHongZXY/AIME2025,TianHongZXY/amc23}"
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
  qwen_ts2)
    MODEL_NAME="MATH-Qwen2.5-math-7B-SimKO-TS2"
    STEP="${STEP:-140}"
    LABEL="qwen-ts2"
    ;;
  qwen_simko)
    MODEL_NAME="MATH-Qwen2.5-math-7B-SimKO"
    STEP="${STEP:-140}"
    LABEL="qwen-simko"
    ;;
  qwen_grpo)
    MODEL_NAME="MATH-Qwen2.5-math-7B-GRPO"
    STEP="${STEP:-140}"
    LABEL="qwen-grpo"
    ;;
  llama3b_grpo)
    MODEL_NAME="MATH-Llama-3.2-3B-GRPO"
    STEP="${STEP:-160}"
    LABEL="llama3b-grpo"
    ;;
  llama3b_simko)
    MODEL_NAME="MATH-Llama-3.2-3B-SimKO"
    STEP="${STEP:-160}"
    LABEL="llama3b-simko"
    ;;
  llama3b_simko_rerun)
    MODEL_NAME="MATH-Llama-3.2-3B-SimKO-Rerun"
    STEP="${STEP:-160}"
    LABEL="llama3b-simko-rerun"
    ;;
  llama3b_ts2)
    MODEL_NAME="MATH-Llama-3.2-3B-SimKO-TS2"
    STEP="${STEP:-160}"
    LABEL="llama3b-ts2"
    ;;
  llama3b_ts2_rerun)
    MODEL_NAME="MATH-Llama-3.2-3B-SimKO-TS2-Rerun"
    STEP="${STEP:-160}"
    LABEL="llama3b-ts2-rerun"
    ;;
  *)
    echo "Unsupported MODEL_SELECT=$MODEL_SELECT" >&2
    echo "Use one of: qwen_grpo, qwen_simko, qwen_ts2, llama3b_grpo, llama3b_simko, llama3b_simko_rerun, llama3b_ts2, llama3b_ts2_rerun" >&2
    exit 1
    ;;
esac

RUN_TAG="${RUN_TAG:-eval-${LABEL}-step${STEP}-${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="$OUTPUT_BASE/$RUN_TAG"
OUTPUT_DIR="$OUTPUT_ROOT/${MODEL_NAME}-step${STEP}"
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

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Model select: $MODEL_SELECT"
echo "Model name: $MODEL_NAME"
echo "Step: $STEP"
echo "Datasets: $DATASETS"
echo "Output dir: $OUTPUT_DIR"
echo "Started at: $(date)"

if [[ ! -d "$ACTOR_DIR" ]]; then
  echo "Missing actor checkpoint: $ACTOR_DIR" >&2
  exit 1
fi

if [[ ! -d "$HF_DIR" ]] || ! has_hf_weights "$HF_DIR"; then
  echo "HF weights not found in $HF_DIR"
  echo "Merging FSDP checkpoint from $ACTOR_DIR"
  python scripts/model_merger.py --local_dir "$ACTOR_DIR"
fi

python eval.py \
  --model_name="$HF_DIR" \
  --datasets="$DATASETS" \
  --split="$SPLIT" \
  --output_dir="$OUTPUT_DIR" \
  --batch_size="$BATCH_SIZE" \
  --max_tokens="$MAX_TOKENS" \
  --num_gpus="$NUM_GPUS" \
  --temperature="$TEMPERATURE" \
  --top_p="$TOP_P" \
  --top_k="$TOP_K" \
  --num_generation="$NUM_GENERATION"

if [[ "$RUN_METRICS" == "1" ]]; then
  echo "Calculating metrics"
  while IFS= read -r file_path; do
    echo "------------------------------------------------------------"
    echo "$file_path"
    python calculate_metrics.py --file_path "$file_path"
  done < <(find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*.jsonl" | sort)
fi

echo "Finished at: $(date)"
find "$OUTPUT_ROOT" -maxdepth 2 -type f -name "*.jsonl" -print | sort
