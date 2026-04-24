#!/bin/bash
#SBATCH -A prj0000000224
#SBATCH --job-name=eval-kingbaicai-ts2
#SBATCH --partition=h200n
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/prj0000000224/eval_outputs_full/rlvr-eval-%x-%j.out
#SBATCH --error=/scratch/prj0000000224/eval_outputs_full/rlvr-eval-%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/users/astar/cfar/stuziyang/RLVR-Decomposed}"
CONDA_ENV="${CONDA_ENV:-verl}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/prj0000000224/eval_outputs_full}"

MODEL_PATH="${MODEL_PATH:-/scratch/prj0000000224/models/Kingbaicai_TS2-method}"
MODEL_LABEL="${MODEL_LABEL:-Kingbaicai_TS2-method}"
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
RUN_TAG="${RUN_TAG:-eval-kingbaicai-ts2-aime-amc-${SLURM_JOB_ID:-manual}}"

OUTPUT_ROOT="$OUTPUT_BASE/$RUN_TAG"
OUTPUT_DIR="$OUTPUT_ROOT/$MODEL_LABEL"

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
echo "Model path: $MODEL_PATH"
echo "Datasets: $DATASETS"
echo "Output dir: $OUTPUT_DIR"
echo "Started at: $(date)"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing model path: $MODEL_PATH" >&2
  exit 1
fi

if [[ "$(basename "$MODEL_PATH")" == "huggingface" ]] && ! has_hf_weights "$MODEL_PATH"; then
  ACTOR_DIR="$(dirname "$MODEL_PATH")"
  echo "HF weights not found in $MODEL_PATH"
  echo "Merging FSDP checkpoint from $ACTOR_DIR"
  python scripts/model_merger.py --local_dir "$ACTOR_DIR"
fi

python eval.py \
  --model_name="$MODEL_PATH" \
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
