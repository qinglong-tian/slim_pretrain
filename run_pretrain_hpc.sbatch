#!/bin/bash
#SBATCH --account=def-qltian
#SBATCH --job-name=slim-pretrain-hpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=qltian2021@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

# Repo path on cluster (override at submit time if needed).
REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

mkdir -p logs artifacts

# Runtime knobs (override with: sbatch --export=ALL,TOTAL_STEPS=...,SAVE_EVERY=...)
TOTAL_STEPS="${TOTAL_STEPS:-250000}"
TOTAL_STAGES="${TOTAL_STAGES:-100}"
STEPS_PER_STAGE="${STEPS_PER_STAGE:-2000}"

BATCH_SIZE="${BATCH_SIZE:-16}"
SEQ_LEN_MIN="${SEQ_LEN_MIN:-500}"
SEQ_LEN_MAX="${SEQ_LEN_MAX:-800}"
NUM_FEATURES_MIN="${NUM_FEATURES_MIN:-8}"
NUM_FEATURES_MAX="${NUM_FEATURES_MAX:-20}"

BASE_LR="${BASE_LR:-1e-4}"
MIN_LR="${MIN_LR:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-12000}"

EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_BATCHES="${EVAL_BATCHES:-8}"
LOG_EVERY="${LOG_EVERY:-200}"
SEED="${SEED:-0}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/pretrain_checkpoints}"
SAVE_EVERY="${SAVE_EVERY:-500}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
mkdir -p "$CHECKPOINT_DIR"

HISTORY_JSON="${HISTORY_JSON:-artifacts/pretrain_history_${SLURM_JOB_ID:-local}.json}"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export PYTHONUNBUFFERED=1

module --force purge
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.2
source ~/venvs/tabpfn/bin/activate

export PATH="$HOME/.local/bin:$PATH"

# This repository is the package itself (slim_pretrain/), so expose its parent.
if [[ -f "$REPO_DIR/__init__.py" && -d "$REPO_DIR/pretrain" ]]; then
  export PYTHONPATH="$(dirname "$REPO_DIR"):${PYTHONPATH:-}"
else
  export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
fi

NPROC="${SLURM_GPUS_ON_NODE:-2}"
if [[ -z "${MASTER_PORT:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MASTER_PORT="$((10000 + (SLURM_JOB_ID % 50000)))"
  else
    MASTER_PORT=29500
  fi
fi
echo "Using MASTER_PORT=${MASTER_PORT}"
echo "Checkpoint dir=${CHECKPOINT_DIR}"
echo "Total steps=${TOTAL_STEPS}, stages=${TOTAL_STAGES}, steps/stage=${STEPS_PER_STAGE}"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# run_pretrain_hpc.py auto-resumes from ${CHECKPOINT_DIR}/latest.pt by default.
torchrun \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  -m slim_pretrain.pretrain.train.run_pretrain_hpc \
  --device cuda \
  --seed "$SEED" \
  --total-steps "$TOTAL_STEPS" \
  --total-stages "$TOTAL_STAGES" \
  --steps-per-stage "$STEPS_PER_STAGE" \
  --batch-size "$BATCH_SIZE" \
  --seq-len-min "$SEQ_LEN_MIN" \
  --seq-len-max "$SEQ_LEN_MAX" \
  --num-features-min "$NUM_FEATURES_MIN" \
  --num-features-max "$NUM_FEATURES_MAX" \
  --base-lr "$BASE_LR" \
  --min-lr "$MIN_LR" \
  --warmup-steps "$WARMUP_STEPS" \
  --log-every "$LOG_EVERY" \
  --eval-every "$EVAL_EVERY" \
  --eval-batches "$EVAL_BATCHES" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --save-every "$SAVE_EVERY" \
  --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS" \
  --history-json "$HISTORY_JSON"
