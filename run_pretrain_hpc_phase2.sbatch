#!/bin/bash
#SBATCH --account=def-qltian
#SBATCH --job-name=slim-pretrain-hpc-p2
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

# Repo path on cluster (override at submit time if needed via REPO_DIR=/path/to/slim_pretrain).
# Auto-detect from common submit locations so "sbatch /path/to/run_pretrain_hpc_phase2.sbatch"
# works even when submitted outside the repo.
INPUT_REPO_DIR="${REPO_DIR:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "$INPUT_REPO_DIR" ]]; then
  for CANDIDATE in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$SCRIPT_DIR"; do
    [[ -z "${CANDIDATE:-}" ]] && continue
    if [[ -f "$CANDIDATE/__init__.py" && -d "$CANDIDATE/pretrain" ]]; then
      INPUT_REPO_DIR="$CANDIDATE"
      break
    fi
    if [[ -f "$CANDIDATE/slim_pretrain/__init__.py" && -d "$CANDIDATE/slim_pretrain/pretrain" ]]; then
      INPUT_REPO_DIR="$CANDIDATE/slim_pretrain"
      break
    fi
  done
fi

if [[ -z "$INPUT_REPO_DIR" || ! -f "$INPUT_REPO_DIR/__init__.py" || ! -d "$INPUT_REPO_DIR/pretrain" ]]; then
  echo "Could not locate slim_pretrain repo root." >&2
  echo "Set REPO_DIR to the slim_pretrain directory, e.g.:" >&2
  echo "  sbatch --export=ALL,REPO_DIR=/path/to/slim_pretrain run_pretrain_hpc_phase2.sbatch" >&2
  exit 2
fi

REPO_DIR="$INPUT_REPO_DIR"
cd "$REPO_DIR"

mkdir -p logs artifacts

PHASE2_BLOCK="${PHASE2_BLOCK:-A}"
PHASE2_BLOCK="${PHASE2_BLOCK^^}"

TOTAL_STAGES="${TOTAL_STAGES:-}"
STEPS_PER_STAGE="${STEPS_PER_STAGE:-2000}"
TRAIN_RATIO_MIN="${TRAIN_RATIO_MIN:-0.6}"
TRAIN_RATIO_MAX="${TRAIN_RATIO_MAX:-0.8}"

case "$PHASE2_BLOCK" in
  A)
    TOTAL_STAGES="${TOTAL_STAGES:-35}"
    TOTAL_STEPS="${TOTAL_STEPS:-320000}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    SEQ_LEN_MIN="${SEQ_LEN_MIN:-500}"
    SEQ_LEN_MAX="${SEQ_LEN_MAX:-900}"
    NUM_FEATURES_MIN="${NUM_FEATURES_MIN:-8}"
    NUM_FEATURES_MAX="${NUM_FEATURES_MAX:-20}"
    NUM_LAYERS_MIN="${NUM_LAYERS_MIN:-2}"
    NUM_LAYERS_MAX="${NUM_LAYERS_MAX:-10}"
    HIDDEN_DIM_MIN="${HIDDEN_DIM_MIN:-8}"
    HIDDEN_DIM_MAX="${HIDDEN_DIM_MAX:-24}"
    BASE_LR="${BASE_LR:-1.6e-4}"
    MIN_LR="${MIN_LR:-1.6e-5}"
    WARMUP_STEPS="${WARMUP_STEPS:-6000}"
    LR_DECAY_POWER="${LR_DECAY_POWER:-1.5}"
    NONLINEARITIES_DEFAULT="tanh,relu,gelu,sine,identity"
    ;;
  B)
    TOTAL_STAGES="${TOTAL_STAGES:-55}"
    TOTAL_STEPS="${TOTAL_STEPS:-430000}"
    BATCH_SIZE="${BATCH_SIZE:-12}"
    SEQ_LEN_MIN="${SEQ_LEN_MIN:-500}"
    SEQ_LEN_MAX="${SEQ_LEN_MAX:-1100}"
    NUM_FEATURES_MIN="${NUM_FEATURES_MIN:-8}"
    NUM_FEATURES_MAX="${NUM_FEATURES_MAX:-22}"
    NUM_LAYERS_MIN="${NUM_LAYERS_MIN:-3}"
    NUM_LAYERS_MAX="${NUM_LAYERS_MAX:-11}"
    HIDDEN_DIM_MIN="${HIDDEN_DIM_MIN:-10}"
    HIDDEN_DIM_MAX="${HIDDEN_DIM_MAX:-30}"
    BASE_LR="${BASE_LR:-1.2e-4}"
    MIN_LR="${MIN_LR:-1.2e-5}"
    WARMUP_STEPS="${WARMUP_STEPS:-5000}"
    LR_DECAY_POWER="${LR_DECAY_POWER:-1.5}"
    NONLINEARITIES_DEFAULT="tanh,relu,gelu,sine,identity,abs,square"
    ;;
  C)
    TOTAL_STAGES="${TOTAL_STAGES:-35}"
    TOTAL_STEPS="${TOTAL_STEPS:-500000}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    SEQ_LEN_MIN="${SEQ_LEN_MIN:-500}"
    SEQ_LEN_MAX="${SEQ_LEN_MAX:-1300}"
    NUM_FEATURES_MIN="${NUM_FEATURES_MIN:-8}"
    NUM_FEATURES_MAX="${NUM_FEATURES_MAX:-24}"
    NUM_LAYERS_MIN="${NUM_LAYERS_MIN:-4}"
    NUM_LAYERS_MAX="${NUM_LAYERS_MAX:-12}"
    HIDDEN_DIM_MIN="${HIDDEN_DIM_MIN:-12}"
    HIDDEN_DIM_MAX="${HIDDEN_DIM_MAX:-36}"
    BASE_LR="${BASE_LR:-8e-5}"
    MIN_LR="${MIN_LR:-8e-6}"
    WARMUP_STEPS="${WARMUP_STEPS:-4000}"
    LR_DECAY_POWER="${LR_DECAY_POWER:-1.5}"
    NONLINEARITIES_DEFAULT="tanh,relu,gelu,sine,identity,abs,square,sign,heaviside,rbf"
    ;;
  *)
    echo "Unknown PHASE2_BLOCK=${PHASE2_BLOCK}; expected A, B, or C." >&2
    exit 2
    ;;
esac

NONLINEARITIES="${NONLINEARITIES:-$NONLINEARITIES_DEFAULT}"

EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_BATCHES="${EVAL_BATCHES:-16}"
LOG_EVERY="${LOG_EVERY:-200}"
SEED="${SEED:-0}"

PHASE1_CHECKPOINT="${PHASE1_CHECKPOINT:-artifacts/pretrain_checkpoints/latest.pt}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-artifacts/pretrain_checkpoints_phase2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/${PHASE2_BLOCK}}"
BLOCK_A_CHECKPOINT="${BLOCK_A_CHECKPOINT:-${CHECKPOINT_ROOT}/A/latest.pt}"
BLOCK_B_CHECKPOINT="${BLOCK_B_CHECKPOINT:-${CHECKPOINT_ROOT}/B/latest.pt}"
LEGACY_PHASE2_CHECKPOINT="${LEGACY_PHASE2_CHECKPOINT:-${CHECKPOINT_ROOT}/latest.pt}"
SAVE_EVERY="${SAVE_EVERY:-500}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
mkdir -p "$CHECKPOINT_DIR"

PHASE_LOCAL_SCHEDULE="${PHASE_LOCAL_SCHEDULE:-1}"  # 1=true, 0=false
PHASE_START_STEP="${PHASE_START_STEP:-}"
INIT_FROM="${INIT_FROM:-}"
RESUME_FROM="${RESUME_FROM:-}"
BLOCK_LATEST="${CHECKPOINT_DIR}/latest.pt"

# Interruption handling for each block (A/B/C): always resume full state from
# that block's own checkpoint directory when latest.pt exists.
if [[ -z "$RESUME_FROM" && -f "$BLOCK_LATEST" ]]; then
  RESUME_FROM="$BLOCK_LATEST"
fi

# Phase transition init (weights only + fresh optimizer) when no block-local resume exists.
if [[ -z "$RESUME_FROM" && -z "$INIT_FROM" ]]; then
  case "$PHASE2_BLOCK" in
    A)
      INIT_FROM="$PHASE1_CHECKPOINT"
      ;;
    B)
      INIT_FROM="$BLOCK_A_CHECKPOINT"
      ;;
    C)
      INIT_FROM="$BLOCK_B_CHECKPOINT"
      ;;
  esac
fi

if [[ -z "$RESUME_FROM" && "$PHASE2_BLOCK" != "A" && -n "$INIT_FROM" && ! -f "$INIT_FROM" && -f "$LEGACY_PHASE2_CHECKPOINT" ]]; then
  echo "Using legacy phase-2 checkpoint fallback: $LEGACY_PHASE2_CHECKPOINT"
  INIT_FROM="$LEGACY_PHASE2_CHECKPOINT"
fi

HISTORY_JSON="${HISTORY_JSON:-artifacts/pretrain_phase2_${PHASE2_BLOCK}_history_${SLURM_JOB_ID:-local}.json}"

if [[ -n "$RESUME_FROM" && -n "$INIT_FROM" ]]; then
  echo "Both RESUME_FROM and INIT_FROM are set; RESUME_FROM takes precedence."
fi
if [[ -n "$RESUME_FROM" && ! -f "$RESUME_FROM" ]]; then
  echo "Requested RESUME_FROM checkpoint does not exist: $RESUME_FROM" >&2
  exit 2
fi
if [[ -z "$RESUME_FROM" && -n "$INIT_FROM" && ! -f "$INIT_FROM" ]]; then
  echo "Requested INIT_FROM checkpoint does not exist: $INIT_FROM" >&2
  if [[ "$PHASE2_BLOCK" == "A" ]]; then
    echo "Expected phase-1 checkpoint at: $PHASE1_CHECKPOINT" >&2
  elif [[ "$PHASE2_BLOCK" == "B" ]]; then
    echo "Expected block-A checkpoint at: $BLOCK_A_CHECKPOINT" >&2
  else
    echo "Expected block-B checkpoint at: $BLOCK_B_CHECKPOINT" >&2
  fi
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export PYTHONUNBUFFERED=1

module --force purge
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.2
source ~/venvs/tabpfn/bin/activate

export PATH="$HOME/.local/bin:$PATH"

# This repository is the package itself (slim_pretrain/), so expose its parent.
export PYTHONPATH="$(dirname "$REPO_DIR"):${PYTHONPATH:-}"

NPROC="${SLURM_GPUS_ON_NODE:-2}"
if [[ -z "${MASTER_PORT:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MASTER_PORT="$((10000 + (SLURM_JOB_ID % 50000)))"
  else
    MASTER_PORT=29500
  fi
fi

echo "Using MASTER_PORT=${MASTER_PORT}"
echo "Repo dir=${REPO_DIR}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "Phase2 block=${PHASE2_BLOCK}"
echo "Checkpoint dir=${CHECKPOINT_DIR}"
echo "Init from=${INIT_FROM:-<none>}"
echo "Resume from=${RESUME_FROM:-<none>}"
echo "Total steps=${TOTAL_STEPS}, stages=${TOTAL_STAGES}, steps/stage=${STEPS_PER_STAGE}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "import slim_pretrain; print('slim_pretrain', slim_pretrain.__file__)"

RUN_ARGS=(
  --device cuda
  --phase2-block "$PHASE2_BLOCK"
  --seed "$SEED"
  --total-steps "$TOTAL_STEPS"
  --total-stages "$TOTAL_STAGES"
  --steps-per-stage "$STEPS_PER_STAGE"
  --batch-size "$BATCH_SIZE"
  --seq-len-min "$SEQ_LEN_MIN"
  --seq-len-max "$SEQ_LEN_MAX"
  --num-features-min "$NUM_FEATURES_MIN"
  --num-features-max "$NUM_FEATURES_MAX"
  --train-ratio-min "$TRAIN_RATIO_MIN"
  --train-ratio-max "$TRAIN_RATIO_MAX"
  --num-layers-min "$NUM_LAYERS_MIN"
  --num-layers-max "$NUM_LAYERS_MAX"
  --hidden-dim-min "$HIDDEN_DIM_MIN"
  --hidden-dim-max "$HIDDEN_DIM_MAX"
  --nonlinearities "$NONLINEARITIES"
  --base-lr "$BASE_LR"
  --min-lr "$MIN_LR"
  --warmup-steps "$WARMUP_STEPS"
  --lr-decay-power "$LR_DECAY_POWER"
  --log-every "$LOG_EVERY"
  --eval-every "$EVAL_EVERY"
  --eval-batches "$EVAL_BATCHES"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --save-every "$SAVE_EVERY"
  --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS"
  --no-auto-resume
  --history-json "$HISTORY_JSON"
)

if [[ "$PHASE_LOCAL_SCHEDULE" == "0" ]]; then
  RUN_ARGS+=(--no-phase-local-schedule)
else
  RUN_ARGS+=(--phase-local-schedule)
fi

if [[ -n "$PHASE_START_STEP" ]]; then
  RUN_ARGS+=(--phase-start-step "$PHASE_START_STEP")
fi

if [[ -n "$INIT_FROM" && -z "$RESUME_FROM" ]]; then
  RUN_ARGS+=(--init-from "$INIT_FROM")
fi

if [[ -n "$RESUME_FROM" ]]; then
  RUN_ARGS+=(--resume-from "$RESUME_FROM")
fi

torchrun \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  -m slim_pretrain.pretrain.train.run_pretrain_hpc_phase2 \
  "${RUN_ARGS[@]}"
