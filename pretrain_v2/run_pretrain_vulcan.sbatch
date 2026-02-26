#!/bin/bash
#SBATCH --account=aip-qltian
#SBATCH --job-name=pretrain-v2-vulcan
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=TERM@120
#SBATCH --mail-user=qltian2021@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

set -euo pipefail

INPUT_REPO_DIR="${REPO_DIR:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYOUT="${LAYOUT:-}"
LAUNCH_MODULE="${LAUNCH_MODULE:-}"
IMPORT_CHECK="${IMPORT_CHECK:-}"

if [[ -z "$INPUT_REPO_DIR" ]]; then
  for CANDIDATE in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(dirname "$SCRIPT_DIR")"; do
    [[ -z "${CANDIDATE:-}" ]] && continue
    if [[ -f "$CANDIDATE/__init__.py" && -d "$CANDIDATE/pretrain_v2" ]]; then
      INPUT_REPO_DIR="$CANDIDATE"
      LAYOUT="full_repo"
      LAUNCH_MODULE="slim_pretrain.pretrain_v2.train.run_pretrain_hpc"
      IMPORT_CHECK="import slim_pretrain; print('slim_pretrain', slim_pretrain.__file__)"
      break
    fi
    if [[ -f "$CANDIDATE/slim_pretrain/__init__.py" && -d "$CANDIDATE/slim_pretrain/pretrain_v2" ]]; then
      INPUT_REPO_DIR="$CANDIDATE/slim_pretrain"
      LAYOUT="full_repo"
      LAUNCH_MODULE="slim_pretrain.pretrain_v2.train.run_pretrain_hpc"
      IMPORT_CHECK="import slim_pretrain; print('slim_pretrain', slim_pretrain.__file__)"
      break
    fi
    if [[ -f "$CANDIDATE/__init__.py" && -d "$CANDIDATE/train" && -d "$CANDIDATE/simplified_prior" && -d "$CANDIDATE/data" ]]; then
      INPUT_REPO_DIR="$CANDIDATE"
      LAYOUT="standalone_v2"
      LAUNCH_MODULE="pretrain_v2.train.run_pretrain_hpc"
      IMPORT_CHECK="import pretrain_v2; print('pretrain_v2', pretrain_v2.__file__)"
      break
    fi
  done
fi

if [[ -n "$INPUT_REPO_DIR" && -z "$LAUNCH_MODULE" ]]; then
  if [[ -f "$INPUT_REPO_DIR/__init__.py" && -d "$INPUT_REPO_DIR/pretrain_v2" ]]; then
    LAYOUT="full_repo"
    LAUNCH_MODULE="slim_pretrain.pretrain_v2.train.run_pretrain_hpc"
    IMPORT_CHECK="import slim_pretrain; print('slim_pretrain', slim_pretrain.__file__)"
  elif [[ -f "$INPUT_REPO_DIR/slim_pretrain/__init__.py" && -d "$INPUT_REPO_DIR/slim_pretrain/pretrain_v2" ]]; then
    INPUT_REPO_DIR="$INPUT_REPO_DIR/slim_pretrain"
    LAYOUT="full_repo"
    LAUNCH_MODULE="slim_pretrain.pretrain_v2.train.run_pretrain_hpc"
    IMPORT_CHECK="import slim_pretrain; print('slim_pretrain', slim_pretrain.__file__)"
  elif [[ -f "$INPUT_REPO_DIR/__init__.py" && -d "$INPUT_REPO_DIR/train" && -d "$INPUT_REPO_DIR/simplified_prior" && -d "$INPUT_REPO_DIR/data" ]]; then
    LAYOUT="standalone_v2"
    LAUNCH_MODULE="pretrain_v2.train.run_pretrain_hpc"
    IMPORT_CHECK="import pretrain_v2; print('pretrain_v2', pretrain_v2.__file__)"
  fi
fi

if [[ -z "$INPUT_REPO_DIR" ]]; then
  echo "Could not locate repository root for pretrain_v2 run." >&2
  echo "Set REPO_DIR to either:" >&2
  echo "  1) slim_pretrain package root, or" >&2
  echo "  2) standalone pretrain_v2 folder." >&2
  exit 2
fi

REPO_DIR="$INPUT_REPO_DIR"
cd "$REPO_DIR"

mkdir -p logs artifacts

# Main phase: 400k steps with 2k steps/stage -> 200 stages.
STEPS_PER_STAGE="${STEPS_PER_STAGE:-2000}"
MAIN_TOTAL_STAGES="${MAIN_TOTAL_STAGES:-200}"
MAIN_TOTAL_STEPS="${MAIN_TOTAL_STEPS:-$((MAIN_TOTAL_STAGES * STEPS_PER_STAGE))}"

# Tail fine-tune phase: keep final-stage settings for +120k steps.
TAIL_EXTRA_STEPS="${TAIL_EXTRA_STEPS:-120000}"
TAIL_TOTAL_STEPS="${TAIL_TOTAL_STEPS:-$((MAIN_TOTAL_STEPS + TAIL_EXTRA_STEPS))}"
TAIL_TOTAL_STAGES="${TAIL_TOTAL_STAGES:-1}"
TAIL_STEPS_PER_STAGE="${TAIL_STEPS_PER_STAGE:-$TAIL_EXTRA_STEPS}"

# Per-GPU micro batch. Set to 3 to reduce late-stage OOM on 48GB L40s.
BATCH_SIZE="${BATCH_SIZE:-3}"
NUM_FEATURES_MIN="${NUM_FEATURES_MIN:-8}"
NUM_FEATURES_MAX="${NUM_FEATURES_MAX:-24}"
POSITIVE_SIZE_MIN="${POSITIVE_SIZE_MIN:-300}"
POSITIVE_SIZE_MAX="${POSITIVE_SIZE_MAX:-500}"

NUM_LAYERS_MIN="${NUM_LAYERS_MIN:-4}"
NUM_LAYERS_MAX="${NUM_LAYERS_MAX:-12}"
HIDDEN_DIM_MIN="${HIDDEN_DIM_MIN:-12}"
HIDDEN_DIM_MAX="${HIDDEN_DIM_MAX:-36}"

# LR defaults linearly scaled from previous (batch_size=4 per GPU):
# base/min LR: 4e-5/4e-6 -> 3e-5/3e-6 for batch_size=3.
BASE_LR="${BASE_LR:-3e-5}"
MIN_LR="${MIN_LR:-3e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-6000}"
LR_DECAY_POWER="${LR_DECAY_POWER:-1.5}"

# Tail phase uses a smaller LR scheme.
TAIL_BASE_LR="${TAIL_BASE_LR:-1e-5}"
TAIL_MIN_LR="${TAIL_MIN_LR:-1e-6}"
TAIL_WARMUP_STEPS="${TAIL_WARMUP_STEPS:-3000}"
TAIL_LR_DECAY_POWER="${TAIL_LR_DECAY_POWER:-1.5}"
NONLINEARITIES="${NONLINEARITIES:-tanh,relu,gelu,sine,identity,abs,square,sign,heaviside,rbf}"

# Stage schedules:
# unlabeled/positive range: [1,1] -> [0.75,3]
UNLABELED_RATIO_START_LO="${UNLABELED_RATIO_START_LO:-1.0}"
UNLABELED_RATIO_START_HI="${UNLABELED_RATIO_START_HI:-1.0}"
UNLABELED_RATIO_END_LO="${UNLABELED_RATIO_END_LO:-0.75}"
UNLABELED_RATIO_END_HI="${UNLABELED_RATIO_END_HI:-3.0}"

# test outlier(class1) range: [0.5,0.5] -> [0.1,0.9]
TEST_OUTLIER_START_LO="${TEST_OUTLIER_START_LO:-0.5}"
TEST_OUTLIER_START_HI="${TEST_OUTLIER_START_HI:-0.5}"
TEST_OUTLIER_END_LO="${TEST_OUTLIER_END_LO:-0.1}"
TEST_OUTLIER_END_HI="${TEST_OUTLIER_END_HI:-0.9}"

EVAL_EVERY="${EVAL_EVERY:-200}"
EVAL_BATCHES="${EVAL_BATCHES:-16}"
LOG_EVERY="${LOG_EVERY:-200}"
SEED="${SEED:-0}"

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  if [[ "$LAYOUT" == "standalone_v2" ]]; then
    CHECKPOINT_DIR="artifacts/checkpoints"
  else
    CHECKPOINT_DIR="pretrain_v2/artifacts/checkpoints"
  fi
fi
SAVE_EVERY="${SAVE_EVERY:-500}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-5}"
mkdir -p "$CHECKPOINT_DIR"

PHASE_LOCAL_SCHEDULE="${PHASE_LOCAL_SCHEDULE:-1}"  # 1=true, 0=false
PHASE_START_STEP="${PHASE_START_STEP:-}"
TAIL_PHASE_LOCAL_SCHEDULE="${TAIL_PHASE_LOCAL_SCHEDULE:-1}"  # 1=true, 0=false
TAIL_PHASE_START_STEP="${TAIL_PHASE_START_STEP:-$MAIN_TOTAL_STEPS}"
INIT_FROM="${INIT_FROM:-}"
RESUME_FROM="${RESUME_FROM:-}"
LATEST_CHECKPOINT="${CHECKPOINT_DIR}/latest.pt"

if [[ -z "$RESUME_FROM" && -f "$LATEST_CHECKPOINT" ]]; then
  RESUME_FROM="$LATEST_CHECKPOINT"
fi

if [[ -z "${HISTORY_JSON_BASE:-}" ]]; then
  if [[ "$LAYOUT" == "standalone_v2" ]]; then
    HISTORY_JSON_BASE="artifacts/history_${SLURM_JOB_ID:-local}.json"
  else
    HISTORY_JSON_BASE="pretrain_v2/artifacts/history_${SLURM_JOB_ID:-local}.json"
  fi
fi
MAIN_HISTORY_JSON="${MAIN_HISTORY_JSON:-${HISTORY_JSON_BASE%.json}_main.json}"
TAIL_HISTORY_JSON="${TAIL_HISTORY_JSON:-${HISTORY_JSON_BASE%.json}_tail.json}"

if [[ -n "$RESUME_FROM" && -n "$INIT_FROM" ]]; then
  echo "Both RESUME_FROM and INIT_FROM are set; RESUME_FROM takes precedence."
fi
if [[ -n "$RESUME_FROM" && ! -f "$RESUME_FROM" ]]; then
  echo "Requested RESUME_FROM checkpoint does not exist: $RESUME_FROM" >&2
  exit 2
fi
if [[ -z "$RESUME_FROM" && -n "$INIT_FROM" && ! -f "$INIT_FROM" ]]; then
  echo "Requested INIT_FROM checkpoint does not exist: $INIT_FROM" >&2
  exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

module --force purge
module load StdEnv/2023
module load python/3.10.13
module load cuda/12.2
source ~/venvs/icl/bin/activate

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$(dirname "$REPO_DIR"):${PYTHONPATH:-}"

NNODES="${SLURM_NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
if [[ -z "${MASTER_PORT:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    MASTER_PORT="$((10000 + (SLURM_JOB_ID % 50000)))"
  else
    MASTER_PORT=29500
  fi
fi

if [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "Could not resolve MASTER_ADDR from SLURM_JOB_NODELIST." >&2
  exit 2
fi

echo "Using MASTER_PORT=${MASTER_PORT}"
echo "Using MASTER_ADDR=${MASTER_ADDR}"
echo "Using NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "Layout=${LAYOUT}"
echo "Repo dir=${REPO_DIR}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "Checkpoint dir=${CHECKPOINT_DIR}"
echo "Init from=${INIT_FROM:-<none>}"
echo "Resume from=${RESUME_FROM:-<none>}"
echo "Main phase: total steps=${MAIN_TOTAL_STEPS}, stages=${MAIN_TOTAL_STAGES}, steps/stage=${STEPS_PER_STAGE}"
echo "Tail phase: total steps=${TAIL_TOTAL_STEPS} (+${TAIL_EXTRA_STEPS}), stages=${TAIL_TOTAL_STAGES}, steps/stage=${TAIL_STEPS_PER_STAGE}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -c "$IMPORT_CHECK"

COMMON_ARGS=(
  --device cuda
  --seed "$SEED"
  --batch-size "$BATCH_SIZE"
  --num-features-min "$NUM_FEATURES_MIN"
  --num-features-max "$NUM_FEATURES_MAX"
  --positive-size-min "$POSITIVE_SIZE_MIN"
  --positive-size-max "$POSITIVE_SIZE_MAX"
  --num-layers-min "$NUM_LAYERS_MIN"
  --num-layers-max "$NUM_LAYERS_MAX"
  --hidden-dim-min "$HIDDEN_DIM_MIN"
  --hidden-dim-max "$HIDDEN_DIM_MAX"
  --nonlinearities "$NONLINEARITIES"
  --unlabeled-ratio-start-lo "$UNLABELED_RATIO_START_LO"
  --unlabeled-ratio-start-hi "$UNLABELED_RATIO_START_HI"
  --unlabeled-ratio-end-lo "$UNLABELED_RATIO_END_LO"
  --unlabeled-ratio-end-hi "$UNLABELED_RATIO_END_HI"
  --test-outlier-start-lo "$TEST_OUTLIER_START_LO"
  --test-outlier-start-hi "$TEST_OUTLIER_START_HI"
  --test-outlier-end-lo "$TEST_OUTLIER_END_LO"
  --test-outlier-end-hi "$TEST_OUTLIER_END_HI"
  --log-every "$LOG_EVERY"
  --eval-every "$EVAL_EVERY"
  --eval-batches "$EVAL_BATCHES"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --save-every "$SAVE_EVERY"
  --keep-last-checkpoints "$KEEP_LAST_CHECKPOINTS"
  --no-auto-resume
)

run_phase() {
  local phase_name="$1"
  shift
  local -a phase_args=("$@")
  local phase_args_str
  phase_args_str="$(printf ' %q' "${phase_args[@]}")"

  echo "========== ${phase_name} phase =========="
  if [[ "$NNODES" -gt 1 ]]; then
    srun --ntasks="$NNODES" --ntasks-per-node=1 \
      bash -lc "torchrun \
        --nnodes=$NNODES \
        --node_rank=\$SLURM_PROCID \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m $LAUNCH_MODULE${phase_args_str}"
  else
    torchrun \
      --nproc_per_node="$NPROC_PER_NODE" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      -m "$LAUNCH_MODULE" \
      "${phase_args[@]}"
  fi
}

MAIN_ARGS=(
  "${COMMON_ARGS[@]}"
  --total-steps "$MAIN_TOTAL_STEPS"
  --total-stages "$MAIN_TOTAL_STAGES"
  --steps-per-stage "$STEPS_PER_STAGE"
  --base-lr "$BASE_LR"
  --min-lr "$MIN_LR"
  --warmup-steps "$WARMUP_STEPS"
  --lr-decay-power "$LR_DECAY_POWER"
  --history-json "$MAIN_HISTORY_JSON"
)

if [[ "$PHASE_LOCAL_SCHEDULE" == "0" ]]; then
  MAIN_ARGS+=(--no-phase-local-schedule)
else
  MAIN_ARGS+=(--phase-local-schedule)
fi
if [[ -n "$PHASE_START_STEP" ]]; then
  MAIN_ARGS+=(--phase-start-step "$PHASE_START_STEP")
fi
if [[ -n "$INIT_FROM" && -z "$RESUME_FROM" ]]; then
  MAIN_ARGS+=(--init-from "$INIT_FROM")
fi
if [[ -n "$RESUME_FROM" ]]; then
  MAIN_ARGS+=(--resume-from "$RESUME_FROM")
fi

run_phase "main" "${MAIN_ARGS[@]}"

if [[ ! -f "$LATEST_CHECKPOINT" ]]; then
  echo "Tail phase requires checkpoint at $LATEST_CHECKPOINT, but it was not found." >&2
  exit 2
fi

TAIL_ARGS=(
  "${COMMON_ARGS[@]}"
  --total-steps "$TAIL_TOTAL_STEPS"
  --total-stages "$TAIL_TOTAL_STAGES"
  --steps-per-stage "$TAIL_STEPS_PER_STAGE"
  --base-lr "$TAIL_BASE_LR"
  --min-lr "$TAIL_MIN_LR"
  --warmup-steps "$TAIL_WARMUP_STEPS"
  --lr-decay-power "$TAIL_LR_DECAY_POWER"
  --history-json "$TAIL_HISTORY_JSON"
  --resume-from "$LATEST_CHECKPOINT"
  --phase-start-step "$TAIL_PHASE_START_STEP"
)

if [[ "$TAIL_PHASE_LOCAL_SCHEDULE" == "0" ]]; then
  TAIL_ARGS+=(--no-phase-local-schedule)
else
  TAIL_ARGS+=(--phase-local-schedule)
fi

run_phase "tail" "${TAIL_ARGS[@]}"
