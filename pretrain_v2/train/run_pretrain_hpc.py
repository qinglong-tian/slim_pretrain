from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from ..data import VariableBatchSpec
from ..simplified_prior import CurriculumBounds, PUCurriculumSchedule
from . import (
    DataCurriculumConfig,
    OptimConfig,
    PretrainConfig,
    default_base_prior_config,
    pretrain_nano_tabpfn_pu,
)


DEFAULT_NONLINEARITIES: Tuple[str, ...] = (
    "tanh",
    "relu",
    "gelu",
    "sine",
    "identity",
    "abs",
    "square",
    "sign",
    "heaviside",
    "rbf",
)


def _rank() -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    return int(dist.get_rank())


def _is_primary() -> bool:
    return _rank() == 0


def _init_distributed(backend: Optional[str]) -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    if dist.is_initialized():
        return True
    resolved_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    dist.init_process_group(backend=resolved_backend, init_method="env://")
    return True


def _resolve_device(device_arg: str, distributed: bool) -> str:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")
    if device_arg == "cuda" and distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return f"cuda:{local_rank}"
    return device_arg


def _parse_nonlinearities(spec: Optional[str], fallback: Sequence[str]) -> Tuple[str, ...]:
    if spec is None:
        return tuple(fallback)
    out = tuple(item.strip() for item in spec.split(",") if len(item.strip()) > 0)
    if len(out) == 0:
        raise ValueError("nonlinearities cannot be empty.")
    return out


def _read_checkpoint_steps(checkpoint_path: str) -> Tuple[Optional[int], Optional[int]]:
    ckpt_path = Path(checkpoint_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu")
    step_raw = payload.get("step", None)
    phase_start_step_raw = payload.get("phase_start_step", None)
    step = int(step_raw) + 1 if step_raw is not None else None
    phase_start_step = int(phase_start_step_raw) if phase_start_step_raw is not None else None
    return step, phase_start_step


def _infer_phase_start_step(
    phase_start_step_arg: Optional[int],
    resume_from: Optional[str],
) -> int:
    if phase_start_step_arg is not None:
        if phase_start_step_arg < 0:
            raise ValueError("phase_start_step must be >= 0.")
        return int(phase_start_step_arg)

    if resume_from is not None:
        resume_step, resume_phase_start = _read_checkpoint_steps(resume_from)
        if resume_phase_start is not None:
            return int(resume_phase_start)
        if resume_step is not None:
            return int(resume_step)

    return 0


def _default_legacy_init_checkpoint() -> Optional[str]:
    candidate = Path(__file__).resolve().parents[1] / "saved_models" / "legacy_model.pt"
    if candidate.exists():
        return str(candidate)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V2 pretraining launcher with stratified PU controls.")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dist-backend", type=str, default=None, choices=["nccl", "gloo"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--total-stages", type=int, default=35)
    parser.add_argument("--steps-per-stage", type=int, default=2000)
    parser.add_argument("--total-steps", type=int, default=500000)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-features-min", type=int, default=8)
    parser.add_argument("--num-features-max", type=int, default=24)
    parser.add_argument("--positive-size-min", type=int, default=300)
    parser.add_argument("--positive-size-max", type=int, default=500)

    parser.add_argument("--num-layers-min", type=int, default=4)
    parser.add_argument("--num-layers-max", type=int, default=12)
    parser.add_argument("--hidden-dim-min", type=int, default=12)
    parser.add_argument("--hidden-dim-max", type=int, default=36)

    parser.add_argument("--base-lr", type=float, default=4e-5)
    parser.add_argument("--min-lr", type=float, default=4e-6)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--lr-decay-power", type=float, default=1.5)
    parser.add_argument("--nonlinearities", type=str, default=None)

    parser.add_argument("--unlabeled-ratio-start-lo", type=float, default=1.0)
    parser.add_argument("--unlabeled-ratio-start-hi", type=float, default=1.0)
    parser.add_argument("--unlabeled-ratio-end-lo", type=float, default=0.75)
    parser.add_argument("--unlabeled-ratio-end-hi", type=float, default=3.0)
    parser.add_argument("--test-outlier-start-lo", type=float, default=0.5)
    parser.add_argument("--test-outlier-start-hi", type=float, default=0.5)
    parser.add_argument("--test-outlier-end-lo", type=float, default=0.1)
    parser.add_argument("--test-outlier-end-hi", type=float, default=0.9)

    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=314159)
    parser.add_argument("--fixed-batch-seed", type=int, default=None)

    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--keep-last-checkpoints", type=int, default=5)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--phase-start-step", type=int, default=None)
    parser.add_argument("--phase-local-schedule", dest="phase_local_schedule", action="store_true")
    parser.add_argument("--no-phase-local-schedule", dest="phase_local_schedule", action="store_false")
    parser.set_defaults(phase_local_schedule=True)
    parser.add_argument("--history-json", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    init_from = args.init_from
    if init_from is None:
        init_from = _default_legacy_init_checkpoint()
        if init_from is not None and _is_primary():
            print(f"Using default legacy init checkpoint: {init_from}")

    distributed = _init_distributed(args.dist_backend)
    device = _resolve_device(args.device, distributed=distributed)

    total_steps = int(args.total_steps)
    phase_start_step = _infer_phase_start_step(
        phase_start_step_arg=args.phase_start_step,
        resume_from=args.resume_from,
    )

    effective_total_stages = int(args.total_stages)
    if args.phase_local_schedule:
        phase_total_steps = max(1, total_steps - phase_start_step)
        max_reachable_stages = max(1, (phase_total_steps + args.steps_per_stage - 1) // args.steps_per_stage)
        if effective_total_stages > max_reachable_stages:
            if _is_primary():
                print(
                    f"Adjusted total_stages from {effective_total_stages} to {max_reachable_stages} "
                    f"to match phase budget (phase_total_steps={phase_total_steps}, "
                    f"steps_per_stage={args.steps_per_stage}, phase_start_step={phase_start_step})."
                )
            effective_total_stages = int(max_reachable_stages)

    nonlinearities = _parse_nonlinearities(spec=args.nonlinearities, fallback=DEFAULT_NONLINEARITIES)
    checkpoint_dir = args.checkpoint_dir if len(args.checkpoint_dir) > 0 else None

    pu_schedule = PUCurriculumSchedule(
        unlabeled_ratio_start=(args.unlabeled_ratio_start_lo, args.unlabeled_ratio_start_hi),
        unlabeled_ratio_end=(args.unlabeled_ratio_end_lo, args.unlabeled_ratio_end_hi),
        test_class1_ratio_start=(args.test_outlier_start_lo, args.test_outlier_start_hi),
        test_class1_ratio_end=(args.test_outlier_end_lo, args.test_outlier_end_hi),
    )

    data_cfg = DataCurriculumConfig(
        total_stages=effective_total_stages,
        steps_per_stage=args.steps_per_stage,
        bounds=CurriculumBounds(
            num_layers_min=args.num_layers_min,
            num_layers_max=args.num_layers_max,
            hidden_dim_min=args.hidden_dim_min,
            hidden_dim_max=args.hidden_dim_max,
        ),
        pu_schedule=pu_schedule,
        stationary_sampler={
            "noise_std": [0.005, 0.01, 0.02],
            "sampling": ["normal", "uniform"],
            "per_layer_activation": [True],
            "y_is_effect": [False, True],
            "in_clique": [False, True],
            "sort_features": [False, True],
        },
        batch_spec=VariableBatchSpec(
            batch_size=args.batch_size,
            num_features_range=(args.num_features_min, args.num_features_max),
            positive_size_range=(args.positive_size_min, args.positive_size_max),
            pu_row_policy="drop",
        ),
    )

    cfg = PretrainConfig(
        device=device,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        eval_seed=args.eval_seed,
        fixed_batch_seed=args.fixed_batch_seed,
        max_steps=total_steps,
        optim=OptimConfig(
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
            decay_power=args.lr_decay_power,
        ),
        data=data_cfg,
    )

    base_cfg = replace(
        default_base_prior_config(),
        device=device,
        nonlinearities=nonlinearities,
    )

    result = pretrain_nano_tabpfn_pu(
        base_cfg=base_cfg,
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.save_every,
        keep_last_checkpoints=args.keep_last_checkpoints,
        init_from=init_from,
        resume_from=args.resume_from,
        auto_resume=not args.no_auto_resume,
        phase_local_schedule=args.phase_local_schedule,
        phase_start_step=phase_start_step,
    )

    if _is_primary():
        history = result["history"]
        if result.get("initialized_from") is not None:
            print(
                f"Initialized from: {result['initialized_from']} "
                f"(start_step={result.get('start_step', 0)}, phase_start_step={result.get('phase_start_step', 0)})"
            )
        if result.get("resumed_from") is not None:
            print(
                f"Resumed from: {result['resumed_from']} "
                f"(start_step={result.get('start_step', 0)}, phase_start_step={result.get('phase_start_step', 0)})"
            )
        print(f"V2 pretraining finished {len(history)} steps on rank 0.")
        print("Last record:", history[-1] if len(history) > 0 else None)

        if args.history_json is not None:
            out_path = Path(args.history_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "nonlinearities": nonlinearities,
                "config": result["config"],
                "history": history,
                "data": {
                    "total_stages": cfg.data.total_stages,
                    "steps_per_stage": cfg.data.steps_per_stage,
                    "total_steps": cfg.total_steps,
                    "pu_schedule": asdict(cfg.data.pu_schedule),
                },
                "optim": asdict(cfg.optim),
                "checkpoint": {
                    "checkpoint_dir": checkpoint_dir,
                    "save_every": args.save_every,
                    "keep_last_checkpoints": args.keep_last_checkpoints,
                    "initialized_from": result.get("initialized_from"),
                    "resumed_from": result.get("resumed_from"),
                    "start_step": result.get("start_step", 0),
                    "phase_start_step": result.get("phase_start_step", 0),
                    "phase_local_schedule": result.get("phase_local_schedule", False),
                },
            }
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"Wrote history JSON to {out_path}")

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
