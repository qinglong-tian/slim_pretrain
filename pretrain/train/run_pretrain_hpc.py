from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

from slim_pretrain.pretrain.data import VariableBatchSpec
from slim_pretrain.pretrain.train import (
    DataCurriculumConfig,
    OptimConfig,
    PretrainConfig,
    default_base_prior_config,
    pretrain_nano_tabpfn_pu,
)
from slim_pretrain.simplified_prior import CurriculumBounds


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HPC-friendly pretraining launcher for slim_pretrain.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dist-backend", type=str, default=None, choices=["nccl", "gloo"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-stages", type=int, default=20)
    parser.add_argument("--steps-per-stage", type=int, default=2000)
    parser.add_argument("--total-steps", type=int, default=80000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len-min", type=int, default=500)
    parser.add_argument("--seq-len-max", type=int, default=800)
    parser.add_argument("--num-features-min", type=int, default=8)
    parser.add_argument("--num-features-max", type=int, default=20)
    parser.add_argument("--train-ratio-min", type=float, default=0.6)
    parser.add_argument("--train-ratio-max", type=float, default=0.8)
    parser.add_argument("--num-layers-min", type=int, default=2)
    parser.add_argument("--num-layers-max", type=int, default=8)
    parser.add_argument("--hidden-dim-min", type=int, default=8)
    parser.add_argument("--hidden-dim-max", type=int, default=16)
    parser.add_argument("--base-lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=8000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=314159)
    parser.add_argument("--fixed-batch-seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--history-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed = _init_distributed(args.dist_backend)
    device = _resolve_device(args.device, distributed=distributed)
    max_steps = args.total_steps if args.total_steps is not None else args.max_steps
    checkpoint_dir = args.checkpoint_dir if len(args.checkpoint_dir) > 0 else None

    data_cfg = DataCurriculumConfig(
        total_stages=args.total_stages,
        steps_per_stage=args.steps_per_stage,
        bounds=CurriculumBounds(
            num_layers_min=args.num_layers_min,
            num_layers_max=args.num_layers_max,
            hidden_dim_min=args.hidden_dim_min,
            hidden_dim_max=args.hidden_dim_max,
        ),
        # Stage-invariant sampler for non-stage-controlled prior fields.
        stationary_sampler={
            "noise_std": [0.005, 0.01, 0.02],
            "sampling": ["normal", "uniform"],
            "per_layer_activation": [False, True],
            "y_is_effect": [False, True],
            "in_clique": [False, True],
            "sort_features": [False, True],
            "balanced_labels": [True],
        },
        batch_spec=VariableBatchSpec(
            batch_size=args.batch_size,
            seq_len_range=(args.seq_len_min, args.seq_len_max),
            num_features_range=(args.num_features_min, args.num_features_max),
            train_ratio_range=(args.train_ratio_min, args.train_ratio_max),
            pu_row_policy="drop",
        ),
    )

    # Core training config. Edit defaults here or override from CLI.
    cfg = PretrainConfig(
        device=device,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        eval_seed=args.eval_seed,
        fixed_batch_seed=args.fixed_batch_seed,
        max_steps=max_steps,
        optim=OptimConfig(
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_steps,
        ),
        data=data_cfg,
    )

    # Base prior defaults. VariableBatchSpec above controls sampled seq_len / num_features / train_size.
    base_cfg = replace(
        default_base_prior_config(),
        device=device,
    )

    result = pretrain_nano_tabpfn_pu(
        base_cfg=base_cfg,
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.save_every,
        keep_last_checkpoints=args.keep_last_checkpoints,
        resume_from=args.resume_from,
        auto_resume=not args.no_auto_resume,
    )
    if _is_primary():
        history = result["history"]
        if result.get("resumed_from") is not None:
            print(f"Resumed from: {result['resumed_from']} (start_step={result.get('start_step', 0)})")
        print(f"Finished {len(history)} steps on rank 0.")
        print("Last record:", history[-1] if len(history) > 0 else None)

        if args.history_json is not None:
            out_path = Path(args.history_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "config": result["config"],
                "history": history,
                "data": {
                    "total_stages": cfg.data.total_stages,
                    "steps_per_stage": cfg.data.steps_per_stage,
                    "total_steps": cfg.total_steps,
                },
                "optim": asdict(cfg.optim),
                "checkpoint": {
                    "checkpoint_dir": checkpoint_dir,
                    "save_every": args.save_every,
                    "keep_last_checkpoints": args.keep_last_checkpoints,
                    "resumed_from": result.get("resumed_from"),
                    "start_step": result.get("start_step", 0),
                },
            }
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"Wrote history JSON to {out_path}")

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
