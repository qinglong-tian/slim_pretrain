from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from slim_pretrain.pretrain.data import generate_variable_padded_batch
from slim_pretrain.pretrain.model import NanoTabPFNPUModel
from slim_pretrain.simplified_prior import SimplifiedPriorConfig, sample_curriculum_config

from .config import PretrainConfig
from .schedule import stage_index_from_step, warmup_cosine_lr


def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_rank() -> int:
    if not _dist_is_initialized():
        return 0
    return int(dist.get_rank())


def _dist_world_size() -> int:
    if not _dist_is_initialized():
        return 1
    return int(dist.get_world_size())


def _is_primary_process() -> bool:
    return _dist_rank() == 0


def _dist_mean(value: float, device: torch.device) -> float:
    if not _dist_is_initialized():
        return float(value)
    t = torch.tensor(float(value), device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= float(_dist_world_size())
    return float(t.item())


def _dist_any_true(flag: bool, device: torch.device) -> bool:
    if not _dist_is_initialized():
        return bool(flag)
    t = torch.tensor(1 if flag else 0, device=device, dtype=torch.int32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(int(t.item()) > 0)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _resolve_resume_checkpoint(
    checkpoint_dir: Optional[str],
    resume_from: Optional[str],
    auto_resume: bool,
) -> Optional[Path]:
    if resume_from is not None:
        resume_path = Path(resume_from).expanduser()
        if not resume_path.exists():
            raise FileNotFoundError(f"Requested resume checkpoint does not exist: {resume_path}")
        return resume_path
    if not auto_resume or checkpoint_dir is None:
        return None

    ckpt_dir = Path(checkpoint_dir).expanduser()
    latest_path = ckpt_dir / "latest.pt"
    if latest_path.exists():
        return latest_path

    step_paths = sorted(ckpt_dir.glob("step_*.pt"))
    if len(step_paths) == 0:
        return None
    return step_paths[-1]


def _load_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[int, Optional[float], Optional[int]]:
    payload = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in payload or "optimizer_state_dict" not in payload or "step" not in payload:
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")

    _unwrap_model(model).load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    step = int(payload["step"])
    ema_loss_raw = payload.get("ema_loss", None)
    ema_loss = float(ema_loss_raw) if ema_loss_raw is not None else None
    phase_start_step_raw = payload.get("phase_start_step", None)
    phase_start_step = int(phase_start_step_raw) if phase_start_step_raw is not None else None
    return step + 1, ema_loss, phase_start_step


def _load_model_state_only(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> Optional[int]:
    payload = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint format (missing model_state_dict): {checkpoint_path}")

    _unwrap_model(model).load_state_dict(payload["model_state_dict"])
    if "step" not in payload:
        return None
    return int(payload["step"]) + 1


def _save_training_checkpoint(
    checkpoint_dir: Path,
    step: int,
    phase_start_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_loss: Optional[float],
    config: PretrainConfig,
    keep_last_checkpoints: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "phase_start_step": int(phase_start_step),
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_loss": None if ema_loss is None else float(ema_loss),
        "config": asdict(config),
    }

    latest_tmp = checkpoint_dir / ".latest.pt.tmp"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, latest_tmp)
    os.replace(latest_tmp, latest_path)

    if keep_last_checkpoints == 0:
        return

    step_path = checkpoint_dir / f"step_{step+1:08d}.pt"
    torch.save(payload, step_path)

    if keep_last_checkpoints > 0:
        step_paths = sorted(checkpoint_dir.glob("step_*.pt"))
        if len(step_paths) > keep_last_checkpoints:
            for old_path in step_paths[: len(step_paths) - keep_last_checkpoints]:
                old_path.unlink(missing_ok=True)


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _train_step_on_batch(
    model: NanoTabPFNPUModel,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    """Train on a padded batch by iterating over per-dataset tasks."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    batch_size = int(batch["X"].shape[0])
    losses = []
    for idx in range(batch_size):
        num_rows = int(batch["seq_lens"][idx].item())
        num_features = int(batch["num_features"][idx].item())
        train_size = int(batch["train_sizes"][idx].item())
        if num_rows <= train_size:
            continue

        x = batch["X"][idx, :num_rows, :num_features].unsqueeze(0).to(device=device, dtype=torch.float32)
        y = batch["y"][idx, :num_rows].to(device=device, dtype=torch.long)

        y_train = y[:train_size].to(torch.float32).unsqueeze(0)
        y_target = y[train_size:]
        if y_target.numel() == 0:
            continue

        logits = model((x, y_train), train_test_split_index=train_size).squeeze(0)
        task_loss = F.cross_entropy(logits, y_target)
        if not torch.isfinite(task_loss):
            return float("nan")
        losses.append(task_loss)

    if len(losses) == 0:
        return 0.0

    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        return float("nan")

    loss.backward()
    for param in model.parameters():
        grad = param.grad
        if grad is not None and not torch.isfinite(grad).all():
            return float("nan")

    if grad_clip_norm > 0:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if isinstance(total_norm, torch.Tensor):
            if not torch.isfinite(total_norm):
                return float("nan")
        elif not np.isfinite(float(total_norm)):
            return float("nan")
    optimizer.step()
    return float(loss.detach().cpu().item())


def _eval_loss_on_batch(
    model: NanoTabPFNPUModel,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    model.eval()
    batch_size = int(batch["X"].shape[0])
    losses = []
    with torch.no_grad():
        for idx in range(batch_size):
            num_rows = int(batch["seq_lens"][idx].item())
            num_features = int(batch["num_features"][idx].item())
            train_size = int(batch["train_sizes"][idx].item())
            if num_rows <= train_size:
                continue

            x = batch["X"][idx, :num_rows, :num_features].unsqueeze(0).to(device=device, dtype=torch.float32)
            y = batch["y"][idx, :num_rows].to(device=device, dtype=torch.long)

            y_train = y[:train_size].to(torch.float32).unsqueeze(0)
            y_target = y[train_size:]
            if y_target.numel() == 0:
                continue

            logits = model((x, y_train), train_test_split_index=train_size).squeeze(0)
            losses.append(F.cross_entropy(logits, y_target))

    if len(losses) == 0:
        return 0.0
    return float(torch.stack(losses).mean().detach().cpu().item())


def _build_fixed_eval_batches(
    base_cfg: SimplifiedPriorConfig,
    config: PretrainConfig,
) -> List[Dict[str, torch.Tensor]]:
    if config.eval_batches <= 0:
        return []
    eval_spec = config.eval_batch_spec if config.eval_batch_spec is not None else config.data.batch_spec
    eval_stage = int(config.data.total_stages)
    batches: List[Dict[str, torch.Tensor]] = []
    for idx in range(config.eval_batches):
        eval_rng = np.random.default_rng(config.eval_seed + idx)

        def _cfg_sampler(local_rng: np.random.Generator) -> SimplifiedPriorConfig:
            return sample_curriculum_config(
                base_cfg=base_cfg,
                stage_idx=eval_stage,
                total_stages=config.data.total_stages,
                bounds=config.data.bounds,
                stationary_sampler=config.data.stationary_sampler,
                rng=local_rng,
            )

        batch = generate_variable_padded_batch(
            base_cfg=base_cfg,
            spec=eval_spec,
            rng=eval_rng,
            config_sampler=_cfg_sampler,
        )
        batches.append(batch)
    return batches


def pretrain_nano_tabpfn_pu(
    base_cfg: SimplifiedPriorConfig,
    config: PretrainConfig,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 0,
    keep_last_checkpoints: int = 3,
    resume_from: Optional[str] = None,
    auto_resume: bool = False,
    init_from: Optional[str] = None,
    phase_local_schedule: bool = False,
    phase_start_step: Optional[int] = None,
) -> Dict[str, object]:
    """Run pretraining with data curriculum + LR curriculum.

    Data curriculum:
    - total_stages (default 10), each with steps_per_stage (default 1000)
    - stage-varying `is_causal`, `num_layers`, `hidden_dim`
    - PU always enabled by design (pu_keep_probability fixed to 0)

    Training curriculum:
    - warmup + cosine learning-rate schedule
    """
    rank = _dist_rank()
    torch.manual_seed(config.seed + rank)
    np.random.seed(config.seed + rank)

    device = torch.device(config.device)
    if device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device.index is None:
            device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    model = NanoTabPFNPUModel(
        embedding_size=config.model.embedding_size,
        num_attention_heads=config.model.num_attention_heads,
        mlp_hidden_size=config.model.mlp_hidden_size,
        num_layers=config.model.num_layers,
        num_outputs=config.model.num_outputs,
    ).to(device)
    if _dist_is_initialized():
        if device.type == "cuda":
            model = DDP(model, device_ids=[device.index], output_device=device.index, broadcast_buffers=False)
        else:
            model = DDP(model, broadcast_buffers=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optim.base_lr,
        betas=(config.optim.beta1, config.optim.beta2),
        weight_decay=config.optim.weight_decay,
    )

    if not (0.0 <= config.ema_decay < 1.0):
        raise ValueError("ema_decay must satisfy 0 <= ema_decay < 1.")
    if config.eval_every < 0:
        raise ValueError("eval_every must be >= 0.")
    if config.eval_batches < 0:
        raise ValueError("eval_batches must be >= 0.")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0.")
    if keep_last_checkpoints < 0:
        raise ValueError("keep_last_checkpoints must be >= 0.")
    if phase_start_step is not None and phase_start_step < 0:
        raise ValueError("phase_start_step must be >= 0 when provided.")

    start_step = 0
    ema_loss: Optional[float] = None
    resolved_phase_start_step: Optional[int] = phase_start_step
    init_path_str: Optional[str] = None
    init_step: Optional[int] = None
    resume_path = _resolve_resume_checkpoint(
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        auto_resume=auto_resume,
    )
    if resume_path is not None:
        start_step, ema_loss, ckpt_phase_start_step = _load_training_state(
            model=model,
            optimizer=optimizer,
            checkpoint_path=resume_path,
            device=device,
        )
        if resolved_phase_start_step is None:
            resolved_phase_start_step = ckpt_phase_start_step
        if _is_primary_process():
            print(f"Resumed from checkpoint {resume_path} at step {start_step}.")
    elif init_from is not None:
        init_path = Path(init_from).expanduser()
        if not init_path.exists():
            raise FileNotFoundError(f"Requested init checkpoint does not exist: {init_path}")
        init_path_str = str(init_path)
        init_step = _load_model_state_only(
            model=model,
            checkpoint_path=init_path,
            device=device,
        )
        if init_step is not None:
            start_step = init_step
            if resolved_phase_start_step is None:
                resolved_phase_start_step = init_step
        if _is_primary_process():
            if init_step is None:
                print(f"Initialized model weights from checkpoint {init_path_str}; optimizer state reset.")
            else:
                print(
                    f"Initialized model weights from checkpoint {init_path_str} "
                    f"at step {init_step}; optimizer state reset."
                )

    if resolved_phase_start_step is None:
        resolved_phase_start_step = 0

    if phase_local_schedule:
        phase_total_steps = max(1, config.total_steps - resolved_phase_start_step)
    else:
        phase_total_steps = config.total_steps

    if start_step >= config.total_steps:
        if _is_primary_process():
            print(f"Checkpoint already reached target total_steps={config.total_steps}. No training steps to run.")
        out_model = _unwrap_model(model)
        return {
            "model": out_model if _is_primary_process() else None,
            "history": [],
            "config": asdict(config),
            "start_step": start_step,
            "phase_start_step": resolved_phase_start_step,
            "phase_local_schedule": phase_local_schedule,
            "initialized_from": init_path_str,
            "resumed_from": None if resume_path is None else str(resume_path),
        }

    eval_batches = _build_fixed_eval_batches(base_cfg=base_cfg, config=config) if _is_primary_process() else []
    history: List[Dict[str, object]] = []
    nonfinite_skip_count = 0
    max_nonfinite_skips = 10
    for step in range(start_step, config.total_steps):
        phase_step = int(step - resolved_phase_start_step) if phase_local_schedule else int(step)
        phase_step = max(0, phase_step)
        stage_idx = stage_index_from_step(
            step=phase_step,
            steps_per_stage=config.data.steps_per_stage,
            total_stages=config.data.total_stages,
        )
        lr = warmup_cosine_lr(
            step=phase_step,
            total_steps=phase_total_steps,
            base_lr=config.optim.base_lr,
            min_lr=config.optim.min_lr,
            warmup_steps=config.optim.warmup_steps,
        )
        _set_lr(optimizer, lr)

        def _cfg_sampler(local_rng: np.random.Generator) -> SimplifiedPriorConfig:
            return sample_curriculum_config(
                base_cfg=base_cfg,
                stage_idx=stage_idx,
                total_stages=config.data.total_stages,
                bounds=config.data.bounds,
                stationary_sampler=config.data.stationary_sampler,
                rng=local_rng,
            )

        batch = generate_variable_padded_batch(
            base_cfg=base_cfg,
            spec=config.data.batch_spec,
            rng=np.random.default_rng(
                config.fixed_batch_seed if config.fixed_batch_seed is not None else config.seed + phase_step
            ),
            config_sampler=_cfg_sampler,
        )

        local_loss = _train_step_on_batch(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=config.optim.grad_clip_norm,
        )
        local_nonfinite = not np.isfinite(local_loss)
        if _dist_any_true(local_nonfinite, device=device):
            nonfinite_skip_count += 1
            msg = (
                f"Skipping step due to non-finite training loss/gradient at "
                f"global_step={step+1}, phase_step={phase_step+1}, stage={stage_idx}, "
                f"lr={lr:.6f} (skip_count={nonfinite_skip_count}/{max_nonfinite_skips})."
            )
            if _is_primary_process():
                print(msg)
            if nonfinite_skip_count > max_nonfinite_skips:
                raise FloatingPointError(
                    "Exceeded max_nonfinite_skips; aborting training to avoid silent divergence."
                )
            continue

        loss = _dist_mean(local_loss, device=device)
        nonfinite_skip_count = 0
        ema_loss = loss if ema_loss is None else (config.ema_decay * ema_loss + (1.0 - config.ema_decay) * loss)

        eval_loss = float("nan")
        should_eval = (
            config.eval_every > 0
            and len(eval_batches) > 0
            and ((phase_step + 1) % config.eval_every == 0 or phase_step == 0)
        )
        if should_eval:
            eval_loss = float(np.mean([_eval_loss_on_batch(model=model, batch=b, device=device) for b in eval_batches]))
        eval_nonfinite = should_eval and (not np.isfinite(eval_loss))
        if _dist_any_true(eval_nonfinite and _is_primary_process(), device=device):
            msg = (
                f"Non-finite eval loss detected at global_step={step+1}, "
                f"phase_step={phase_step+1}, stage={stage_idx}, lr={lr:.6f}."
            )
            if _is_primary_process():
                print(msg)
            raise FloatingPointError(msg)

        rec = {
            "step": step,
            "phase_step": phase_step,
            "stage": stage_idx,
            "lr": lr,
            "loss": loss,
            "loss_ema": float(ema_loss),
            "eval_loss": eval_loss,
            "is_causal": float(batch["cfg_is_causal"].float().mean().item()),
            "num_layers": float(batch["cfg_num_layers"].float().mean().item()),
            "hidden_dim": float(batch["cfg_hidden_dim"].float().mean().item()),
            "pu_keep_probability": float(batch["cfg_pu_keep_probability"].mean().item()),
            "batch_pu_rate": float(batch["is_pu"].float().mean().item()),
            "batch_removed_rows_mean": float(batch["removed_train_rows"].float().mean().item()),
        }
        if _is_primary_process():
            history.append(rec)
        if _is_primary_process() and config.log_every > 0 and ((phase_step + 1) % config.log_every == 0 or phase_step == 0):
            eval_str = f"{eval_loss:.4f}" if np.isfinite(eval_loss) else "nan"
            msg = (
                f"step={step+1}/{config.total_steps} phase_step={phase_step+1}/{phase_total_steps} "
                f"stage={stage_idx} lr={lr:.6f} "
                f"loss={loss:.4f} loss_ema={float(ema_loss):.4f} eval_loss={eval_str} "
                f"pu_rate={rec['batch_pu_rate']:.2f}"
            )
            print(msg)
        if (
            _is_primary_process()
            and checkpoint_dir is not None
            and checkpoint_every > 0
            and (phase_step + 1) % checkpoint_every == 0
        ):
            _save_training_checkpoint(
                checkpoint_dir=Path(checkpoint_dir).expanduser(),
                step=step,
                phase_start_step=resolved_phase_start_step,
                model=model,
                optimizer=optimizer,
                ema_loss=ema_loss,
                config=config,
                keep_last_checkpoints=keep_last_checkpoints,
            )

    if _is_primary_process() and checkpoint_dir is not None:
        _save_training_checkpoint(
            checkpoint_dir=Path(checkpoint_dir).expanduser(),
            step=config.total_steps - 1,
            phase_start_step=resolved_phase_start_step,
            model=model,
            optimizer=optimizer,
            ema_loss=ema_loss,
            config=config,
            keep_last_checkpoints=keep_last_checkpoints,
        )

    out_model = _unwrap_model(model)
    return {
        "model": out_model if _is_primary_process() else None,
        "history": history,
        "config": asdict(config),
        "start_step": start_step,
        "phase_start_step": resolved_phase_start_step,
        "phase_local_schedule": phase_local_schedule,
        "initialized_from": init_path_str,
        "resumed_from": None if resume_path is None else str(resume_path),
    }
