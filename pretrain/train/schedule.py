from __future__ import annotations

import math


def stage_index_from_step(step: int, steps_per_stage: int, total_stages: int) -> int:
    """Map 0-based global step to 1-based stage index."""
    if step < 0:
        raise ValueError("step must be >= 0.")
    if steps_per_stage <= 0:
        raise ValueError("steps_per_stage must be > 0.")
    if total_stages <= 0:
        raise ValueError("total_stages must be > 0.")

    stage = (step // steps_per_stage) + 1
    return int(min(stage, total_stages))


def warmup_cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    """Common schedule for curriculum-style training: warmup then cosine decay."""
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0.")
    if not (0.0 <= min_lr <= base_lr):
        raise ValueError("Require 0 <= min_lr <= base_lr.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0.")

    s = max(0, min(step, total_steps - 1))
    if warmup_steps > 0 and s < warmup_steps:
        return float(base_lr * (s + 1) / warmup_steps)

    decay_den = max(1, total_steps - warmup_steps)
    progress = (s - warmup_steps) / decay_den
    progress = max(0.0, min(1.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + (base_lr - min_lr) * cosine)
