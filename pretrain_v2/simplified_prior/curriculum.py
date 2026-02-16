from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .generator import SimplifiedPriorConfig, generate_simplified_prior_data


@dataclass(frozen=True)
class CurriculumBounds:
    num_layers_min: int
    num_layers_max: int
    hidden_dim_min: int
    hidden_dim_max: int


@dataclass(frozen=True)
class PUCurriculumSchedule:
    """Stage schedule for PU-composition controls.

    The ranges widen linearly by stage from start -> end.
    """

    unlabeled_ratio_start: Tuple[float, float] = (1.0, 1.0)
    unlabeled_ratio_end: Tuple[float, float] = (0.75, 3.0)
    test_class1_ratio_start: Tuple[float, float] = (0.5, 0.5)
    test_class1_ratio_end: Tuple[float, float] = (0.2, 0.8)


GROUP1_STAGE_FACTORS = {
    "is_causal",
    "num_layers",
    "hidden_dim",
    "unlabeled_to_positive_ratio",
    "test_class1_ratio",
}
STAGE_CONTROLLED_FACTORS = GROUP1_STAGE_FACTORS


def _is_sequence_like(x: object) -> bool:
    if isinstance(x, (str, bytes, dict)):
        return False
    return isinstance(x, (list, tuple, np.ndarray))


def _sample_value(spec: object, rng: np.random.Generator) -> object:
    if callable(spec):
        return spec(rng)
    if _is_sequence_like(spec):
        values = list(spec)  # type: ignore[arg-type]
        if len(values) == 0:
            raise ValueError("Stationary sampler list cannot be empty.")
        idx = int(rng.integers(0, len(values)))
        return values[idx]
    return spec


def sample_stationary_hyperparameters(
    base_cfg: SimplifiedPriorConfig,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Sample stage-invariant fields from a stationary sampler."""
    if rng is None:
        rng = np.random.default_rng()

    cfg_dict = asdict(base_cfg)
    sampler = stationary_sampler or {}
    for key, spec in sampler.items():
        if key in STAGE_CONTROLLED_FACTORS:
            raise ValueError(f"'{key}' is stage-controlled and cannot be in stationary_sampler.")
        if key not in cfg_dict:
            raise ValueError(f"Unknown config key in stationary_sampler: '{key}'")
        cfg_dict[key] = _sample_value(spec, rng)
    return cfg_dict


def is_causal_false_probability(stage_idx: int, total_stages: int) -> float:
    """P(is_causal=False) at stage s in {1,...,K}: 1 - s/(2K)."""
    if total_stages < 1:
        raise ValueError("total_stages must be >= 1.")
    if not (1 <= stage_idx <= total_stages):
        raise ValueError("stage_idx must satisfy 1 <= stage_idx <= total_stages.")
    prob = 1.0 - (stage_idx / (2.0 * total_stages))
    return float(np.clip(prob, 0.0, 1.0))


def stage_linear_value(stage_idx: int, total_stages: int, start: float, end: float) -> float:
    """Linear stage schedule from `start` (stage 1) to `end` (stage K)."""
    if total_stages < 1:
        raise ValueError("total_stages must be >= 1.")
    if not (1 <= stage_idx <= total_stages):
        raise ValueError("stage_idx must satisfy 1 <= stage_idx <= total_stages.")

    if total_stages == 1:
        return float(end)
    frac = (stage_idx - 1) / (total_stages - 1)
    return float((1.0 - frac) * start + frac * end)


def _sample_stage_range(
    stage_idx: int,
    total_stages: int,
    start: Tuple[float, float],
    end: Tuple[float, float],
    rng: np.random.Generator,
) -> float:
    lo = stage_linear_value(stage_idx=stage_idx, total_stages=total_stages, start=float(start[0]), end=float(end[0]))
    hi = stage_linear_value(stage_idx=stage_idx, total_stages=total_stages, start=float(start[1]), end=float(end[1]))
    if lo > hi:
        raise ValueError(f"Invalid stage range: lower {lo} > upper {hi}.")
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def sample_curriculum_config(
    base_cfg: SimplifiedPriorConfig,
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    pu_schedule: Optional[PUCurriculumSchedule] = None,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
) -> SimplifiedPriorConfig:
    """Sample one stage-specific config for v2 pretraining."""
    if rng is None:
        rng = np.random.default_rng()
    pu_schedule = pu_schedule if pu_schedule is not None else PUCurriculumSchedule()

    p_false = is_causal_false_probability(stage_idx=stage_idx, total_stages=total_stages)
    sampled_is_causal = bool(rng.random() >= p_false)

    # Full last-block ranges are used from stage 1 (no progressive widening for these).
    sampled_num_layers = int(rng.integers(int(bounds.num_layers_min), int(bounds.num_layers_max) + 1))
    sampled_hidden_dim = int(rng.integers(int(bounds.hidden_dim_min), int(bounds.hidden_dim_max) + 1))

    sampled_unlabeled_ratio = _sample_stage_range(
        stage_idx=stage_idx,
        total_stages=total_stages,
        start=pu_schedule.unlabeled_ratio_start,
        end=pu_schedule.unlabeled_ratio_end,
        rng=rng,
    )
    sampled_test_class1_ratio = _sample_stage_range(
        stage_idx=stage_idx,
        total_stages=total_stages,
        start=pu_schedule.test_class1_ratio_start,
        end=pu_schedule.test_class1_ratio_end,
        rng=rng,
    )

    cfg_dict = sample_stationary_hyperparameters(
        base_cfg=base_cfg,
        stationary_sampler=stationary_sampler,
        rng=rng,
    )
    cfg_dict["difficulty"] = None
    cfg_dict["is_causal"] = sampled_is_causal
    cfg_dict["num_layers"] = sampled_num_layers
    cfg_dict["hidden_dim"] = sampled_hidden_dim
    cfg_dict["unlabeled_to_positive_ratio"] = sampled_unlabeled_ratio
    cfg_dict["test_class1_ratio"] = sampled_test_class1_ratio

    return SimplifiedPriorConfig(**cfg_dict)


def generate_curriculum_stage_batch(
    base_cfg: SimplifiedPriorConfig,
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    num_datasets: int,
    pu_schedule: Optional[PUCurriculumSchedule] = None,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[SimplifiedPriorConfig, Dict[str, object]]:
    """Sample a stage config and generate datasets with that config."""
    stage_cfg = sample_curriculum_config(
        base_cfg=base_cfg,
        stage_idx=stage_idx,
        total_stages=total_stages,
        bounds=bounds,
        pu_schedule=pu_schedule,
        stationary_sampler=stationary_sampler,
        rng=rng,
    )
    batch = generate_simplified_prior_data(stage_cfg, num_datasets=num_datasets)
    return stage_cfg, batch
