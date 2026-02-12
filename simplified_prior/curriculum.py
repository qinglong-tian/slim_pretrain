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


GROUP1_STAGE_FACTORS = {"is_causal", "num_layers", "hidden_dim"}
STAGE_CONTROLLED_FACTORS = GROUP1_STAGE_FACTORS | {"pu_keep_probability"}


def _is_sequence_like(x: object) -> bool:
    if isinstance(x, (str, bytes, dict)):
        return False
    return isinstance(x, (list, tuple, np.ndarray))


def _sample_value(spec: object, rng: np.random.Generator) -> object:
    """Sample a value from a stationary sampler spec.

    Allowed specs:
    - Callable: f(rng) -> value
    - Sequence: sample uniformly from values
    - Scalar/object: treated as constant
    """
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
    """Sample group-2 hyperparameters with a stage-invariant sampler.

    Keys in `stationary_sampler` can target any config field except stage-controlled
    factors (`is_causal`, `num_layers`, `hidden_dim`, `pu_keep_probability`).
    Keys not listed keep base_cfg values.
    """
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
    """P(is_causal=False) at stage s in {1,...,K}: 1 - (s-1)/(2K)."""
    if total_stages < 1:
        raise ValueError("total_stages must be >= 1.")
    if not (1 <= stage_idx <= total_stages):
        raise ValueError("stage_idx must satisfy 1 <= stage_idx <= total_stages.")
    prob = 1.0 - ((stage_idx - 1) / (2.0 * total_stages))
    return float(np.clip(prob, 0.0, 1.0))


def stage_upper_limit(stage_idx: int, total_stages: int, lo: int, hi: int) -> int:
    """Linear growth of upper limit from lo to hi over stages."""
    if total_stages < 1:
        raise ValueError("total_stages must be >= 1.")
    if not (1 <= stage_idx <= total_stages):
        raise ValueError("stage_idx must satisfy 1 <= stage_idx <= total_stages.")
    if lo > hi:
        raise ValueError("Lower bound must be <= upper bound.")

    if total_stages == 1:
        return int(hi)
    frac = (stage_idx - 1) / (total_stages - 1)
    return int(round(lo + frac * (hi - lo)))


def stage_linear_probability(stage_idx: int, total_stages: int, start: float, end: float) -> float:
    """Linear stage schedule from `start` (stage 1) to `end` (stage K)."""
    if total_stages < 1:
        raise ValueError("total_stages must be >= 1.")
    if not (1 <= stage_idx <= total_stages):
        raise ValueError("stage_idx must satisfy 1 <= stage_idx <= total_stages.")
    if not (0.0 <= start <= 1.0 and 0.0 <= end <= 1.0):
        raise ValueError("start and end must be in [0, 1].")

    if total_stages == 1:
        return float(end)
    frac = (stage_idx - 1) / (total_stages - 1)
    return float((1.0 - frac) * start + frac * end)


def sample_curriculum_config(
    base_cfg: SimplifiedPriorConfig,
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
) -> SimplifiedPriorConfig:
    """Sample one stage-specific config.

    Group 1 (changes by stage):
    - is_causal
    - num_layers
    - hidden_dim

    Group 2 (stationary sampler): all other fields are sampled with the same
    sampler across all stages (`stationary_sampler`).
    PU behavior is fixed to always-PU (pu_keep_probability=0.0) and is not part
    of the curriculum schedule.
    """
    if rng is None:
        rng = np.random.default_rng()

    p_false = is_causal_false_probability(stage_idx=stage_idx, total_stages=total_stages)
    sampled_is_causal = bool(rng.random() >= p_false)

    layer_upper = stage_upper_limit(
        stage_idx=stage_idx,
        total_stages=total_stages,
        lo=int(bounds.num_layers_min),
        hi=int(bounds.num_layers_max),
    )
    hidden_upper = stage_upper_limit(
        stage_idx=stage_idx,
        total_stages=total_stages,
        lo=int(bounds.hidden_dim_min),
        hi=int(bounds.hidden_dim_max),
    )

    sampled_num_layers = int(rng.integers(int(bounds.num_layers_min), int(layer_upper) + 1))
    sampled_hidden_dim = int(rng.integers(int(bounds.hidden_dim_min), int(hidden_upper) + 1))

    cfg_dict = sample_stationary_hyperparameters(
        base_cfg=base_cfg,
        stationary_sampler=stationary_sampler,
        rng=rng,
    )
    cfg_dict["difficulty"] = None
    cfg_dict["is_causal"] = sampled_is_causal
    cfg_dict["num_layers"] = sampled_num_layers
    cfg_dict["hidden_dim"] = sampled_hidden_dim
    cfg_dict["pu_keep_probability"] = 0.0
    return SimplifiedPriorConfig(**cfg_dict)


def generate_curriculum_stage_batch(
    base_cfg: SimplifiedPriorConfig,
    stage_idx: int,
    total_stages: int,
    bounds: CurriculumBounds,
    num_datasets: int,
    stationary_sampler: Optional[Dict[str, object]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[SimplifiedPriorConfig, Dict[str, object]]:
    """Sample a stage config and generate datasets with that config."""
    stage_cfg = sample_curriculum_config(
        base_cfg=base_cfg,
        stage_idx=stage_idx,
        total_stages=total_stages,
        bounds=bounds,
        stationary_sampler=stationary_sampler,
        rng=rng,
    )
    batch = generate_simplified_prior_data(stage_cfg, num_datasets=num_datasets)
    return stage_cfg, batch
