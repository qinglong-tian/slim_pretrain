from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from slim_pretrain.pretrain.data import VariableBatchSpec
from slim_pretrain.simplified_prior import CurriculumBounds, SimplifiedPriorConfig


@dataclass(frozen=True)
class ModelConfig:
    embedding_size: int = 128
    num_attention_heads: int = 8
    mlp_hidden_size: int = 256
    num_layers: int = 6
    num_outputs: int = 2


@dataclass(frozen=True)
class OptimConfig:
    base_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 8000
    decay_power: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class DataCurriculumConfig:
    total_stages: int = 10
    steps_per_stage: int = 1000
    bounds: CurriculumBounds = field(
        default_factory=lambda: CurriculumBounds(
            num_layers_min=2,
            num_layers_max=8,
            hidden_dim_min=8,
            hidden_dim_max=16,
        )
    )
    stationary_sampler: Dict[str, object] = field(
        default_factory=lambda: {
            "noise_std": [0.005, 0.01, 0.02],
            "sampling": ["normal", "uniform"],
            "per_layer_activation": [False, True],
            "y_is_effect": [False, True],
            "in_clique": [False, True],
            "sort_features": [False, True],
            "balanced_labels": [True],
        }
    )
    batch_spec: VariableBatchSpec = field(
        default_factory=lambda: VariableBatchSpec(
            batch_size=16,
            seq_len_range=(128, 320),
            num_features_range=(8, 48),
            train_ratio_range=(0.6, 0.8),
            pu_row_policy="drop",
        )
    )

    @property
    def total_steps(self) -> int:
        return int(self.total_stages * self.steps_per_stage)


@dataclass(frozen=True)
class PretrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataCurriculumConfig = field(default_factory=DataCurriculumConfig)
    device: str = "cpu"
    seed: int = 0
    log_every: int = 100
    max_steps: Optional[int] = None
    ema_decay: float = 0.95
    fixed_batch_seed: Optional[int] = None
    eval_every: int = 0
    eval_batches: int = 0
    eval_seed: int = 314159
    eval_batch_spec: Optional[VariableBatchSpec] = None

    @property
    def total_steps(self) -> int:
        if self.max_steps is not None:
            return int(self.max_steps)
        return int(self.data.total_steps)


def default_base_prior_config() -> SimplifiedPriorConfig:
    """Base prior config for pretraining data generation.

    PU is always on by default (`pu_keep_probability=0.0`).
    """
    return SimplifiedPriorConfig(
        seq_len=256,
        train_size=0.7,
        num_features=20,
        num_causes=20,
        num_layers=4,
        hidden_dim=32,
        is_causal=False,
        noncausal_feature_source="head",
        y_is_effect=True,
        in_clique=False,
        sort_features=True,
        nonlinearities=(
            "tanh",
            "relu",
            "gelu",
            "sine",
        ),
        per_layer_activation=True,
        noise_std=0.01,
        init_std=0.8,
        sampling="normal",
        balanced_labels=True,
        pu_keep_probability=0.0,
        seed=0,
        device="cpu",
        difficulty=None,
    )
