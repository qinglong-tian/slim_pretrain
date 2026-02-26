from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from ..data import VariableBatchSpec
from ..simplified_prior import CurriculumBounds, PUCurriculumSchedule, SimplifiedPriorConfig


@dataclass(frozen=True)
class ModelConfig:
    embedding_size: int = 128
    num_attention_heads: int = 8
    mlp_hidden_size: int = 256
    num_layers: int = 6
    num_outputs: int = 2
    max_categorical_classes: int = 64


@dataclass(frozen=True)
class OptimConfig:
    base_lr: float = 4e-5
    min_lr: float = 4e-6
    weight_decay: float = 1e-4
    warmup_steps: int = 4000
    decay_power: float = 1.5
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class DataCurriculumConfig:
    total_stages: int = 35
    steps_per_stage: int = 2000
    bounds: CurriculumBounds = field(
        default_factory=lambda: CurriculumBounds(
            num_layers_min=4,
            num_layers_max=12,
            hidden_dim_min=12,
            hidden_dim_max=36,
        )
    )
    pu_schedule: PUCurriculumSchedule = field(default_factory=PUCurriculumSchedule)
    stationary_sampler: Dict[str, object] = field(
        default_factory=lambda: {
            "noise_std": [0.005, 0.01, 0.02],
            "sampling": ["normal", "uniform"],
            "per_layer_activation": [True],
            "y_is_effect": [False, True],
            "in_clique": [False, True],
            "sort_features": [False, True],
        }
    )
    batch_spec: VariableBatchSpec = field(
        default_factory=lambda: VariableBatchSpec(
            batch_size=4,
            num_features_range=(8, 24),
            positive_size_range=(300, 500),
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
    """Base prior config for v2 PU pretraining data generation."""
    return SimplifiedPriorConfig(
        seq_len=1100,
        train_size=0.7,
        min_test_size=1,
        split_strategy="stratified",
        positive_train_size=160,
        unlabeled_to_positive_ratio=1.0,
        test_class1_ratio=0.5,
        class1_ratio=0.5,
        num_features=20,
        num_causes=20,
        num_layers=8,
        hidden_dim=24,
        is_causal=False,
        noncausal_feature_source="head",
        y_is_effect=True,
        in_clique=False,
        sort_features=True,
        categorical_feature_ratio_range=(0.0, 1.0),
        categorical_cardinality_range=(2, 10),
        nonlinearities=(
            "tanh",
            "relu",
            "gelu",
            "identity",
            "sign",
            "heaviside",
            "rbf",
            "sine",
            "square",
            "abs",
        ),
        per_layer_activation=True,
        noise_std=0.01,
        init_std=0.8,
        sampling="normal",
        seed=0,
        device="cpu",
        difficulty=None,
    )
